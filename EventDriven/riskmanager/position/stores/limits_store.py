"""Persistence layer for position limits and sizing metadata.

Separates save/get concerns from limits cogs. In-memory stores mirror the
dict-based behavior in ``LimitsAndSizingCog``; database stores delegate to
the ``limits`` table and ``position_metadata`` table in ``portfolio_data``.

Public API:
    Reads and in-memory writes are keyed by ``(signal_id, trade_id)``.
    Database stores also require ``strategy_name`` at construction time.
    Metadata saves carry ``signal_id`` inside the payload; limits saves require
    ``signal_id`` explicitly. Underlying tickers are parsed from ``trade_id``.

Post-save verification:
    Database stores re-read persisted rows after each write. On mismatch,
    the local cache entry is dropped and ``StoreVerificationError`` is raised
    so callers do not proceed with unverified state. Recommended caller
    behavior: log the failure, skip or halt sizing for that trade, and retry
    once; do not silently fall back to in-memory limits in live trading.

DB persistence toggle:
    ``DatabaseLimitsStore.PERSIST_TO_DB`` and ``DatabaseMetadataStore.PERSIST_TO_DB``
    are class-level flags (default ``True``). Use ``disable_storing_to_db``,
    ``enable_storing_to_db``, and ``reset_storing_to_db`` to toggle globally.
"""

from __future__ import annotations

import hashlib
import json
import math
from abc import ABC, abstractmethod
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from dbase.database.SQLHelpers import DatabaseAdapter, dynamic_batch_update, get_engine
from sqlalchemy import text
from EventDriven.dataclasses.limits import PositionLimits
from EventDriven.riskmanager.position.cogs.vars import MEASURES_SET
from EventDriven.types import TradeID
from EventDriven.riskmanager.position.live_cogs.save_utils import store_position_limits
from trade.helpers.Logging import setup_logger
from trade.helpers.helper import to_datetime

logger = setup_logger("EventDriven.riskmanager.position.stores.limits_store")

PORTFOLIO_DATA_DB = "portfolio_data"
POSITION_METADATA_TABLE = "position_metadata"
METADATA_TYPE_LIMITS = "LIMITS"
METADATA_TYPE_METADATA = "METADATA"
# The live DB column is misspelled; keep aligned with the schema.
POSITION_METADATA_SIGNAL_ID_COL = "signal_id"
LIMITS_TABLE = "limits"
PositionStoreKey = Tuple[str, str]


def _store_key(signal_id: str, trade_id: str) -> PositionStoreKey:
    """Build the canonical in-memory store key.

    Args:
        signal_id: Originating signal identifier.
        trade_id: Unique trade identifier.

    Returns:
        Tuple key ``(signal_id, trade_id)``.
    """
    return (signal_id, trade_id)


class StoreVerificationError(RuntimeError):
    """Raised when a database write cannot be confirmed by a post-save read."""

    def __init__(
        self,
        store_type: str,
        trade_id: str,
        errors: List[str],
        *,
        strategy_name: Optional[str] = None,
    ) -> None:
        """Initialize the verification error.

        Args:
            store_type: Store label, e.g. ``limits`` or ``metadata``.
            trade_id: Trade identifier associated with the failed write.
            errors: Human-readable verification failure messages.
            strategy_name: Strategy identifier when the failure occurred in a DB store.
        """
        self.store_type = store_type
        self.trade_id = trade_id
        self.strategy_name = strategy_name
        self.errors = errors
        scope = f"strategy_name={strategy_name}, " if strategy_name else ""
        message = (
            f"{store_type} store verification failed for {scope}trade_id={trade_id}: "
            f"{'; '.join(errors)}"
        )
        super().__init__(message)


class LimitsStore(ABC):
    """Abstract store for ``PositionLimits`` persistence."""

    @abstractmethod
    def save(self, trade_id: str, signal_id: str, limits: PositionLimits) -> None:
        """Persist position limits for a trade.

        Args:
            trade_id: Unique trade identifier.
            signal_id: Originating signal identifier.
            limits: Limit values to store.
        """

    @abstractmethod
    def get(self, trade_id: str, signal_id: str) -> Optional[PositionLimits]:
        """Retrieve position limits for a trade.

        Args:
            trade_id: Unique trade identifier.
            signal_id: Originating signal identifier.

        Returns:
            Stored limits, or ``None`` when no record exists.
        """


class MetadataStore(ABC):
    """Abstract store for position sizing metadata persistence."""

    @abstractmethod
    def save(self, trade_id: str, metadata: Any) -> None:
        """Persist metadata for a trade.

        Args:
            trade_id: Unique trade identifier.
            metadata: Dataclass or dict payload to store; must include ``signal_id``.
        """

    @abstractmethod
    def get(self, trade_id: str, signal_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve metadata for a trade.

        Args:
            trade_id: Unique trade identifier.
            signal_id: Originating signal identifier.

        Returns:
            Metadata dictionary, or ``None`` when no record exists.
        """


class PositionStore(ABC):
    """Combined limits and metadata store interface."""

    limits: LimitsStore
    metadata: MetadataStore

    def save_limits(self, trade_id: str, signal_id: str, limits: PositionLimits) -> None:
        """Persist position limits for a trade.

        Args:
            trade_id: Unique trade identifier.
            signal_id: Originating signal identifier.
            limits: Limit values to store.
        """
        self.limits.save(trade_id, signal_id, limits)

    def get_limits(self, trade_id: str, signal_id: str) -> Optional[PositionLimits]:
        """Retrieve position limits for a trade.

        Args:
            trade_id: Unique trade identifier.
            signal_id: Originating signal identifier.

        Returns:
            Stored limits, or ``None`` when no record exists.
        """
        return self.limits.get(trade_id, signal_id)

    def save_metadata(self, trade_id: str, metadata: Any) -> None:
        """Persist metadata for a trade.

        Args:
            trade_id: Unique trade identifier.
            metadata: Dataclass or dict payload to store; must include ``signal_id``.
        """
        self.metadata.save(trade_id, metadata)

    def get_metadata(self, trade_id: str, signal_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve metadata for a trade.

        Args:
            trade_id: Unique trade identifier.
            signal_id: Originating signal identifier.

        Returns:
            Metadata dictionary, or ``None`` when no record exists.
        """
        return self.metadata.get(trade_id, signal_id)


class InMemoryLimitsStore(LimitsStore):
    """Dict-backed limits store used by backtests."""

    def __init__(self) -> None:
        """Initialize an empty in-memory limits cache."""
        self._limits: Dict[PositionStoreKey, PositionLimits] = {}

    def save(self, trade_id: str, signal_id: str, limits: PositionLimits) -> None:
        """Store limits keyed by ``(signal_id, trade_id)``.

        Args:
            trade_id: Unique trade identifier.
            signal_id: Originating signal identifier.
            limits: Limit values to store.
        """
        self._limits[_store_key(signal_id, trade_id)] = limits

    def get(self, trade_id: str, signal_id: str) -> Optional[PositionLimits]:
        """Return cached limits for ``(signal_id, trade_id)``.

        Args:
            trade_id: Unique trade identifier.
            signal_id: Originating signal identifier.

        Returns:
            Cached limits, or ``None`` when absent.
        """
        return self._limits.get(_store_key(signal_id, trade_id))


class InMemoryMetadataStore(MetadataStore):
    """Dict-backed metadata store used by backtests."""

    def __init__(self) -> None:
        """Initialize an empty in-memory metadata cache."""
        self._metadata: Dict[PositionStoreKey, Any] = {}

    def save(self, trade_id: str, metadata: Any) -> None:
        """Store metadata keyed by ``(signal_id, trade_id)``.

        Args:
            trade_id: Unique trade identifier.
            metadata: Dataclass or dict payload to store; must include ``signal_id``.
        """
        signal_id = _signal_id_from_metadata(metadata)
        self._metadata[_store_key(signal_id, trade_id)] = metadata

    def get(self, trade_id: str, signal_id: str) -> Optional[Dict[str, Any]]:
        """Return cached metadata for ``(signal_id, trade_id)``.

        Args:
            trade_id: Unique trade identifier.
            signal_id: Originating signal identifier.

        Returns:
            Stored metadata object or dict, or ``None`` when absent.
        """
        value = self._metadata.get(_store_key(signal_id, trade_id))
        if value is None:
            return None
        if isinstance(value, dict):
            return value
        if is_dataclass(value):
            return metadata_to_dict(value)
        return {"value": value}


class InMemoryPositionStore(PositionStore):
    """Combined in-memory store for limits and metadata."""

    def __init__(self) -> None:
        """Initialize empty in-memory limits and metadata stores."""
        self.limits = InMemoryLimitsStore()
        self.metadata = InMemoryMetadataStore()


class DatabaseLimitsStore(LimitsStore):
    """Database-backed limits store with optional in-memory cache."""

    PERSIST_TO_DB: bool = True

    def __init__(
        self,
        strategy_name: str,
        *,
        default_dte: Optional[int] = None,
        default_moneyness: Optional[float] = None,
        use_cache: bool = True,
        verify_after_save: bool = True,
        db: Optional[DatabaseAdapter] = None,
    ) -> None:
        """Initialize the database limits store.

        Args:
            strategy_name: Strategy identifier used in the limits table.
            default_dte: Fallback DTE when limits are loaded from the database.
            default_moneyness: Fallback moneyness when limits are loaded from the database.
            use_cache: When ``True``, cache reads and write-through saves locally.
            verify_after_save: Re-read the database after each write and raise on mismatch.
            db: Optional database adapter override for testing.
        """
        self.strategy_name = strategy_name
        self.default_dte = default_dte
        self.default_moneyness = default_moneyness
        self.use_cache = use_cache
        self.verify_after_save = verify_after_save
        self._db = db or DatabaseAdapter()
        self._cache: Dict[PositionStoreKey, PositionLimits] = {}

    def save(self, trade_id: str, signal_id: str, limits: PositionLimits) -> None:
        """Persist limits to the database and optional cache.

        Args:
            trade_id: Unique trade identifier.
            signal_id: Originating signal identifier.
            limits: Limit values to store.

        Raises:
            StoreVerificationError: If post-save verification fails.
        """
        cache_key = _store_key(signal_id, trade_id)
        if self.use_cache:
            self._cache[cache_key] = limits

        if not self.PERSIST_TO_DB:
            return

        store_position_limits(
            delta_limit=limits.delta,
            gamma_limit=limits.gamma,
            vega_limit=limits.vega,
            theta_limit=limits.theta,
            trade_id=trade_id,
            signal_id=signal_id,
            strategy_name=self.strategy_name,
            date=limits.creation_date,
        )

        if not self.verify_after_save:
            return

        errors = _collect_limits_verification_errors(
            trade_id=trade_id,
            signal_id=signal_id,
            strategy_name=self.strategy_name,
            expected=limits,
        )
        if errors:
            self._cache.pop(cache_key, None)
            _raise_store_verification_error(
                "limits",
                trade_id,
                errors,
                strategy_name=self.strategy_name,
                signal_id=signal_id,
            )

    def get(self, trade_id: str, signal_id: str) -> Optional[PositionLimits]:
        """Load limits from cache or the database.

        Args:
            trade_id: Unique trade identifier.
            signal_id: Originating signal identifier.

        Returns:
            Position limits with configured defaults applied, or ``None`` when absent.
        """
        cache_key = _store_key(signal_id, trade_id)
        if self.use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        limits = _limits_from_rows(
            _load_limits_rows(
                trade_id=trade_id,
                signal_id=signal_id,
                strategy_name=self.strategy_name,
            )
        )
        if limits is None:
            return None

        limits.dte = self.default_dte
        limits.moneyness = self.default_moneyness

        if self.use_cache:
            self._cache[cache_key] = limits

        return limits


class DatabaseMetadataStore(MetadataStore):
    """Database-backed metadata store using ``position_metadata``."""

    PERSIST_TO_DB: bool = True

    def __init__(
        self,
        strategy_name: str,
        *,
        use_cache: bool = True,
        enabled: bool = True,
        verify_after_save: bool = True,
        db: Optional[DatabaseAdapter] = None,
    ) -> None:
        """Initialize the database metadata store.

        Args:
            strategy_name: Strategy identifier used in ``position_metadata``.
            use_cache: When ``True``, cache reads and write-through saves locally.
            enabled: Value stored in the ``enabled`` column for new rows.
            verify_after_save: Re-read the database after each write and raise on mismatch.
            db: Optional database adapter override for testing.
        """
        self.strategy_name = strategy_name
        self.use_cache = use_cache
        self.enabled = enabled
        self.verify_after_save = verify_after_save
        self._db = db or DatabaseAdapter()
        self._cache: Dict[PositionStoreKey, Dict[str, Any]] = {}

    def save(self, trade_id: str, metadata: Any) -> None:
        """Persist metadata to ``position_metadata``.

        Args:
            trade_id: Unique trade identifier.
            metadata: Dataclass or dict payload to store; must include ``signal_id``.

        Raises:
            ValueError: If ``signal_id`` cannot be resolved from ``metadata``.
            StoreVerificationError: If post-save verification fails.
        """
        metadata_dict = metadata_to_dict(metadata)
        metadata_date = _resolve_metadata_date(metadata, metadata_dict)
        signal_id = _signal_id_from_metadata(metadata, metadata_dict)
        cache_key = _store_key(signal_id, trade_id)

        if self.use_cache:
            self._cache[cache_key] = metadata_dict

        if not self.PERSIST_TO_DB:
            return

        ticker = ticker_from_trade_id(trade_id)
        metadata_hash = compute_metadata_hash(metadata_dict)
        row = {
            POSITION_METADATA_SIGNAL_ID_COL: signal_id,
            "trade_id": trade_id,
            "ticker": ticker,
            "strategy_name": self.strategy_name,
            "date": metadata_date,
            "metadata_type": METADATA_TYPE_METADATA,
            "metadata_dict": json.dumps(metadata_dict),
            "metadata_hash": metadata_hash,
            "enabled": int(self.enabled),
        }

        if _metadata_row_exists(
            signal_id=signal_id,
            trade_id=trade_id,
            ticker=ticker,
            strategy_name=self.strategy_name,
            metadata_date=metadata_date,
            metadata_type=METADATA_TYPE_METADATA,
        ):
            dynamic_batch_update(
                db=PORTFOLIO_DATA_DB,
                table_name=POSITION_METADATA_TABLE,
                update_values={
                    "metadata_dict": row["metadata_dict"],
                    "metadata_hash": row["metadata_hash"],
                    "enabled": row["enabled"],
                },
                condition={
                    POSITION_METADATA_SIGNAL_ID_COL: signal_id,
                    "trade_id": trade_id,
                    "ticker": ticker,
                    "strategy_name": self.strategy_name,
                    "date": metadata_date,
                    "metadata_type": METADATA_TYPE_METADATA,
                },
            )
            logger.info(
                "Updated position metadata for trade_id=%s, strategy_name=%s, ticker=%s, date=%s",
                trade_id,
                self.strategy_name,
                ticker,
                metadata_date,
            )
        else:
            self._db.save_to_database(
                db=PORTFOLIO_DATA_DB,
                table_name=POSITION_METADATA_TABLE,
                data=pd.DataFrame([row]),
                filter_data=False,
            )
            logger.info(
                "Stored position metadata for trade_id=%s, strategy_name=%s, ticker=%s, date=%s",
                trade_id,
                self.strategy_name,
                ticker,
                metadata_date,
            )

        if not self.verify_after_save:
            return

        errors = _collect_metadata_verification_errors(
            trade_id=trade_id,
            signal_id=signal_id,
            strategy_name=self.strategy_name,
            metadata_date=metadata_date,
            expected_metadata=metadata_dict,
            expected_hash=metadata_hash,
            expected_enabled=int(self.enabled),
        )
        if errors:
            self._cache.pop(cache_key, None)
            _raise_store_verification_error(
                "metadata",
                trade_id,
                errors,
                strategy_name=self.strategy_name,
                signal_id=signal_id,
            )

    def get(self, trade_id: str, signal_id: str) -> Optional[Dict[str, Any]]:
        """Load metadata from cache or the database.

        Args:
            trade_id: Unique trade identifier.
            signal_id: Originating signal identifier.

        Returns:
            Metadata dictionary, or ``None`` when no record exists.
        """
        cache_key = _store_key(signal_id, trade_id)
        if self.use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        ticker = ticker_from_trade_id(trade_id)
        row = _load_latest_metadata_row(
            trade_id=trade_id,
            signal_id=signal_id,
            ticker=ticker,
            strategy_name=self.strategy_name,
            metadata_type=METADATA_TYPE_METADATA,
        )
        if row is None:
            return None

        metadata_dict = _parse_metadata_dict(row["metadata_dict"])
        if self.use_cache:
            self._cache[cache_key] = metadata_dict
        return metadata_dict


class DatabasePositionStore(PositionStore):
    """Combined database-backed store for limits and metadata."""

    def __init__(
        self,
        strategy_name: str,
        *,
        default_dte: Optional[int] = None,
        default_moneyness: Optional[float] = None,
        use_cache: bool = True,
        metadata_enabled: bool = True,
        verify_after_save: bool = True,
        db: Optional[DatabaseAdapter] = None,
    ) -> None:
        """Initialize combined database stores.

        Args:
            strategy_name: Strategy identifier used in database tables.
            default_dte: Fallback DTE when loading limits from the database.
            default_moneyness: Fallback moneyness when loading limits from the database.
            use_cache: Enable in-memory caching for both stores.
            metadata_enabled: Value stored in the ``enabled`` column for metadata rows.
            verify_after_save: Re-read the database after each write and raise on mismatch.
            db: Optional database adapter override for testing.
        """
        adapter = db or DatabaseAdapter()
        self.limits = DatabaseLimitsStore(
            strategy_name=strategy_name,
            default_dte=default_dte,
            default_moneyness=default_moneyness,
            use_cache=use_cache,
            verify_after_save=verify_after_save,
            db=adapter,
        )
        self.metadata = DatabaseMetadataStore(
            strategy_name=strategy_name,
            use_cache=use_cache,
            enabled=metadata_enabled,
            verify_after_save=verify_after_save,
            db=adapter,
        )


def enable_storing_to_db(*args: object, **kwargs: object) -> None:
    """Enable database persistence for limits and metadata stores globally."""
    DatabaseLimitsStore.PERSIST_TO_DB = True
    DatabaseMetadataStore.PERSIST_TO_DB = True


def disable_storing_to_db(*args: object, **kwargs: object) -> None:
    """Disable database persistence for limits and metadata stores globally."""
    DatabaseLimitsStore.PERSIST_TO_DB = False
    DatabaseMetadataStore.PERSIST_TO_DB = False


def reset_storing_to_db() -> None:
    """Reset database persistence for limits and metadata stores to default (enabled)."""
    DatabaseLimitsStore.PERSIST_TO_DB = True
    DatabaseMetadataStore.PERSIST_TO_DB = True


def build_position_store(
    *,
    live: bool = False,
    strategy_name: Optional[str] = None,
    default_dte: Optional[int] = None,
    default_moneyness: Optional[float] = None,
    position_store: Optional[PositionStore] = None,
    verify_after_save: bool = True,
) -> PositionStore:
    """Build an in-memory or database position store from a live flag.

    Args:
        live: When ``True``, return a ``DatabasePositionStore``; otherwise in-memory.
        strategy_name: Strategy identifier required for database stores.
        default_dte: Fallback DTE applied when loading limits from the database.
        default_moneyness: Fallback moneyness applied when loading limits from the database.
        position_store: Optional explicit store override (wins over ``live``).
        verify_after_save: Re-read the database after each write and raise on mismatch.

    Returns:
        Configured ``PositionStore`` implementation.

    Raises:
        ValueError: If ``live`` is ``True`` and ``strategy_name`` is missing.

    Examples:
        >>> store = build_position_store(live=False)
        >>> isinstance(store, InMemoryPositionStore)
        True
    """
    if position_store is not None:
        return position_store
    if live:
        if not strategy_name:
            raise ValueError("strategy_name is required when live=True")
        return DatabasePositionStore(
            strategy_name=strategy_name,
            default_dte=default_dte,
            default_moneyness=default_moneyness,
            verify_after_save=verify_after_save,
        )
    return InMemoryPositionStore()


def ticker_from_trade_id(trade_id: str) -> str:
    """Extract the underlying ticker from a trade identifier.

    Args:
        trade_id: Trade identifier in ``TradeID`` format.

    Returns:
        Underlying ticker parsed from the first long or short leg.

    Raises:
        ValueError: If no option legs are found in ``trade_id``.

    Examples:
        >>> ticker_from_trade_id("&L:AAPL20261218C380")
        'AAPL'
    """
    parsed = TradeID(trade_id)
    for side in ("L", "S"):
        legs = parsed.meta.get(side) or []
        if legs:
            return legs[0]["ticker"]
    raise ValueError(f"Could not extract ticker from trade_id: {trade_id}")


def _resolve_ticker(trade_id: str, ticker: Optional[str]) -> str:
    """Resolve ticker from an explicit value or trade identifier.

    Args:
        trade_id: Trade identifier used for parsing when ``ticker`` is omitted.
        ticker: Optional explicit ticker override.

    Returns:
        Resolved underlying ticker.
    """
    if ticker is not None:
        return ticker
    return ticker_from_trade_id(trade_id)


def _signal_id_from_metadata(metadata: Any, metadata_dict: Optional[Dict[str, Any]] = None) -> str:
    """Extract ``signal_id`` from a metadata payload.

    Args:
        metadata: Dataclass or dict payload.
        metadata_dict: Optional pre-serialized metadata dictionary.

    Returns:
        Originating signal identifier.

    Raises:
        ValueError: If ``signal_id`` is missing from the payload.
    """
    if is_dataclass(metadata):
        signal_id = getattr(metadata, "signal_id", None)
        if signal_id:
            return signal_id
    payload = metadata_dict if metadata_dict is not None else metadata_to_dict(metadata)
    signal_id = payload.get("signal_id")
    if not signal_id:
        raise ValueError("metadata must include signal_id for database persistence")
    return str(signal_id)


def metadata_to_dict(metadata: Any) -> Dict[str, Any]:
    """Convert a metadata dataclass or dict to a JSON-safe dictionary.

    Args:
        metadata: Dataclass instance or dictionary payload.

    Returns:
        JSON-serializable metadata dictionary.

    Raises:
        TypeError: If ``metadata`` is neither a dataclass nor a dict.
    """
    if is_dataclass(metadata):
        raw = asdict(metadata)
    elif isinstance(metadata, dict):
        raw = metadata
    else:
        raise TypeError(f"Unsupported metadata type: {type(metadata)}")

    return {key: _serialize_metadata_value(value) for key, value in raw.items()}


def compute_metadata_hash(metadata_dict: Dict[str, Any]) -> str:
    """Compute a stable hash for a metadata dictionary.

    Args:
        metadata_dict: JSON-serializable metadata payload.

    Returns:
        SHA-256 hex digest of the canonical JSON representation.
    """
    payload = json.dumps(metadata_dict, sort_keys=True, default=_json_default)
    return hashlib.sha256(payload.encode()).hexdigest()


def _serialize_metadata_value(value: Any) -> Any:
    """Convert metadata values into JSON-safe primitives.

    Args:
        value: Metadata field value.

    Returns:
        JSON-safe representation of ``value``.
    """
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _json_default(value: Any) -> str:
    """Serialize non-standard JSON values for hashing.

    Args:
        value: Value that cannot be encoded by the default JSON encoder.

    Returns:
        String representation of ``value``.
    """
    return _serialize_metadata_value(value) if isinstance(value, (datetime, date, pd.Timestamp)) else str(value)


def _resolve_metadata_date(metadata: Any, metadata_dict: Dict[str, Any]) -> Union[datetime, date]:
    """Resolve the metadata row date from payload fields.

    Args:
        metadata: Original metadata object passed to the store.
        metadata_dict: Serialized metadata dictionary.

    Returns:
        Date associated with the metadata row.

    Raises:
        ValueError: If no date can be resolved from the payload.
    """
    raw_date = metadata_dict.get("date")
    if raw_date is None and is_dataclass(metadata):
        raw_date = getattr(metadata, "date", None)
    if raw_date is None:
        raise ValueError("metadata must include a date for database persistence")

    if isinstance(raw_date, str):
        parsed = to_datetime(raw_date)
        return parsed.date() if hasattr(parsed, "date") else parsed
    if isinstance(raw_date, pd.Timestamp):
        return raw_date.date()
    if isinstance(raw_date, datetime):
        return raw_date.date()
    if isinstance(raw_date, date):
        return raw_date

    raise ValueError(f"Unsupported metadata date type: {type(raw_date)}")


def _parse_metadata_dict(raw_value: Any) -> Dict[str, Any]:
    """Parse a metadata JSON payload from the database.

    Args:
        raw_value: Raw ``metadata_dict`` column value.

    Returns:
        Parsed metadata dictionary.
    """
    if raw_value is None:
        return {}
    if isinstance(raw_value, dict):
        return raw_value
    if isinstance(raw_value, str):
        return json.loads(raw_value)
    return dict(raw_value)


def _metadata_row_exists(
    signal_id: str,
    trade_id: str,
    ticker: str,
    strategy_name: str,
    metadata_date: Union[datetime, date],
    metadata_type: str,
) -> bool:
    """Check whether a metadata row already exists for the composite key.

    Args:
        signal_id: Originating signal identifier.
        trade_id: Unique trade identifier.
        ticker: Underlying ticker.
        strategy_name: Strategy identifier.
        metadata_date: Metadata row date.
        metadata_type: ``LIMITS`` or ``METADATA`` enum value.

    Returns:
        ``True`` when a matching row exists.
    """
    query = text(f"""
        SELECT 1
        FROM `{POSITION_METADATA_TABLE}`
        WHERE `{POSITION_METADATA_SIGNAL_ID_COL}` = :signal_id
          AND `trade_id` = :trade_id
          AND `ticker` = :ticker
          AND `strategy_name` = :strategy_name
          AND `date` = :metadata_date
          AND `metadata_type` = :metadata_type
        LIMIT 1
    """)
    params = {
        "signal_id": signal_id,
        "trade_id": trade_id,
        "ticker": ticker,
        "strategy_name": strategy_name,
        "metadata_date": metadata_date,
        "metadata_type": metadata_type,
    }
    engine = get_engine(PORTFOLIO_DATA_DB)
    rows = pd.read_sql(query, engine, params=params)
    return len(rows) > 0


def _load_latest_metadata_row(
    trade_id: str,
    signal_id: str,
    ticker: str,
    strategy_name: str,
    metadata_type: str,
) -> Optional[Dict[str, Any]]:
    """Load the latest metadata row for a trade.

    Args:
        trade_id: Unique trade identifier.
        signal_id: Originating signal identifier.
        ticker: Underlying ticker parsed from ``trade_id``.
        strategy_name: Strategy identifier.
        metadata_type: ``LIMITS`` or ``METADATA`` enum value.

    Returns:
        Latest matching row as a dictionary, or ``None`` when absent.
    """
    query = text(f"""
        SELECT `metadata_dict`, `date`, `metadata_hash`, `enabled`
        FROM `{POSITION_METADATA_TABLE}`
        WHERE `{POSITION_METADATA_SIGNAL_ID_COL}` = :signal_id
          AND `trade_id` = :trade_id
          AND `ticker` = :ticker
          AND `strategy_name` = :strategy_name
          AND `metadata_type` = :metadata_type
        ORDER BY `date` DESC
        LIMIT 1
    """)
    params = {
        "signal_id": signal_id,
        "trade_id": trade_id,
        "ticker": ticker,
        "strategy_name": strategy_name,
        "metadata_type": metadata_type,
    }
    engine = get_engine(PORTFOLIO_DATA_DB)
    rows = pd.read_sql(query, engine, params=params)
    if len(rows) == 0:
        return None
    return rows.iloc[0].to_dict()


def _raise_store_verification_error(
    store_type: str,
    trade_id: str,
    errors: List[str],
    *,
    strategy_name: Optional[str] = None,
    signal_id: Optional[str] = None,
) -> None:
    """Log and raise a store verification failure.

    Recommended caller handling:
        1. Log the exception with trade context for operations review.
        2. Do not proceed with sizing or analysis for that trade in live mode.
        3. Retry the save once if the failure looks transient (connection blip).
        4. Escalate if repeated; do not silently substitute in-memory values live.

    Args:
        store_type: Store label, e.g. ``limits`` or ``metadata``.
        trade_id: Trade identifier associated with the failed write.
        errors: Human-readable verification failure messages.
        strategy_name: Strategy identifier when the failure occurred in a DB store.

    Raises:
        StoreVerificationError: Always raised with the supplied error details.
    """
    logger.error(
        "%s store verification failed for trade_id=%s signal_id=%s strategy_name=%s: %s",
        store_type,
        trade_id,
        signal_id,
        strategy_name,
        "; ".join(errors),
    )
    raise StoreVerificationError(
        store_type,
        trade_id,
        errors,
        strategy_name=strategy_name,
    )


def _collect_limits_verification_errors(
    trade_id: str,
    signal_id: str,
    strategy_name: str,
    expected: PositionLimits,
) -> List[str]:
    """Compare expected limits against a direct database read.

    Args:
        trade_id: Unique trade identifier.
        signal_id: Originating signal identifier.
        strategy_name: Strategy identifier.
        expected: Limits that were just written.

    Returns:
        Verification error messages; empty when the stored row matches.
    """
    stored_rows = _load_limits_rows(
        trade_id=trade_id,
        signal_id=signal_id,
        strategy_name=strategy_name,
    )
    if stored_rows.empty:
        return ["no limits rows found after save"]

    errors: List[str] = []
    for risk_measure in MEASURES_SET:
        expected_value = _normalize_limit_value(getattr(expected, risk_measure))
        measure_rows = stored_rows[stored_rows["risk_measure"] == risk_measure]
        if measure_rows.empty:
            if expected_value is not None:
                errors.append(f"missing stored row for risk_measure={risk_measure}")
            continue

        stored_value = _normalize_limit_value(measure_rows.iloc[0]["value"])
        if not _limit_values_equal(expected_value, stored_value):
            errors.append(
                f"{risk_measure} mismatch: expected={expected_value}, stored={stored_value}"
            )

    if expected.creation_date is not None:
        stored_dates = pd.to_datetime(stored_rows["date"]).dt.date.unique().tolist()
        expected_date = _normalize_row_date(expected.creation_date)
        if expected_date not in stored_dates:
            errors.append(
                f"creation_date mismatch: expected={expected_date}, stored_dates={stored_dates}"
            )

    return errors


def _collect_metadata_verification_errors(
    trade_id: str,
    signal_id: str,
    strategy_name: str,
    metadata_date: Union[datetime, date],
    expected_metadata: Dict[str, Any],
    expected_hash: str,
    expected_enabled: int,
) -> List[str]:
    """Compare expected metadata against a direct database read.

    Args:
        trade_id: Unique trade identifier.
        signal_id: Originating signal identifier.
        strategy_name: Strategy identifier.
        metadata_date: Metadata row date.
        expected_metadata: Metadata payload that was just written.
        expected_hash: Expected metadata hash.
        expected_enabled: Expected enabled flag.

    Returns:
        Verification error messages; empty when the stored row matches.
    """
    ticker = ticker_from_trade_id(trade_id)
    row = _load_metadata_row_by_date(
        trade_id=trade_id,
        signal_id=signal_id,
        ticker=ticker,
        strategy_name=strategy_name,
        metadata_date=metadata_date,
        metadata_type=METADATA_TYPE_METADATA,
    )
    if row is None:
        return ["no metadata row found after save"]

    errors: List[str] = []
    stored_metadata = _parse_metadata_dict(row.get("metadata_dict"))
    if stored_metadata != expected_metadata:
        errors.append("metadata_dict mismatch after save")

    stored_hash = row.get("metadata_hash")
    if stored_hash != expected_hash:
        errors.append(f"metadata_hash mismatch: expected={expected_hash}, stored={stored_hash}")

    stored_enabled = row.get("enabled")
    if stored_enabled is not None and int(stored_enabled) != expected_enabled:
        errors.append(f"enabled mismatch: expected={expected_enabled}, stored={stored_enabled}")

    return errors


def _load_limits_rows(trade_id: str, signal_id: str, strategy_name: str) -> pd.DataFrame:
    """Load all limit rows for a trade directly from the database.

    Args:
        trade_id: Unique trade identifier.
        signal_id: Originating signal identifier.
        strategy_name: Strategy identifier.

    Returns:
        DataFrame of matching ``limits`` rows.
    """
    query = text(f"""
        SELECT `date`, `risk_measure`, `value`
        FROM `{LIMITS_TABLE}`
        WHERE `trade_id` = :trade_id
          AND `signal_id` = :signal_id
          AND `strategy_name` = :strategy_name
        ORDER BY `date` DESC
    """)
    params = {
        "trade_id": trade_id,
        "signal_id": signal_id,
        "strategy_name": strategy_name,
    }
    engine = get_engine(PORTFOLIO_DATA_DB)
    return pd.read_sql(query, engine, params=params)


def _limits_from_rows(stored_rows: pd.DataFrame) -> Optional[PositionLimits]:
    """Build ``PositionLimits`` from database rows for a trade.

    Args:
        stored_rows: Raw limit rows ordered by descending date.

    Returns:
        Parsed limits using the latest date snapshot, or ``None`` when empty.
    """
    if stored_rows.empty:
        return None

    latest_date = stored_rows.iloc[0]["date"]
    latest_rows = stored_rows[stored_rows["date"] == latest_date]
    limits = PositionLimits(creation_date=latest_date)
    found_any = False
    for _, row in latest_rows.iterrows():
        limit_value = _normalize_limit_value(row["value"])
        if limit_value is not None:
            setattr(limits, row["risk_measure"], limit_value)
            found_any = True
    if not found_any and limits.delta is None:
        return None
    return limits


def _load_metadata_row_by_date(
    trade_id: str,
    signal_id: str,
    ticker: str,
    strategy_name: str,
    metadata_date: Union[datetime, date],
    metadata_type: str,
) -> Optional[Dict[str, Any]]:
    """Load a metadata row for an exact composite key.

    Args:
        trade_id: Unique trade identifier.
        signal_id: Originating signal identifier.
        ticker: Underlying ticker parsed from ``trade_id``.
        strategy_name: Strategy identifier.
        metadata_date: Metadata row date.
        metadata_type: ``LIMITS`` or ``METADATA`` enum value.

    Returns:
        Matching row as a dictionary, or ``None`` when absent.
    """
    query = text(f"""
        SELECT `metadata_dict`, `metadata_hash`, `enabled`, `date`
        FROM `{POSITION_METADATA_TABLE}`
        WHERE `{POSITION_METADATA_SIGNAL_ID_COL}` = :signal_id
          AND `trade_id` = :trade_id
          AND `ticker` = :ticker
          AND `strategy_name` = :strategy_name
          AND `date` = :metadata_date
          AND `metadata_type` = :metadata_type
        ORDER BY `date` DESC
        LIMIT 1
    """)
    params = {
        "signal_id": signal_id,
        "trade_id": trade_id,
        "ticker": ticker,
        "strategy_name": strategy_name,
        "metadata_date": metadata_date,
        "metadata_type": metadata_type,
    }
    engine = get_engine(PORTFOLIO_DATA_DB)
    rows = pd.read_sql(query, engine, params=params)
    if len(rows) == 0:
        return None
    return rows.iloc[0].to_dict()


def _normalize_limit_value(value: Any) -> Optional[float]:
    """Normalize a limit value for comparison.

    Args:
        value: Raw limit value from code or database.

    Returns:
        Float limit value, or ``None`` when absent or NaN.
    """
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    return float(value)


def _limit_values_equal(expected: Optional[float], stored: Optional[float]) -> bool:
    """Return whether two limit values are equivalent.

    Args:
        expected: Expected limit value.
        stored: Stored limit value.

    Returns:
        ``True`` when both values are equivalent, including both absent.
    """
    if expected is None and stored is None:
        return True
    if expected is None or stored is None:
        return False
    return math.isclose(expected, stored, rel_tol=1e-9, abs_tol=1e-9)


def _normalize_row_date(value: Any) -> date:
    """Normalize a date-like value to ``date``.

    Args:
        value: Date, datetime, timestamp, or ISO string.

    Returns:
        Normalized calendar date.
    """
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, pd.Timestamp):
        return value.date()
    parsed = to_datetime(value)
    return parsed.date() if hasattr(parsed, "date") else parsed
