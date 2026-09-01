"""Short-index-equity dollar-multiplier sizing and optional PnL management cog.

Sizes new short Donchian equity option positions from a fixed ``trade_size`` and
``assign_dollar_multiplier``, then optionally emits full ROLL or a one-shot
waterfall (qty 1 ROLL / else CLOSE half) when PnL percent clears a threshold.

Comment density: orchestration.

Core Classes:
    ShortIdxEqCog: Sizing + optional profit-roll / waterfall analysis.

Core Dataclasses:
    _ShortIdxEqMetaData: Per-trade sizing snapshot including setup-feature values,
        waterfall trim state, armed stop level, and threshold/stop trigger events.

Core Functions:
    metadata_from_store_payload: Rebuild metadata from a PositionStore payload.

Processing Flow:
    1. Skip signals whose slug does not contain ``strategy_slug_token``.
    2. Resolve ticker strategy from ``MultiAssetStrategy.asset_strategies``.
    3. Inspect that instance for a child matching ``SignalID.strategy_slug``
       (composites like DualShortStrategy); use the match for multiplier and
       setup-feature snapshot, else the ticker strategy itself.
    4. Temporarily apply ``multiplier_version`` if set, call
       ``assign_dollar_multiplier`` on the signal-id date, then restore version.
    5. Effective trade size is ``min(tick_cash, config.trade_size)`` (tick cash
       scaled to dollars when needed). Calculator receives
       ``(multiplier, option_price, trade_size)`` with both money args in
       dollars (option premium * 100) and returns quantity.
       Default: ``trade_size * multiplier / 3 / option_price``.
       Qty 0 becomes 1.
    6. Snapshot ``REQUIRED_SETUP_FEATURES`` values at signal-id date.
    7. Persist metadata (including ``initial_quantity`` / ``half_closed``) on the
       cog dict and ``PositionStore``.
    8. During analysis: ``enable_profit_roll`` emits full ROLL vs threshold;
       ``enable_profit_waterfall`` emits qty-1 ROLL or one-time ceil-half CLOSE
       vs ``waterfall_profit_threshold`` on initial cost basis. The waterfall
       can also arm a metadata-backed profit stop at crossing PnL times a fixed
       multiplier. Both primary profit flags False emits nothing.

Usage:
    >>> cog = ShortIdxEqCog(eq_strategy=multi, config=ShortIdxEqCogConfig(trade_size=3000))
    >>> cog.on_new_position(new_position_state)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from EventDriven.configs.core import ShortIdxEqCogConfig
from EventDriven.dataclasses.states import CogActions, NewPositionState, PositionAnalysisContext, PositionState
from EventDriven.riskmanager.actions import CLOSE, Changes, ROLL
from EventDriven.riskmanager.position.base import BaseCog
from EventDriven.riskmanager.position.cogs.pnl_utils import correct_position_pnl
from EventDriven.types import SignalID
from trade.backtester_._multi_asset_strategy import MultiAssetStrategy
from trade.helpers.helper import to_datetime
from trade.helpers.Logging import setup_logger

if TYPE_CHECKING:
    from EventDriven.riskmanager.position.stores.limits_store import PositionStore

logger = setup_logger("EventDriven.riskmanager.position.cogs.short_idx_eq", stream_log_level="INFO")

QuantityCalculator = Callable[[int, float, float], int]

_CONTRACT_MULTIPLIER = 100
_DEFAULT_DIVISOR = 3


@dataclass
class _ShortIdxEqMetaData:
    """Per-trade sizing metadata for ShortIdxEqCog.

    Stores the dollar-multiplier decision, resulting quantity, at-date
    ``REQUIRED_SETUP_FEATURES`` values, waterfall trim/roll state, armed stop
    level, and threshold/stop trigger event fields. ``trade_size`` is the
    effective budget ``min(tick_cash, config_trade_size)``. ``option_price`` is
    the raw per-share premium (not * 100); calculator calls receive dollars.
    """

    trade_id: str
    date: datetime
    signal_id: str
    ticker: str
    multiplier: int
    multiplier_version: int
    option_price: float
    tick_cash: float
    config_trade_size: float
    trade_size: float
    allowed_trade_size: float
    new_quantity: int
    signal_date: str
    setup_features_at_date: Dict[str, float] = field(default_factory=dict)
    initial_quantity: int = 0
    half_closed: bool = False
    ## Armed stop level (set when threshold crosses and stop-loss is enabled).
    waterfall_stop_reference_pnl_pct: Optional[float] = None
    waterfall_stop_pnl_pct: Optional[float] = None
    waterfall_stop_set_date: Optional[datetime] = None
    ## Threshold-crossing event (trim/roll fire).
    threshold_triggered: bool = False
    threshold_triggered_pct: Optional[float] = None
    threshold_triggered_date: Optional[datetime] = None
    ## Profit-stop event (remaining qty closed after stop armed).
    stop_triggered: bool = False
    stop_triggered_pct: Optional[float] = None
    stop_triggered_date: Optional[datetime] = None


def _optional_float(payload: Dict[str, Any], key: str) -> Optional[float]:
    """Return a float from ``payload[key]`` when present, else ``None``.

    Args:
        payload: Metadata dictionary.
        key: Field name to read.

    Returns:
        Parsed float, or ``None`` when the key is missing or null.
    """
    value = payload.get(key)
    if value is None:
        return None
    return float(value)


def _optional_datetime(payload: Dict[str, Any], key: str) -> Optional[datetime]:
    """Return a datetime from ``payload[key]`` when present, else ``None``.

    Args:
        payload: Metadata dictionary.
        key: Field name to read.

    Returns:
        Parsed datetime, or ``None`` when the key is missing or null.
    """
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, str):
        return to_datetime(value)
    return value


def metadata_from_store_payload(payload: object) -> Optional[_ShortIdxEqMetaData]:
    """Convert a store payload back to ``_ShortIdxEqMetaData``.

    Older rows missing waterfall fields default ``initial_quantity`` from
    ``new_quantity`` and all waterfall/trigger state to inactive values.
    Legacy ``waterfall_stop_triggered`` maps to ``stop_triggered``.

    Args:
        payload: Dataclass, dict, or ``None`` value from a metadata store.

    Returns:
        Reconstructed metadata object, or ``None`` when ``payload`` is ``None``.
    """
    if payload is None:
        return None
    if isinstance(payload, _ShortIdxEqMetaData):
        return payload
    if not isinstance(payload, dict):
        return None

    raw_date = payload.get("date")
    if isinstance(raw_date, str):
        raw_date = to_datetime(raw_date)

    new_quantity = int(payload.get("new_quantity") or 0)
    initial_quantity = payload.get("initial_quantity")
    if initial_quantity is None:
        initial_quantity = new_quantity
    else:
        initial_quantity = int(initial_quantity)

    setup_features = payload.get("setup_features_at_date") or {}
    if not isinstance(setup_features, dict):
        setup_features = {}

    ## Prefer the new stop_triggered flag; fall back to legacy waterfall_stop_triggered.
    if "stop_triggered" in payload:
        stop_triggered = bool(payload.get("stop_triggered", False))
    else:
        stop_triggered = bool(payload.get("waterfall_stop_triggered", False))

    return _ShortIdxEqMetaData(
        trade_id=str(payload["trade_id"]),
        date=raw_date,
        signal_id=str(payload["signal_id"]),
        ticker=str(payload.get("ticker") or ""),
        multiplier=int(payload.get("multiplier") or 0),
        multiplier_version=int(payload.get("multiplier_version") or 0),
        option_price=float(payload.get("option_price") or 0.0),
        tick_cash=float(payload.get("tick_cash") or 0.0),
        config_trade_size=float(payload.get("config_trade_size") or 0.0),
        trade_size=float(payload.get("trade_size") or 0.0),
        allowed_trade_size=float(payload.get("allowed_trade_size") or 0.0),
        new_quantity=new_quantity,
        signal_date=str(payload.get("signal_date") or ""),
        setup_features_at_date={str(k): float(v) if v is not None else float("nan") for k, v in setup_features.items()},
        initial_quantity=initial_quantity,
        half_closed=bool(payload.get("half_closed", False)),
        waterfall_stop_reference_pnl_pct=_optional_float(payload, "waterfall_stop_reference_pnl_pct"),
        waterfall_stop_pnl_pct=_optional_float(payload, "waterfall_stop_pnl_pct"),
        waterfall_stop_set_date=_optional_datetime(payload, "waterfall_stop_set_date"),
        threshold_triggered=bool(payload.get("threshold_triggered", False)),
        threshold_triggered_pct=_optional_float(payload, "threshold_triggered_pct"),
        threshold_triggered_date=_optional_datetime(payload, "threshold_triggered_date"),
        stop_triggered=stop_triggered,
        stop_triggered_pct=_optional_float(payload, "stop_triggered_pct"),
        stop_triggered_date=_optional_datetime(payload, "stop_triggered_date"),
    )


class ShortIdxEqCog(BaseCog):
    """Size short-index-equity options and optionally manage profit via roll/waterfall.

    Expects a ``MultiAssetStrategy`` whose per-ticker strategies expose
    ``assign_dollar_multiplier`` and ``REQUIRED_SETUP_FEATURES``. Composite
    ticker strategies may nest child strategies; the cog inspects for a child
    whose ``strategy_slug`` matches the signal. Only signals whose slug
    contains ``config.strategy_slug_token`` are processed.
    """

    default_config = ShortIdxEqCogConfig(trade_size=1.0)

    def __init__(
        self,
        eq_strategy: MultiAssetStrategy,
        config: Optional[ShortIdxEqCogConfig] = None,
        calculator: Optional[QuantityCalculator] = None,
        *,
        live: bool = False,
        position_store: Optional["PositionStore"] = None,
        verify_after_save: bool = True,
    ) -> None:
        """Initialize the short-index-equity cog.

        Args:
            eq_strategy: Multi-asset container used to look up ticker strategies.
            config: Runtime config. ``trade_size`` is required; do not omit this.
            calculator: Optional ``(multiplier, option_price, trade_size) -> quantity``
                override. Both ``option_price`` and ``trade_size`` are dollar-scale:
                ``trade_size`` is ``min(tick_cash, config.trade_size)`` and
                ``option_price`` is premium * 100 (one-contract notional).
            live: When ``True``, persist metadata via ``DatabasePositionStore``.
            position_store: Optional store override for testing.
            verify_after_save: Re-read the database after each write when ``live``.

        Raises:
            TypeError: If ``config`` is omitted or not a ``ShortIdxEqCogConfig``.
        """
        ## Lazy import avoids circular RiskManager -> cogs -> stores -> backtest load
        from EventDriven.riskmanager.position.stores.limits_store import build_position_store

        if config is None:
            raise TypeError(
                "ShortIdxEqCog requires config=ShortIdxEqCogConfig(trade_size=...). trade_size is required."
            )
        if not isinstance(config, ShortIdxEqCogConfig):
            raise TypeError("Config must be of type ShortIdxEqCogConfig")

        super().__init__(config)
        self.config: ShortIdxEqCogConfig = config
        if self.config.trade_size is None or self.config.trade_size <= 0:
            raise ValueError("ShortIdxEqCogConfig.trade_size is required and must be > 0")
        self.eq_strategy = eq_strategy
        self.calculator: QuantityCalculator = calculator or self._default_calculator
        self.live = live
        self.position_metadata: Dict[str, _ShortIdxEqMetaData] = {}
        strategy_name = self.config.run_name or "short_idx_eq_cog"
        self._position_store = build_position_store(
            live=live,
            strategy_name=strategy_name,
            position_store=position_store,
            verify_after_save=verify_after_save,
        )

    def _scaled_tick_cash(self, tick_cash: float, is_tick_cash_scaled: bool) -> float:
        """Return tick cash in dollars.

        Unscaled tick cash is stored in hundredths of a dollar, matching other
        sizing cogs. Scaled tick cash is already a dollar amount.

        Args:
            tick_cash: Raw tick cash from the order request.
            is_tick_cash_scaled: Whether ``tick_cash`` is already in dollars.

        Returns:
            Tick cash in dollars.
        """
        cash = float(tick_cash)
        return cash if is_tick_cash_scaled else cash * 100

    def _effective_trade_size(self, tick_cash: float) -> float:
        """Return the dollar budget used for sizing.

        Args:
            tick_cash: Dollar-scaled tick cash available for the request.

        Returns:
            ``min(tick_cash, config.trade_size)``.
        """
        return min(tick_cash, float(self.config.trade_size))

    def _default_calculator(self, multiplier: int, option_price: float, trade_size: float) -> int:
        """Return default contract quantity from dollar multiplier and option close.

        Formula: ``floor((trade_size * multiplier / 3) / option_price)``.
        Both money args are dollar-scale (premium already * 100).

        Args:
            multiplier: Integer dollar multiplier from the asset strategy.
            option_price: Dollar notional of one contract (premium * 100).
            trade_size: Effective dollar budget ``min(tick_cash, config.trade_size)``.

        Returns:
            Non-negative integer quantity. Zero is converted to 1 by the caller.
        """
        if option_price is None or option_price <= 0 or trade_size <= 0:
            return 0
        allowed_trade_size = trade_size * multiplier / _DEFAULT_DIVISOR
        return int(math.floor(allowed_trade_size / option_price))

    def _is_target_strategy(self, signal_id: str) -> bool:
        """Return whether the signal belongs to the short Donchian equity slug.

        Args:
            signal_id: Raw signal identifier, optionally slug-prefixed.

        Returns:
            True when ``strategy_slug_token`` is contained in the parsed slug.
        """
        try:
            slug = SignalID(signal_id).strategy_slug or ""
        except Exception:
            logger.warning(f"Unable to parse signal id {signal_id} for ShortIdxEqCog slug check.", exc_info=True)
            return False
        return self.config.strategy_slug_token in slug

    def _resolve_asset_strategy(self, ticker: str):
        """Return the per-ticker strategy or raise an informative error.

        Args:
            ticker: Underlying ticker to look up on ``eq_strategy``.

        Returns:
            Asset strategy instance for ``ticker``.

        Raises:
            KeyError: If ``ticker`` is missing from ``asset_strategies``.
        """
        strategies = getattr(self.eq_strategy, "asset_strategies", None)
        if not isinstance(strategies, dict) or ticker not in strategies:
            available = sorted(strategies.keys()) if isinstance(strategies, dict) else []
            raise KeyError(
                f"ShortIdxEqCog: ticker {ticker!r} not found in MultiAssetStrategy.asset_strategies. "
                f"Available tickers: {available}"
            )
        return strategies[ticker]

    def _inspect_strategy_for_signal(self, asset_strat: Any, signal_id: str) -> Any:
        """Return the strategy that owns this signal's slug for sizing lookups.

        Single-strategy tickers are returned unchanged. Composites (for example
        DualShortStrategy) keep children as instance attributes with their own
        ``strategy_slug`` and ``assign_dollar_multiplier``; when the signal slug
        matches a child, that child is used so momentum and mean-reversion each
        get their own multiplier.

        Args:
            asset_strat: Per-ticker strategy from ``asset_strategies``.
            signal_id: Raw signal identifier, optionally slug-prefixed.

        Returns:
            Matching child strategy when found, otherwise ``asset_strat``.
        """
        try:
            slug = SignalID(signal_id).strategy_slug
        except Exception:
            logger.warning(
                f"Unable to parse signal id {signal_id} for ShortIdxEqCog strategy inspect.",
                exc_info=True,
            )
            return asset_strat
        if not slug:
            return asset_strat

        ## Own slug match with a multiplier method — use the ticker strategy as-is.
        own_slug = getattr(asset_strat, "strategy_slug", None)
        if own_slug == slug and callable(getattr(asset_strat, "assign_dollar_multiplier", None)):
            return asset_strat

        ## Walk instance attrs for nested strategies that own this signal slug.
        for value in getattr(asset_strat, "__dict__", {}).values():
            child_slug = getattr(value, "strategy_slug", None)
            if child_slug != slug:
                continue
            if callable(getattr(value, "assign_dollar_multiplier", None)):
                return value

        return asset_strat

    def _signal_lookup_date(self, signal_id: str) -> str:
        """Return YYYY-MM-DD for ``assign_dollar_multiplier`` from the signal id.

        Rolls keep the original signal id, so order date is the wrong lookback.
        Strategy bars are calendar-indexed; use a date-only string.

        Args:
            signal_id: Raw signal identifier.

        Returns:
            Signal date as ``YYYY-MM-DD``.
        """
        parsed = SignalID(signal_id)
        return pd.Timestamp(to_datetime(parsed.date)).strftime("%Y-%m-%d")

    def _coerce_multiplier(self, raw: object, *, ticker: str, lookup_date: str) -> int:
        """Validate and coerce ``assign_dollar_multiplier`` output to int.

        Fractional multipliers are rejected; the default quantity formula
        expects an integer dollar multiplier.

        Args:
            raw: Value returned by ``assign_dollar_multiplier``.
            ticker: Ticker used for error context.
            lookup_date: Signal date used for error context.

        Returns:
            Integer multiplier.

        Raises:
            TypeError: If ``raw`` is not numeric.
            ValueError: If ``raw`` is not an integer-valued number.
        """
        if isinstance(raw, bool) or not isinstance(raw, (int, float, np.integer, np.floating)):
            raise TypeError(
                f"assign_dollar_multiplier for {ticker} on {lookup_date} returned non-numeric "
                f"value {raw!r}. Expected an integer."
            )
        raw_float = float(raw)
        if not raw_float.is_integer():
            raise ValueError(
                f"assign_dollar_multiplier for {ticker} on {lookup_date} returned non-integer "
                f"{raw}. Expected an integer multiplier."
            )
        return int(raw_float)

    def _call_assign_dollar_multiplier(self, asset_strat, lookup_date: str, ticker: str) -> tuple[int, int]:
        """Call ``assign_dollar_multiplier`` with an optional temporary version.

        Sets ``MULTIPLIER_VERSION`` only for the call when config requests it,
        then restores the previous value.

        Args:
            asset_strat: Per-ticker strategy exposing ``assign_dollar_multiplier``.
            lookup_date: Signal-id date as ``YYYY-MM-DD``.
            ticker: Ticker used for error context.

        Returns:
            Tuple of ``(multiplier, effective_version)``.
        """
        instance_dict = getattr(asset_strat, "__dict__", {})
        had_instance_version = "MULTIPLIER_VERSION" in instance_dict
        prev_version = getattr(asset_strat, "MULTIPLIER_VERSION", None)
        version_overridden = self.config.multiplier_version is not None
        try:
            if version_overridden:
                asset_strat.MULTIPLIER_VERSION = self.config.multiplier_version
            effective_version = getattr(asset_strat, "MULTIPLIER_VERSION", None)
            raw = asset_strat.assign_dollar_multiplier(date=lookup_date)
            multiplier = self._coerce_multiplier(raw, ticker=ticker, lookup_date=lookup_date)
            version_for_meta = int(effective_version) if effective_version is not None else 0
            return multiplier, version_for_meta
        finally:
            ## Undo version stamp so the strategy object is unchanged after sizing.
            if version_overridden:
                if had_instance_version:
                    asset_strat.MULTIPLIER_VERSION = prev_version
                elif hasattr(asset_strat, "MULTIPLIER_VERSION"):
                    try:
                        delattr(asset_strat, "MULTIPLIER_VERSION")
                    except Exception:
                        if prev_version is not None:
                            asset_strat.MULTIPLIER_VERSION = prev_version

    def _snapshot_setup_features(self, asset_strat, lookup_date: str) -> Dict[str, float]:
        """Snapshot ``REQUIRED_SETUP_FEATURES`` values at the signal date.

        Missing indicators or resolve failures become NaN so sizing can continue.

        Args:
            asset_strat: Per-ticker strategy with ``indicators`` and ``_resolve``.
            lookup_date: Signal-id date as ``YYYY-MM-DD``.

        Returns:
            Feature name to at-date value mapping.
        """
        feature_names = tuple(getattr(asset_strat, "REQUIRED_SETUP_FEATURES", ()) or ())
        if not feature_names:
            return {}

        try:
            idx, _ = asset_strat._resolve(date=lookup_date)
        except Exception:
            logger.warning(
                f"Unable to resolve setup-feature date {lookup_date} on asset strategy. "
                "Storing NaN for REQUIRED_SETUP_FEATURES.",
                exc_info=True,
            )
            return {name: float("nan") for name in feature_names}

        indicators = getattr(asset_strat, "indicators", {}) or {}
        snapshot: Dict[str, float] = {}
        for name in feature_names:
            ind = indicators.get(name)
            if ind is None:
                snapshot[name] = float("nan")
                continue
            values = getattr(ind, "values", None)
            try:
                if values is None:
                    snapshot[name] = float("nan")
                elif isinstance(values, pd.Series):
                    snapshot[name] = float(values.iloc[idx])
                else:
                    snapshot[name] = float(values[idx])
            except Exception:
                snapshot[name] = float("nan")
        return snapshot

    def _store_metadata(self, metadata: _ShortIdxEqMetaData) -> None:
        """Persist metadata in-process and through the configured store.

        Args:
            metadata: Sizing snapshot for one position.
        """
        self.position_metadata[metadata.trade_id] = metadata
        self._position_store.save_metadata(metadata.trade_id, metadata)

    def _get_metadata(self, trade_id: str, signal_id: str) -> Optional[_ShortIdxEqMetaData]:
        """Load metadata from in-process cache or the position store.

        Args:
            trade_id: Unique trade identifier.
            signal_id: Originating signal identifier for store lookup.

        Returns:
            Metadata for the trade, or ``None`` when absent.
        """
        cached = self.position_metadata.get(trade_id)
        if cached is not None:
            return cached
        payload = self._position_store.get_metadata(trade_id, signal_id)
        metadata = metadata_from_store_payload(payload)
        if metadata is not None:
            self.position_metadata[trade_id] = metadata
        return metadata

    def on_new_position(self, state: NewPositionState) -> None:
        """Size a new short-index-equity option from dollar multiplier.

        Process:
            1. Skip non-matching strategy slugs.
            2. Look up ticker strategy; error if missing.
            3. Inspect for a child matching the signal strategy slug.
            4. Call ``assign_dollar_multiplier`` on signal-id date.
            5. Effective trade size is ``min(tick_cash, config.trade_size)``.
            6. Scale option premium to dollars (* 100), then compute quantity via
               calculator; force 1 when it returns 0.
            7. Snapshot setup-feature values and store metadata with waterfall fields.

        Args:
            state: Newly created position container. Quantity is updated in place.
        """
        order = state.order
        signal_id = order["signal_id"]
        if not self._is_target_strategy(signal_id):
            logger.debug(
                f"Skipping ShortIdxEqCog sizing for non-matching signal {signal_id}. "
                f"Trade ID: {order['data']['trade_id']}"
            )
            return

        ticker = state.symbol or SignalID(signal_id).ticker
        asset_strat = self._inspect_strategy_for_signal(
            self._resolve_asset_strategy(ticker),
            signal_id,
        )
        lookup_date = self._signal_lookup_date(signal_id)
        option_chain = state.at_time_data
        if option_chain is None:
            raise ValueError(
                f"ShortIdxEqCog: at_time_data is missing for trade {order['data']['trade_id']}; "
                "cannot source option close."
            )
        ## Raw premium (per share); metadata keeps this. Calculator gets dollars.
        option_price = float(option_chain.get_price())
        option_price_dollars = option_price * _CONTRACT_MULTIPLIER
        request = state.request
        tick_cash = self._scaled_tick_cash(float(request.tick_cash), request.is_tick_cash_scaled)
        config_trade_size = float(self.config.trade_size)
        trade_size = self._effective_trade_size(tick_cash)

        multiplier, effective_version = self._call_assign_dollar_multiplier(
            asset_strat, lookup_date=lookup_date, ticker=ticker
        )
        allowed_trade_size = trade_size * multiplier / _DEFAULT_DIVISOR
        ## Both money args dollar-scale so custom calculators need no * 100.
        qty = int(self.calculator(multiplier, option_price_dollars, trade_size))
        if qty <= 0:
            logger.info(
                f"Quantity was {qty} for trade {order['data']['trade_id']} after ShortIdxEq sizing. Defaulting to 1."
            )
            qty = 1

        order["data"]["quantity"] = qty
        setup_features = self._snapshot_setup_features(asset_strat, lookup_date)
        metadata = _ShortIdxEqMetaData(
            trade_id=order["data"]["trade_id"],
            date=to_datetime(order["date"]),
            signal_id=signal_id,
            ticker=ticker,
            multiplier=multiplier,
            multiplier_version=effective_version,
            option_price=option_price,
            tick_cash=tick_cash,
            config_trade_size=config_trade_size,
            trade_size=trade_size,
            allowed_trade_size=allowed_trade_size,
            new_quantity=qty,
            signal_date=lookup_date,
            setup_features_at_date=setup_features,
            ## Always seed waterfall fields so schema is stable across mode toggles / reloads.
            initial_quantity=qty,
            half_closed=False,
            waterfall_stop_reference_pnl_pct=None,
            waterfall_stop_pnl_pct=None,
            waterfall_stop_set_date=None,
            threshold_triggered=False,
            threshold_triggered_pct=None,
            threshold_triggered_date=None,
            stop_triggered=False,
            stop_triggered_pct=None,
            stop_triggered_date=None,
        )
        logger.info(
            f"ShortIdxEq sized trade {metadata.trade_id}: ticker={ticker}, signal_date={lookup_date}, "
            f"multiplier={multiplier}, version={effective_version}, option_price={option_price}, effective_trade_size={option_price * qty:.2f}, "
            f"tick_cash={tick_cash:.2f}, config_trade_size={config_trade_size:.2f}, "
            f"trade_size={trade_size:.2f}, allowed_trade_size={allowed_trade_size:.2f}, quantity={qty}"
        )
        self._store_metadata(metadata)

    def _analyze_impl(self, context: PositionAnalysisContext) -> CogActions:
        """Emit ROLL or waterfall CLOSE/ROLL opinions based on config flags.

        Both profit flags False returns empty opinions. Positions below the
        active threshold are left untouched (no HOLD spam).

        Args:
            context: Portfolio snapshot for the current analysis cycle.

        Returns:
            CogActions containing ROLL/CLOSE opinions, if any.
        """
        if self.config.enable_profit_waterfall:
            return self._analyze_waterfall(context)
        if self.config.enable_profit_roll:
            return self._analyze_full_roll(context)
        return CogActions(date=context.date, source_cog=self.name, opinions=[])

    def _analysis_schedule(self, context: PositionAnalysisContext) -> tuple[Optional[pd.Timestamp], pd.offsets.BusinessDay]:
        """Return last-updated stamp and t+n business-day offset for action dating.

        Args:
            context: Portfolio analysis context.

        Returns:
            Tuple of ``(last_updated, t_plus_n_bdays)``.
        """
        portfolio_meta = context.portfolio_meta
        last_updated = context.portfolio.last_updated
        t_plus_n = portfolio_meta.t_plus_n if portfolio_meta.t_plus_n is not None else 1
        t_plus_n_bdays = pd.offsets.BusinessDay(max(t_plus_n, 1))
        return last_updated, t_plus_n_bdays

    def _stamp_action(
        self,
        action: Union[CLOSE, ROLL],
        *,
        context: PositionAnalysisContext,
        last_updated: Optional[pd.Timestamp],
        t_plus_n_bdays: pd.offsets.BusinessDay,
        reason: str,
        verbose_info: str,
    ) -> None:
        """Stamp analysis metadata onto a CLOSE/ROLL action.

        Args:
            action: Action to stamp.
            context: Current analysis context.
            last_updated: Portfolio last-updated timestamp.
            t_plus_n_bdays: Business-day offset for effective date.
            reason: Human-readable reason string.
            verbose_info: Extra observability text.
        """
        action.analysis_date = context.date
        action.reason = reason
        action.verbose_info = verbose_info
        if last_updated is not None:
            action.effective_date = last_updated + t_plus_n_bdays

    def _analyze_full_roll(self, context: PositionAnalysisContext) -> CogActions:
        """Emit full ROLL when PnL% versus remaining cost basis clears the threshold.

        Args:
            context: Portfolio snapshot for the current analysis cycle.

        Returns:
            CogActions containing ROLL opinions, if any.
        """
        opinions: List[PositionState] = []
        last_updated, t_plus_n_bdays = self._analysis_schedule(context)

        for pos_state in context.portfolio.positions:
            if not self._is_target_strategy(pos_state.signal_id):
                continue
            ## Prefer ledger-backed % so partial closes do not inflate entry_price.
            pnl_pct = correct_position_pnl(pos_state).pnl_pct
            if pnl_pct <= self.config.roll_profit_threshold:
                continue

            logger.info(
                f"Position {pos_state.trade_id} has PnL of {pnl_pct:.2%} which is greater than "
                f"{self.config.roll_profit_threshold:.2%} of entry price. Rolling the position."
            )
            action = ROLL(
                trade_id=pos_state.trade_id,
                action=Changes(
                    quantity_diff=-abs(pos_state.quantity),
                    new_quantity=pos_state.quantity,
                ),
            )
            self._stamp_action(
                action,
                context=context,
                last_updated=last_updated,
                t_plus_n_bdays=t_plus_n_bdays,
                reason=(
                    f"PnL is {pnl_pct:.2%} which is greater than {self.config.roll_profit_threshold:.2%} "
                    f"of entry price. Rolling the position."
                ),
                verbose_info=f"Position {pos_state.trade_id} has PnL of {pnl_pct:.2%}. Rolling the position.",
            )
            pos_state.action = action
            opinions.append(pos_state)

        return CogActions(date=context.date, source_cog=self.name, opinions=opinions)

    def _ensure_waterfall_metadata(self, pos_state: PositionState) -> _ShortIdxEqMetaData:
        """Return waterfall metadata, synthesizing and persisting when missing.

        Args:
            pos_state: Open position being analyzed.

        Returns:
            Metadata with ``initial_quantity`` and ``half_closed`` populated.
        """
        metadata = self._get_metadata(pos_state.trade_id, pos_state.signal_id)
        if metadata is not None:
            if metadata.initial_quantity <= 0:
                metadata.initial_quantity = metadata.new_quantity or pos_state.quantity
            return metadata

        ## Backtest / reload edge: seed from open quantity so later cycles stay idempotent.
        metadata = _ShortIdxEqMetaData(
            trade_id=pos_state.trade_id,
            date=to_datetime(pos_state.last_updated) if pos_state.last_updated is not None else to_datetime(datetime.now()),
            signal_id=pos_state.signal_id,
            ticker=pos_state.underlier_tick or "",
            multiplier=0,
            multiplier_version=0,
            option_price=float(pos_state.entry_price or 0.0),
            tick_cash=0.0,
            config_trade_size=float(self.config.trade_size or 0.0),
            trade_size=0.0,
            allowed_trade_size=0.0,
            new_quantity=pos_state.quantity,
            signal_date="",
            initial_quantity=pos_state.quantity,
            half_closed=False,
            waterfall_stop_reference_pnl_pct=None,
            waterfall_stop_pnl_pct=None,
            waterfall_stop_set_date=None,
            threshold_triggered=False,
            threshold_triggered_pct=None,
            threshold_triggered_date=None,
            stop_triggered=False,
            stop_triggered_pct=None,
            stop_triggered_date=None,
        )
        self._store_metadata(metadata)
        return metadata

    def _waterfall_pnl_pct(self, pos_state: PositionState, initial_qty: int) -> float:
        """Return PnL% versus the frozen initial entry notional.

        Delegates to ``correct_position_pnl`` so waterfall shares the same
        ledger-backed ratio as other cogs. See that helper for why open-book
        ``pos_state.pnl / (entry_price * qty)`` is unsafe after a partial SELL.

        Args:
            pos_state: Open position being analyzed.
            initial_qty: Frozen initial quantity from sizing metadata, used only
                by the fallback path.

        Returns:
            PnL as a fraction of the initial entry notional.
        """
        return correct_position_pnl(
            pos_state,
            fallback_initial_qty=initial_qty,
        ).pnl_pct

    def _analyze_waterfall(self, context: PositionAnalysisContext) -> CogActions:
        """Emit one-shot waterfall opinions versus initial cost basis.

        PnL% is measured by ``_waterfall_pnl_pct`` (trade-ledger total PnL over
        the frozen initial notional) so it stays stable across a partial trim.

        Rules when PnL% ``>= waterfall_profit_threshold`` and ``half_closed`` is False:
            - ``initial_qty == 1``: ROLL remaining
            - else: CLOSE ``ceil(initial_qty * waterfall_close_fraction)`` (clamped to remaining)
            - when enabled, store ``crossing PnL% * waterfall_stop_loss_offset``
              as the stop for the remaining position

        On later cycles, PnL at or below the stored stop fully closes the
        remaining quantity. Stop actions are latched in metadata to avoid
        duplicate opinions before execution.

        Marks ``half_closed=True`` and re-stores metadata so the action does not
        re-fire on later cycles.

        Args:
            context: Portfolio snapshot for the current analysis cycle.

        Returns:
            CogActions containing CLOSE/ROLL opinions, if any.
        """
        opinions: List[PositionState] = []
        last_updated, t_plus_n_bdays = self._analysis_schedule(context)
        threshold = float(self.config.waterfall_profit_threshold)

        for pos_state in context.portfolio.positions:
            if not self._is_target_strategy(pos_state.signal_id):
                continue

            metadata = self._ensure_waterfall_metadata(pos_state)
            initial_qty = int(metadata.initial_quantity)
            if initial_qty <= 0:
                continue

            ## Ledger-based PnL% vs frozen initial notional; stays stable after a trim.
            pnl_pct = self._waterfall_pnl_pct(pos_state, initial_qty)
            remaining = int(pos_state.quantity)
            if remaining <= 0:
                continue

            ## After the one-shot trim/roll, enforce the exact profit stop frozen
            ## from the threshold-crossing look. It never ratchets on later highs.
            stop_pnl_pct = metadata.waterfall_stop_pnl_pct
            if (
                metadata.half_closed
                and self.config.enable_waterfall_stop_loss
                and stop_pnl_pct is not None
                and not metadata.stop_triggered
                and pnl_pct <= stop_pnl_pct
            ):
                action = CLOSE(
                    trade_id=pos_state.trade_id,
                    action=Changes(quantity_diff=-abs(remaining), new_quantity=0),
                )
                reason = (
                    f"Waterfall profit stop {stop_pnl_pct:.2%} hit "
                    f"(PnL {pnl_pct:.2%} vs initial). Closing remaining {remaining}."
                )
                self._stamp_action(
                    action,
                    context=context,
                    last_updated=last_updated,
                    t_plus_n_bdays=t_plus_n_bdays,
                    reason=reason,
                    verbose_info=f"Position {pos_state.trade_id} waterfall profit-stop CLOSE.",
                )
                logger.info(reason)
                metadata.new_quantity = 0
                metadata.stop_triggered = True
                metadata.stop_triggered_pct = pnl_pct
                metadata.stop_triggered_date = to_datetime(context.date)
                self._store_metadata(metadata)
                pos_state.action = action
                opinions.append(pos_state)
                continue

            ## Waterfall is one-shot: once we ROLL (qty 1) or CLOSE half (qty > 1),
            ## half_closed stays True so later analysis cycles cannot trim again.
            if metadata.half_closed:
                continue

            if pnl_pct < threshold:
                continue

            ## Latch threshold-crossing event before arming stop / emitting trim-or-roll.
            metadata.threshold_triggered = True
            metadata.threshold_triggered_pct = pnl_pct
            metadata.threshold_triggered_date = to_datetime(context.date)

            if self.config.enable_waterfall_stop_loss:
                ## Multiplicative stop: crossing 110% with offset 0.5 -> stop at 55%.
                stop_mult = float(self.config.waterfall_stop_loss_offset)
                metadata.waterfall_stop_reference_pnl_pct = pnl_pct
                metadata.waterfall_stop_pnl_pct = pnl_pct * stop_mult
                metadata.waterfall_stop_set_date = to_datetime(context.date)

            if initial_qty == 1:
                action: Union[CLOSE, ROLL] = ROLL(
                    trade_id=pos_state.trade_id,
                    action=Changes(quantity_diff=-abs(remaining), new_quantity=remaining),
                )
                reason = (
                    f"Waterfall threshold {threshold:.2%} hit (PnL {pnl_pct:.2%} vs initial). "
                    f"initial_qty=1; rolling the position."
                )
                verbose = f"Position {pos_state.trade_id} waterfall ROLL."
                metadata.new_quantity = remaining
            else:
                ## Size from frozen initial_quantity * configured fraction (ceil), then clamp to open qty.
                fraction = float(self.config.waterfall_close_fraction)
                close_qty = min(math.ceil(initial_qty * fraction), remaining)
                if close_qty <= 0:
                    continue
                new_remaining = remaining - close_qty
                action = CLOSE(
                    trade_id=pos_state.trade_id,
                    action=Changes(quantity_diff=-close_qty, new_quantity=new_remaining),
                )
                reason = (
                    f"Waterfall threshold {threshold:.2%} hit (PnL {pnl_pct:.2%} vs initial). "
                    f"initial_qty={initial_qty}; closing {close_qty} "
                    f"(ceil {fraction:.4g} of initial)."
                )
                verbose = f"Position {pos_state.trade_id} waterfall CLOSE {close_qty}."
                metadata.new_quantity = new_remaining

            if metadata.waterfall_stop_pnl_pct is not None:
                reason += (
                    f" Profit stop armed at {metadata.waterfall_stop_pnl_pct:.2%} "
                    f"from crossing PnL {pnl_pct:.2%}."
                )
            logger.info(reason)
            self._stamp_action(
                action,
                context=context,
                last_updated=last_updated,
                t_plus_n_bdays=t_plus_n_bdays,
                reason=reason,
                verbose_info=verbose,
            )
            ## Latch half_closed before re-store so the next cycle hits the guard above.
            ## Keep metadata.date (entry) unchanged so DB upsert updates the same row
            ## rather than inserting a second metadata record under a new analysis date.
            metadata.half_closed = True
            self._store_metadata(metadata)

            pos_state.action = action
            opinions.append(pos_state)

        return CogActions(date=context.date, source_cog=self.name, opinions=opinions)
