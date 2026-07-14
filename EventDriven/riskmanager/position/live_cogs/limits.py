"""Live trading limits cog with database-backed position store persistence.

``LiveCOGLimitsAndSizingCog`` extends ``LimitsAndSizingCog`` and delegates
limits and sizing metadata persistence to ``DatabasePositionStore``. That store
writes through to the ``limits`` and ``position_metadata`` tables, verifies
saves, and keeps an optional in-memory cache keyed by ``(signal_id, trade_id)``.

Core Classes:
    LiveCOGLimitsAndSizingCog: Live limits enforcement with store persistence.

Core Functions:
    enable_storing_to_db: Enable global DB persistence for limits/metadata stores.
    disable_storing_to_db: Disable global DB persistence (cache-only writes).
    reset_storing_to_db: Restore default DB persistence (enabled).

Processing Flow:
    1. Construct ``DatabasePositionStore`` from ``config.run_name`` and defaults.
    2. On save, write limits/metadata through the store (DB + cache when enabled).
    3. On get, read from store cache or DB; fall back to config DTE/moneyness defaults.
    4. Keep ``position_metadata`` in-process for same-session failsafe updates.

Usage:
    >>> from EventDriven.riskmanager.position.live_cogs.limits import LiveCOGLimitsAndSizingCog
    >>> from EventDriven.configs.core import LimitsEnabledConfig
    >>> cog = LiveCOGLimitsAndSizingCog(config=LimitsEnabledConfig(run_name="demo"))
"""

from __future__ import annotations

from typing import List, Optional, Union

from EventDriven.configs.core import (
    BaseSizerConfigs,
    DefaultSizerConfigs,
    LimitsEnabledConfig,
    ZscoreSizerConfigs,
)
from EventDriven.dataclasses.limits import PositionLimits
from EventDriven.riskmanager.position.cogs.limits import LimitsAndSizingCog, _LimitsMetaData, metadata_from_store_payload
from EventDriven.riskmanager.position.stores.limits_store import (
    PositionStore,
    build_position_store,
)
from EventDriven.riskmanager.position.stores.limits_store import (
    disable_storing_to_db as _disable_storing_to_db,
)
from EventDriven.riskmanager.position.stores.limits_store import (
    enable_storing_to_db as _enable_storing_to_db,
)
from EventDriven.riskmanager.position.stores.limits_store import (
    reset_storing_to_db as _reset_storing_to_db,
)
from trade.helpers.Logging import setup_logger

logger = setup_logger("EventDriven.riskmanager.position.live_cogs.limits")


def enable_storing_to_db(*args: object, **kwargs: object) -> None:
    """Enable database persistence for limits and metadata stores globally.

    Args:
        *args: Ignored; accepted for call-site compatibility.
        **kwargs: Ignored; accepted for call-site compatibility.
    """
    _enable_storing_to_db(*args, **kwargs)


def disable_storing_to_db(*args: object, **kwargs: object) -> None:
    """Disable database persistence for limits and metadata stores globally.

    Args:
        *args: Ignored; accepted for call-site compatibility.
        **kwargs: Ignored; accepted for call-site compatibility.
    """
    _disable_storing_to_db(*args, **kwargs)


def reset_storing_to_db() -> None:
    """Reset database persistence for limits and metadata stores to enabled."""
    _reset_storing_to_db()


class LiveCOGLimitsAndSizingCog(LimitsAndSizingCog):
    """Live limits cog that persists via ``DatabasePositionStore``.

    Attributes:
        cool_off_period: Reserved cool-off window in seconds (0 = disabled).
        _position_store: Combined limits and metadata persistence backend.
    """

    def __init__(
        self,
        config: Optional[LimitsEnabledConfig] = None,
        sizer_configs: Optional[Union[BaseSizerConfigs, DefaultSizerConfigs, ZscoreSizerConfigs]] = None,
        underlier_list: Optional[List[str]] = None,
        position_store: Optional[PositionStore] = None,
        *,
        live: bool = True,
        # ponytail: False unblocks live !mw until NaN serialization + FLOAT tolerance fixes
        # at market close; re-enable verify_after_save=True — audit whats-this-error-2026-07-14
        verify_after_save: bool = False,
    ) -> None:
        """Initialize the live database-backed limits cog.

        Args:
            config: Limits cog configuration; ``run_name`` is used as strategy name.
            sizer_configs: Sizer configuration object.
            underlier_list: Underliers used by vol-adjusted sizers.
            position_store: Optional store override for testing.
            live: When ``True`` (default), persist via ``DatabasePositionStore``.
            verify_after_save: Re-read the database after each write and raise on mismatch.
                Defaults to ``False`` for live trading until store serialization fixes land.
        """
        super().__init__(config=config, sizer_configs=sizer_configs, underlier_list=underlier_list)
        strategy_name = self.config.run_name or "live_limits_cog"
        self.live = live
        self.cool_off_period = 0
        self._position_store = build_position_store(
            live=live,
            strategy_name=strategy_name,
            default_dte=self.config.default_dte,
            default_moneyness=self.config.default_moneyness,
            position_store=position_store,
            verify_after_save=verify_after_save,
        )

    def _save_position_limits(self, trade_id: str, signal_id: str, limits: PositionLimits) -> None:
        """Persist limits through the configured position store.

        Args:
            trade_id: Unique trade identifier.
            signal_id: Originating signal identifier.
            limits: Limit values to store.
        """
        self._position_store.save_limits(trade_id, signal_id, limits)

    def _get_position_limits(self, trade_id: str, signal_id: str) -> Optional[PositionLimits]:
        """Load limits from the store, falling back to config defaults when absent.

        Args:
            trade_id: Unique trade identifier.
            signal_id: Originating signal identifier.

        Returns:
            Stored limits, or a defaults-only ``PositionLimits`` when no row exists
            so analyze paths do not dereference ``None``.
        """
        limits = self._position_store.get_limits(trade_id, signal_id)
        if limits is not None:
            return limits

        ## No DB/cache row yet — analyze still expects an object with dte/moneyness
        return PositionLimits(
            dte=self.config.default_dte,
            moneyness=self.config.default_moneyness,
        )

    def _store_metadata(self, metadata: _LimitsMetaData) -> None:
        """Persist metadata in-process and through the position store.

        Args:
            metadata: Limits sizing metadata payload.
        """
        ## Keep the in-process dict for same-session failsafe lookups by trade_id only
        self.position_metadata[metadata.trade_id] = metadata
        self._position_store.save_metadata(metadata.trade_id, metadata)

    def _get_metadata(self, trade_id: str) -> Optional[_LimitsMetaData]:
        """Load metadata from the in-process cache.

        The base cog only passes ``trade_id``. Same-session failsafe updates rely
        on the in-process dict populated by ``_store_metadata``. Cross-session
        reload should use ``_position_store.get_metadata(trade_id, signal_id)``.

        Args:
            trade_id: Unique trade identifier.

        Returns:
            Stored metadata, or ``None`` when absent from the in-process cache.
        """
        cached = self.position_metadata.get(trade_id)
        if cached is not None:
            return cached
        return None

    def get_stored_metadata(self, trade_id: str, signal_id: str) -> Optional[_LimitsMetaData]:
        """Load metadata from the position store by ``(trade_id, signal_id)``.

        Args:
            trade_id: Unique trade identifier.
            signal_id: Originating signal identifier.

        Returns:
            Reconstructed metadata, or ``None`` when absent.
        """
        payload = self._position_store.get_metadata(trade_id, signal_id)
        return metadata_from_store_payload(payload)

    def _analyze_impl(self, portfolio_context):
        """Delegate portfolio analysis to the base limits cog.

        Args:
            portfolio_context: Portfolio analysis context from the position analyzer.

        Returns:
            Cog actions produced by the base implementation.
        """
        return super()._analyze_impl(portfolio_context)
