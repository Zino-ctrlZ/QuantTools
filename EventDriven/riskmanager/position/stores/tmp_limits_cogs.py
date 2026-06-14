"""Temporary limits cogs wired to the position store layer.

These classes mirror ``LimitsAndSizingCog`` and ``LiveCOGLimitsAndSizingCog`` but
delegate persistence to ``PositionStore`` implementations. They exist to validate
the store integration before replacing the production cogs.

Core Classes:
    TmpLimitsAndSizingCog: Backtest cog backed by ``InMemoryPositionStore``.
    TmpLiveLimitsAndSizingCog: Live cog backed by ``DatabasePositionStore``.

Usage:
    >>> from EventDriven.riskmanager.position.stores.tmp_limits_cogs import TmpLimitsAndSizingCog
    >>> cog = TmpLimitsAndSizingCog()
"""

from __future__ import annotations

from typing import List, Optional, Union

from EventDriven.configs.core import DefaultSizerConfigs, LimitsEnabledConfig, ZscoreSizerConfigs
from EventDriven.dataclasses.limits import PositionLimits
from EventDriven.riskmanager.position.cogs.limits import LimitsAndSizingCog, _LimitsMetaData
from EventDriven.riskmanager.position.stores.limits_store import (
    DatabasePositionStore,
    InMemoryPositionStore,
    PositionStore,
)
from trade.helpers.helper import to_datetime


class TmpLimitsAndSizingCog(LimitsAndSizingCog):
    """Backtest limits cog that persists via ``InMemoryPositionStore``."""

    def __init__(
        self,
        config: Optional[LimitsEnabledConfig] = None,
        sizer_configs: Optional[Union[DefaultSizerConfigs, ZscoreSizerConfigs]] = None,
        underlier_list: Optional[List[str]] = None,
        position_store: Optional[PositionStore] = None,
    ) -> None:
        """Initialize the temporary in-memory limits cog.

        Args:
            config: Limits cog configuration.
            sizer_configs: Sizer configuration object.
            underlier_list: Underliers used by vol-adjusted sizers.
            position_store: Optional store override for testing.
        """
        super().__init__(config=config, sizer_configs=sizer_configs, underlier_list=underlier_list)
        self._position_store = position_store or InMemoryPositionStore()

    def _save_position_limits(self, trade_id: str, signal_id: str, limits: PositionLimits) -> None:
        """Persist limits through the configured store.

        Args:
            trade_id: Unique trade identifier.
            signal_id: Originating signal identifier.
            limits: Limit values to store.
        """
        self._position_store.save_limits(trade_id, signal_id, limits)

    def _get_position_limits(self, trade_id: str, signal_id: str) -> Optional[PositionLimits]:
        """Load limits from the configured store.

        Args:
            trade_id: Unique trade identifier.
            signal_id: Originating signal identifier.

        Returns:
            Stored limits, or ``None`` when absent.
        """
        return self._position_store.get_limits(trade_id, signal_id)

    def _store_metadata(self, metadata: _LimitsMetaData) -> None:
        """Persist metadata through the configured store.

        Args:
            metadata: Limits sizing metadata payload.
        """
        self._position_store.save_metadata(metadata.trade_id, metadata)

    def _get_metadata(self, trade_id: str) -> Optional[_LimitsMetaData]:
        """Load metadata from the configured store.

        The base cog only passes ``trade_id`` today. Once ``LimitsAndSizingCog``
        is updated to pass ``signal_id`` as well, this should delegate to
        ``get_metadata(trade_id, signal_id)``.

        Args:
            trade_id: Unique trade identifier.

        Returns:
            Stored metadata, or ``None`` when absent or ``signal_id`` is unavailable.
        """
        return None


class TmpLiveLimitsAndSizingCog(LimitsAndSizingCog):
    """Live limits cog that persists via ``DatabasePositionStore``."""

    def __init__(
        self,
        config: Optional[LimitsEnabledConfig] = None,
        sizer_configs: Optional[Union[DefaultSizerConfigs, ZscoreSizerConfigs]] = None,
        underlier_list: Optional[List[str]] = None,
        position_store: Optional[PositionStore] = None,
        *,
        verify_after_save: bool = True,
    ) -> None:
        """Initialize the temporary database-backed limits cog.

        Args:
            config: Limits cog configuration; ``run_name`` is used as strategy name.
            sizer_configs: Sizer configuration object.
            underlier_list: Underliers used by vol-adjusted sizers.
            position_store: Optional store override for testing.
            verify_after_save: Re-read the database after each write and raise on mismatch.
        """
        super().__init__(config=config, sizer_configs=sizer_configs, underlier_list=underlier_list)
        strategy_name = self.config.run_name or "tmp_live_limits_cog"
        self._position_store = position_store or DatabasePositionStore(
            strategy_name=strategy_name,
            default_dte=self.config.default_dte,
            default_moneyness=self.config.default_moneyness,
            verify_after_save=verify_after_save,
        )

    def _save_position_limits(self, trade_id: str, signal_id: str, limits: PositionLimits) -> None:
        """Persist limits through the configured store.

        Args:
            trade_id: Unique trade identifier.
            signal_id: Originating signal identifier.
            limits: Limit values to store.
        """
        self._position_store.save_limits(trade_id, signal_id, limits)

    def _get_position_limits(self, trade_id: str, signal_id: str) -> Optional[PositionLimits]:
        """Load limits from the configured store.

        Args:
            trade_id: Unique trade identifier.
            signal_id: Originating signal identifier.

        Returns:
            Stored limits, or ``None`` when absent.
        """
        return self._position_store.get_limits(trade_id, signal_id)

    def _store_metadata(self, metadata: _LimitsMetaData) -> None:
        """Persist metadata through the configured store.

        Args:
            metadata: Limits sizing metadata payload.
        """
        self._position_store.save_metadata(metadata.trade_id, metadata)

    def _get_metadata(self, trade_id: str) -> Optional[_LimitsMetaData]:
        """Load metadata from the configured store.

        The base cog only passes ``trade_id`` today. Once ``LimitsAndSizingCog``
        is updated to pass ``signal_id`` as well, this should delegate to
        ``get_metadata(trade_id, signal_id)``.

        Args:
            trade_id: Unique trade identifier.

        Returns:
            Stored metadata, or ``None`` when absent or ``signal_id`` is unavailable.
        """
        return None


def metadata_from_store_payload(payload: object) -> Optional[_LimitsMetaData]:
    """Convert a store payload back to ``_LimitsMetaData``.

    Args:
        payload: Dataclass, dict, or ``None`` value from a metadata store.

    Returns:
        Reconstructed metadata object, or ``None`` when ``payload`` is ``None``.
    """
    if payload is None:
        return None
    if isinstance(payload, _LimitsMetaData):
        return payload
    if not isinstance(payload, dict):
        return None

    raw_date = payload.get("date")
    if isinstance(raw_date, str):
        raw_date = to_datetime(raw_date)

    return _LimitsMetaData(
        trade_id=payload["trade_id"],
        date=raw_date,
        signal_id=payload["signal_id"],
        scalar=payload["scalar"],
        sizing_lev=payload["sizing_lev"],
        delta_lmt=payload["delta_lmt"],
        delta_per_contract=payload.get("delta_per_contract"),
        option_price=payload.get("option_price"),
        undl_price=payload.get("undl_price"),
        prev_quantity=payload.get("prev_quantity"),
        new_quantity=payload.get("new_quantity"),
        rvol=payload.get("rvol"),
    )
