"""Temporary and compatibility limits cogs wired to the position store layer.

``TmpLimitsAndSizingCog`` remains the in-memory store harness for backtests.
Production live persistence uses ``LiveCOGLimitsAndSizingCog``; ``stores.__init__``
re-exports it as ``TmpLiveLimitsAndSizingCog`` for compatibility.

Core Classes:
    TmpLimitsAndSizingCog: Backtest cog backed by ``InMemoryPositionStore``.

Core Functions:
    metadata_from_store_payload: Rebuild ``_LimitsMetaData`` from a store dict.

Usage:
    >>> from EventDriven.riskmanager.position.stores.tmp_limits_cogs import TmpLimitsAndSizingCog
    >>> cog = TmpLimitsAndSizingCog()
"""

from __future__ import annotations

from typing import List, Optional, Union

from EventDriven.configs.core import DefaultSizerConfigs, LimitsEnabledConfig, ZscoreSizerConfigs
from EventDriven.dataclasses.limits import PositionLimits
from EventDriven.riskmanager.position.cogs.limits import (
    LimitsAndSizingCog,
    _LimitsMetaData,
    metadata_from_store_payload,
)
from EventDriven.riskmanager.position.stores.limits_store import (
    InMemoryPositionStore,
    PositionStore,
)

__all__ = [
    "TmpLimitsAndSizingCog",
    "metadata_from_store_payload",
]


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
