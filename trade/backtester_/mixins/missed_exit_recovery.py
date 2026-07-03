"""Mixin for recovering live positions when a pure exit signal was missed.

Comment density: orchestration

Processing Flow:
    1. Subclass implements ``_pure_close_signal`` (position-free exit event).
    2. ``setup`` calls ``register_pure_close_signal_indicator`` after indicators exist.
    3. ``should_close`` falls back to ``_kill_stale_positions`` when normal exit paths fail.
    4. While still in a position, any pure close signal since entry triggers another close request.

Core Functions:
    register_pure_close_signal_indicator: Precompute historical pure close flags.
    _kill_stale_positions: Detect missed exit fills and request close again.

Usage:
    class MyStrategy(MissedExitRecoveryMixin, StrategyBase):
        def setup(self) -> None:
            ...
            self.register_pure_close_signal_indicator()

        def _pure_close_signal(self, idx: int) -> bool:
            ...

        def should_close(self, *, date=None, index=None) -> TradeDecision:
            ok = ...  # normal exit logic
            if not ok:
                ok = self._kill_stale_positions(date=date, index=index)
            ...
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Optional

import pandas as pd


class MissedExitRecoveryMixin:
    """Opt-in live recovery when a pure exit signal fired but the position stayed open.

    Intended for strategies with event-style exit signals (e.g. band crosses) where
    live execution may miss or fail to fill on the signal bar. After a miss, this
    mixin keeps requesting close on subsequent bars until flat.
    """

    CLOSE_SIGNAL_INDICATOR: str = "all_close_signals"

    @abstractmethod
    def _pure_close_signal(self, idx: int) -> bool:
        """Return whether a position-free exit event is true at bar ``idx``.

        Must not read ``position_open``, ``position_info``, stops, or other state
        set in ``open_action`` / ``close_action``.

        Args:
            idx: Integer bar index into the strategy dataset.

        Returns:
            True when the pure market exit event is active at ``idx``.
        """

    def register_pure_close_signal_indicator(self, *, color: str = "green") -> None:
        """Precompute and register the pure close-signal indicator series.

        Args:
            color: Plot color for the registered indicator.
        """
        series = pd.Series(
            [self._pure_close_signal(i) for i in range(self._n)],
            index=self._index,
        )
        self.add_indicator(self.CLOSE_SIGNAL_INDICATOR, series, overlay=False, color=color)

    def _kill_stale_positions(
        self,
        *,
        date: pd.Timestamp = None,
        index: Optional[int] = None,
    ) -> bool:
        """Return True when a pure close signal fired since entry but position remains open.

        Args:
            date: Bar timestamp to evaluate.
            index: Optional bar index to evaluate.

        Returns:
            True if any pure close signal occurred from entry through the current bar.
        """
        idx, _ = self._resolve(date=date, index=index)
        if not self.have_position():
            return False

        indicator = self.indicators.get(self.CLOSE_SIGNAL_INDICATOR)
        if indicator is None:
            return False

        entry_idx, _ = self._resolve(date=self.position_info.entry_date, index=None)
        if entry_idx is None:
            return False

        ## Inclusive through current bar so a signal on this bar or any prior bar since entry counts.
        close_signals = indicator.values[entry_idx : idx + 1]
        return bool(close_signals.any())
