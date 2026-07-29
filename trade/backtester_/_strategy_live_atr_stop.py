"""In-memory live ATR trail for daily-remade strategy instances.

``stop_triggered`` ensures ``atr_trail`` is entry through TODAY-1 (last bar in
data minus one), then computes ``self.stop`` from that series plus TODAY's bar.
No SQL. Does not override ``open_action``.

Core Classes:
        LiveAtrTrailStrategyBase: Remake-safe ATR trail attribute + stop check.

Usage:
        >>> strat.stop_triggered(date=check_date)
        >>> strat.atr_trail  # entry .. TODAY-1
"""

from __future__ import annotations

from abc import ABC
from typing import ClassVar, Optional

import numpy as np
import pandas as pd

from trade.backtester_._strategy import StrategyBase
from trade.backtester_.indicators import update_atr_trail_long, update_atr_trail_short


class LiveAtrTrailStrategyBase(StrategyBase, ABC):
    """Live ATR trail: ``atr_trail`` through T-1, dynamic stop on today.

    Subclasses must register loss under ``ATR_LOSS_INDICATOR``. Do not define
    ``stop_triggered`` on the child if you want this remake-safe behavior.

    Attributes:
        ATR_LOSS_INDICATOR: Name of the ATR-loss indicator.
        atr_trail: Instance series from entry through T-1 (set in ``stop_triggered``).
    """

    ATR_LOSS_INDICATOR: ClassVar[str] = "atr_loss"

    def setup(self) -> None:
        """Initialize instance ``atr_trail`` to NaNs.

        Subclasses should call ``super().setup()`` at the start of their ``setup``
        before registering indicators. Does not call ``StrategyBase.setup`` (that
        method raises ``NotImplementedError`` by design).

        Returns:
            None.
        """
        ## Instance attribute — empty until stop_triggered rebuilds entry..T-1
        self.atr_trail = self._nan_atr_trail()

    def _atr_side_int(self) -> int:
        """Return ``+1`` / ``-1`` for the open position.

        Returns:
            Integer side.

        Raises:
            ValueError: If flat or side missing.
        """
        if not self.have_position() or self.position_info.side is None:
            raise ValueError("ATR trail requires an open position with side")
        side = int(self.position_info.side)
        if side not in (1, -1):
            raise ValueError(f"unsupported position side {side!r}")
        return side

    def _atr_loss_array(self) -> np.ndarray:
        """Return ATR-loss values aligned to strategy bars.

        Returns:
            Float loss array.

        Raises:
            ValueError: If the indicator is missing.
        """
        ind = self.indicators.get(self.ATR_LOSS_INDICATOR)
        if ind is None or ind.values is None:
            raise ValueError(
                f"{type(self).__name__} missing indicator {self.ATR_LOSS_INDICATOR!r}"
            )
        return np.asarray(ind.values, dtype=float)

    def _nan_atr_trail(self) -> pd.Series:
        """Return an all-NaN trail aligned to the full dataset index.

        Returns:
            NaN series named ``atr_trail``.
        """
        return pd.Series(np.nan, index=self._index, name="atr_trail")

    def _build_atr_trail_to(self, end_idx: int) -> pd.Series:
        """Build trail levels from entry through ``end_idx`` (inclusive).

        Args:
            end_idx: Last completed bar index (T-1).

        Returns:
            Series indexed from entry date through ``end_idx``.
        """
        if not self.have_position() or self.position_info.entry_date is None:
            return self._nan_atr_trail()

        entry_idx, _ = self._resolve(date=self.position_info.entry_date, index=None)
        if end_idx < entry_idx:
            return pd.Series(dtype=float, name="atr_trail")

        side = self._atr_side_int()
        loss = self._atr_loss_array()
        closes = np.asarray(self.close, dtype=float)
        update_fn = update_atr_trail_long if side == 1 else update_atr_trail_short

        out = np.full(end_idx - entry_idx + 1, np.nan, dtype=float)
        trail: Optional[float] = None
        for j, i in enumerate(range(entry_idx, end_idx + 1)):
            bar_close = float(closes[i])
            bar_loss = float(loss[i])
            if np.isnan(bar_close) or np.isnan(bar_loss):
                continue
            reset = trail is None or (isinstance(trail, float) and np.isnan(trail))
            trail = update_fn(
                close=bar_close,
                loss=bar_loss,
                prev_trail=trail,
                reset=reset,
            )
            out[j] = trail

        return pd.Series(out, index=self._index[entry_idx : end_idx + 1], name="atr_trail")

    def _atr_trail_covers_through(self, end_idx: int) -> bool:
        """Return True when ``atr_trail`` is set and ends on ``end_idx``.

        Args:
            end_idx: Required last completed bar index (TODAY-1).

        Returns:
            Whether the cached trail is usable through ``end_idx``.
        """
        if not hasattr(self, "atr_trail") or self.atr_trail is None or self.atr_trail.empty:
            return False
        if end_idx < 0:
            return False
        return self.atr_trail.index[-1] == self._index[end_idx]

    def stop_triggered(
        self,
        *,
        date: pd.Timestamp = None,
        index: Optional[int] = None,
    ) -> bool:
        """Ensure ``atr_trail`` is entry..TODAY-1, then compute stop and test breach.

        TODAY is always the last bar in the loaded dataset — independent of the
        ``date``/``index`` passed in. Rebuilds the trail only when missing or
        not yet through TODAY-1. Then sets ``self.stop`` from the last
        completed level plus TODAY's close/loss and compares the evaluation bar.

        Args:
            date: Evaluation timestamp (breach check only).
            index: Evaluation bar index (breach check only).

        Returns:
            ``True`` when the evaluation bar breaches the trail; ``False`` if flat.
        """
        if not self.have_position():
            self.atr_trail = self._nan_atr_trail()
            return False

        idx, _ = self._resolve(date=date, index=index)
        today_idx = self._n - 1
        completed_end = today_idx - 1
        entry_idx, _ = self._resolve(date=self.position_info.entry_date, index=None)

        ## Trail is always entry → TODAY-1 (not the caller's check date).
        if not self._atr_trail_covers_through(completed_end):
            if completed_end >= entry_idx:
                self.atr_trail = self._build_atr_trail_to(completed_end)
            else:
                self.atr_trail = pd.Series(dtype=float, name="atr_trail")

        ## Stop = last completed trail level, advanced with TODAY's bar.
        side = self._atr_side_int()
        loss = self._atr_loss_array()
        update_fn = update_atr_trail_long if side == 1 else update_atr_trail_short
        prev: Optional[float] = None
        if self.atr_trail is not None and not self.atr_trail.empty:
            prev = float(self.atr_trail.iloc[-1])
        reset = prev is None or np.isnan(prev)
        stop = update_fn(
            close=float(self.close[today_idx]),
            loss=float(loss[today_idx]),
            prev_trail=prev,
            reset=reset,
        )
        if stop is None or (isinstance(stop, float) and np.isnan(stop)):
            return False

        self.stop = float(stop)
        close = float(self.close[idx])
        if side == 1:
            return close < self.stop
        return close > self.stop


# Backward-compatible alias
StrategyBaseLiveAtrStop = LiveAtrTrailStrategyBase
