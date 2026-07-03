"""Tests for MissedExitRecoveryMixin."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trade.backtester_._strategy import StrategyBase, TradeDecision
from trade.backtester_.data import PTDataset
from trade.backtester_.mixins import MissedExitRecoveryMixin


class _StubRecoveryStrategy(MissedExitRecoveryMixin, StrategyBase):
    """Minimal strategy exercising pure close signal recovery."""

    bt_params = {}

    def __init__(self, data: PTDataset, *, close_flags: list[bool]) -> None:
        self._close_flags = close_flags
        super().__init__(data)

    def setup(self) -> None:
        self.register_pure_close_signal_indicator()

    def is_open_signal(self, *, date=None, index=None) -> bool:
        return False

    def is_close_signal(self, *, date=None, index=None) -> bool:
        idx, _ = self._resolve(date=date, index=index)
        return self._pure_close_signal(idx)

    def _pure_close_signal(self, idx: int) -> bool:
        return bool(self._close_flags[idx])

    def open_action(self, *, signal_id=None, entry_price=None, side=None, date=None, index=None) -> None:
        super().open_action(signal_id=signal_id, entry_price=entry_price, side=side or 1, date=date, index=index)

    def close_action(self, *, date=None, index=None) -> None:
        super().close_action(date=date, index=index)

    def should_open(self, *, date=None, index=None) -> TradeDecision:
        return TradeDecision(ok=False, side=1)


def _make_strategy(close_flags: list[bool]) -> _StubRecoveryStrategy:
    dates = pd.date_range("2024-01-02", periods=len(close_flags), freq="B")
    closes = np.linspace(100, 100 + len(close_flags) - 1, len(close_flags))
    frame = pd.DataFrame(
        {
            "Open": closes,
            "High": closes,
            "Low": closes,
            "Close": closes,
            "Volume": 1_000,
        },
        index=dates,
    )
    strategy = _StubRecoveryStrategy(data=PTDataset("TEST", frame), close_flags=close_flags)
    return strategy


def test_kill_stale_positions_true_after_missed_signal() -> None:
    strategy = _make_strategy([False, False, True, False, False])
    strategy.open_action(index=0, entry_price=100.0, side=1, signal_id="sig-1")

    assert strategy._kill_stale_positions(index=1) is False
    assert strategy._kill_stale_positions(index=4) is True


def test_kill_stale_positions_false_when_flat() -> None:
    strategy = _make_strategy([False, True, False])

    assert strategy._kill_stale_positions(index=1) is False
