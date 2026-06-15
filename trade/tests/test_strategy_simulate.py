"""Tests for StrategyBase.simulate() equity compounding and close trade returns."""

from __future__ import annotations

import unittest

import pandas as pd

from EventDriven.types import PositionEffect, SignalID
from trade.backtester_._strategy import StrategyBase, TradeDecision
from trade.backtester_._types import SideInt
from trade.backtester_.data import PTDataset


def _make_dataset(closes: list[float]) -> PTDataset:
    """Build a minimal OHLCV dataset from close prices."""
    dates = pd.date_range("2020-01-01", periods=len(closes), freq="B")
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
    return PTDataset("TEST", frame)


class _ScriptedStrategy(StrategyBase):
    """Open/close on fixed bar indices for deterministic simulate() tests."""

    bt_params = {}

    def __init__(
        self,
        data: PTDataset,
        *,
        open_at: int,
        close_at: int,
        side: SideInt,
        signal_id_str: str = "test::USO20200101SHORT",
    ) -> None:
        self._open_at = open_at
        self._close_at = close_at
        self._trade_side = side
        self._signal_id = SignalID(signal_id_str)
        super().__init__(data, tplusn=0)

    def setup(self) -> None:
        """No indicators required for scripted tests."""

    def is_open_signal(self, *, date: pd.Timestamp = None, index: int = None) -> bool:
        idx, _ = self._resolve(date=date, index=index)
        return idx == self._open_at

    def is_close_signal(self, *, date: pd.Timestamp = None, index: int = None) -> bool:
        idx, _ = self._resolve(date=date, index=index)
        return idx == self._close_at

    def should_open(self, *, date: pd.Timestamp = None, index: int = None) -> TradeDecision:
        if not self.should_trade(date=date, index=index):
            return TradeDecision(ok=False, side=0)
        if not self.is_open_signal(date=date, index=index) or self.position_open:
            return TradeDecision(ok=False, side=0)
        return TradeDecision(
            ok=True,
            side=int(self._trade_side),
            pos_effect=PositionEffect.OPEN,
            signal_id=self._signal_id,
        )

    def should_close(self, *, date: pd.Timestamp = None, index: int = None) -> TradeDecision:
        if not self.position_open or not self.is_close_signal(date=date, index=index):
            return TradeDecision(ok=False, side=0)
        return TradeDecision(
            ok=True,
            side=int(self._trade_side),
            pos_effect=PositionEffect.CLOSE,
            signal_id="N/A",
        )


class TestStrategySimulate(unittest.TestCase):
    """Validate compounded equity and close returns for long and short legs."""

    def test_short_single_bar_compounds_to_twenty_percent(self) -> None:
        """Short 100 -> 80 over one held bar should compound equity by +20%."""
        data = _make_dataset([100.0, 80.0])
        strategy = _ScriptedStrategy(data, open_at=0, close_at=1, side=SideInt.SELL)

        trades, equity = strategy.simulate()

        self.assertAlmostEqual(equity.iloc[-1], 1.2)
        close_trade = next(trade for trade in trades if trade["action"] == "close")
        self.assertAlmostEqual(close_trade["return_pct"], 0.2)
        self.assertEqual(close_trade["side"], SideInt.SELL)

    def test_short_multi_bar_compounds_bar_by_bar(self) -> None:
        """Short 100 -> 90 -> 80 should compound to 22.22%, not inverse-ratio 25%."""
        data = _make_dataset([100.0, 90.0, 80.0])
        strategy = _ScriptedStrategy(data, open_at=0, close_at=2, side=SideInt.SELL)

        trades, equity = strategy.simulate()

        expected_equity = 1.1 * (10.0 / 9.0)
        self.assertAlmostEqual(equity.iloc[-1], expected_equity)

        close_trade = next(trade for trade in trades if trade["action"] == "close")
        self.assertAlmostEqual(close_trade["return_pct"], 0.2)

    def test_long_multi_bar_matches_simple_return(self) -> None:
        """Long path compounding should match simple entry-to-exit return."""
        data = _make_dataset([100.0, 110.0, 120.0])
        strategy = _ScriptedStrategy(
            data,
            open_at=0,
            close_at=2,
            side=SideInt.BUY,
            signal_id_str="test::USO20200101LONG",
        )

        trades, equity = strategy.simulate()

        self.assertAlmostEqual(equity.iloc[-1], 1.2)
        close_trade = next(trade for trade in trades if trade["action"] == "close")
        self.assertAlmostEqual(close_trade["return_pct"], 0.2)


if __name__ == "__main__":
    unittest.main()
