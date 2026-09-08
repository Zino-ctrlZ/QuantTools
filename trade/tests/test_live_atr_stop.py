"""Tests for remake-safe ATR trail evaluation-bar TODAY.

``LiveAtrTrailStrategyBase.stop_triggered`` must not use the last row of a
full-history dataset as TODAY. That look-ahead ratchets a long trail with
later prices and closes early bars against a future stop.

Comment density: domain policy.

Core Classes:
    _LiveAtrLongStrategy: Minimal long ATR strategy for stop tests.

Usage:
    Run with ``pytest trade/tests/test_live_atr_stop.py``.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from EventDriven.types import SignalID
from trade.backtester_._strategy_live_atr_stop import LiveAtrTrailStrategyBase
from trade.backtester_.data import PTDataset


def _make_rising_dataset() -> PTDataset:
    """Build five bars where later closes ratchet a long ATR trail far above early prices.

    Returns:
        Dataset with deterministic OHLCV and a steep late rally.
    """
    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    closes = [100.0, 101.0, 102.0, 150.0, 160.0]
    frame = pd.DataFrame(
        {
            "Open": closes,
            "High": closes,
            "Low": closes,
            "Close": closes,
            "Volume": [1_000] * 5,
        },
        index=dates,
    )
    return PTDataset("TEST", frame)


class _LiveAtrLongStrategy(LiveAtrTrailStrategyBase):
    """Long-only ATR strategy with a constant loss for deterministic trail math."""

    bt_params: dict = {}

    def __init__(
        self,
        data: PTDataset,
        start_trading_date: Optional[str] = None,
        ticker: Optional[str] = None,
        tplusn: Optional[int] = 0,
    ) -> None:
        """Initialize the test strategy.

        Args:
            data: Input OHLCV dataset.
            start_trading_date: Optional first trading date.
            ticker: Optional ticker override.
            tplusn: Optional execution lag.
        """
        super().__init__(
            data=data,
            start_trading_date=start_trading_date,
            ticker=ticker or data.name,
            tplusn=tplusn,
        )

    def setup(self) -> None:
        """Register a constant ATR loss of 10 on every bar."""
        super().setup()
        loss = pd.Series(10.0, index=self.data.data.index)
        self.add_indicator("atr_loss", loss)

    def open_action(
        self,
        *,
        signal_id: Optional[str] = None,
        entry_price: Optional[float] = None,
        side: Optional[int] = None,
        date: Optional[pd.Timestamp] = None,
        index: Optional[int] = None,
    ) -> None:
        """Open via ``StrategyBase.open_action``."""
        super().open_action(
            signal_id=signal_id,
            entry_price=entry_price,
            side=side,
            date=date,
            index=index,
        )

    def is_open_signal(
        self,
        *,
        date: Optional[pd.Timestamp] = None,
        index: Optional[int] = None,
    ) -> bool:
        """Unused; tests call ``open_action`` directly."""
        return False

    def is_close_signal(
        self,
        *,
        date: Optional[pd.Timestamp] = None,
        index: Optional[int] = None,
    ) -> bool:
        """Unused; tests call ``stop_triggered`` directly."""
        return False

    def close_action(
        self,
        *,
        date: Optional[pd.Timestamp] = None,
        index: Optional[int] = None,
    ) -> None:
        """Clear position state after a close."""
        super().close_action(date=date, index=index)


def test_stop_triggered_uses_evaluation_bar_not_sample_end() -> None:
    """A mid-sample bar must not be compared to the end-of-sample ratcheted stop.

    Entry at bar 0 (close 100, loss 10) sets trail 90. Bar 1 close 101 stays
    above the same-bar trail (~91). The last bars rally to 160, which would
    ratchet the sample-end stop to 150 and falsely trip bar 1 if TODAY were
    ``self._n - 1``.
    """
    strat = _LiveAtrLongStrategy(_make_rising_dataset())
    entry_date = strat.data.data.index[0]
    check_date = strat.data.data.index[1]
    strat.open_action(
        signal_id=SignalID.generate(
            underlier="TEST",
            date=entry_date,
            signal_type="LONG",
            strategy_slug="test_live_atr",
        ),
        entry_price=100.0,
        side=1,
        date=entry_date,
        index=0,
    )

    hit = strat.stop_triggered(date=check_date, index=1)
    assert hit is False
    ## Same-bar long stop is close - loss = 91, not the later 150 trail.
    assert strat.stop == 91.0


def test_stop_triggered_on_last_bar_matches_live_remake_shape() -> None:
    """When the check date is the last loaded bar, TODAY equals ``self._n - 1``.

    Live remakes load through ``check_date``, so this is the production call.
    """
    strat = _LiveAtrLongStrategy(_make_rising_dataset())
    entry_date = strat.data.data.index[0]
    last_date = strat.data.data.index[-1]
    strat.open_action(
        signal_id=SignalID.generate(
            underlier="TEST",
            date=entry_date,
            signal_type="LONG",
            strategy_slug="test_live_atr",
        ),
        entry_price=100.0,
        side=1,
        date=entry_date,
        index=0,
    )

    hit = strat.stop_triggered(date=last_date, index=4)
    ## Last close 160 vs ratcheted stop max(150, 160-10) = 150; no breach.
    assert hit is False
    assert strat.stop == 150.0
