"""Tests for lightweight strategy rule registration and evaluation.

These tests verify that ``StrategyBase`` creates its rule filter before
``setup()`` and that rules receive the resolved ticker, date, index, and
indicator values for each evaluated bar.

Core Classes:
    _RuleStrategy: Minimal concrete strategy that registers a setup-time rule.

Usage:
    Run with ``pytest trade/tests/test_strategy_rules.py``.
"""

from typing import Optional

import pandas as pd

from trade.backtester_._strategy import RuleFilter, StrategyBase
from trade.backtester_.data import PTDataset


def _make_dataset() -> PTDataset:
    """Build a minimal two-bar OHLCV dataset.

    Returns:
        Dataset with deterministic prices and business-day timestamps.
    """
    dates = pd.date_range("2020-01-01", periods=2, freq="B")
    frame = pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [101.0, 102.0],
            "Low": [99.0, 100.0],
            "Close": [100.0, 101.0],
            "Volume": [1_000, 1_100],
        },
        index=dates,
    )
    return PTDataset("TEST", frame)


class _RuleStrategy(StrategyBase):
    """Minimal strategy that registers a rule during ``setup()``."""

    bt_params = {}

    def __init__(
        self,
        data: PTDataset,
        start_trading_date: Optional[str] = None,
        ticker: Optional[str] = None,
        tplusn: Optional[int | float] = 0,
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
        """Register an indicator and a rule before initialization completes."""
        score = pd.Series([0.5, 1.5], index=self.data.data.index)
        self.add_indicator("score", score)
        self.rules.add(lambda bar: bar.ind("score") > 1.0)

    def is_open_signal(
        self,
        *,
        date: Optional[pd.Timestamp] = None,
        index: Optional[int] = None,
    ) -> bool:
        """Return whether registered rules pass for the requested bar."""
        return self.rules.check(date=date, index=index)

    def is_close_signal(
        self,
        *,
        date: Optional[pd.Timestamp] = None,
        index: Optional[int] = None,
    ) -> bool:
        """Return no close signal for rule-filter tests."""
        return False

    def open_action(
        self,
        *,
        signal_id: Optional[str] = None,
        entry_price: Optional[float] = None,
        side: Optional[int] = None,
        date: Optional[pd.Timestamp] = None,
        index: Optional[int] = None,
    ) -> None:
        """Delegate position opening to ``StrategyBase``."""
        super().open_action(
            signal_id=signal_id,
            entry_price=entry_price,
            side=side,
            date=date,
            index=index,
        )

    def close_action(
        self,
        *,
        date: Optional[pd.Timestamp] = None,
        index: Optional[int] = None,
    ) -> None:
        """Delegate position closing to ``StrategyBase``."""
        super().close_action(date=date, index=index)


def test_rule_filter_exists_before_setup_and_reads_indicator() -> None:
    """A setup-time rule should evaluate the resolved indicator value."""
    strategy = _RuleStrategy(_make_dataset())

    assert isinstance(strategy.rules, RuleFilter)
    assert strategy.is_open_signal(index=0) is False
    assert strategy.is_open_signal(index=1) is True


def test_rule_bar_exposes_ticker_date_and_index() -> None:
    """Rules should receive ticker and resolved bar metadata."""
    strategy = _RuleStrategy(_make_dataset(), ticker="XYZ")
    expected_date = strategy.data.data.index[1]
    strategy.rules.add(
        lambda bar: (
            bar.ticker == "XYZ"
            and bar.index == 1
            and bar.date == expected_date
        )
    )

    assert strategy.rules.check(date=expected_date) is True


def test_empty_rule_filter_passes() -> None:
    """A newly created rule filter should preserve existing entry behavior."""
    strategy = _RuleStrategy(_make_dataset())
    strategy.rules = RuleFilter(strategy)

    assert strategy.rules.check(index=0) is True
