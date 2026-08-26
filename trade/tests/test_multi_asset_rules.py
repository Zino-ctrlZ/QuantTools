"""Tests for MultiAssetStrategy universe-wide and ticker-specific rule assignment.

These tests verify that ``set_rules`` adds callables onto each asset's existing
``RuleFilter`` rather than sharing one filter instance.

Core Classes:
    _BlankRuleStrategy: Minimal strategy with an empty rule filter.

Usage:
    Run with ``pytest trade/tests/test_multi_asset_rules.py``.
"""

from typing import Optional

import pandas as pd
import pytest

from trade.backtester_._multi_asset_strategy import MultiAssetStrategy
from trade.backtester_._strategy import RuleFilter, StrategyBase
from trade.backtester_.data import PTDataset


def _make_dataset(name: str) -> PTDataset:
    """Build a minimal two-bar OHLCV dataset.

    Args:
        name: Dataset name used as the default ticker.

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
    return PTDataset(name, frame)


class _BlankRuleStrategy(StrategyBase):
    """Minimal strategy that starts with an empty rule filter."""

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
        """Register a dummy indicator without adding rules."""
        score = pd.Series([0.5, 1.5], index=self.data.data.index)
        self.add_indicator("score", score)

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
        """Return no close signal for rule-assignment tests."""
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


def _make_multi() -> MultiAssetStrategy:
    """Build a two-ticker multi-asset container for rule tests.

    Returns:
        Multi-asset strategy covering ``FXI`` and ``IWM``.
    """
    return MultiAssetStrategy(
        name="rule_test",
        start_date="2020-01-01",
        params={"FXI": {}, "IWM": {}},
        strategy_class=_BlankRuleStrategy,
        data={"FXI": _make_dataset("FXI"), "IWM": _make_dataset("IWM")},
        tplusn=0,
    )


def test_set_rules_broadcasts_to_every_ticker() -> None:
    """A shared callable should be added to every asset filter."""
    multi = _make_multi()
    multi.set_rules(lambda bar: bar.ind("score") > 1.0)

    assert multi.rules["FXI"].check(index=0) is False
    assert multi.rules["IWM"].check(index=0) is False
    assert multi.rules["FXI"].check(index=1) is True
    assert multi.rules["IWM"].check(index=1) is True


def test_set_rules_ticker_dict_is_local() -> None:
    """A ticker dict should add rules only to the named assets."""
    multi = _make_multi()
    multi.set_rules({"FXI": lambda bar: False})

    assert multi.rules["FXI"].check(index=1) is False
    assert multi.rules["IWM"].check(index=1) is True


def test_set_rules_combines_shared_and_ticker_specific() -> None:
    """Universe-wide and ticker-specific calls should stack on the same filters."""
    multi = _make_multi()
    multi.set_rules(lambda bar: bar.ind("score") > 1.0)
    multi.set_rules({"FXI": lambda bar: bar.ticker == "FXI"})

    assert isinstance(multi.rules["FXI"], RuleFilter)
    assert multi.rules["FXI"].check(index=1) is True
    assert multi.rules["IWM"].check(index=1) is True
    assert multi.rules["FXI"].check(index=0) is False


def test_set_rules_unknown_ticker_raises() -> None:
    """Unknown tickers in a dict should raise KeyError."""
    multi = _make_multi()

    with pytest.raises(KeyError, match="Unknown ticker"):
        multi.set_rules({"SPY": lambda bar: True})


def test_set_rules_list_of_dicts_is_ticker_local() -> None:
    """A list of ticker dicts should apply each dict in order."""
    multi = _make_multi()
    multi.set_rules(
        [
            {"FXI": lambda bar: bar.ind("score") > 1.0},
            {"IWM": lambda bar: False},
        ]
    )

    assert multi.rules["FXI"].check(index=0) is False
    assert multi.rules["FXI"].check(index=1) is True
    assert multi.rules["IWM"].check(index=1) is False


def test_set_rules_list_of_lists_is_shared() -> None:
    """Nested lists of callables should flatten into universe-wide rules."""
    multi = _make_multi()
    multi.set_rules(
        [
            [lambda bar: bar.ind("score") > 0.0],
            [lambda bar: bar.ind("score") > 1.0],
        ]
    )

    assert multi.rules["FXI"].check(index=0) is False
    assert multi.rules["IWM"].check(index=0) is False
    assert multi.rules["FXI"].check(index=1) is True
    assert multi.rules["IWM"].check(index=1) is True


def test_set_rules_flattens_nested_ticker_lists() -> None:
    """Ticker values that are nested lists should flatten to one rule list."""
    multi = _make_multi()
    multi.set_rules(
        {
            "FXI": [
                [lambda bar: bar.ind("score") > 1.0],
                lambda bar: bar.ticker == "FXI",
            ]
        }
    )

    assert multi.rules["FXI"].check(index=1) is True
    assert multi.rules["IWM"].check(index=1) is True
    assert multi.rules["FXI"].check(index=0) is False

