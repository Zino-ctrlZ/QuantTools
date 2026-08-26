"""Tests for StrategyBase filter-threshold ClassVar getters and setters.

These tests lock the production convention: concrete threshold bundles subclass
``FilterThresholds``, bind on ``FILTER_THRESHOLDS``, and are mutated only via
``set_filter_thresholds`` with concrete-type checking.

Usage:
    Run with ``pytest trade/tests/test_filter_thresholds.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import ClassVar, Optional

import pandas as pd
import pytest

from trade.backtester_._strategy import FilterThresholds, StrategyBase
from trade.backtester_.data import PTDataset


def _make_dataset() -> PTDataset:
    """Build a minimal one-bar OHLCV dataset.

    Returns:
        Dataset with deterministic prices and a business-day timestamp.
    """
    dates = pd.date_range("2020-01-01", periods=1, freq="B")
    frame = pd.DataFrame(
        {
            "Open": [100.0],
            "High": [101.0],
            "Low": [99.0],
            "Close": [100.0],
            "Volume": [1_000],
        },
        index=dates,
    )
    return PTDataset("TEST", frame)


@dataclass(frozen=True, slots=True)
class _AlphaThresholds(FilterThresholds):
    """Concrete threshold bundle for alpha-style filter tests."""

    zscore_min: float = 2.0


@dataclass(frozen=True, slots=True)
class _BetaThresholds(FilterThresholds):
    """Concrete threshold bundle for beta-style filter tests."""

    fear_max: float = 1.5


class _BareStrategy(StrategyBase):
    """Strategy that does not configure filter thresholds."""

    bt_params = {}

    def __init__(
        self,
        data: PTDataset,
        start_trading_date: Optional[str] = None,
        ticker: Optional[str] = None,
        tplusn: Optional[int | float] = 0,
    ) -> None:
        """Initialize the bare test strategy.

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
        """No indicators required for threshold API tests."""
        return None

    def is_open_signal(
        self,
        *,
        date: Optional[pd.Timestamp] = None,
        index: Optional[int] = None,
    ) -> bool:
        """Return no open signal."""
        return False

    def is_close_signal(
        self,
        *,
        date: Optional[pd.Timestamp] = None,
        index: Optional[int] = None,
    ) -> bool:
        """Return no close signal."""
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


class _AlphaStrategy(_BareStrategy):
    """Strategy with a concrete alpha threshold bundle."""

    FILTER_THRESHOLDS: ClassVar[FilterThresholds] = _AlphaThresholds()


def test_unconfigured_get_raises() -> None:
    """Strategies without FILTER_THRESHOLDS should fail closed on get."""
    with pytest.raises(AttributeError, match="FILTER_THRESHOLDS"):
        _BareStrategy.get_filter_thresholds()


def test_set_rejects_non_filter_thresholds() -> None:
    """Setter should reject values that are not FilterThresholds instances."""
    with pytest.raises(TypeError, match="FilterThresholds"):
        _BareStrategy.set_filter_thresholds({"zscore_min": 1.0})  # type: ignore[arg-type]


def test_set_then_get_on_unconfigured_class() -> None:
    """First set on a None class should accept any FilterThresholds subclass."""
    original = _BareStrategy.FILTER_THRESHOLDS
    try:
        bundle = _AlphaThresholds(zscore_min=3.0)
        _BareStrategy.set_filter_thresholds(bundle)
        assert _BareStrategy.get_filter_thresholds() is bundle
        assert _BareStrategy.FILTER_THRESHOLDS is bundle
    finally:
        _BareStrategy.FILTER_THRESHOLDS = original


def test_concrete_type_enforced_after_configured() -> None:
    """Once configured, setter should reject a different concrete subclass."""
    original = _AlphaStrategy.FILTER_THRESHOLDS
    try:
        updated = replace(original, zscore_min=4.0)
        _AlphaStrategy.set_filter_thresholds(updated)
        assert _AlphaStrategy.get_filter_thresholds() == updated

        with pytest.raises(TypeError, match="_AlphaThresholds"):
            _AlphaStrategy.set_filter_thresholds(_BetaThresholds())
    finally:
        _AlphaStrategy.set_filter_thresholds(original)


def test_instance_property_reads_class_thresholds() -> None:
    """Instance property should mirror the class-level bundle."""
    strategy = _AlphaStrategy(_make_dataset())
    assert strategy.filter_thresholds is _AlphaStrategy.get_filter_thresholds()
    assert isinstance(strategy.filter_thresholds, _AlphaThresholds)
