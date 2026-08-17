"""Unit tests for static position attribution (no portfolio, no Trade)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from EventDriven.attribution import (
    BacktestPositionAttribution,
    LegTimeseries,
    StaticPosition,
    StaticPositionAttributionAnalyzer,
    _normalize_leg_direction,
    _quantity_time_series_from_static,
    compute_static_position_attribution,
    create_static_position_attribution,
    structure_mark_at,
)


def _market_frame(n: int = 5, mid: float = 2.0) -> pd.DataFrame:
    """Build a synthetic market option frame.

    Args:
        n: Number of business days.
        mid: Constant Midpoint level.

    Returns:
        Market-shaped DataFrame.
    """
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    return pd.DataFrame(
        {
            "Rho": 0.01,
            "Gamma": 0.02,
            "Delta": 0.3,
            "Theta": -0.05,
            "Volga": 0.0,
            "Vega": 0.1,
            "Midpoint": mid,
            "Vol": 0.25,
        },
        index=idx,
    )


def _unit_attr(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Build unit attribution over an index.

    Args:
        index: Date index.

    Returns:
        Unit attribution frame with required columns used downstream.
    """
    n = len(index)
    return pd.DataFrame(
        {
            "opt_dod_change": [1.0] * n,
            "delta_pnl": [0.25] * n,
            "gamma_pnl": [0.0] * n,
            "vega_pnl": [0.0] * n,
            "theta_pnl": [0.0] * n,
            "volga_pnl": [0.0] * n,
            "vanna_pnl": [0.0] * n,
            "rho_pnl": [0.0] * n,
            "unexplained_pnl": [0.05] * n,
            "opt_spot": [2.0] * n,
        },
        index=index,
    )


def test_normalize_leg_direction() -> None:
    """Direction tokens normalize to L/S."""
    assert _normalize_leg_direction("LONG") == "L"
    assert _normalize_leg_direction("l") == "L"
    assert _normalize_leg_direction("SHORT") == "S"
    assert _normalize_leg_direction("S") == "S"
    with pytest.raises(ValueError):
        _normalize_leg_direction("B")


def test_structure_mark_at_signed_sum() -> None:
    """Structure mark is long mid minus short mid."""
    idx = pd.date_range("2024-01-02", periods=3, freq="B")
    long = LegTimeseries("AAA", "LONG", _market_frame(3, mid=3.0).reindex(idx))
    short = LegTimeseries("BBB", "SHORT", _market_frame(3, mid=1.0).reindex(idx))
    assert structure_mark_at([long, short], idx[1]) == pytest.approx(2.0)


def test_create_static_position_attribution_sums_and_flips() -> None:
    """create_static sums legs and flips short units."""
    frame = _market_frame(5)
    idx = frame.index
    unit = _unit_attr(idx)
    long = LegTimeseries("AAA", "L", frame)
    short = LegTimeseries("BBB", "S", frame)
    pos = StaticPosition(
        position_id="p1",
        legs=[long, short],
        entry_date=idx[1],
        exit_date=idx[3],
    )

    with patch(
        "EventDriven.attribution.load_option_pnl_data",
        return_value=SimpleNamespace(attribution=unit.copy()),
    ) as mock_load:
        combined = create_static_position_attribution(pos)
        assert mock_load.call_count == 2
        ## long +1 and short -1 cancel on equal units
        assert float(combined["opt_dod_change"].iloc[1]) == pytest.approx(0.0)


def test_quantity_series_open_close_events() -> None:
    """Synthetic qty series opens first day and closes last day."""
    frame = _market_frame(4)
    pos = StaticPosition(
        position_id="p1",
        legs=[LegTimeseries("AAA", "LONG", frame)],
        entry_date=frame.index[0],
        exit_date=frame.index[-1],
        quantity=2.0,
        commission=0.05,
        slippage_pct=0.01,
    )
    qty = _quantity_time_series_from_static(pos, frame.index)
    assert float(qty.quantity_change.iloc[0]) == pytest.approx(2.0)
    assert float(qty.quantity_change.iloc[-1]) == pytest.approx(-2.0)
    assert float(qty.daily_qty.iloc[1]) == pytest.approx(2.0)
    assert float(qty.daily_qty.iloc[-1]) == pytest.approx(0.0)
    assert float(qty.commission.iloc[0]) == pytest.approx(0.05)
    ## mark mid=2, slip_pct=0.01 -> 0.02 per unit
    assert float(qty.slippage.iloc[0]) == pytest.approx(0.02)


def test_compute_static_reuses_open_close_path() -> None:
    """Open day zeros greek components; mid days scale by quantity."""
    frame = _market_frame(5)
    idx = frame.index
    unit = _unit_attr(idx)
    pos = StaticPosition(
        position_id="p1",
        legs=[LegTimeseries("AAA", "LONG", frame)],
        entry_date=idx[1],
        exit_date=idx[3],
        quantity=3.0,
        commission=0.0,
        slippage_pct=0.0,
        signal_id="sigA",
    )

    with patch(
        "EventDriven.attribution.load_option_pnl_data",
        return_value=SimpleNamespace(attribution=unit.copy()),
    ):
        result = compute_static_position_attribution(pos)

    assert isinstance(result, BacktestPositionAttribution)
    assert result.trade_id == "p1"
    assert result.signal_id == "sigA"
    attr = result.attribution
    ## Entry day just_opened: non-trade components zeroed by compute_position_attribution.
    entry = pd.to_datetime(idx[1])
    assert float(attr.loc[entry, "delta_pnl"]) == pytest.approx(0.0)
    ## Mid hold day: scaled by qty=3.
    mid = pd.to_datetime(idx[2])
    assert float(attr.loc[mid, "opt_dod_change"]) == pytest.approx(3.0)
    assert float(attr.loc[mid, "delta_pnl"]) == pytest.approx(0.75)


def test_static_analyzer_workflow() -> None:
    """Analyzer analyze_all and convert_attribution_to_df match portfolio UX."""
    frame = _market_frame(4)
    idx = frame.index
    unit = _unit_attr(idx)
    pos = StaticPosition(
        position_id="p1",
        legs=[LegTimeseries("AAA", "SHORT", frame)],
        entry_date=idx[0],
        exit_date=idx[-1],
        quantity=1.0,
        signal_id="sigA",
    )
    with patch(
        "EventDriven.attribution.load_option_pnl_data",
        return_value=SimpleNamespace(attribution=unit.copy()),
    ):
        analyzer = StaticPositionAttributionAnalyzer(positions={"p1": pos})
        cache = analyzer.analyze_all()
        assert "p1" in cache
        by_signal = analyzer.convert_attribution_to_df(groupby="signal_id")
        assert "sigA" in by_signal.index


def test_static_analyzer_rejects_non_static_position() -> None:
    """Raw dataframes rejected."""
    with pytest.raises(TypeError):
        StaticPositionAttributionAnalyzer(positions={"p1": _market_frame(2)})  # type: ignore[arg-type]
