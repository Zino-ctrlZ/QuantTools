"""Smoke tests for datamanager NA logging on retrieval APIs."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from trade.datamanager._enums import RealTimeFallbackOption
from trade.datamanager.market_data import AtIndexResult, TimeseriesData
from trade.datamanager.result import (
    ForwardResult,
    GreekResultSet,
    ModelResultPack,
    OptionSpotResult,
    RatesResult,
    SpotResult,
)
from trade.datamanager.utils.logging import MODEL_NA_LOGGER_NAME
from trade.datamanager.utils.na_logging import log_retrieval_na
from trade.helpers.Logging import get_logger_base_location


def _log_file_path() -> Path:
    """Return the NA logger file path for the current environment."""
    return get_logger_base_location() / f"{MODEL_NA_LOGGER_NAME}_test.log"


def _count_na_log_lines() -> int:
    """Count lines in the NA log file that mention NA detection."""
    path = _log_file_path()
    if not path.exists():
        return 0
    text = path.read_text()
    return sum(1 for line in text.splitlines() if "NA detected" in line)


@pytest.fixture
def na_log_baseline() -> int:
    """Snapshot NA log line count before each test."""
    return _count_na_log_lines()


def _assert_new_na_log(baseline: int, *, label: str) -> None:
    """Assert at least one new NA log entry was written."""
    current = _count_na_log_lines()
    assert current > baseline, f"Expected new NA log for {label}; baseline={baseline}, current={current}"


def test_log_retrieval_na_result_dispatch(na_log_baseline: int) -> None:
    """Result subclasses route through log_result_na."""
    idx = pd.to_datetime(["2024-01-02"])
    result = SpotResult(
        symbol="TEST",
        timeseries=pd.Series([float("nan")], index=idx, name="spot"),
    )
    log_retrieval_na(result, manager="spot", method="synthetic")
    _assert_new_na_log(na_log_baseline, label="Result dispatch")


def test_log_retrieval_na_model_pack_dispatch(na_log_baseline: int) -> None:
    """ModelResultPack routes through log_model_result_pack_na."""
    idx = pd.to_datetime(["2024-01-02"])
    packet = ModelResultPack(
        spot=SpotResult(
            symbol="TEST",
            timeseries=pd.Series([float("nan")], index=idx, name="spot"),
        ),
        undo_adjust=True,
    )
    log_retrieval_na(packet, manager="model", method="synthetic")
    _assert_new_na_log(na_log_baseline, label="ModelResultPack dispatch")


def test_log_retrieval_na_timeseries_duplicate_index_does_not_raise(na_log_baseline: int) -> None:
    """Duplicate-index spot rows do not crash NA snapshot logging at that date."""
    idx = pd.to_datetime(["2026-06-14", "2026-06-15", "2026-06-15"])
    spot = pd.DataFrame({"close": [100.0, 101.0, 102.0]}, index=idx)
    rates_idx = pd.to_datetime(["2026-06-14", "2026-06-15"])
    rates = pd.Series([0.05, float("nan")], index=rates_idx, name="r")
    data = TimeseriesData(
        spot=spot,
        chain_spot=None,
        dividends=None,
        dividend_yield=None,
        split_factor=None,
        rates=rates,
    )
    log_retrieval_na(data, manager="market_timeseries", method="synthetic", symbol="NVDA")
    _assert_new_na_log(na_log_baseline, label="TimeseriesData duplicate index")


def test_log_retrieval_na_timeseries_data_dispatch(na_log_baseline: int) -> None:
    """TimeseriesData routes through log_timeseries_data_na."""
    idx = pd.to_datetime(["2024-01-02"])
    data = TimeseriesData(
        spot=pd.DataFrame({"close": [float("nan")]}, index=idx),
        chain_spot=None,
        dividends=None,
        dividend_yield=None,
        split_factor=None,
    )
    log_retrieval_na(data, manager="market_timeseries", method="synthetic", symbol="TEST")
    _assert_new_na_log(na_log_baseline, label="TimeseriesData dispatch")


def test_log_retrieval_na_at_index_dispatch(na_log_baseline: int) -> None:
    """AtIndexResult routes through log_at_index_result_na."""
    result = AtIndexResult(
        sym="TEST",
        date="2024-01-02",
        spot=pd.Series({"close": float("nan")}),
        chain_spot=pd.Series({"close": 100.0}),
        dividends=0.0,
        dividend_yield=0.0,
        split_factor=1.0,
        rates=float("nan"),
    )
    log_retrieval_na(result, manager="market_timeseries", method="synthetic")
    _assert_new_na_log(na_log_baseline, label="AtIndexResult dispatch")


def test_at_index_rates_nan_placeholder_not_logged(na_log_baseline: int) -> None:
    """Placeholder rates=np.nan from get_at_index should not trigger NA logging."""
    result = AtIndexResult(
        sym="TEST",
        date="2024-01-02",
        spot=pd.Series({"close": 100.0}),
        chain_spot=pd.Series({"close": 100.0}),
        dividends=0.0,
        dividend_yield=0.0,
        split_factor=1.0,
        rates=float("nan"),
    )
    log_retrieval_na(result, manager="market_timeseries", method="get_at_index")
    assert _count_na_log_lines() == na_log_baseline


def test_log_retrieval_na_series_dispatch(na_log_baseline: int) -> None:
    """Raw Series routes through log_pandas_series_na."""
    idx = pd.to_datetime(["2024-01-02"])
    series = pd.Series([float("nan")], index=idx, name="div_yield")
    log_retrieval_na(series, manager="dividend", method="synthetic")
    _assert_new_na_log(na_log_baseline, label="Series dispatch")


def test_log_retrieval_na_scalar_dispatch(na_log_baseline: int) -> None:
    """Scalar NA routes through log_scalar_na."""
    log_retrieval_na(float("nan"), manager="market_timeseries", method="synthetic")
    _assert_new_na_log(na_log_baseline, label="Scalar dispatch")


def test_log_includes_thetadata_use_v3(na_log_baseline: int) -> None:
    """NA snapshots include THETADATA_USE_V3 from the environment."""
    os.environ["THETADATA_USE_V3"] = "true"
    idx = pd.to_datetime(["2024-01-02"])
    result = RatesResult(timeseries=pd.Series([float("nan")], index=idx, name="r"))
    log_retrieval_na(result, manager="rates", method="env_check")
    _assert_new_na_log(na_log_baseline, label="environment context")
    assert "'thetadata_use_v3': 'true'" in _log_file_path().read_text()


def test_rates_get_rate_decorator_logs_nan_fallback(na_log_baseline: int) -> None:
    """Decorated get_rate logs when NAN fallback is intentionally returned."""
    from trade.datamanager.rates import RatesDataManager

    # Saturday — triggers non-business-day branch with explicit NAN fallback.
    result = RatesDataManager().get_rate(
        date="2024-01-06",
        fallback_option=RealTimeFallbackOption.NAN,
    )
    assert result.timeseries is not None and result.timeseries.isna().all()
    _assert_new_na_log(na_log_baseline, label="rates.get_rate decorator")


def test_spot_get_spot_timeseries_decorator(na_log_baseline: int) -> None:
    """Decorated get_spot_timeseries logs when returned series contains NA."""
    from trade.datamanager.spot import SpotDataManager

    idx = pd.to_datetime(["2024-01-02"])
    nan_df = pd.DataFrame({"close": [float("nan")]}, index=idx)
    mock_ts = MagicMock()
    mock_ts._get_chain_spot_timeseries.return_value = nan_df

    with patch("trade.datamanager.spot.TS", mock_ts), patch("trade.datamanager.spot.load_name"):
        result = SpotDataManager("TEST").get_spot_timeseries(
            start_date="2024-01-02",
            end_date="2024-01-02",
            undo_adjust=True,
        )

    assert result.timeseries is not None and result.timeseries.isna().any()
    _assert_new_na_log(na_log_baseline, label="spot.get_spot_timeseries decorator")


def test_forward_get_forward_decorator(na_log_baseline: int) -> None:
    """Decorated get_forward logs when inner timeseries contains NA."""
    from trade.datamanager.forward import ForwardDataManager

    idx = pd.to_datetime(["2024-01-02"])
    mock_result = ForwardResult(
        symbol="TEST",
        timeseries=pd.Series([float("nan")], index=idx, name="forward"),
    )

    with (
        patch.object(ForwardDataManager, "get_forward_timeseries", return_value=mock_result),
        patch("trade.datamanager.forward.load_name"),
    ):
        result = ForwardDataManager("TEST").get_forward(
            date="2024-01-02",
            maturity_date="2024-06-21",
        )

    assert result.timeseries is not None and result.timeseries.isna().any()
    _assert_new_na_log(na_log_baseline, label="forward.get_forward decorator")


def test_dividend_get_div_yield_history_decorator(na_log_baseline: int) -> None:
    """Decorated get_div_yield_history logs NA in returned Series."""
    from trade.datamanager.dividend import DividendDataManager

    idx = pd.to_datetime(["2024-01-02"])
    nan_series = pd.Series([float("nan")], index=idx, name="div_yield")

    with patch("trade.datamanager.dividend.TS") as mock_ts:
        mock_ts._get_dividend_yield_timeseries.return_value = nan_series
        series = DividendDataManager("TEST").get_div_yield_history("TEST")

    assert series.isna().any()
    _assert_new_na_log(na_log_baseline, label="dividend.get_div_yield_history decorator")


def test_option_spot_decorator(na_log_baseline: int) -> None:
    """Decorated option_spot API logs NA in returned OptionSpotResult."""
    from trade.datamanager.option_spot import OptionSpotDataManager

    idx = pd.to_datetime(["2024-01-02"])
    nan_df = pd.DataFrame(
        {"close": [float("nan")], "midpoint": [float("nan")]},
        index=idx,
    )
    mock_result = OptionSpotResult(
        symbol="TEST",
        strike=100.0,
        expiration=datetime(2024, 6, 21),
        right="C",
        timeseries=nan_df,
    )

    with patch.object(
        OptionSpotDataManager,
        "get_option_spot_timeseries",
        return_value=mock_result,
    ):
        result = OptionSpotDataManager("TEST").get_option_spot(
            date="2024-01-02",
            strike=100.0,
            expiration="2024-06-21",
            right="C",
        )

    assert result.timeseries is not None and result.timeseries.isna().any().any()
    _assert_new_na_log(na_log_baseline, label="option_spot.get_option_spot decorator")


def test_vol_decorator(na_log_baseline: int) -> None:
    """Decorated get_at_time_implied_volatility logs when NAN fallback is returned."""
    from trade.datamanager.vol import VolDataManager

    result = VolDataManager("TEST").get_at_time_implied_volatility(
        as_of="2024-01-06",
        expiration="2024-06-21",
        strike=100.0,
        right="C",
        fallback_option=RealTimeFallbackOption.NAN,
    )

    assert result.timeseries is not None and result.timeseries.isna().any()
    _assert_new_na_log(na_log_baseline, label="vol.get_at_time_implied_volatility decorator")


def test_greeks_decorator(na_log_baseline: int) -> None:
    """Decorated get_at_time_greeks logs when NAN fallback is returned."""
    from trade.datamanager.greeks import GreekDataManager

    result = GreekDataManager("TEST").get_at_time_greeks(
        as_of="2024-01-06",
        expiration="2024-06-21",
        strike=100.0,
        right="C",
        fallback_option=RealTimeFallbackOption.NAN,
    )

    assert result.timeseries is not None and result.timeseries.isna().any().any()
    _assert_new_na_log(na_log_baseline, label="greeks.get_at_time_greeks decorator")


def test_market_timeseries_get_timeseries_decorator(na_log_baseline: int) -> None:
    """Decorated MarketTimeseries.get_timeseries logs NA across populated fields."""
    from trade.datamanager.market_data import MarketTimeseries

    idx = pd.to_datetime(["2024-01-02"])
    nan_spot = pd.DataFrame({"close": [float("nan")]}, index=idx)

    with patch.object(MarketTimeseries, "_get_spot_timeseries", return_value=nan_spot):
        result = MarketTimeseries().get_timeseries("TEST", factor="spot")

    assert result.spot is not None and result.spot.isna().any().any()
    _assert_new_na_log(na_log_baseline, label="market_timeseries.get_timeseries decorator")


def test_clean_result_does_not_log(na_log_baseline: int) -> None:
    """Clean results should not append NA log entries."""
    idx = pd.to_datetime(["2024-01-02"])
    result = RatesResult(timeseries=pd.Series([0.05], index=idx, name="r"))
    log_retrieval_na(result, manager="rates", method="clean_control")
    assert _count_na_log_lines() == na_log_baseline
