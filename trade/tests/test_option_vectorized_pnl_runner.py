from types import SimpleNamespace

import pandas as pd

from trade.backtester_.option_vectorized_pnl_runner import OptionVectorizedPnLRunner
from trade.backtester_.option_vectorized_retrieval import OptionVectorizedRetrievalResult


def _make_retrieval_result() -> OptionVectorizedRetrievalResult:
    selected_contracts = pd.DataFrame(
        [
            {
                "signal_id": "sig_a",
                "ticker": "AAPL",
                "right": "C",
                "strike": 100.0,
                "expiration": "2026-01-17",
                "roll_index": 0,
                "segment_start": "2026-01-02",
                "segment_end": "2026-01-05",
            },
            {
                "signal_id": "sig_a",
                "ticker": "AAPL",
                "right": "C",
                "strike": 105.0,
                "expiration": "2026-01-24",
                "roll_index": 1,
                "segment_start": "2026-01-05",
                "segment_end": "2026-01-07",
            },
        ]
    )

    dummy_seg_0 = pd.DataFrame(
        {
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
        },
        index=pd.to_datetime(["2026-01-02"]),
    )
    dummy_seg_1 = pd.DataFrame(
        {
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
        },
        index=pd.to_datetime(["2026-01-05"]),
    )

    return OptionVectorizedRetrievalResult(
        normalized_signals=selected_contracts[["signal_id", "ticker"]].drop_duplicates().copy(),
        selected_contracts=selected_contracts,
        signal_ohlc={"sig_a": [dummy_seg_0, dummy_seg_1]},
        unmatched_signals=pd.DataFrame(),
    )


def test_runner_aggregates_rolls_and_builds_normalized_equity(monkeypatch):
    retrieval_result = _make_retrieval_result()

    def _mock_opttick(*, symbol, right, exp, strike):
        return f"{symbol}_{right}_{exp}_{int(strike)}"

    def _mock_load_option_pnl_data(*, yesterday, today, opttick, payload=None):
        dates = pd.date_range(pd.Timestamp(yesterday), pd.Timestamp(today), freq="D")
        attribution = pd.DataFrame(
            {
                "delta_pnl": [0.4] * len(dates),
                "total_pnl_excl_trade_pnl": [1.0] * len(dates),
                "trade_pnl_adjustment": [7.0] * len(dates),
                "unexplained_pnl": [0.1] * len(dates),
            },
            index=dates,
        )
        return SimpleNamespace(attribution=attribution)

    monkeypatch.setattr(
        "trade.backtester_.option_vectorized_pnl_runner.generate_opttick_new",
        _mock_opttick,
    )
    monkeypatch.setattr(
        "trade.backtester_.option_vectorized_pnl_runner.load_option_pnl_data",
        _mock_load_option_pnl_data,
    )

    runner = OptionVectorizedPnLRunner(starting_nav=100.0)
    result = runner.run(retrieval_result)

    assert result.failures.empty
    assert len(result.trade_summary) == 2

    assert "trade_pnl_adjustment" not in result.trade_daily.columns
    assert (result.trade_daily["total_pnl"] == result.trade_daily["total_pnl_excl_trade_pnl"]).all()

    overlap_row = result.signal_daily[
        (result.signal_daily["date"] == pd.Timestamp("2026-01-05")) & (result.signal_daily["signal_id"] == "sig_a")
    ]
    assert len(overlap_row) == 1
    assert float(overlap_row.iloc[0]["total_pnl"]) == 2.0

    eq = result.portfolio_equity_curve
    first_day = eq[eq["date"] == pd.Timestamp("2026-01-02")].iloc[0]
    assert float(first_day["normalized_equity"]) == 1.01


def test_runner_collects_failures_when_roll_segment_missing(monkeypatch):
    retrieval_result = _make_retrieval_result()
    retrieval_result.signal_ohlc = {"sig_a": [retrieval_result.signal_ohlc["sig_a"][0]]}

    def _mock_opttick(*, symbol, right, exp, strike):
        return f"{symbol}_{right}_{exp}_{int(strike)}"

    def _mock_load_option_pnl_data(*, yesterday, today, opttick, payload=None):
        dates = pd.date_range(pd.Timestamp(yesterday), pd.Timestamp(today), freq="D")
        attribution = pd.DataFrame(
            {
                "total_pnl_excl_trade_pnl": [1.0] * len(dates),
                "trade_pnl_adjustment": [0.0] * len(dates),
            },
            index=dates,
        )
        return SimpleNamespace(attribution=attribution)

    monkeypatch.setattr(
        "trade.backtester_.option_vectorized_pnl_runner.generate_opttick_new",
        _mock_opttick,
    )
    monkeypatch.setattr(
        "trade.backtester_.option_vectorized_pnl_runner.load_option_pnl_data",
        _mock_load_option_pnl_data,
    )

    result = OptionVectorizedPnLRunner().run(retrieval_result)

    assert len(result.failures) == 1
    assert "missing segment" in str(result.failures.iloc[0]["reason"]).lower()
    assert len(result.trade_summary) == 1
