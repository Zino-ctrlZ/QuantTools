from datetime import datetime

import pandas as pd
import pytest

from trade.backtester_.option_vectorized_retrieval import (
    OptionRetrievalSignal,
    OptionVectorizedRetriever,
)


def test_run_requires_exactly_one_source() -> None:
    retriever = OptionVectorizedRetriever(target_dte=30, target_moneyness=1.0)
    with pytest.raises(ValueError):
        retriever.run(signals=None, strategy=None)


def test_explicit_right_precedence() -> None:
    retriever = OptionVectorizedRetriever(target_dte=30, target_moneyness=1.0, right="P")
    signal = OptionRetrievalSignal(
        ticker="AAPL",
        start_date="2026-01-02",
        end_date="2026-01-30",
        side=1,
        quantity=1.0,
        signal_id="s1",
        right="C",
    )
    assert retriever._resolve_signal_right(signal) == "C"


def test_run_selects_contract_and_returns_ohlc(monkeypatch: pytest.MonkeyPatch) -> None:
    retriever = OptionVectorizedRetriever(
        target_dte=30,
        target_moneyness=1.0,
        right="C",
        dte_tolerance=5,
        moneyness_tolerance=0.2,
    )

    chain = pd.DataFrame(
        {
            "Root": ["AAPL", "AAPL"],
            "Expiration": ["20260201", "20260215"],
            "Strike": [100.0, 110.0],
            "Right": ["C", "C"],
            "Spot": [100.0, 100.0],
            "Open_interest": [1000, 500],
        }
    )

    ohlc = pd.DataFrame(
        {
            "Open": [1.0, 1.1],
            "High": [1.2, 1.3],
            "Low": [0.9, 1.0],
            "Close": [1.1, 1.2],
            "Midpoint": [1.05, 1.15],
        },
        index=pd.DatetimeIndex([datetime(2026, 1, 2), datetime(2026, 1, 3)]),
    )

    chain_spot_ts = pd.DataFrame(
        {
            "close": [100.0, 100.0],
        },
        index=pd.DatetimeIndex([datetime(2026, 1, 2), datetime(2026, 1, 3)]),
    )

    def _mock_chain_bulk(**kwargs):
        return chain

    def _mock_eod_ohlc(**kwargs):
        return ohlc

    class _MockTsResult:
        def __init__(self, chain_spot: pd.DataFrame):
            self.chain_spot = chain_spot

    class _MockMarketTimeseries:
        def get_timeseries(self, **kwargs):
            return _MockTsResult(chain_spot=chain_spot_ts)

    def _mock_get_timeseries_obj(*args, **kwargs):
        return _MockMarketTimeseries()

    monkeypatch.setattr(
        "trade.backtester_.option_vectorized_retrieval.retrieve_chain_bulk",
        _mock_chain_bulk,
    )
    monkeypatch.setattr(
        "trade.backtester_.option_vectorized_retrieval.retrieve_eod_ohlc",
        _mock_eod_ohlc,
    )
    monkeypatch.setattr(
        "trade.backtester_.option_vectorized_retrieval.get_timeseries_obj",
        _mock_get_timeseries_obj,
    )

    signal = OptionRetrievalSignal(
        ticker="AAPL",
        start_date="2026-01-02",
        end_date="2026-01-10",
        side=1,
        quantity=1.0,
        signal_id="sig-123",
    )
    result = retriever.run(signals=[signal])

    assert len(result.selected_contracts) == 1
    assert result.unmatched_signals.empty
    assert "sig-123" in result.signal_ohlc
    assert len(result.signal_ohlc["sig-123"]) == 1
    assert not result.signal_ohlc["sig-123"][0].empty


def test_roll_enabled_builds_multiple_segments(monkeypatch: pytest.MonkeyPatch) -> None:
    retriever = OptionVectorizedRetriever(
        target_dte=5,
        target_moneyness=1.0,
        right="C",
        roll_enabled=True,
        roll_on_dte=0,
        dte_tolerance=50,
        moneyness_tolerance=1.0,
    )

    chain_first = pd.DataFrame(
        {
            "Root": ["AAPL"],
            "Expiration": ["20260105"],
            "Strike": [100.0],
            "Right": ["C"],
            "Open_interest": [1000],
        }
    )
    chain_second = pd.DataFrame(
        {
            "Root": ["AAPL"],
            "Expiration": ["20260120"],
            "Strike": [100.0],
            "Right": ["C"],
            "Open_interest": [900],
        }
    )

    def _mock_chain_bulk(**kwargs):
        if kwargs["start_date"] <= "2026-01-05":
            return chain_first
        return chain_second

    def _mock_eod_ohlc(**kwargs):
        start_dt = pd.to_datetime(kwargs["start_date"])
        end_dt = pd.to_datetime(kwargs["end_date"])
        idx = pd.date_range(start=start_dt, end=end_dt, freq="D")
        return pd.DataFrame(
            {
                "Open": [1.0] * len(idx),
                "High": [1.1] * len(idx),
                "Low": [0.9] * len(idx),
                "Close": [1.0] * len(idx),
                "Midpoint": [1.0] * len(idx),
            },
            index=idx,
        )

    chain_spot_ts = pd.DataFrame(
        {
            "close": [100.0] * 30,
        },
        index=pd.date_range(start="2026-01-01", periods=30, freq="D"),
    )

    class _MockTsResult:
        def __init__(self, chain_spot: pd.DataFrame):
            self.chain_spot = chain_spot

    class _MockMarketTimeseries:
        def get_timeseries(self, **kwargs):
            return _MockTsResult(chain_spot=chain_spot_ts)

    def _mock_get_timeseries_obj(*args, **kwargs):
        return _MockMarketTimeseries()

    monkeypatch.setattr(
        "trade.backtester_.option_vectorized_retrieval.retrieve_chain_bulk",
        _mock_chain_bulk,
    )
    monkeypatch.setattr(
        "trade.backtester_.option_vectorized_retrieval.retrieve_eod_ohlc",
        _mock_eod_ohlc,
    )
    monkeypatch.setattr(
        "trade.backtester_.option_vectorized_retrieval.get_timeseries_obj",
        _mock_get_timeseries_obj,
    )

    signal = OptionRetrievalSignal(
        ticker="AAPL",
        start_date="2026-01-02",
        end_date="2026-01-15",
        side=1,
        quantity=1.0,
        signal_id="sig-roll",
    )
    result = retriever.run(signals=[signal])

    assert result.unmatched_signals.empty
    assert "sig-roll" in result.signal_ohlc
    assert len(result.signal_ohlc["sig-roll"]) == 2
    assert len(result.selected_contracts) == 2
    assert set(result.selected_contracts["roll_index"].tolist()) == {0, 1}


def test_run_n_size_samples_signals(monkeypatch: pytest.MonkeyPatch) -> None:
    retriever = OptionVectorizedRetriever(
        target_dte=30,
        target_moneyness=1.0,
        right="C",
        dte_tolerance=5,
        moneyness_tolerance=0.2,
    )

    chain = pd.DataFrame(
        {
            "Root": ["AAPL", "MSFT"],
            "Expiration": ["20260201", "20260201"],
            "Strike": [100.0, 100.0],
            "Right": ["C", "C"],
            "Spot": [100.0, 100.0],
            "Open_interest": [1000, 1000],
        }
    )

    ohlc = pd.DataFrame(
        {
            "Open": [1.0],
            "High": [1.2],
            "Low": [0.9],
            "Close": [1.1],
            "Midpoint": [1.05],
        },
        index=pd.DatetimeIndex([datetime(2026, 1, 2)]),
    )

    chain_spot_ts = pd.DataFrame(
        {"close": [100.0, 100.0]},
        index=pd.DatetimeIndex([datetime(2026, 1, 2), datetime(2026, 1, 3)]),
    )

    def _mock_chain_bulk(**kwargs):
        return chain

    def _mock_eod_ohlc(**kwargs):
        return ohlc

    class _MockTsResult:
        def __init__(self, chain_spot: pd.DataFrame):
            self.chain_spot = chain_spot

    class _MockMarketTimeseries:
        def get_timeseries(self, **kwargs):
            return _MockTsResult(chain_spot=chain_spot_ts)

    def _mock_get_timeseries_obj(*args, **kwargs):
        return _MockMarketTimeseries()

    def _sample_first(self, n=None, *args, **kwargs):
        return self.iloc[:n].copy()

    monkeypatch.setattr(
        "trade.backtester_.option_vectorized_retrieval.retrieve_chain_bulk",
        _mock_chain_bulk,
    )
    monkeypatch.setattr(
        "trade.backtester_.option_vectorized_retrieval.retrieve_eod_ohlc",
        _mock_eod_ohlc,
    )
    monkeypatch.setattr(
        "trade.backtester_.option_vectorized_retrieval.get_timeseries_obj",
        _mock_get_timeseries_obj,
    )
    monkeypatch.setattr(pd.DataFrame, "sample", _sample_first)

    signals = [
        OptionRetrievalSignal(
            ticker="AAPL",
            start_date="2026-01-02",
            end_date="2026-01-10",
            side=1,
            quantity=1.0,
            signal_id="sig-a",
        ),
        OptionRetrievalSignal(
            ticker="MSFT",
            start_date="2026-01-02",
            end_date="2026-01-10",
            side=1,
            quantity=1.0,
            signal_id="sig-b",
        ),
    ]

    full_result = retriever.run(signals=signals)
    assert set(full_result.normalized_signals["signal_id"].tolist()) == {"sig-a", "sig-b"}
    assert set(full_result.selected_contracts["signal_id"].tolist()) == {"sig-a", "sig-b"}

    sampled_result = retriever.run(signals=signals, n_size=1)
    assert len(sampled_result.normalized_signals) == 1
    assert len(sampled_result.selected_contracts) == 1
