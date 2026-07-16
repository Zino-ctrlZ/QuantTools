"""Comprehensive tests for live_diag checkpoints and load-path instrumentation.

Usage:
    >>> pytest EventDriven/tests/test_live_diag.py -q
"""

from __future__ import annotations

import logging
import math
from datetime import date
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import patch

import pandas as pd
import pytest

from EventDriven.riskmanager import live_diag
from EventDriven.riskmanager.live_diag import (
    GREEK_ARTIFACT_COLUMNS,
    JOINED_OPTION_FIELD_COLUMNS,
    LEG_LOADED_FIELD_COLUMNS,
    OPTION_TRADE_FIELD_COLUMNS,
    disable_live_diag,
    enable_live_diag,
    fields_at_current_date,
    get_row_at_date,
    log_live_checkpoint,
    log_option_data_checkpoint,
)


FIXED_TODAY = date(2026, 7, 15)


@pytest.fixture(autouse=True)
def _reset_live_diag_gate() -> None:
    """Ensure each test starts with live diag disabled."""
    disable_live_diag()


@pytest.fixture(autouse=True)
def _pin_ny_today(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin live_diag row lookup to a fixed NY calendar day for deterministic tests."""
    monkeypatch.setattr(
        "trade.helpers.helper.ny_now",
        lambda: pd.Timestamp("2026-07-15 14:30:00"),
    )


@pytest.fixture
def live_diag_caplog(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    """Capture live_diag logger output (custom setup_logger bypasses root caplog)."""
    caplog.set_level(logging.INFO, logger=live_diag.logger.name)
    live_diag.logger.addHandler(caplog.handler)
    try:
        yield caplog
    finally:
        live_diag.logger.removeHandler(caplog.handler)


def _live_chk_lines(caplog: pytest.LogCaptureFixture) -> List[str]:
    return [rec.message for rec in caplog.records if "live_chk" in rec.message]


def _sample_frame(*, delta: float = 0.42, gamma: float = 0.01) -> pd.DataFrame:
    idx = pd.DatetimeIndex([pd.Timestamp("2026-07-15")])
    return pd.DataFrame(
        {
            "Delta": [delta],
            "Gamma": [gamma],
            "Vega": [0.12],
            "Theta": [-0.03],
            "Rho": [0.02],
            "Vol": [0.25],
            "Midpoint": [2.58],
            "Closebid": [2.47],
            "Closeask": [2.69],
            "s": [211.0],
        },
        index=idx,
    )


class TestLiveDiagHelpers:
    def test_value_flag_classifications(self) -> None:
        assert live_diag._value_flag(1.0) == "ok"
        assert live_diag._value_flag(float("nan")) == "nan"
        assert live_diag._value_flag(float("inf")) == "inf"
        assert live_diag._value_flag(None) == "missing"
        assert live_diag._value_flag("n/a") == "nonnum"

    def test_get_row_at_date_handles_timestamp_and_date_index(self) -> None:
        frame = _sample_frame()
        row = get_row_at_date(frame, FIXED_TODAY)
        assert row is not None
        assert math.isclose(float(row["Delta"]), 0.42)

    def test_get_row_at_date_defaults_to_ny_today(self) -> None:
        frame = _sample_frame()
        row = get_row_at_date(frame)
        assert row is not None
        assert math.isclose(float(row["Vol"]), 0.25)

    def test_get_row_at_date_missing_returns_none(self) -> None:
        frame = _sample_frame()
        assert get_row_at_date(frame, date(2026, 1, 1)) is None
        assert get_row_at_date(None, FIXED_TODAY) is None

    def test_fields_at_current_date_case_insensitive(self) -> None:
        frame = _sample_frame().rename(columns=str.lower)
        out = fields_at_current_date(frame, ["delta", "gamma"], date_value=FIXED_TODAY)
        assert out["row_found"] is True
        assert math.isclose(float(out["delta"]), 0.42)
        assert math.isclose(float(out["gamma"]), 0.01)

    def test_fields_at_current_date_on_series(self) -> None:
        idx = pd.DatetimeIndex([pd.Timestamp("2026-07-15")])
        vol = pd.Series([0.25], index=idx, name="iv")
        out = fields_at_current_date(vol, ["iv"], date_value=FIXED_TODAY)
        assert out["row_found"] is True
        assert math.isclose(float(out["iv"]), 0.25)

    def test_fields_at_current_date_missing_row(self) -> None:
        out = fields_at_current_date(_sample_frame(), ["Delta"], date_value=date(2020, 1, 1))
        assert out["row_found"] is False
        assert out["Delta"] is None

    def test_fields_at_current_date_swallows_unexpected_errors(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("boom")

        monkeypatch.setattr(live_diag, "get_row_at_date", _boom)
        out = fields_at_current_date(_sample_frame(), ["Delta"])
        assert out["row_found"] is False
        assert out["Delta"] is None


class TestLogLiveCheckpoint:
    def test_no_emit_when_disabled(self, live_diag_caplog: pytest.LogCaptureFixture) -> None:
        log_live_checkpoint("leg_loaded", trade_id="X", data={"delta": 1.0})
        assert not _live_chk_lines(live_diag_caplog)

    def test_emit_with_flags_when_enabled(self, live_diag_caplog: pytest.LogCaptureFixture) -> None:
        enable_live_diag()
        log_live_checkpoint(
            "leg_loaded",
            trade_id="&L:AAPL20270115C430",
            data={"delta": float("nan"), "gamma": 0.01, "vol": 0.25},
        )
        lines = _live_chk_lines(live_diag_caplog)
        assert len(lines) == 1
        assert "stage=leg_loaded" in lines[0]
        assert "date=2026-07-15" in lines[0]
        assert "'delta': 'nan'" in lines[0]
        assert "'gamma': 'ok'" in lines[0]

    def test_log_live_checkpoint_swallows_errors(
        self, monkeypatch: pytest.MonkeyPatch, live_diag_caplog: pytest.LogCaptureFixture
    ) -> None:
        enable_live_diag()

        def _boom(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("logger boom")

        monkeypatch.setattr(live_diag.logger, "log", _boom)
        log_live_checkpoint("leg_loaded", trade_id="X", data={"delta": 1.0})

    def test_log_option_data_checkpoint_includes_greeks_vol_spot(
        self, live_diag_caplog: pytest.LogCaptureFixture
    ) -> None:
        enable_live_diag()
        log_option_data_checkpoint(
            "option_data_raw",
            opttick="AAPL20270115C430",
            data=_sample_frame(),
        )
        line = _live_chk_lines(live_diag_caplog)[0]
        assert "stage=option_data_raw" in line
        assert "'Delta': 'ok'" in line
        assert "'Gamma': 'ok'" in line
        assert "'Vol': 'ok'" in line
        assert "'Midpoint': 'ok'" in line
        assert "'s': 'ok'" in line


class TestColumnContracts:
    def test_joined_columns_cover_greeks_vol_quotes_spot(self) -> None:
        assert JOINED_OPTION_FIELD_COLUMNS == (
            "Delta",
            "Gamma",
            "Vega",
            "Theta",
            "Rho",
            "Vol",
            "Midpoint",
            "Closebid",
            "Closeask",
            "s",
        )
        assert LEG_LOADED_FIELD_COLUMNS == JOINED_OPTION_FIELD_COLUMNS

    def test_option_trade_columns_include_greeks_and_spot(self) -> None:
        for col in ("Delta", "Gamma", "Vega", "Theta", "Vol", "Midpoint", "s"):
            assert col in OPTION_TRADE_FIELD_COLUMNS


class TestLoadPositionDataNew:
    def test_emits_artifact_and_position_data_loaded_stages(
        self,
        live_diag_caplog: pytest.LogCaptureFixture,
    ) -> None:
        enable_live_diag()

        idx = pd.DatetimeIndex([pd.Timestamp("2026-07-15")])
        greeks = pd.DataFrame(
            {c: [float("nan") if c == "delta" else 0.1] for c in GREEK_ARTIFACT_COLUMNS},
            index=idx,
        )
        option_spot = pd.DataFrame({"midpoint": [2.58], "closebid": [2.47], "closeask": [2.69]}, index=idx)
        vol = pd.Series([0.25], index=idx, name="iv")
        spot = pd.Series([211.0], index=idx, name="close")
        rates = pd.Series([0.04], index=idx, name="rate")
        dividends = pd.Series([0.0], index=idx, name="div")

        pack = SimpleNamespace(
            greek=SimpleNamespace(timeseries=greeks),
            option_spot=SimpleNamespace(timeseries=option_spot),
            spot=SimpleNamespace(timeseries=spot),
            dividend=SimpleNamespace(timeseries=dividends),
            rates=SimpleNamespace(timeseries=rates),
            vol=SimpleNamespace(timeseries=vol),
        )

        cache: Dict[str, pd.DataFrame] = {}
        with patch(
            "EventDriven.riskmanager.utils.load_full_option_data",
            return_value=pack,
        ) as load_mock:
            from EventDriven.riskmanager.utils import load_position_data_new

            out = load_position_data_new(
                "AAPL20270115C430",
                cache,
                start="2026-01-01",
                end="2026-12-31",
            )

        load_mock.assert_called_once()
        assert not out.empty

        stages = [line.split("stage=")[1].split(" ")[0] for line in _live_chk_lines(live_diag_caplog)]
        assert stages == [
            "greek_artifact_loaded",
            "option_spot_artifact_loaded",
            "vol_artifact_loaded",
            "spot_artifact_loaded",
            "position_data_loaded",
        ]
        def _stage_line(stage: str) -> str:
            return next(
                line for line in _live_chk_lines(live_diag_caplog) if f"stage={stage}" in line
            )

        assert "'delta': 'nan'" in _stage_line("greek_artifact_loaded")
        assert "'iv': 'ok'" in _stage_line("vol_artifact_loaded")
        assert "'midpoint': 'ok'" in _stage_line("option_spot_artifact_loaded")
        assert "'s': 'ok'" in _stage_line("spot_artifact_loaded")
        pos_line = _stage_line("position_data_loaded")
        assert "'Gamma': 'ok'" in pos_line
        assert "'Vol': 'ok'" in pos_line
        assert "'s': 'ok'" in pos_line

    def test_cache_hit_skips_artifact_checkpoints(self, live_diag_caplog: pytest.LogCaptureFixture) -> None:
        enable_live_diag()
        cached = _sample_frame()
        cache = {"AAPL20270115C430": cached}

        from EventDriven.riskmanager.utils import load_position_data_new

        out = load_position_data_new(
            "AAPL20270115C430",
            cache,
            start="2026-01-01",
            end="2026-12-31",
        )
        assert out is cached
        assert not _live_chk_lines(live_diag_caplog)


class TestGenerateOptionDataForTrade:
    def _make_timeseries(self) -> SimpleNamespace:
        return SimpleNamespace(
            start_date=pd.Timestamp("2026-01-01"),
            end_date=pd.Timestamp("2026-12-31"),
            options_cache={},
            session_loaded_option_cache={},
            adjusted_strike_cache={},
            splits={},
        )

    def test_no_split_path_emits_intermediate_stages(
        self,
        live_diag_caplog: pytest.LogCaptureFixture,
    ) -> None:
        enable_live_diag()

        from EventDriven.riskmanager.market_timeseries import BacktestTimeseries

        ts = self._make_timeseries()
        frame = _sample_frame()
        check_date = FIXED_TODAY
        ts.get_splits = lambda ticker, bkt_end_date: []
        ts.get_special_dividends = lambda ticker: {}
        ts.load_position_data = lambda opttick: frame.copy()
        ts._option_data_sanity_check = lambda **props: None
        ts._ffill_adj_strike_business_days = lambda strike_series, start_date, end_date: strike_series

        out = BacktestTimeseries.generate_option_data_for_trade(ts, "AAPL20270115C430", check_date)

        assert not out.empty
        stages = [line.split("stage=")[1].split(" ")[0] for line in _live_chk_lines(live_diag_caplog)]
        assert stages == ["option_data_raw", "option_data_windowed", "option_data_ready"]
        for stage in stages:
            line = next(line for line in _live_chk_lines(live_diag_caplog) if f"stage={stage}" in line)
            assert "'Delta': 'ok'" in line
            assert "'Gamma': 'ok'" in line
            assert "'Vol': 'ok'" in line

    def test_session_cache_hit_skips_intermediate_stages(
        self,
        live_diag_caplog: pytest.LogCaptureFixture,
    ) -> None:
        enable_live_diag()

        from EventDriven.riskmanager.market_timeseries import BacktestTimeseries

        ts = self._make_timeseries()
        frame = _sample_frame()
        key = ("AAPL20270115C430", "2026-07-15")
        ts.session_loaded_option_cache[key] = frame

        out = BacktestTimeseries.generate_option_data_for_trade(ts, "AAPL20270115C430", FIXED_TODAY)
        assert out is frame
        assert not _live_chk_lines(live_diag_caplog)


class TestLegLoadedFieldExtraction:
    def test_leg_loaded_payload_includes_greeks_vol_spot(self) -> None:
        frame = _sample_frame()
        payload = fields_at_current_date(frame, LEG_LOADED_FIELD_COLUMNS)
        assert payload["row_found"] is True
        for col in ("Delta", "Gamma", "Vega", "Theta", "Rho", "Vol", "Midpoint", "s"):
            assert col in payload
            assert payload[col] is not None
