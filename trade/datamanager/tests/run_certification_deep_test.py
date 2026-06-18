"""Deep integration smoke test for certification + point-in-time wiring.

Run:
    python -m trade.datamanager.tests.run_certification_deep_test
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, List, Optional

import pandas as pd

from trade.datamanager.config import OptionDataConfig, CertificationLevel, OptionPricingModel
from trade.datamanager._enums import DivType
from trade.datamanager.rates import RatesDataManager
from trade.datamanager.spot import SpotDataManager
from trade.datamanager.option_spot import OptionSpotDataManager
from trade.datamanager.forward import ForwardDataManager
from trade.datamanager.dividend import DividendDataManager
from trade.datamanager.vol import VolDataManager
from trade.datamanager.greeks import GreekDataManager
from trade.datamanager.vars import get_times_series
from trade.optionlib.config.types import DivType as DivTypeLib


TEST_OPTION_META = {
    "symbol": "AAPL",
    "right": "C",
    "expiration": "2026-12-18",
    "strike": 380.0,
}
TEST_START = "2026-01-02"
TEST_END = "2026-06-17"
TEST_AS_OF = "2026-06-17"
MATURITY = TEST_OPTION_META["expiration"]
SYMBOL = TEST_OPTION_META["symbol"]


@dataclass
class TestRow:
    """One test case outcome."""

    category: str
    name: str
    ok: bool
    detail: str = ""
    error: Optional[str] = None


@dataclass
class TestReport:
    """Aggregated test report."""

    rows: List[TestRow] = field(default_factory=list)

    def add(self, category: str, name: str, fn: Callable[[], Any]) -> None:
        """Run ``fn`` and record pass/fail."""
        try:
            result = fn()
            detail = _summarize(result)
            self.rows.append(TestRow(category=category, name=name, ok=True, detail=detail))
        except Exception as exc:  # noqa: BLE001 — integration harness
            self.rows.append(
                TestRow(
                    category=category,
                    name=name,
                    ok=False,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    def print_summary(self) -> int:
        """Print table and return exit code (0 pass, 1 fail)."""
        width = 72
        print("=" * width)
        print("CERTIFICATION + POINT-IN-TIME DEEP TEST")
        print(f"Option: {TEST_OPTION_META}  |  window: {TEST_START} .. {TEST_END}  |  as_of: {TEST_AS_OF}")
        print("=" * width)
        fails = 0
        for cat in sorted({r.category for r in self.rows}):
            print(f"\n## {cat}")
            for row in [r for r in self.rows if r.category == cat]:
                mark = "PASS" if row.ok else "FAIL"
                print(f"  [{mark}] {row.name}")
                if row.detail:
                    print(f"         {row.detail}")
                if row.error:
                    print(f"         {row.error}")
                    fails += 1
                elif not row.ok:
                    fails += 1
        passed = sum(1 for r in self.rows if r.ok)
        print("\n" + "=" * width)
        print(f"TOTAL: {passed}/{len(self.rows)} passed, {fails} failed")
        print("=" * width)
        return 1 if fails else 0


def _summarize(result: Any) -> str:
    """Compact one-line summary of a manager result."""
    if isinstance(result, pd.DataFrame):
        return f"DataFrame rows={len(result)}, cols={list(result.columns)[:4]}"
    if isinstance(result, pd.Series):
        return f"Series len={len(result)}"
    certified = getattr(result, "is_certified", None)
    parts = [f"is_certified={certified}"]
    if hasattr(result, "timeseries") and result.timeseries is not None:
        ts = result.timeseries
        parts.append(f"timeseries len={len(ts)}")
    if hasattr(result, "daily_option_spot") and result.daily_option_spot is not None:
        parts.append(f"option_spot rows={len(result.daily_option_spot)}")
    if hasattr(result, "daily_spot") and result.daily_spot is not None:
        parts.append(f"daily_spot len={len(result.daily_spot)}")
    if hasattr(result, "daily_discrete_forward") and result.daily_discrete_forward is not None:
        parts.append(f"fwd len={len(result.daily_discrete_forward)}")
    if hasattr(result, "daily_discrete_dividends") and result.daily_discrete_dividends is not None:
        parts.append(f"div len={len(result.daily_discrete_dividends)}")
    if hasattr(result, "endpoint_source") and result.endpoint_source is not None:
        parts.append(f"endpoint={result.endpoint_source}")
    if hasattr(result, "key") and result.key:
        parts.append(f"key={result.key[:48]}...")
    if isinstance(result, (int, float)):
        parts.append(f"value={result}")
    if hasattr(result, "sym"):
        parts.append(f"sym={result.sym}")
    return ", ".join(parts)


def _assert_certified(result: Any, *, expect: bool = True) -> Any:
    """Assert ``is_certified`` flag when present."""
    flag = getattr(result, "is_certified", None)
    if flag is not None and flag is not expect:
        raise AssertionError(f"expected is_certified={expect}, got {flag}")
    return result


def _assert_nonempty_timeseries(result: Any) -> Any:
    """Assert result carries non-empty primary payload."""
    if isinstance(result, pd.DataFrame):
        if len(result) == 0:
            raise AssertionError("empty DataFrame")
        return result
    if isinstance(result, pd.Series):
        if len(result) == 0:
            raise AssertionError("empty Series")
        return result
    for attr in (
        "timeseries",
        "daily_option_spot",
        "daily_spot",
        "daily_discrete_forward",
        "daily_continuous_forward",
        "daily_discrete_dividends",
        "daily_continuous_dividends",
    ):
        payload = getattr(result, attr, None)
        if payload is not None and hasattr(payload, "__len__") and len(payload) > 0:
            return result
    if isinstance(result, (int, float)):
        return result
    raise AssertionError("no non-empty timeseries payload on result")


def _assert_single_row(result: Any) -> Any:
    """Assert point-in-time result is a single row/value."""
    for attr in ("timeseries", "daily_option_spot", "daily_spot", "daily_discrete_forward"):
        payload = getattr(result, attr, None)
        if payload is not None:
            n = len(payload)
            if n != 1:
                raise AssertionError(f"{attr} expected 1 row, got {n}")
            return result
    raise AssertionError("no point-in-time payload found")


def main() -> int:
    """Run all integration checks."""
    conf = OptionDataConfig()
    conf.certification_level = CertificationLevel.L3
    conf.option_model = OptionPricingModel.BSM
    conf.allow_rates_resample_on_missing = True
    conf.dividend_type = DivTypeLib.DISCRETE

    report = TestReport()
    rates = RatesDataManager()
    spot = SpotDataManager(SYMBOL)
    opt = OptionSpotDataManager(SYMBOL)
    fwd = ForwardDataManager(SYMBOL)
    div = DividendDataManager(SYMBOL)
    vol = VolDataManager(SYMBOL)
    greek = GreekDataManager(SYMBOL)
    mt = get_times_series()

    strike = TEST_OPTION_META["strike"]
    right = TEST_OPTION_META["right"]
    expiration = TEST_OPTION_META["expiration"]

    ## --- Timeseries (certified) ---
    report.add(
        "timeseries",
        "rates.get_risk_free_rate_timeseries",
        lambda: _assert_nonempty_timeseries(
            _assert_certified(rates.get_risk_free_rate_timeseries(TEST_START, TEST_END))
        ),
    )
    report.add(
        "timeseries",
        "spot.get_spot_timeseries",
        lambda: _assert_nonempty_timeseries(_assert_certified(spot.get_spot_timeseries(TEST_START, TEST_END))),
    )
    report.add(
        "timeseries",
        "option_spot.get_option_spot_timeseries",
        lambda: _assert_nonempty_timeseries(
            _assert_certified(
                opt.get_option_spot_timeseries(
                    TEST_START, TEST_END, strike=strike, expiration=expiration, right=right
                )
            )
        ),
    )
    report.add(
        "timeseries",
        "forward.get_forward_timeseries",
        lambda: _assert_nonempty_timeseries(
            _assert_certified(
                fwd.get_forward_timeseries(TEST_START, TEST_END, maturity_date=MATURITY)
            )
        ),
    )
    report.add(
        "timeseries",
        "dividend.get_schedule_timeseries",
        lambda: _assert_nonempty_timeseries(
            _assert_certified(
                div.get_schedule_timeseries(TEST_START, TEST_END, maturity_date=MATURITY)
            )
        ),
    )
    report.add(
        "timeseries",
        "vol.get_implied_volatility_timeseries",
        lambda: _assert_nonempty_timeseries(
            _assert_certified(
                vol.get_implied_volatility_timeseries(
                    TEST_START, TEST_END, expiration=expiration, strike=strike, right=right, american=False
                )
            )
        ),
    )
    report.add(
        "timeseries",
        "greeks.get_greeks_timeseries",
        lambda: _assert_nonempty_timeseries(
            _assert_certified(
                greek.get_greeks_timeseries(
                    TEST_START, TEST_END, expiration=expiration, strike=strike, right=right
                )
            )
        ),
    )
    report.add(
        "timeseries",
        "market_timeseries._get_spot_timeseries",
        lambda: _assert_nonempty_timeseries(
            mt._get_spot_timeseries(SYMBOL, start=TEST_START, end=TEST_END)  # noqa: SLF001
        ),
    )
    report.add(
        "timeseries",
        "market_timeseries._get_chain_spot_timeseries",
        lambda: _assert_nonempty_timeseries(
            mt._get_chain_spot_timeseries(SYMBOL, start=TEST_START, end=TEST_END)  # noqa: SLF001
        ),
    )

    ## --- Point-in-time (L1 lookback) ---
    report.add(
        "get_at_time",
        "rates.get_rate",
        lambda: _assert_single_row(_assert_certified(rates.get_rate(TEST_AS_OF))),
    )
    report.add(
        "get_at_time",
        "spot.get_at_time",
        lambda: _assert_single_row(spot.get_at_time(TEST_AS_OF)),
    )
    report.add(
        "get_at_time",
        "option_spot.get_option_spot",
        lambda: _assert_single_row(
            _assert_certified(
                opt.get_option_spot(TEST_AS_OF, strike=strike, expiration=expiration, right=right)
            )
        ),
    )
    report.add(
        "get_at_time",
        "forward.get_forward",
        lambda: _assert_single_row(
            _assert_certified(fwd.get_forward(TEST_AS_OF, maturity_date=MATURITY))
        ),
    )
    report.add(
        "get_at_time",
        "dividend.get_schedule",
        lambda: _assert_single_row(
            _assert_certified(div.get_schedule(TEST_AS_OF, maturity_date=MATURITY))
        ),
    )
    report.add(
        "get_at_time",
        "vol.get_at_time_implied_volatility",
        lambda: _assert_single_row(
            _assert_certified(
                vol.get_at_time_implied_volatility(
                    TEST_AS_OF, expiration=expiration, strike=strike, right=right, american=False
                )
            )
        ),
    )
    report.add(
        "get_at_time",
        "greeks.get_at_time_greeks",
        lambda: _assert_single_row(
            _assert_certified(
                greek.get_at_time_greeks(TEST_AS_OF, expiration=expiration, strike=strike, right=right)
            )
        ),
    )
    report.add(
        "get_at_time",
        "market_timeseries.get_at_index",
        lambda: mt.get_at_index(SYMBOL, pd.Timestamp(TEST_AS_OF)),
    )
    report.add(
        "get_at_time",
        "market_timeseries.get_split_factor_at_index",
        lambda: mt.get_split_factor_at_index(SYMBOL, pd.Timestamp(TEST_AS_OF)),
    )

    ## --- rt() ---
    report.add("rt", "rates.rt", lambda: _assert_single_row(rates.rt()))
    report.add("rt", "spot.rt", lambda: spot.rt())
    report.add(
        "rt",
        "option_spot.rt",
        lambda: opt.rt(strike=strike, right=right, expiration=expiration),
    )
    report.add("rt", "forward.rt", lambda: fwd.rt(maturity_date=MATURITY))
    report.add("rt", "dividend.rt", lambda: div.rt(maturity_date=MATURITY))
    report.add(
        "rt",
        "vol.rt",
        lambda: vol.rt(expiration=expiration, strike=strike, right=right, american=False),
    )
    report.add(
        "rt",
        "greeks.rt",
        lambda: greek.rt(expiration=expiration, strike=strike, right=right),
    )

    return report.print_summary()


if __name__ == "__main__":
    raise SystemExit(main())
