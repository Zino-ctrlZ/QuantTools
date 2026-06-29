"""Forced-error test for the list_dates truncation bug.

Deterministically simulates the ThetaData fault we observed (list_dates randomly
drops the in-window tail of dates) by monkeypatching list_dates to ALWAYS drop
2026-06-12..2026-06-22 for the contract, while leaving the EOD price endpoint
(retrieve_eod_ohlc) real — which always returns the true prices.

Asserts, under the forced fault:
  A) option spot keeps all real price rows (no NaN placeholders for priced days),
  B) IV and greeks have no NaN,
  C) certification PASSES at L2 and L3,
  D) the vol/greeks checked-missing exemption is NOT inflated by the fault
     (the fix reconciles against observed prices).

Run (QuantTools, env openbb_new_use):
    python .sandbox_vol_debug/test_forced_error.py
"""

from __future__ import annotations

import sys
import traceback
import numpy as np
import pandas as pd

pd.set_option("display.width", 200)

from dbase.database.db_utils import set_environment_context
from trade.datamanager.config import setup_config_for_live

set_environment_context("long_bbands_v2")
setup_config_for_live()

from trade.datamanager._enums import CertificationLevel, OptionSpotEndpointSource
from trade.datamanager.base import BaseDataManager
from trade.datamanager.option_spot import OptionSpotDataManager
from trade.datamanager.vol import VolDataManager
from trade.datamanager.greeks import GreekDataManager
from trade.datamanager.utils.vol_helpers import resolve_checked_missing_dates_for_option_artifact

import trade.datamanager.utils.classification as classmod
import trade.datamanager.utils.date as datemod
from dbase.DataAPI.ThetaData import list_dates as _real_list_dates

LEG = dict(symbol="NVDA", strike=280.0, right="C", expiration="2026-12-18")
START, END = "2026-06-08", "2026-06-22"
DROP_START, DROP_END = pd.Timestamp("2026-06-12"), pd.Timestamp("2026-06-22")

_force_calls = {"n": 0}


def _truncated_list_dates(*args, **kwargs):
    """Real list_dates, then forcibly drop the in-window tail to simulate the fault."""
    full = _real_list_dates(*args, **kwargs)
    dates = pd.to_datetime(list(full))
    kept = [d for d in dates if not (DROP_START <= pd.Timestamp(d).normalize() <= DROP_END)]
    _force_calls["n"] += 1
    return np.array(kept, dtype="datetime64[ns]")


def _nan_dates(ts) -> list:
    if ts is None or len(ts) == 0:
        return []
    mask = ts.isna().any(axis=1) if isinstance(ts, pd.DataFrame) else ts.isna()
    return [d.strftime("%Y-%m-%d") for d in pd.to_datetime(ts.index[mask])]


def _all_dates(ts) -> list:
    if ts is None or len(ts) == 0:
        return []
    return [d.strftime("%Y-%m-%d") for d in pd.to_datetime(ts.index)]


def main() -> int:
    failures: list[str] = []
    exp = LEG["expiration"]

    # Patch BOTH modules that bound `list_dates` via `from ... import list_dates`.
    orig_classmod = classmod.list_dates
    orig_datemod = datemod.list_dates
    classmod.list_dates = _truncated_list_dates
    datemod.list_dates = _truncated_list_dates

    try:
        BaseDataManager.clear_all_caches()
        print(">> cleared caches; list_dates is FORCED-truncated (drops 06-12..06-22)\n")

        # Sanity: prove the fault is active.
        forced = _truncated_list_dates(symbol=LEG["symbol"], strike=LEG["strike"], right=LEG["right"], exp=pd.Timestamp(exp))
        in_window = sorted(
            d.strftime("%Y-%m-%d")
            for d in pd.to_datetime(list(forced))
            if pd.Timestamp(START) <= pd.Timestamp(d).normalize() <= pd.Timestamp(END)
        )
        print(f"[fault] forced list_dates in-window: {in_window}")
        assert in_window == ["2026-06-08", "2026-06-09", "2026-06-10", "2026-06-11"], in_window

        # (A) Option spot under the fault.
        osm = OptionSpotDataManager(LEG["symbol"])
        os_res = osm.get_option_spot_timeseries(
            start_date=START, end_date=END,
            strike=LEG["strike"], right=LEG["right"], expiration=exp,
            endpoint_source=OptionSpotEndpointSource.EOD,
            certification_level=CertificationLevel.L1,
        )
        osd = os_res.daily_option_spot
        os_rows, os_nan = _all_dates(osd), _nan_dates(osd)
        print(f"\n[A] option_spot rows ({len(os_rows)}): {os_rows}")
        print(f"[A] option_spot NaN-row dates: {os_nan}")
        ## The 10 in-window business days the fault tried to drop must all be present and priced.
        expected_in_window = [
            "2026-06-08", "2026-06-09", "2026-06-10", "2026-06-11", "2026-06-12",
            "2026-06-15", "2026-06-16", "2026-06-17", "2026-06-18", "2026-06-22",
        ]
        missing_priced = [d for d in expected_in_window if d not in os_rows]
        if missing_priced:
            failures.append(f"A: in-window priced days dropped by fault: {missing_priced}")
        if os_nan:
            failures.append(f"A: option_spot has NaN placeholder rows: {os_nan}")

        # (D) Exemption set must NOT be inflated by the fault.
        cm = resolve_checked_missing_dates_for_option_artifact(
            symbol=LEG["symbol"], strike=LEG["strike"], right=LEG["right"],
            expiration=exp, start_date=START, end_date=END,
        )
        cm = sorted(str(d) for d in cm)
        print(f"\n[D] artifact checked_missing under fault: {cm}")
        if cm:
            failures.append(f"D: checked_missing inflated by fault (should be []): {cm}")

        # (B) IV under the fault.
        vm = VolDataManager(LEG["symbol"])
        iv_res = vm.get_implied_volatility_timeseries(
            start_date=START, end_date=END,
            expiration=exp, strike=LEG["strike"], right=LEG["right"],
            certification_level=CertificationLevel.L1,
        )
        iv = iv_res.timeseries
        iv_nan = _nan_dates(iv)
        print(f"\n[B] IV rows ({len(_all_dates(iv))}); NaN dates: {iv_nan}; is_certified={getattr(iv_res,'is_certified',None)}")
        if iv_nan:
            failures.append(f"B: IV has NaN dates under fault: {iv_nan}")

        # (B) Greeks under the fault.
        gm = GreekDataManager(LEG["symbol"])
        gr_res = gm.get_greeks_timeseries(
            start_date=START, end_date=END,
            expiration=exp, strike=LEG["strike"], right=LEG["right"],
            certification_level=CertificationLevel.L1,
        )
        gr = gr_res.timeseries
        gr_nan = _nan_dates(gr)
        print(f"[B] greeks rows ({len(_all_dates(gr))}); NaN dates: {gr_nan}; is_certified={getattr(gr_res,'is_certified',None)}")
        if gr_nan:
            failures.append(f"B: greeks has NaN dates under fault: {gr_nan}")

        # (C) Full L2 and L3 certification must pass (no raise) under the fault.
        for lvl in (CertificationLevel.L2, CertificationLevel.L3):
            try:
                vm.get_implied_volatility_timeseries(
                    start_date=START, end_date=END,
                    expiration=exp, strike=LEG["strike"], right=LEG["right"],
                    certification_level=lvl,
                )
                print(f"[C] IV {lvl}: PASSED")
            except Exception as e:
                failures.append(f"C: IV {lvl} raised {type(e).__name__}: {e}")
                print(f"[C] IV {lvl}: RAISED {type(e).__name__}: {e}")
            try:
                gm.get_greeks_timeseries(
                    start_date=START, end_date=END,
                    expiration=exp, strike=LEG["strike"], right=LEG["right"],
                    certification_level=lvl,
                )
                print(f"[C] greeks {lvl}: PASSED")
            except Exception as e:
                failures.append(f"C: greeks {lvl} raised {type(e).__name__}: {e}")
                print(f"[C] greeks {lvl}: RAISED {type(e).__name__}: {e}")

    finally:
        classmod.list_dates = orig_classmod
        datemod.list_dates = orig_datemod

    print("\n" + "=" * 78)
    if failures:
        print(f"RESULT: FAIL ({len(failures)} issue(s))")
        for f in failures:
            print("  - " + f)
        return 1
    print("RESULT: PASS — forced ThetaData truncation no longer breaks spot/IV/greeks/cert")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
