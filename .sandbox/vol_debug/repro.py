"""Sandbox repro for the vol/greeks NaN certification failure.

Recreates the failing path for the legs seen in the cert logs
(QQQ 985C, NVDA 280C, exp 2026-12-18) and traces WHERE the option-spot
blank originates and whether it is consistently flagged checked-missing.

Run (from QuantTools, env openbb_new_use):
    python .sandbox_vol_debug/repro.py
"""

from __future__ import annotations

import sys
import traceback
import numpy as np
import pandas as pd

pd.set_option("display.max_rows", 60)
pd.set_option("display.width", 200)


def banner(msg: str) -> None:
    print("\n" + "=" * 78)
    print(msg)
    print("=" * 78)


# Faithful env setup (matches the notebook the user pointed at).
from dbase.database.db_utils import set_environment_context
from trade.datamanager.config import setup_config_for_live, OptionDataConfig

set_environment_context("long_bbands_v2")
setup_config_for_live()
cfg = OptionDataConfig()
print("config certification_level:", cfg.certification_level)
print("config is_live:", cfg.is_live)

from trade.datamanager._enums import CertificationLevel, OptionSpotEndpointSource
from trade.datamanager.base import BaseDataManager
from trade.datamanager.option_spot import OptionSpotDataManager
from trade.datamanager.vol import VolDataManager
from trade.datamanager.greeks import GreekDataManager
from trade.datamanager.utils.classification import (
    get_option_dates,
    resolve_checked_missing_dates_for_option_contract,
)

# Legs + window from the failing 2026-06-22 cert reports.
LEGS = [
    dict(symbol="QQQ", strike=985.0, right="C", expiration="2026-12-18"),
    dict(symbol="NVDA", strike=280.0, right="C", expiration="2026-12-18"),
]
START = "2026-06-08"
END = "2026-06-22"


def nan_dates(ts) -> list:
    if ts is None or len(ts) == 0:
        return []
    if isinstance(ts, pd.DataFrame):
        mask = ts.isna().any(axis=1)
    else:
        mask = ts.isna()
    return [d.strftime("%Y-%m-%d") for d in pd.to_datetime(ts.index[mask])]


def all_dates(ts) -> list:
    if ts is None or len(ts) == 0:
        return []
    return [d.strftime("%Y-%m-%d") for d in pd.to_datetime(ts.index)]


def run_leg(leg: dict) -> None:
    banner(f"LEG {leg['symbol']} {leg['strike']}{leg['right']} exp={leg['expiration']}  window {START}..{END}")

    # Force a clean recheck so nothing is served from a stale placeholder.
    BaseDataManager.clear_all_caches()
    print(">> cleared all caches")

    exp_dt = pd.to_datetime(leg["expiration"])

    # 1) Raw vendor calendar (the source of truth for checked-missing).
    try:
        vendor = get_option_dates(leg["symbol"], leg["strike"], leg["right"], exp_dt)
        vendor_in_window = sorted(
            d.strftime("%Y-%m-%d")
            for d in pd.to_datetime(list(vendor))
            if pd.Timestamp(START) <= pd.Timestamp(d).normalize() <= pd.Timestamp(END)
        )
        print(f"\n[1] vendor list_dates in window ({len(vendor_in_window)}): {vendor_in_window}")
    except Exception as e:
        print(f"[1] list_dates FAILED: {e!r}")
        vendor_in_window = None

    # 2) Resolved checked-missing for the contract (twice -> detect non-determinism).
    cm1 = resolve_checked_missing_dates_for_option_contract(
        symbol=leg["symbol"], strike=leg["strike"], right=leg["right"],
        expiration=exp_dt, valid_start=START, valid_end=END,
    )
    cm2 = resolve_checked_missing_dates_for_option_contract(
        symbol=leg["symbol"], strike=leg["strike"], right=leg["right"],
        expiration=exp_dt, valid_start=START, valid_end=END,
    )
    print(f"[2] checked_missing call#1: {sorted(str(d) for d in cm1)}")
    print(f"[2] checked_missing call#2: {sorted(str(d) for d in cm2)}")
    print(f"[2] deterministic across calls: {sorted(map(str, cm1)) == sorted(map(str, cm2))}")

    # 3) Option spot series (EOD) at L1 -> what the option chain produced.
    osm = OptionSpotDataManager(leg["symbol"])
    os_res = osm.get_option_spot_timeseries(
        start_date=START, end_date=END,
        strike=leg["strike"], right=leg["right"], expiration=leg["expiration"],
        endpoint_source=OptionSpotEndpointSource.EOD,
        certification_level=CertificationLevel.L1,
    )
    osd = os_res.daily_option_spot
    print(f"\n[3] option_spot rows: {all_dates(osd)}")
    print(f"[3] option_spot NaN-row dates: {nan_dates(osd)}")
    if osd is not None and len(osd):
        print(osd.to_string())

    # 4) IV series at L1 -> where NaN propagates and whether cert flags it.
    vm = VolDataManager(leg["symbol"])
    iv_res = vm.get_implied_volatility_timeseries(
        start_date=START, end_date=END,
        expiration=leg["expiration"], strike=leg["strike"], right=leg["right"],
        certification_level=CertificationLevel.L1,
    )
    iv = iv_res.timeseries
    iv_nan = nan_dates(iv)
    print(f"\n[4] IV rows: {all_dates(iv)}")
    print(f"[4] IV NaN dates: {iv_nan}")
    print(f"[4] IV is_certified: {getattr(iv_res, 'is_certified', None)}")

    # 5) Verdict for each IV-NaN date: listed by vendor? exempt as checked-missing?
    print("\n[5] PER-NaN-DATE VERDICT:")
    cm_set = set(str(d) for d in cm1)
    vendor_set = set(vendor_in_window) if vendor_in_window is not None else set()
    os_present = set(all_dates(osd))
    os_nan = set(nan_dates(osd))
    for d in iv_nan:
        listed = d in vendor_set
        exempt = d in cm_set
        in_os = d in os_present
        os_blank = d in os_nan
        print(
            f"    {d}: vendor_lists={listed}  exempt_checked_missing={exempt}  "
            f"in_option_spot={in_os}  option_spot_blank={os_blank}  "
            f"--> {'UNEXPLAINED (cert FAIL)' if (not exempt) else 'explained'}"
        )

    # 6) What L2 would do (the config default / latent failure).
    try:
        vm.get_implied_volatility_timeseries(
            start_date=START, end_date=END,
            expiration=leg["expiration"], strike=leg["strike"], right=leg["right"],
            certification_level=CertificationLevel.L2,
        )
        print("\n[6] L2 certification: PASSED (no raise)")
    except Exception as e:
        print(f"\n[6] L2 certification: RAISED -> {type(e).__name__}: {e}")


if __name__ == "__main__":
    for leg in LEGS:
        try:
            run_leg(leg)
        except Exception:
            print(f"\n!!! leg {leg} crashed:")
            traceback.print_exc()
    print("\nDONE")
