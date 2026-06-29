"""Probe: is list_dates (the source of checked_missing) self-consistent,
and does it agree with the EOD endpoint that actually produces spot prices?

Hammers list_dates N times for a non-expired contract and contrasts with
retrieve_eod_ohlc over the same window.
"""

from __future__ import annotations
import pandas as pd
from dbase.database.db_utils import set_environment_context
from trade.datamanager.config import setup_config_for_live

set_environment_context("long_bbands_v2")
setup_config_for_live()

from trade.datamanager.base import BaseDataManager
from trade.datamanager.utils.classification import get_option_dates
from dbase.DataAPI.ThetaData import list_dates, retrieve_eod_ohlc

CONTRACTS = [
    dict(symbol="NVDA", strike=280.0, right="C", expiration=pd.Timestamp("2026-12-18")),
    dict(symbol="QQQ", strike=985.0, right="C", expiration=pd.Timestamp("2026-12-18")),
]
START, END = "2026-06-08", "2026-06-22"
N = 8


def window(dates):
    s, e = pd.Timestamp(START), pd.Timestamp(END)
    return sorted(
        d.strftime("%Y-%m-%d")
        for d in pd.to_datetime(list(dates))
        if s <= pd.Timestamp(d).normalize() <= e
    )


for c in CONTRACTS:
    print("\n" + "=" * 78)
    print(f"{c['symbol']} {c['strike']}{c['right']} exp={c['expiration'].date()}  window {START}..{END}")
    print("=" * 78)
    BaseDataManager.clear_all_caches()

    # Hammer the raw vendor list_dates endpoint.
    seen = {}
    for i in range(N):
        raw = list_dates(symbol=c["symbol"], strike=c["strike"], right=c["right"], exp=c["expiration"])
        w = tuple(window(raw))
        seen.setdefault(w, 0)
        seen[w] += 1
        print(f"  list_dates#{i+1}: in-window count={len(w)} -> {list(w)}")

    print(f"\n  DISTINCT in-window results across {N} calls: {len(seen)}")
    for w, n in seen.items():
        print(f"    x{n}: {list(w)}")

    # The endpoint that actually produces spot prices.
    eod = retrieve_eod_ohlc(
        symbol=c["symbol"], start_date=START, end_date=END,
        strike=c["strike"], exp=c["expiration"], right=c["right"],
    )
    eod_dates = window(eod.index) if eod is not None and len(eod) else []
    print(f"\n  retrieve_eod_ohlc in-window rows ({len(eod_dates)}): {eod_dates}")

    # Disagreement: dates EOD has real prices for, but the (most common) list_dates omits.
    most_common = max(seen.items(), key=lambda kv: kv[1])[0]
    missing_from_listdates = sorted(set(eod_dates) - set(most_common))
    print(f"  EOD has data but list_dates(most-common) omits: {missing_from_listdates}")

print("\nDONE")
