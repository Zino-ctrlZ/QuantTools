"""Prove list_dates flakiness is live ThetaData API, not local datamanager cache."""

from __future__ import annotations

import pandas as pd
from dbase.database.db_utils import set_environment_context
from trade.datamanager.config import setup_config_for_live

set_environment_context("long_bbands_v2")
setup_config_for_live()

from trade.datamanager.base import BaseDataManager
from trade.datamanager.utils.date import LIST_DATE_CACHE
from trade.datamanager.utils.classification import get_option_dates
from dbase.DataAPI.ThetaData import list_dates

SYMBOL, STRIKE, RIGHT, EXP = "NVDA", 280.0, "C", pd.Timestamp("2026-12-18")
START, END = "2026-06-08", "2026-06-22"


def in_window(dates) -> list[str]:
    s, e = pd.Timestamp(START), pd.Timestamp(END)
    return sorted(
        d.strftime("%Y-%m-%d")
        for d in pd.to_datetime(list(dates))
        if s <= pd.Timestamp(d).normalize() <= e
    )


BaseDataManager.clear_all_caches()
print("LIST_DATE_CACHE size after clear:", len(LIST_DATE_CACHE))
print("option expired?:", EXP < pd.Timestamp.now())
print()

for i in range(3):
  # Direct API wrapper (what switcher logs as v3).
  raw = list_dates(symbol=SYMBOL, strike=STRIKE, right=RIGHT, exp=EXP, print_url=(i == 0))
  print(f"direct list_dates#{i+1}: {len(in_window(raw))} dates in window")

print()
print("LIST_DATE_CACHE keys after direct calls:", list(LIST_DATE_CACHE.keys())[:3], "... total", len(LIST_DATE_CACHE))

for i in range(3):
  via_get = get_option_dates(SYMBOL, STRIKE, RIGHT, EXP.to_pydatetime())
  print(f"get_option_dates#{i+1}: {len(in_window(via_get))} dates in window")

print()
print("LIST_DATE_CACHE entry for contract (non-expired should NOT store full range):")
opttick = f"{SYMBOL}{EXP.strftime('%Y%m%d')}{RIGHT}{int(STRIKE*1000):08d}"
entry = LIST_DATE_CACHE.get(opttick)
print("  key present:", entry is not None)
if entry:
    print("  keys in entry:", list(entry.keys()) if isinstance(entry, dict) else type(entry))
    print("  has 'range':", isinstance(entry, dict) and "range" in entry)
