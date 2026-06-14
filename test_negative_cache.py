"""Test script for the negative-cache implementation.

Runs two consecutive calls for a date range that has no option data,
then reports whether the second call avoids an API hit.
"""

import os
import logging

logging.basicConfig(level=logging.INFO, format="%(name)s [%(levelname)s] %(message)s")

os.environ["THETADATA_USE_V3"] = "true"

from dbase.DataAPI.ThetaData import list_dates, set_use_proxy, retrieve_chain_bulk, list_contracts, retrieve_eod_ohlc
from trade.datamanager.utils.date import _sync_date, LIST_DATE_CACHE
from trade.datamanager.option_spot import OptionSpotDataManager

set_use_proxy("")

# ── Check the actual valid window from the API ──────────────────────────────
print("=" * 60)
print("Step 0: list_dates for QQQ 440C 2024-03-28")
d = list_dates(symbol="QQQ", exp="2024-03-28", strike=440, right="C", print_url=True)
d = list(d) if d is not None else []
if len(d) > 0:
    print(f"  first 5 : {d[:5]}")
    print(f"  min={min(d)}  max={max(d)}  count={len(d)}")
else:
    print("  No dates returned")

# ── Test params ──────────────────────────────────────────────────────────────
symbol = "QQQ"
start_date = "2023-12-27"
end_date = "2024-02-08"
opt_params = dict(strike=440.00, right="C", expiration="2024-03-28")

# ── Call 1: should hit the API and cache checked_missing_dates ───────────────
print("\n" + "=" * 60)
print("Call 1: Fetching from API (expect cache miss)")
result1 = OptionSpotDataManager(symbol=symbol).get_option_spot_timeseries(
    start_date=start_date, end_date=end_date, **opt_params
)
print("  daily_option_spot:")
print(result1.daily_option_spot)

# ── Inspect cache entry directly ─────────────────────────────────────────────
from trade.datamanager.utils.cache import _CachedData

mgr = OptionSpotDataManager(symbol=symbol)
from trade.datamanager._enums import ArtifactType, Interval, SeriesId, OptionSpotEndpointSource
from trade.datamanager.config import OptionDataConfig
from dbase.DataAPI.ThetaData.utils import _handle_opttick_param

strike, right, sym, expiration = _handle_opttick_param(
    strike=440.0, right="C", symbol=symbol, exp="2024-03-28", opttick=None, enforce_single_option=True
)
key = mgr.make_key(
    symbol=symbol,
    artifact_type=ArtifactType.OPTION_SPOT,
    series_id=SeriesId.HIST,
    endpoint_source=OptionSpotEndpointSource.EOD.value,
    interval=Interval.EOD,
    strike=strike,
    right=right,
    expiration=expiration,
)
print(f"\nCache key: {key}")
cached = mgr.get(key, default=None)
if isinstance(cached, _CachedData):
    print(f"  _CachedData found")
    print(f"  data shape     : {cached.data.shape}")
    print(f"  checked_missing: {cached.checked_missing_dates[:10]}")  # first 10
    print(f"  len(checked_missing): {len(cached.checked_missing_dates)}")
else:
    print(f"  Cached type: {type(cached)}")

# ── Call 2: should return from cache WITHOUT calling the API ─────────────────
print("\n" + "=" * 60)
print("Call 2: Should be a full cache hit (no API call)")
result2 = OptionSpotDataManager(symbol=symbol).get_option_spot_timeseries(
    start_date=start_date, end_date=end_date, **opt_params
)
print("  daily_option_spot:")
print(result2.daily_option_spot)
print("\nDone.")

# ── Call 3: wider window that includes both real data and checked-missing dates ──
print("\n" + "=" * 60)
print("Call 3: Wider range 2023-11-27 to 2024-02-08 (real data + checked-missing)")
result3 = OptionSpotDataManager(symbol=symbol).get_option_spot_timeseries(
    start_date="2023-11-27",
    end_date="2024-02-08",
    **opt_params,
)
df3 = result3.daily_option_spot
print(f"  shape: {df3.shape}")
print(f"  date range: {df3.index.min().date()} to {df3.index.max().date()}")
real_rows = df3.dropna(how="all")
nan_rows = df3[df3.isna().all(axis=1)]
print(f"  real rows   : {len(real_rows)}")
print(f"  NaN rows    : {len(nan_rows)}")
if len(nan_rows):
    print(f"  first NaN   : {nan_rows.index.min().date()}")
    print(f"  last NaN    : {nan_rows.index.max().date()}")
print("\nDone (all calls).")
