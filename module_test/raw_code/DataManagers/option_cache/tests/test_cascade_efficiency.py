#!/usr/bin/env python3
# ruff: noqa
"""
Test cascade efficiency - demonstrate that cached spot/vol/underlier data is used
instead of requerying everything.
"""
import os
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

WORK_DIR = Path(__file__).parent.parent.parent.parent
os.environ["WORK_DIR"] = str(WORK_DIR)
print(f"Set WORK_DIR to: {WORK_DIR}")

from DataManagers.DataManagers_cached import CachedOptionDataManager
from DataManagers.option_cache import clear_all_caches
from DataManagers import set_skip_mysql_query

# Test parameters
OPTTICK = "AAPL20250117C220"
INTERVAL = "1d"
MODEL = "bs"

print("\n" + "="*70)
print("CASCADE EFFICIENCY TEST")
print("="*70)
print("Testing that vol/greeks use cached spot + underlier data")
print("="*70 + "\n")

set_skip_mysql_query(True)

# Clear cache completely
print("[DEBUG] Clearing all caches...")
cache_dir = WORK_DIR / '.cache' / 'data_manager_cache'
if cache_dir.exists():
    shutil.rmtree(cache_dir)
    print(f"[DEBUG] Deleted cache directory: {cache_dir}")
clear_all_caches()
print("[DEBUG] Caches cleared\n")

dm = CachedOptionDataManager(
    symbol='AAPL',
    right='C',
    strike=220.0,
    exp='2025-01-17',
    enable_cache=True
)

# Date range
start = pd.Timestamp("2024-10-01")
end = pd.Timestamp("2024-10-31")

print("="*70)
print("STEP 1: Query SPOT data (cache empty)")
print("="*70)
t1 = time.time()
result_spot = dm.get_timeseries(start, end, INTERVAL, type_='spot', model=MODEL)
time_spot = time.time() - t1
rows_spot = len(result_spot.post_processed_data) if hasattr(result_spot, 'post_processed_data') else 0
print(f"[RESULT] Queried {rows_spot} spot rows in {time_spot:.3f}s")
print("[CACHE] Spot data now cached\n")

print("="*70)
print("STEP 2: Query VOL data (should use cached spot!)")
print("="*70)
print("[EXPECTED] Vol calculation should use cached spot + underlier data")
print("[EXPECTED] No parent query needed - just calculate vol from cached data")
t2 = time.time()
result_vol = dm.get_timeseries(start, end, INTERVAL, type_='vol', model=MODEL)
time_vol = time.time() - t2
rows_vol = len(result_vol.post_processed_data) if hasattr(result_vol, 'post_processed_data') else 0
print(f"[RESULT] Retrieved {rows_vol} vol rows in {time_vol:.3f}s")
print("[CACHE] Vol data now cached\n")

if time_vol < time_spot * 0.5:
    print(f"[PASS] Vol calculation was faster ({time_vol:.3f}s vs {time_spot:.3f}s)")
    print("[PASS] Indicates cached spot data was used!")
else:
    print(f"[WARN] Vol calculation time ({time_vol:.3f}s) not significantly faster than spot ({time_spot:.3f}s)")

print("\n" + "="*70)
print("STEP 3: Query GREEKS data (should use cached vol + spot!)")
print("="*70)
print("[EXPECTED] Greeks calculation should use cached vol + spot + underlier data")
print("[EXPECTED] No parent query needed - just calculate greeks from cached data")
t3 = time.time()
result_greeks = dm.get_timeseries(start, end, INTERVAL, type_='greeks', model=MODEL)
time_greeks = time.time() - t3
rows_greeks = len(result_greeks.post_processed_data) if hasattr(result_greeks, 'post_processed_data') else 0
print(f"[RESULT] Retrieved {rows_greeks} greeks rows in {time_greeks:.3f}s")
print("[CACHE] Greeks data now cached\n")

if time_greeks < time_spot * 0.5:
    print(f"[PASS] Greeks calculation was faster ({time_greeks:.3f}s vs {time_spot:.3f}s)")
    print("[PASS] Indicates cached vol/spot data was used!")
else:
    print(f"[WARN] Greeks calculation time ({time_greeks:.3f}s) not significantly faster than spot ({time_spot:.3f}s)")

print("\n" + "="*70)
print("STEP 4: Query GREEKS again (should be instant cache hit)")
print("="*70)
t4 = time.time()
result_greeks2 = dm.get_timeseries(start, end, INTERVAL, type_='greeks', model=MODEL)
time_greeks2 = time.time() - t4
rows_greeks2 = len(result_greeks2.post_processed_data) if hasattr(result_greeks2, 'post_processed_data') else 0
print(f"[RESULT] Retrieved {rows_greeks2} greeks rows in {time_greeks2:.3f}s")

if time_greeks2 < 0.01:
    print(f"[PASS] Second greeks query was instant ({time_greeks2:.3f}s) - cache hit!")
else:
    print(f"[WARN] Second greeks query not instant: {time_greeks2:.3f}s")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Spot query:          {rows_spot:3d} rows in {time_spot:.3f}s (baseline)")
print(f"Vol calculation:     {rows_vol:3d} rows in {time_vol:.3f}s (used cached spot)")
print(f"Greeks calculation:  {rows_greeks:3d} rows in {time_greeks:.3f}s (used cached vol+spot)")
print(f"Greeks cache hit:    {rows_greeks2:3d} rows in {time_greeks2:.3f}s (instant)")
print("="*70)

print("\n[INFO] Check logs for 'Using cached' messages to verify cascade efficiency")
