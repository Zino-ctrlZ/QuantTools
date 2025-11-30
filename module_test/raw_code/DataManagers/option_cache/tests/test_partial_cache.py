#!/usr/bin/env python3
# ruff: noqa
"""
Test to verify partial cache behavior - only missing dates are queried.
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
from DataManagers.option_cache import clear_all_caches, get_cache_instances
from DataManagers import set_skip_mysql_query

# Test parameters
OPTTICK = "AAPL20260821C300"
INTERVAL = "1d"
MODEL = "bs"

print("\n" + "="*70)
print("PARTIAL CACHE TEST")
print("="*70)
print("Testing that only missing dates are queried, not entire range")
print("="*70 + "\n")

set_skip_mysql_query(True)

print("\n[DEBUG] Clearing all caches...")
# Delete cache files from disk completely
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
    exp='2025-01-17',  # Jan 2025 expiry - existed in Sep-Nov 2024
    enable_cache=True
)

# Verify cache is empty by querying a known date range
print("[DEBUG] Verifying cache is empty...")
caches = get_cache_instances()
spot_cache = caches[0]
test_key = ('1d', 'AAPL20250117C220')
cached_data = spot_cache.get(test_key)
if cached_data:
    print(f"[WARN] Cache is NOT empty! Found data: {len(cached_data)} rows")
else:
    print("[DEBUG] Cache is empty ✓\n")

# Step 1: Query October data (cache miss - full query)
print("[STEP 1] Query October 2024 data (cache empty)")
print("-" * 70)
start1 = pd.Timestamp("2024-10-01")
end1 = pd.Timestamp("2024-10-31")

start_time = time.time()
result1 = dm.get_timeseries(start1, end1, INTERVAL, type_='spot', model=MODEL)
time1 = time.time() - start_time

if hasattr(result1, 'post_processed_data') and result1.post_processed_data is not None:
    rows1 = len(result1.post_processed_data)
    print(f"[DEBUG] Result1 data shape: {result1.post_processed_data.shape}")
    print(f"[DEBUG] Result1 data index range: {result1.post_processed_data.index.min()} to {result1.post_processed_data.index.max()}")
else:
    rows1 = 0
    print("[DEBUG] Result1 has no post_processed_data!")

print(f"[RESULT] Queried {rows1} rows in {time1:.3f}s")
print(f"         Cache now contains: Oct 1-31 ({rows1} days)")

# Check what's actually in cache after Step 1
cached_after_step1 = spot_cache.get(test_key)
if cached_after_step1 is not None:
    print(f"[DEBUG] Actual cache after Step 1: {len(cached_after_step1)} rows, date range: {cached_after_step1.index.min()} to {cached_after_step1.index.max()}")

# Step 2: Query November data (cache miss - but only Nov should be queried)
print("\n[STEP 2] Query September 2024 data (cache empty for Sep)")
print("-" * 70)
start2 = pd.Timestamp("2024-09-01")
end2 = pd.Timestamp("2024-09-30")

start_time = time.time()
result2 = dm.get_timeseries(start2, end2, INTERVAL, type_='spot', model=MODEL)
time2 = time.time() - start_time
rows2 = len(result2.post_processed_data) if hasattr(result2, 'post_processed_data') else 0

print(f"[RESULT] Queried {rows2} rows in {time2:.3f}s")
print("         Cache now contains: Sep 1-30 + Oct 1-31")

# Step 3: Query Sep + Oct combined (cache hit - all cached)
print("\n[STEP 3] Query Sep + Oct combined (all cached)")
print("-" * 70)
start3 = pd.Timestamp("2024-09-01")
end3 = pd.Timestamp("2024-10-31")

start_time = time.time()
result3 = dm.get_timeseries(start3, end3, INTERVAL, type_='spot', model=MODEL)
time3 = time.time() - start_time

if hasattr(result3, 'post_processed_data') and result3.post_processed_data is not None:
    rows3 = len(result3.post_processed_data)
    print(f"[DEBUG] Result3 data shape: {result3.post_processed_data.shape}")
    print(f"[DEBUG] Result3 data index range: {result3.post_processed_data.index.min()} to {result3.post_processed_data.index.max()}")
else:
    rows3 = 0
    print("[DEBUG] Result3 has no post_processed_data!")

print(f"[RESULT] Retrieved {rows3} rows in {time3:.3f}s")
print("         Should be INSTANT (cache hit)")

# Step 4: Query extended range Sep-Nov (partial cache - only Nov missing)
print("\n[STEP 4] Query Sep-Nov 2024 (partial cache - only Nov missing)")
print("-" * 70)
start4 = pd.Timestamp("2024-09-01")
end4 = pd.Timestamp("2024-11-15")

print("[EXPECTED] Should query ONLY November dates (not Sep or Oct)")
start_time = time.time()
result4 = dm.get_timeseries(start4, end4, INTERVAL, type_='spot', model=MODEL)
time4 = time.time() - start_time

if hasattr(result4, 'post_processed_data') and result4.post_processed_data is not None:
    rows4 = len(result4.post_processed_data)
    print(f"[DEBUG] Result4 data shape: {result4.post_processed_data.shape}")
    print(f"[DEBUG] Result4 data index range: {result4.post_processed_data.index.min()} to {result4.post_processed_data.index.max()}")
else:
    rows4 = 0
    print("[DEBUG] Result4 has no post_processed_data!")

print(f"[RESULT] Retrieved {rows4} rows in {time4:.3f}s")
print("         Cache now contains: Sep 1-30 + Oct 1-31 + Nov 1-15")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Step 1 (Oct only):       {rows1} rows, {time1:.3f}s - FULL QUERY")
print(f"Step 2 (Sep only):       {rows2} rows, {time2:.3f}s - FULL QUERY")
print(f"Step 3 (Sep+Oct):        {rows3} rows, {time3:.3f}s - CACHE HIT")
print(f"Step 4 (Sep+Oct+Nov):    {rows4} rows, {time4:.3f}s - PARTIAL QUERY (Nov only)")

if rows3 == (rows1 + rows2):
    print("\n[PASS] Step 3 correctly returned combined cached data")
else:
    print(f"\n[WARN] Step 3 expected {rows1 + rows2} rows, got {rows3}")

if time3 < time1 * 0.1:  # Cache hit should be <10% of query time
    print("[PASS] Step 3 was instant (cache hit working)")
else:
    print("[WARN] Step 3 was slow (cache hit may not be working)")

expected_nov_rows = 11  # Nov 1-15 is roughly 11 business days
if abs(rows4 - (rows1 + rows2 + expected_nov_rows)) <= 2:  # Allow 2-day tolerance
    print("[PASS] Step 4 correctly added only November data")
else:
    print(f"[WARN] Step 4 expected ~{rows1 + rows2 + expected_nov_rows} rows, got {rows4}")

print("="*70)
print("\nCheck logs for 'Querying spot data for missing dates' to confirm")
print("that only missing date ranges are being queried.")
