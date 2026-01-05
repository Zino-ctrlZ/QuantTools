#!/usr/bin/env python3
"""
Test full cascade caching: vol -> greeks with cache hits.
Run from: QuantTools/module_test/raw_code/
"""
import os
import sys
from pathlib import Path

# Set up paths
WORK_DIR = Path(__file__).parent.parent.parent
os.environ["WORK_DIR"] = str(WORK_DIR)
sys.path.insert(0, str(Path(__file__).parent))

print(f"WORK_DIR: {WORK_DIR}")
print(f"sys.path[0]: {sys.path[0]}\n")

import pandas as pd
import time
import shutil

from DataManagers.DataManagers_cached import CachedOptionDataManager
from DataManagers.option_cache import clear_all_caches
from DataManagers import set_skip_mysql_query

print("="*70)
print("FULL CASCADE CACHING TEST")
print("="*70)
print("Testing bottleneck caching: vol -> greeks")
print("="*70 + "\n")

set_skip_mysql_query(True)

# Clear cache
print("[1] Clearing all caches...")
cache_dir = WORK_DIR / '.cache' / 'data_manager_cache'
if cache_dir.exists():
    shutil.rmtree(cache_dir)
    print(f"    Deleted cache directory: {cache_dir}")
clear_all_caches()
print("    Caches cleared\n")

# Create data manager
print("[2] Creating CachedOptionDataManager...")
dm = CachedOptionDataManager(
    symbol='AAPL',
    right='C',
    strike=220.0,
    exp='2025-01-17',
    enable_cache=True,
    use_market_timeseries=True
)
print(f"    Created manager for {dm.opttick}\n")

# Date range
start = pd.Timestamp("2024-10-01")
end = pd.Timestamp("2024-10-31")

print("="*70)
print("STEP 1: Query VOL (cache empty)")
print("="*70)
t1 = time.time()
result_vol = dm.get_timeseries(start, end, '1d', type_='vol', model='bs')
time_vol = time.time() - t1
rows_vol = len(result_vol.post_processed_data) if hasattr(result_vol, 'post_processed_data') else 0
print(f"Vol query: {rows_vol} rows in {time_vol:.3f}s\n")

print("="*70)
print("STEP 2: Query VOL again (should hit cache)")
print("="*70)
t2 = time.time()
result_vol2 = dm.get_timeseries(start, end, '1d', type_='vol', model='bs')
time_vol2 = time.time() - t2
rows_vol2 = len(result_vol2.post_processed_data) if hasattr(result_vol2, 'post_processed_data') else 0
print(f"Vol query: {rows_vol2} rows in {time_vol2:.3f}s")
if time_vol2 < 0.01:
    print("PASS: Vol cache hit!\n")
else:
    print(f"WARN: Vol query not instant: {time_vol2:.3f}s\n")

print("="*70)
print("STEP 3: Query GREEKS (vol cache should help)")
print("="*70)
t3 = time.time()
result_greeks = dm.get_timeseries(start, end, '1d', type_='greeks', model='bs')
time_greeks = time.time() - t3
rows_greeks = len(result_greeks.post_processed_data) if hasattr(result_greeks, 'post_processed_data') else 0
print(f"Greeks query: {rows_greeks} rows in {time_greeks:.3f}s\n")

print("="*70)
print("STEP 4: Query GREEKS again (should hit cache)")
print("="*70)
t4 = time.time()
result_greeks2 = dm.get_timeseries(start, end, '1d', type_='greeks', model='bs')
time_greeks2 = time.time() - t4
rows_greeks2 = len(result_greeks2.post_processed_data) if hasattr(result_greeks2, 'post_processed_data') else 0
print(f"Greeks query: {rows_greeks2} rows in {time_greeks2:.3f}s")
if time_greeks2 < 0.01:
    print("PASS: Greeks cache hit!\n")
else:
    print(f"WARN: Greeks query not instant: {time_greeks2:.3f}s\n")

print("="*70)
print("SUMMARY")
print("="*70)
print(f"Vol (1st):       {rows_vol:3d} rows in {time_vol:.3f}s")
print(f"Vol (2nd):       {rows_vol2:3d} rows in {time_vol2:.3f}s  <- cache hit")
print(f"Greeks (1st):    {rows_greeks:3d} rows in {time_greeks:.3f}s")
print(f"Greeks (2nd):    {rows_greeks2:3d} rows in {time_greeks2:.3f}s  <- cache hit")
print("="*70)

if time_vol2 < 0.01 and time_greeks2 < 0.01:
    print("\nSUCCESS: Both vol and greeks benefit from caching!")
else:
    print("\nIssUE: Some cache hits not working as expected")
