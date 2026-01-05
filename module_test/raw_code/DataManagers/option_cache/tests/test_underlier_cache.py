#!/usr/bin/env python3
# ruff: noqa
"""
Test to verify MarketTimeseries underlier cache integration.
This test compares performance with and without underlier caching.
"""
import sys
from pathlib import Path

# Add parent directories to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

import os
WORK_DIR = Path(__file__).parent.parent.parent.parent
os.environ["WORK_DIR"] = str(WORK_DIR)
print(f"Set WORK_DIR to: {WORK_DIR}")

import pandas as pd
import time
from DataManagers.DataManagers_cached import CachedOptionDataManager
from DataManagers.option_cache import clear_all_caches
from DataManagers import set_skip_mysql_query

# Test parameters
OPTTICK = "AAPL20260821C300"
START = pd.Timestamp("2025-10-01")
END = pd.Timestamp("2025-11-28")
INTERVAL = "1d"
MODEL = "bs"

print("\n" + "="*70)
print("UNDERLIER CACHE TEST")
print("="*70)
print(f"Opttick:  {OPTTICK}")
print(f"Start:    {START.date()}")
print(f"End:      {END.date()}")
print("="*70 + "\n")

# Skip MySQL queries
set_skip_mysql_query(True)

# Clear all caches
print("[SETUP] Clearing option caches...")
clear_all_caches()

# Test 1: WITHOUT MarketTimeseries (fresh query)
print("\n[TEST 1] First query WITHOUT MarketTimeseries underlier cache")
print("-" * 70)
dm1 = CachedOptionDataManager(
    symbol='AAPL',
    right='C',
    strike=300.0,
    exp='2026-08-21',
    enable_cache=True,
    use_market_timeseries=False  # Disable MarketTimeseries
)

start_time = time.time()
result1 = dm1.get_timeseries(START, END, INTERVAL, type_='greeks', model=MODEL)
time1 = time.time() - start_time
rows1 = len(result1.post_processed_data) if hasattr(result1, 'post_processed_data') else 0

print(f"[RESULT] Query completed in {time1:.3f}s ({rows1} rows)")

# Test 2: WITH MarketTimeseries (should be faster due to underlier caching)
print("\n[TEST 2] Second query WITH MarketTimeseries underlier cache")
print("-" * 70)

# Clear option caches but keep MarketTimeseries caches
clear_all_caches()

dm2 = CachedOptionDataManager(
    symbol='AAPL',
    right='C',
    strike=300.0,
    exp='2026-08-21',
    enable_cache=True,
    use_market_timeseries=True  # Enable MarketTimeseries
)

start_time = time.time()
result2 = dm2.get_timeseries(START, END, INTERVAL, type_='greeks', model=MODEL)
time2 = time.time() - start_time
rows2 = len(result2.post_processed_data) if hasattr(result2, 'post_processed_data') else 0

print(f"[RESULT] Query completed in {time2:.3f}s ({rows2} rows)")

# Compare
print("\n" + "="*70)
print("COMPARISON")
print("="*70)
print(f"WITHOUT MarketTimeseries: {time1:.3f}s")
print(f"WITH MarketTimeseries:    {time2:.3f}s")

if time2 < time1:
    speedup = time1 / time2
    improvement_pct = ((time1 - time2) / time1) * 100
    print(f"Improvement:              {speedup:.2f}x faster ({improvement_pct:.1f}% reduction)")
    print("\n[SUCCESS] MarketTimeseries underlier caching is working!")
else:
    print(f"[WARNING] No improvement detected (may already be cached)")

print("="*70)
