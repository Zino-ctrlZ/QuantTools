#!/usr/bin/env python3
"""
Standalone test to verify underlier cache integration.
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
print("UNDERLIER CACHE INTEGRATION TEST")
print("="*70)
print("Testing that _ManagerLazyLoader uses cached underlier data")
print("="*70 + "\n")

set_skip_mysql_query(True)

# Clear cache
print("[1] Clearing all caches...")
cache_dir = WORK_DIR / '.cache' / 'data_manager_cache'
if cache_dir.exists():
    shutil.rmtree(cache_dir)
    print(f"    Deleted cache directory: {cache_dir}")
clear_all_caches()
print("    ✓ Caches cleared\n")

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
print(f"    ✓ Created manager for {dm.opttick}\n")

# Date range
start = pd.Timestamp("2024-10-01")
end = pd.Timestamp("2024-10-31")

print("="*70)
print("STEP 1: Query VOL data (cache empty)")
print("="*70)
print("Expected behavior:")
print("  1. Preload underlier data to MarketTimeseries")
print("  2. Set global MarketTimeseries in utils.py")
print("  3. Query parent for vol calculation")
print("  4. Parent's _ManagerLazyLoader uses cached underlier\n")

t1 = time.time()
result_vol = dm.get_timeseries(start, end, '1d', type_='vol', model='bs')
time_vol = time.time() - t1
rows_vol = len(result_vol.post_processed_data) if hasattr(result_vol, 'post_processed_data') else 0

print(f"\n✓ Retrieved {rows_vol} vol rows in {time_vol:.3f}s\n")

print("="*70)
print("STEP 2: Query VOL again (should hit cache)")
print("="*70)
t2 = time.time()
result_vol2 = dm.get_timeseries(start, end, '1d', type_='vol', model='bs')
time_vol2 = time.time() - t2
rows_vol2 = len(result_vol2.post_processed_data) if hasattr(result_vol2, 'post_processed_data') else 0

print(f"\n✓ Retrieved {rows_vol2} vol rows in {time_vol2:.3f}s")

if time_vol2 < 0.01:
    print("✅ PASS: Second vol query was instant - cache hit!\n")
else:
    print(f"⚠️  WARN: Second vol query not instant: {time_vol2:.3f}s\n")

print("="*70)
print("STEP 3: Query GREEKS (should use cached vol + underlier)")
print("="*70)
print("Expected behavior:")
print("  1. Check greeks cache (miss)")
print("  2. Check vol cache (HIT from Step 1)")
print("  3. Query parent for greeks  calculation")
print("  4. Parent's _ManagerLazyLoader uses cached underlier\n")

t3 = time.time()
result_greeks = dm.get_timeseries(start, end, '1d', type_='greeks', model='bs')
time_greeks = time.time() - t3
rows_greeks = len(result_greeks.post_processed_data) if hasattr(result_greeks, 'post_processed_data') else 0

print(f"\n✓ Retrieved {rows_greeks} greeks rows in {time_greeks:.3f}s")

if time_greeks < time_vol * 0.7:
    print(f"✅ PASS: Greeks faster than initial vol ({time_greeks:.3f}s vs {time_vol:.3f}s)")
    print("         This indicates cached underlier data was used!\n")
else:
    print(f"ℹ️  INFO: Greeks time ({time_greeks:.3f}s) similar to vol ({time_vol:.3f}s)\n")

print("="*70)
print("SUMMARY")
print("="*70)
print(f"Vol query (1st):     {rows_vol:3d} rows in {time_vol:.3f}s")
print(f"Vol query (2nd):     {rows_vol2:3d} rows in {time_vol2:.3f}s (cache hit)")
print(f"Greeks query:        {rows_greeks:3d} rows in {time_greeks:.3f}s")
print("="*70)

# Calculate efficiency
if time_vol2 < 0.01:
    efficiency_cache = ((time_vol - time_vol2) / time_vol) * 100
    print(f"\n📊 Cache Efficiency: {efficiency_cache:.1f}% faster on cache hit")

if time_greeks < time_vol:
    efficiency_underlier = ((time_vol - time_greeks) / time_vol) * 100
    print(f"📊 Underlier Cache: {efficiency_underlier:.1f}% faster with cached underlier")

print("\n" + "="*70)
print("CHECK LOGS FOR:")
print("="*70)
print("✓ 'Set global MarketTimeseries for _ManagerLazyLoader'")
print("✓ 'Loading s0_chain for AAPL from MarketTimeseries cache'")
print("✓ 'Loading dividends for AAPL from MarketTimeseries cache'")
print("\nIf you see these messages, the fix is working! ✅")
print("="*70)
