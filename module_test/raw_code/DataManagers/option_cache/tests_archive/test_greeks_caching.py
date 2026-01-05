"""
Test CachedGreeksManager - Verify greeks caching at factor level with cascade
"""

import pandas as pd
import time
import cProfile
import pstats
from io import StringIO
from pandas.tseries.offsets import BDay
from trade.helpers.helper import parse_option_tick
from DataManagers.DataManagers_cached import CachedOptionDataManager
from DataManagers import set_skip_mysql_query
from DataManagers.option_cache.helpers import load_option_data_cache

# Skip MySQL for performance
set_skip_mysql_query(True)

print("="*80)
print("GREEKS CACHING TEST")
print("="*80)

# Setup
opttick = 'AAPL20241220C215'
parsed = parse_option_tick(opttick)
exp = pd.to_datetime(parsed['exp_date'])
start = exp - BDay(50)  # 50 days for thorough testing
end = exp

print(f"\nTicker: {opttick}")
print(f"Date range: {start.date()} to {end.date()}")

# Clear the actual caches used by the managers
spot_cache, bs_vol, bs_greeks, binom_vol, binom_greeks = load_option_data_cache()
spot_cache.clear()
bs_vol.clear()
bs_greeks.clear()
binom_vol.clear()
binom_greeks.clear()
print("✓ All caches cleared\n")

# =============================================================================
# TEST 1: First Greeks Query (Cache MISS - will query spot, calc vol, calc greeks)
# =============================================================================
print("="*80)
print("TEST 1: First Greeks Query (Cache MISS)")
print("="*80)

manager1 = CachedOptionDataManager(opttick=opttick)
print(f"Manager type: {type(manager1.greek_manager).__name__}")

start_time = time.time()
result1 = manager1.get_timeseries(type_='greeks', start=start, end=end, interval='1d')
time1 = time.time() - start_time

if result1 and hasattr(result1, 'post_processed_data') and result1.post_processed_data is not None:
    data1 = result1.post_processed_data
    print(f"\n✓ Got greeks data: {len(data1)} rows in {time1:.3f}s")
    print(f"Columns ({len(data1.columns)}): {list(data1.columns)}")
    
    # Check for NaN
    nan_count = data1.isna().sum().sum()
    if nan_count == 0:
        print("✓ NO NaN VALUES")
    else:
        print("NaN breakdown:")
        for col in data1.columns:
            col_nans = data1[col].isna().sum()
            if col_nans > 0:
                print(f"  {col}: {col_nans} NaN")
    
    # Check for expected greek columns
    greek_cols = ['delta', 'gamma', 'vega', 'theta', 'rho']
    present_greeks = [col for col in greek_cols if col in data1.columns]
    print(f"\nGreek columns present: {present_greeks}")
    
    print("\nFirst 3 rows:")
    print(data1.head(3).to_string())
else:
    print("✗ FAILED to get data")

# =============================================================================
# TEST 2: Second Greeks Query (Cache HIT - should be much faster)
# =============================================================================
print("\n" + "="*80)
print("TEST 2: Second Greeks Query (Cache HIT) - WITH PROFILING")
print("="*80)

manager2 = CachedOptionDataManager(opttick=opttick)

# Profile the cache HIT
profiler = cProfile.Profile()
profiler.enable()

start_time = time.time()
result2 = manager2.get_timeseries(type_='greeks', start=start, end=end, interval='1d')
time2 = time.time() - start_time

profiler.disable()

# Print profiling results
print("\n" + "="*80)
print("PROFILING RESULTS - Top 30 time consumers")
print("="*80)
s = StringIO()
ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
ps.print_stats(30)
print(s.getvalue())

if result2 and hasattr(result2, 'post_processed_data') and result2.post_processed_data is not None:
    data2 = result2.post_processed_data
    print(f"\n✓ Got greeks data: {len(data2)} rows in {time2:.3f}s")
    print(f"Columns ({len(data2.columns)}): {list(data2.columns)}")
    
    # Check for NaN
    nan_count = data2.isna().sum().sum()
    if nan_count == 0:
        print("✓ NO NaN VALUES")
    else:
        print("NaN breakdown:")
        for col in data2.columns:
            col_nans = data2[col].isna().sum()
            if col_nans > 0:
                print(f"  {col}: {col_nans} NaN")
    
    # Performance comparison
    speedup = ((time1 - time2) / time1) * 100
    print("\n⚡ Performance:")
    print(f"  First query:  {time1:.3f}s")
    print(f"  Second query: {time2:.3f}s")
    print(f"  Speedup:      {speedup:.1f}%")
    
    if time2 < time1 * 0.4:  # Should be 60%+ faster
        print("  ✓ Cache is working!")
    else:
        print("  ⚠ Cache may not be working (expected much faster)")
    
    # Data comparison
    if data1.equals(data2):
        print("\n✓ Data is IDENTICAL between queries")
    else:
        print("\n⚠ Data differs between queries")
else:
    print("✗ FAILED to get data")

# =============================================================================
# TEST 3: Cascade Check - Query Spot & Vol After Greeks (Should use cache)
# =============================================================================
print("\n" + "="*80)
print("TEST 3: Cascade Check (Spot & Vol should be cached)")
print("="*80)

manager3 = CachedOptionDataManager(opttick=opttick)

# Spot query
start_time = time.time()
result3_spot = manager3.get_timeseries(type_='spot', start=start, end=end, interval='1d')
time3_spot = time.time() - start_time

if result3_spot and hasattr(result3_spot, 'post_processed_data'):
    print(f"✓ Spot: {len(result3_spot.post_processed_data)} rows in {time3_spot:.3f}s")
    if time3_spot < 0.05:
        print("  ✓ Fast (spot was cached from greeks calc)")
    else:
        print("  ⚠ Slow (spot may not have been cached)")

# Vol query
start_time = time.time()
result3_vol = manager3.get_timeseries(type_='vol', start=start, end=end, interval='1d')
time3_vol = time.time() - start_time

if result3_vol and hasattr(result3_vol, 'post_processed_data'):
    print(f"✓ Vol:  {len(result3_vol.post_processed_data)} rows in {time3_vol:.3f}s")
    if time3_vol < 0.1:
        print("  ✓ Fast (vol was cached from greeks calc)")
    else:
        print("  ⚠ Slow (vol may not have been cached)")

print("\n" + "="*80)
print("GREEKS CACHING TEST COMPLETE")
print("="*80)
print("\n✓ Expected: Second greeks query 60-95% faster than first")
print("✓ Expected: NO NaN values in clean data")
print("✓ Expected: Spot & vol queries are fast (cascade caching works)")
print("✓ Expected: Greek columns present (delta, gamma, vega, theta, rho)")
