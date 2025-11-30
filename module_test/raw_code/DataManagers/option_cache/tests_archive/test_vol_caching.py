"""
Test CachedVolManager - Verify vol data caching at factor level
"""

import pandas as pd
import time
import shutil
from pandas.tseries.offsets import BDay
from trade.helpers.helper import parse_option_tick
from DataManagers.DataManagers_cached import CachedOptionDataManager
from DataManagers import set_skip_mysql_query

# Skip MySQL for performance
set_skip_mysql_query(True)

print("="*80)
print("VOL CACHING TEST")
print("="*80)

# Setup
opttick = 'AAPL20241220C215'
parsed = parse_option_tick(opttick)
exp = pd.to_datetime(parsed['exp_date'])
start = exp - BDay(50)  # 50 days for thorough testing
end = exp

print(f"\nTicker: {opttick}")
print(f"Date range: {start.date()} to {end.date()}")

# Clear cache
try:
    shutil.rmtree('__pycache__/option_cache')
    print("✓ Cache cleared\n")
except Exception:
    print("✓ No cache to clear\n")

# =============================================================================
# TEST 1: First Vol Query (Cache MISS - will query spot, then calculate vol)
# =============================================================================
print("="*80)
print("TEST 1: First Vol Query (Cache MISS)")
print("="*80)

manager1 = CachedOptionDataManager(opttick=opttick)
print(f"Manager type: {type(manager1.vol_manager).__name__}")

start_time = time.time()
result1 = manager1.get_timeseries(type_='vol', start=start, end=end, interval='1d')
time1 = time.time() - start_time

if result1 and hasattr(result1, 'post_processed_data') and result1.post_processed_data is not None:
    data1 = result1.post_processed_data
    print(f"\n✓ Got vol data: {len(data1)} rows in {time1:.3f}s")
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
    
    print("\nFirst 3 rows:")
    print(data1.head(3).to_string())
else:
    print("✗ FAILED to get data")

# =============================================================================
# TEST 2: Second Vol Query (Cache HIT - should be much faster)
# =============================================================================
print("\n" + "="*80)
print("TEST 2: Second Vol Query (Cache HIT)")
print("="*80)

manager2 = CachedOptionDataManager(opttick=opttick)

start_time = time.time()
result2 = manager2.get_timeseries(type_='vol', start=start, end=end, interval='1d')
time2 = time.time() - start_time

if result2 and hasattr(result2, 'post_processed_data') and result2.post_processed_data is not None:
    data2 = result2.post_processed_data
    print(f"\n✓ Got vol data: {len(data2)} rows in {time2:.3f}s")
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
    
    if time2 < time1 * 0.5:  # Should be 50%+ faster
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
# TEST 3: Spot Query After Vol (Should use cached spot)
# =============================================================================
print("\n" + "="*80)
print("TEST 3: Spot Query After Vol (Spot should be cached)")
print("="*80)

manager3 = CachedOptionDataManager(opttick=opttick)

start_time = time.time()
result3 = manager3.get_timeseries(type_='spot', start=start, end=end, interval='1d')
time3 = time.time() - start_time

if result3 and hasattr(result3, 'post_processed_data') and result3.post_processed_data is not None:
    data3 = result3.post_processed_data
    print(f"\n✓ Got spot data: {len(data3)} rows in {time3:.3f}s")
    print(f"Columns: {list(data3.columns)}")
    
    if time3 < 0.05:  # Should be very fast from cache
        print("✓ Fast retrieval (spot was cached from vol query)")
    else:
        print("⚠ Slow retrieval (spot may not have been cached)")
else:
    print("✗ FAILED to get data")

print("\n" + "="*80)
print("VOL CACHING TEST COMPLETE")
print("="*80)
print("\n✓ Expected: Second vol query 50-90% faster than first")
print("✓ Expected: NO NaN values in clean data")
print("✓ Expected: Spot query is fast (uses spot cached from vol calc)")
