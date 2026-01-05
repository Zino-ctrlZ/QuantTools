"""
Test to verify cached managers are intercepting at factor level
"""

import pandas as pd
from pandas.tseries.offsets import BDay
from trade.helpers.helper import parse_option_tick
from DataManagers.DataManagers_cached import CachedOptionDataManager

print("="*80)
print("CACHED MANAGERS TEST - Verifying Factor-Level Interception")
print("="*80)

# Setup
opttick = 'AAPL20241220C215'
parsed = parse_option_tick(opttick)
exp = pd.to_datetime(parsed['exp_date'])
start = exp - BDay(30)
end = exp

print(f"\nTicker: {opttick}")
print(f"Date range: {start.date()} to {end.date()}")

# Clear cache
import shutil
try:
    shutil.rmtree('__pycache__/option_cache')
    print("\n✓ Cache cleared\n")
except:
    print("\n✓ No cache to clear\n")

# =============================================================================
# TEST 1: SPOT - First Query (Cache Miss, Should Query ThetaData)
# =============================================================================
print("="*80)
print("TEST 1: SPOT - First Query (should query ThetaData)")
print("="*80)

manager = CachedOptionDataManager(opttick=opttick)
result = manager.get_timeseries(type_='spot', start=start, end=end, interval='1d')

if result and hasattr(result, 'post_processed_data') and result.post_processed_data is not None:
    data = result.post_processed_data
    print(f"✓ Got spot data: {len(data)} rows")
    print(f"Columns: {list(data.columns)}")
    nan_count = data.isna().sum().sum()
    print(f"NaN count: {nan_count}")
    if nan_count == 0:
        print("✓ NO NaN VALUES")
    else:
        print(f"✗ Has {nan_count} NaN values")

# =============================================================================
# TEST 2: SPOT - Second Query (Cache Hit, Should Be Faster)
# =============================================================================
print("\n" + "="*80)
print("TEST 2: SPOT - Second Query (should hit cache)")
print("="*80)

manager2 = CachedOptionDataManager(opttick=opttick)
result2 = manager2.get_timeseries(type_='spot', start=start, end=end, interval='1d')

if result2 and hasattr(result2, 'post_processed_data') and result2.post_processed_data is not None:
    data2 = result2.post_processed_data
    print(f"✓ Got spot data: {len(data2)} rows")
    print(f"Columns: {list(data2.columns)}")
    nan_count = data2.isna().sum().sum()
    print(f"NaN count: {nan_count}")
    if nan_count == 0:
        print("✓ NO NaN VALUES")
    else:
        print(f"✗ Has {nan_count} NaN values")

# =============================================================================
# TEST 3: VOL - First Query (Should Use Cached Spot)
# =============================================================================
print("\n" + "="*80)
print("TEST 3: VOL - First Query (should use cached spot)")
print("="*80)

manager3 = CachedOptionDataManager(opttick=opttick)
result3 = manager3.get_timeseries(type_='vol', start=start, end=end, interval='1d')

if result3 and hasattr(result3, 'post_processed_data') and result3.post_processed_data is not None:
    data3 = result3.post_processed_data
    print(f"✓ Got vol data: {len(data3)} rows")
    print(f"Columns: {list(data3.columns)}")
    nan_count = data3.isna().sum().sum()
    print(f"NaN count: {nan_count}")
    if nan_count == 0:
        print("✓ NO NaN VALUES")
    else:
        print(f"✗ Has {nan_count} NaN values")

# =============================================================================
# TEST 4: GREEKS - First Query (Should Use Cached Spot + Vol)
# =============================================================================
print("\n" + "="*80)
print("TEST 4: GREEKS - First Query (should use cached spot + vol)")
print("="*80)

manager4 = CachedOptionDataManager(opttick=opttick)
result4 = manager4.get_timeseries(type_='greeks', start=start, end=end, interval='1d')

if result4 and hasattr(result4, 'post_processed_data') and result4.post_processed_data is not None:
    data4 = result4.post_processed_data
    print(f"✓ Got greeks data: {len(data4)} rows")
    print(f"Columns: {list(data4.columns)}")
    nan_count = data4.isna().sum().sum()
    print(f"NaN count: {nan_count}")
    if nan_count == 0:
        print("✓ NO NaN VALUES")
    else:
        print(f"✗ Has {nan_count} NaN values")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
print("\nExpected results:")
print("1. All queries should return data with NO NaN values")
print("2. Spot 2nd query should be faster (cache hit)")
print("3. Vol should be fast (uses cached spot)")
print("4. Greeks should be fast (uses cached spot + vol)")
