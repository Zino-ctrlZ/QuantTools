"""
Test to see what columns are being saved to cache
"""

import pandas as pd
from pandas.tseries.offsets import BDay
from trade.helpers.helper import parse_option_tick
from DataManagers.DataManagers_cached import CachedOptionDataManager

print("="*80)
print("CACHE COLUMN TEST")
print("="*80)

# Setup
opttick = 'AAPL20241220C215'
parsed = parse_option_tick(opttick)
exp = pd.to_datetime(parsed['exp_date'])
start = exp - BDay(30)
end = exp

print(f"\nTicker: {opttick}")
print(f"Date range: {start.date()} to {end.date()}")

# Clear cache first
import shutil
try:
    shutil.rmtree('__pycache__/option_cache')
    print("\n✓ Cache cleared")
except:
    print("\n✓ No cache to clear")

# First query - will save to cache
print("\n" + "="*80)
print("FIRST QUERY - Will save to cache")
print("="*80)

manager = CachedOptionDataManager(opttick=opttick)
result = manager.get_timeseries(
    type_='spot',
    start=start,
    end=end,
    interval='1d'
)

if result and hasattr(result, 'post_processed_data') and result.post_processed_data is not None:
    data = result.post_processed_data
    print(f"\n✓ Got data: {len(data)} rows")
    print(f"Columns ({len(data.columns)}): {list(data.columns)}")
    print(f"\nNaN counts:")
    nan_counts = data.isna().sum()
    for col in data.columns:
        count = nan_counts[col]
        if count > 0:
            print(f"  {col}: {count} NaN ({count/len(data)*100:.1f}%)")
    if nan_counts.sum() == 0:
        print("  ✓ NO NaN VALUES!")

# Second query - will retrieve from cache
print("\n" + "="*80)
print("SECOND QUERY - Should retrieve from cache")
print("="*80)

manager2 = CachedOptionDataManager(opttick=opttick)
result2 = manager2.get_timeseries(
    type_='spot',
    start=start,
    end=end,
    interval='1d'
)

if result2 and hasattr(result2, 'post_processed_data') and result2.post_processed_data is not None:
    data2 = result2.post_processed_data
    print(f"\n✓ Got data: {len(data2)} rows")
    print(f"Columns ({len(data2.columns)}): {list(data2.columns)}")
    print(f"\nNaN counts:")
    nan_counts2 = data2.isna().sum()
    for col in data2.columns:
        count = nan_counts2[col]
        if count > 0:
            print(f"  {col}: {count} NaN ({count/len(data2)*100:.1f}%)")
    if nan_counts2.sum() == 0:
        print("  ✓ NO NaN VALUES!")

# Compare
print("\n" + "="*80)
print("COMPARISON")
print("="*80)

if data.columns.equals(data2.columns):
    print("✓ Columns MATCH between first and second query")
else:
    print("✗ Columns DIFFER!")
    print(f"  First query: {len(data.columns)} columns")
    print(f"  Second query: {len(data2.columns)} columns")
    
if data.equals(data2):
    print("✓ Data IDENTICAL")
else:
    print("✗ Data differs (expected for values, but structure should match)")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
