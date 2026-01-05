"""
Compare OptionDataManager vs CachedOptionDataManager outputs
to identify source of NaN values in spot data
"""

import pandas as pd
from pandas.tseries.offsets import BDay
from trade.helpers.helper import parse_option_tick

# Import BOTH managers
from module_test.raw_code.DataManagers.DataManagers import OptionDataManager
from DataManagers.DataManagers_cached import CachedOptionDataManager

print("="*80)
print("COMPARISON TEST: Parent DM vs Cached DM")
print("="*80)

# Setup
opttick = 'AAPL20241220C215'
parsed = parse_option_tick(opttick)
exp = pd.to_datetime(parsed['exp_date'])
start = exp - BDay(30)
end = exp

print(f"\nTicker: {opttick}")
print(f"Start: {start}")
print(f"End: {end}")
print(f"Expected range: ~30 business days")

# ============================================================================
# TEST 1: Parent OptionDataManager (NO CACHE)
# ============================================================================
print("\n" + "="*80)
print("TEST 1: Parent OptionDataManager.get_timeseries() - NO CACHE")
print("="*80)

parent_dm = OptionDataManager(opttick=opttick)
parent_result = parent_dm.get_timeseries(
    type_='spot',
    start=start,
    end=end,
    interval='1d'
)

# Extract data
parent_data = None
if parent_result:
    if hasattr(parent_result, 'post_processed_data') and parent_result.post_processed_data is not None:
        parent_data = parent_result.post_processed_data
        print("✓ Got data from post_processed_data")
    elif hasattr(parent_result, 'data') and parent_result.data is not None:
        parent_data = parent_result.data
        print("✓ Got data from data attribute")
    else:
        print("✗ No data found in result")
else:
    print("✗ Result is None")

if parent_data is not None:
    print(f"\nParent DM Results:")
    print(f"  Rows: {len(parent_data)}")
    print(f"  Columns: {list(parent_data.columns)}")
    print(f"  Date range: {parent_data.index.min()} to {parent_data.index.max()}")
    
    # Check for NaN values
    nan_counts = parent_data.isna().sum()
    print(f"\n  NaN counts per column:")
    for col, count in nan_counts.items():
        if count > 0:
            print(f"    {col}: {count} NaN values ({count/len(parent_data)*100:.1f}%)")
    
    total_nans = nan_counts.sum()
    if total_nans == 0:
        print("  ✓ NO NaN VALUES - Parent DM data is clean!")
    else:
        print(f"  ✗ Total NaN values: {total_nans}")
    
    # Show sample
    print("\n  First 3 rows:")
    print(parent_data.head(3).to_string())
    print("\n  Last 3 rows:")
    print(parent_data.tail(3).to_string())

# ============================================================================
# TEST 2: CachedOptionDataManager
# ============================================================================
print("\n" + "="*80)
print("TEST 2: CachedOptionDataManager.get_timeseries() - WITH CACHE")
print("="*80)

cached_dm = CachedOptionDataManager(opttick=opttick)
cached_result = cached_dm.get_timeseries(
    type_='spot',
    start=start,
    end=end,
    interval='1d'
)

# Extract data
cached_data = None
if cached_result:
    if hasattr(cached_result, 'post_processed_data') and cached_result.post_processed_data is not None:
        cached_data = cached_result.post_processed_data
        print("✓ Got data from post_processed_data (CACHED)")
    elif hasattr(cached_result, 'data') and cached_result.data is not None:
        cached_data = cached_result.data
        print("✓ Got data from data attribute")
    else:
        print("✗ No data found in result")
else:
    print("✗ Result is None")

if cached_data is not None:
    print(f"\nCached DM Results:")
    print(f"  Rows: {len(cached_data)}")
    print(f"  Columns: {list(cached_data.columns)}")
    print(f"  Date range: {cached_data.index.min()} to {cached_data.index.max()}")
    
    # Check for NaN values
    nan_counts = cached_data.isna().sum()
    print(f"\n  NaN counts per column:")
    for col, count in nan_counts.items():
        if count > 0:
            print(f"    {col}: {count} NaN values ({count/len(cached_data)*100:.1f}%)")
    
    total_nans = nan_counts.sum()
    if total_nans == 0:
        print("  ✓ NO NaN VALUES - Cached DM data is clean!")
    else:
        print(f"  ✗ Total NaN values: {total_nans}")
    
    # Show sample
    print("\n  First 3 rows:")
    print(cached_data.head(3).to_string())
    print("\n  Last 3 rows:")
    print(cached_data.tail(3).to_string())

# ============================================================================
# TEST 3: COMPARISON
# ============================================================================
if parent_data is not None and cached_data is not None:
    print("\n" + "="*80)
    print("TEST 3: COMPARISON - Parent vs Cached")
    print("="*80)
    
    # Compare shapes
    print(f"\nShape comparison:")
    print(f"  Parent:  {parent_data.shape}")
    print(f"  Cached:  {cached_data.shape}")
    
    if parent_data.shape == cached_data.shape:
        print("  ✓ Shapes match")
    else:
        print("  ✗ SHAPES DIFFER!")
    
    # Compare columns
    parent_cols = set(parent_data.columns)
    cached_cols = set(cached_data.columns)
    
    print(f"\nColumn comparison:")
    print(f"  Parent has {len(parent_cols)} columns")
    print(f"  Cached has {len(cached_cols)} columns")
    
    missing_in_cached = parent_cols - cached_cols
    extra_in_cached = cached_cols - parent_cols
    
    if missing_in_cached:
        print(f"  ✗ Columns in Parent but NOT in Cached: {missing_in_cached}")
    if extra_in_cached:
        print(f"  ✗ Columns in Cached but NOT in Parent: {extra_in_cached}")
    if not missing_in_cached and not extra_in_cached:
        print("  ✓ Columns match exactly")
    
    # Compare NaN counts
    parent_nans = parent_data.isna().sum().sum()
    cached_nans = cached_data.isna().sum().sum()
    
    print(f"\nNaN comparison:")
    print(f"  Parent total NaNs:  {parent_nans}")
    print(f"  Cached total NaNs:  {cached_nans}")
    print(f"  Difference:         {cached_nans - parent_nans}")
    
    if cached_nans > parent_nans:
        print("  ✗ CACHED HAS MORE NaN VALUES!")
        print("\n  NaN difference per column:")
        parent_nan_counts = parent_data.isna().sum()
        cached_nan_counts = cached_data.isna().sum()
        
        for col in cached_cols:
            if col in parent_cols:
                diff = cached_nan_counts.get(col, 0) - parent_nan_counts.get(col, 0)
                if diff != 0:
                    print(f"    {col}: {diff:+d} more NaNs in cached")
    elif cached_nans < parent_nans:
        print("  ✗ PARENT HAS MORE NaN VALUES!")
    else:
        print("  ✓ Same NaN count")
    
    # Compare actual values for overlapping columns
    common_cols = parent_cols & cached_cols
    if common_cols:
        print(f"\nValue comparison for {len(common_cols)} common columns:")
        
        for col in sorted(common_cols):
            # Compare only non-NaN values
            parent_values = parent_data[col].dropna()
            cached_values = cached_data[col].dropna()
            
            if len(parent_values) > 0 and len(cached_values) > 0:
                # Check if values are close (for floating point)
                try:
                    parent_sample = parent_values.iloc[0]
                    cached_sample = cached_values.iloc[0]
                    if pd.notna(parent_sample) and pd.notna(cached_sample):
                        if abs(parent_sample - cached_sample) < 0.01:
                            status = "✓"
                        else:
                            status = "✗"
                        print(f"  {status} {col}: Parent={parent_sample:.2f}, Cached={cached_sample:.2f}")
                except (TypeError, ValueError):
                    # Non-numeric column
                    if parent_sample == cached_sample:
                        print(f"  ✓ {col}: Values match")
                    else:
                        print(f"  ✗ {col}: Values differ")

print("\n" + "="*80)
print("COMPARISON COMPLETE")
print("="*80)
