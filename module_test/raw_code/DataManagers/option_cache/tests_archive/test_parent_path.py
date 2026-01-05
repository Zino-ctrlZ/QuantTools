"""
Debug test to see which path the data is taking
"""

import pandas as pd
from pandas.tseries.offsets import BDay
from trade.helpers.helper import parse_option_tick
from DataManagers.DataManagers import OptionDataManager

print("="*80)
print("DEBUG: Parent OptionDataManager Path")
print("="*80)

# Setup
opttick = 'AAPL20241220C215'
parsed = parse_option_tick(opttick)
exp = pd.to_datetime(parsed['exp_date'])
start = exp - BDay(30)
end = exp

print(f"\nTicker: {opttick}")
print(f"Date range: {start.date()} to {end.date()}")

# Query parent
manager = OptionDataManager(opttick=opttick)
result = manager.get_timeseries(
    type_='spot',
    start=start,
    end=end,
    interval='1d'
)

# Check what path it took
print(f"\n--- DATA REQUEST ATTRIBUTES ---")
print(f"is_complete: {result.pre_process.get('is_complete')}")
print(f"is_empty: {result.pre_process.get('is_empty')}")
print(f"requested_col: {result.requested_col}")

if hasattr(result, 'post_processed_data') and result.post_processed_data is not None:
    data = result.post_processed_data
    print(f"\n--- POST PROCESSED DATA ---")
    print(f"Rows: {len(data)}")
    print(f"Columns ({len(data.columns)}): {list(data.columns)}")
    
    nan_counts = data.isna().sum()
    total_nans = nan_counts.sum()
    if total_nans > 0:
        print(f"\nNaN counts:")
        for col in data.columns:
            count = nan_counts[col]
            if count > 0:
                print(f"  {col}: {count} NaN ({count/len(data)*100:.1f}%)")
        print(f"✗ Total: {total_nans} NaN values")
    else:
        print("\n✓ NO NaN VALUES!")

if hasattr(result, 'raw_spot_data') and result.raw_spot_data is not None and not result.raw_spot_data.empty:
    raw = result.raw_spot_data
    print(f"\n--- RAW SPOT DATA ---")
    print(f"Rows: {len(raw)}")
    print(f"Columns ({len(raw.columns)}): {list(raw.columns)}")

if hasattr(result, 'pre_processed_data') and result.pre_processed_data is not None and not result.pre_processed_data.empty:
    pre = result.pre_processed_data
    print(f"\n--- PRE PROCESSED DATA (from database) ---")
    print(f"Rows: {len(pre)}")
    print(f"Columns ({len(pre.columns)}): {list(pre.columns)}")

print("\n" + "="*80)
