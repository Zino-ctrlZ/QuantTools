#!/usr/bin/env python3
"""
Targeted test: Does EODData._lazy_load() actually use cached MarketTimeseries data?

This test directly checks if the _ManagerLazyLoader lazy loading mechanism
correctly retrieves data from the global MarketTimeseries cache.
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

from DataManagers.utils import _ManagerLazyLoader, set_global_market_timeseries, get_global_market_timeseries

print("="*70)
print("TARGETED TEST: EODData + MarketTimeseries Integration")
print("="*70)
print("Testing _ManagerLazyLoader._lazy_load() with cached underlier\n")

# Step 1: Create a _ManagerLazyLoader instance
print("[1] Creating _ManagerLazyLoader for AAPL...")
loader = _ManagerLazyLoader(symbol='AAPL')
loader.exp = pd.Timestamp('2025-01-17')  # Set expiration for date range calculation
print(f"    ✓ Created loader for {loader.symbol}\n")

# Step 2: Check if global MarketTimeseries is set
print("[2] Checking global MarketTimeseries...")
market_ts = get_global_market_timeseries()
if market_ts is None:
    print("    ⚠️  Global MarketTimeseries is None!")
    print("    This means CachedOptionDataManager hasn't set it yet")
    print("    Let's try to create one manually...\n")
    
    # Try to import and create MarketTimeseries
    try:
        from EventDriven.riskmanager.market_data import get_timeseries_obj
        market_ts = get_timeseries_obj()
        set_global_market_timeseries(market_ts)
        print(f"    ✓ Created and set MarketTimeseries: {market_ts}\n")
    except Exception as e:
        print(f"    ❌ Failed to create MarketTimeseries: {e}")
        print("    Test cannot proceed without MarketTimeseries\n")
        sys.exit(1)
else:
    print(f"    ✓ Global MarketTimeseries exists: {market_ts}\n")

# Step 3: Preload some data into MarketTimeseries
print("[3] Preloading AAPL data into MarketTimeseries...")
start = pd.Timestamp("2024-10-01")
end = pd.Timestamp("2024-10-31")
interval = '1d'

try:
    # Check if already loaded
    if market_ts.already_loaded('AAPL', interval, start, end):
        print(f"    ✓ Data already cached for AAPL {start.date()} to {end.date()}")
    else:
        print(f"    Loading AAPL data for {start.date()} to {end.date()}...")
        t_load = time.time()
        market_ts.load_timeseries(
            sym='AAPL',
            start_date=start,
            end_date=end,
            interval=interval,
            force=False
        )
        time_load = time.time() - t_load
        print(f"    ✓ Loaded data in {time_load:.3f}s")
    
    # Verify data is there
    result = market_ts.get_timeseries(
        sym='AAPL',
        factor='chain_spot',
        interval=interval
    )
    if result and result.chain_spot is not None:
        print(f"    ✓ Verified: {len(result.chain_spot)} rows in MarketTimeseries cache\n")
    else:
        print(f"    ⚠️  No data found in MarketTimeseries cache\n")
except Exception as e:
    print(f"    ❌ Error loading data: {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 4: Test EODData access (should use cached data)
print("="*70)
print("[4] Testing EODData['s0_chain'] access...")
print("="*70)
print("This should:")
print("  1. Call EODData.__getitem__('s0_chain')")
print("  2. Check parent._eod dictionary (empty initially)")
print("  3. Call parent._lazy_load('s0_chain', intra_flag=False)")
print("  4. _lazy_load checks global MarketTimeseries")
print("  5. Returns cached data (NO API CALL!)\n")

# Access eod data - this triggers lazy loading
t1 = time.time()
try:
    chain_data = loader.eod['s0_chain']
    time_access = time.time() - t1
    
    print(f"\n✓ Got chain data: {len(chain_data)} rows in {time_access:.3f}s")
    print(f"  Date range: {chain_data.index.min()} to {chain_data.index.max()}")
    
    if time_access < 1.0:
        print(f"✅ FAST ACCESS ({time_access:.3f}s) - Likely used cached data!")
    else:
        print(f"⚠️  SLOW ACCESS ({time_access:.3f}s) - May have hit API instead of cache")
        
except Exception as e:
    time_access = time.time() - t1
    print(f"\n❌ Error accessing eod['s0_chain']: {e}")
    print(f"   Time before error: {time_access:.3f}s")
    import traceback
    traceback.print_exc()

# Step 5: Test dividend access
print("\n" + "="*70)
print("[5] Testing EODData['y'] (dividends) access...")
print("="*70)

t2 = time.time()
try:
    div_data = loader.eod['y']
    time_div = time.time() - t2
    
    print(f"\n✓ Got dividend data: {len(div_data)} rows in {time_div:.3f}s")
    print(f"  Date range: {div_data.index.min()} to {div_data.index.max()}")
    
    if time_div < 1.0:
        print(f"✅ FAST ACCESS ({time_div:.3f}s) - Likely used cached data!")
    else:
        print(f"⚠️  SLOW ACCESS ({time_div:.3f}s) - May have hit API instead of cache")
        
except Exception as e:
    time_div = time.time() - t2
    print(f"\n❌ Error accessing eod['y']: {e}")
    print(f"   Time before error: {time_div:.3f}s")
    import traceback
    traceback.print_exc()

# Step 6: Check what's in parent._eod
print("\n" + "="*70)
print("[6] Checking loader._eod dictionary...")
print("="*70)
print(f"Keys in _eod: {list(loader._eod.keys())}")
print(f"Number of cached items: {len(loader._eod)}")

if loader._eod:
    for key, value in loader._eod.items():
        if isinstance(value, pd.DataFrame):
            print(f"  {key}: DataFrame with {len(value)} rows")
        else:
            print(f"  {key}: {type(value)}")

print("\n" + "="*70)
print("ANALYSIS")
print("="*70)
print("If access times are < 1s, MarketTimeseries cache is working!")
print("If access times are > 2s, it's hitting APIs (cache not working)")
print("\nCheck logs above for:")
print("  ✓ 'Loading s0_chain for AAPL from MarketTimeseries cache'")
print("  ✓ 'Loading dividends for AAPL from MarketTimeseries cache'")
print("="*70)
