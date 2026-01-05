"""
Debug single ticker to troubleshoot empty results.
Following exact steps:
1. Initialize with CachedOptionDataManager(opttick=opttick)
2. Parse ticker to get exp
3. Calculate start = exp - BDay(30)
4. Run queries for Spot, Vol, Greeks
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import time
import traceback
from pandas.tseries.offsets import BDay
from trade.helpers.helper import parse_option_tick
from DataManagers.DataManagers_cached import CachedOptionDataManager
from tabulate import tabulate

def test_single_ticker():
    """Test a single ticker following exact steps."""
    
    # Pick a ticker
    opttick = 'AAPL20241220C215'
    print("="*100)
    print(f"TESTING SINGLE TICKER: {opttick}")
    print("="*100)
    
    # Step 1: Initialize with CachedOptionDataManager(opttick=opttick)
    print("\n[STEP 1] Initialize CachedOptionDataManager(opttick=opttick)")
    print("-"*100)
    manager = CachedOptionDataManager(opttick=opttick)
    print(f"✓ Manager created")
    print(f"  Symbol: {manager.symbol}")
    print(f"  Expiration: {manager.exp}")
    print(f"  Strike: {manager.strike}")
    print(f"  Right: {manager.right}")
    
    # Step 2a: Parse ticker
    print("\n[STEP 2a] Parse option tick")
    print("-"*100)
    parsed = parse_option_tick(opttick)
    print(f"Parsed components:")
    for key, val in parsed.items():
        print(f"  {key}: {val} (type: {type(val).__name__})")
    
    # Step 2b: Extract exp
    print("\n[STEP 2b] Extract exp from parsed['exp_date']")
    print("-"*100)
    exp_str = parsed['exp_date']
    print(f"exp (raw) = {exp_str} (type: {type(exp_str).__name__})")
    
    # Convert to pandas Timestamp for BDay arithmetic
    import pandas as pd
    exp = pd.to_datetime(exp_str)
    print(f"exp (converted) = {exp} (type: {type(exp).__name__})")
    
    # Step 2c: Calculate start = exp - BDay(30)
    print("\n[STEP 2c] Calculate start = exp - BDay(30)")
    print("-"*100)
    start = exp - BDay(30)
    end = exp  # Use expiration as end
    print(f"start = {start} (type: {type(start).__name__})")
    print(f"end = {end} (type: {type(end).__name__})")
    print(f"Date range: {start} to {end}")
    
    # Step 2d: Run queries
    print("\n" + "="*100)
    print("[STEP 2d] RUN SPOT QUERY")
    print("="*100)
    try:
        start_time = time.time()
        result_spot = manager.get_timeseries(
            type_='spot',
            start=start,
            end=end,
            interval='1d'
        )
        elapsed = time.time() - start_time
        
        # Check for data in post_processed_data attribute (cached data returns this)
        data = None
        if result_spot:
            if hasattr(result_spot, 'post_processed_data') and result_spot.post_processed_data is not None:
                data = result_spot.post_processed_data
            elif hasattr(result_spot, 'data') and result_spot.data is not None:
                data = result_spot.data
        
        if data is not None and not data.empty:
            print(f"✓ SUCCESS - Spot query completed in {elapsed:.3f}s")
            print("\nData Summary:")
            print(f"  Rows: {len(data)}")
            print(f"  Columns: {list(data.columns)}")
            print(f"  Date range: {data.index.min()} to {data.index.max()}")
            
            # Save to CSV
            csv_path = 'spot_data.csv'
            data.to_csv(csv_path)
            print(f"  Saved to: {csv_path}")
            
            print("\n" + "="*100)
            print("SPOT DATA TABLE (Sample - First 10 & Last 5 rows)")
            print("="*100)
            
            # Show sample: first 10 and last 5 rows
            sample_data = pd.concat([data.head(10), data.tail(5)])
            sample_data_reset = sample_data.reset_index()
            
            # Format for display
            print(tabulate(
                sample_data_reset, 
                headers='keys', 
                tablefmt='grid',
                floatfmt='.2f',
                showindex=False
            ))
        else:
            print(f"✗ EMPTY RESULT - Spot query returned empty or None in {elapsed:.3f}s")
            print(f"  Result object: {result_spot}")
            print(f"  Has data attr: {hasattr(result_spot, 'data') if result_spot else False}")
            print(f"  Has post_processed_data attr: {hasattr(result_spot, 'post_processed_data') if result_spot else False}")
            if result_spot and hasattr(result_spot, 'post_processed_data'):
                print(f"  post_processed_data is None: {result_spot.post_processed_data is None}")
                if result_spot.post_processed_data is not None:
                    print(f"  post_processed_data is empty: {result_spot.post_processed_data.empty}")
            
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*100)
    print("[STEP 2d] RUN VOL QUERY")
    print("="*100)
    try:
        start_time = time.time()
        result_vol = manager.get_timeseries(
            type_='vol',
            start=start,
            end=end,
            interval='1d'
        )
        elapsed = time.time() - start_time
        
        # Check for data in post_processed_data attribute (cached data returns this)
        data_vol = None
        if result_vol:
            if hasattr(result_vol, 'post_processed_data') and result_vol.post_processed_data is not None:
                data_vol = result_vol.post_processed_data
            elif hasattr(result_vol, 'data') and result_vol.data is not None:
                data_vol = result_vol.data
        
        if data_vol is not None and not data_vol.empty:
            print(f"✓ SUCCESS - Vol query completed in {elapsed:.3f}s")
            print("\nData Summary:")
            print(f"  Rows: {len(data_vol)}")
            print(f"  Columns: {list(data_vol.columns)}")
            print(f"  Date range: {data_vol.index.min()} to {data_vol.index.max()}")
            
            # Save to CSV
            csv_path = 'vol_data.csv'
            data_vol.to_csv(csv_path)
            print(f"  Saved to: {csv_path}")
            
            print("\n" + "="*100)
            print("VOL DATA TABLE (Sample - First 10 & Last 5 rows)")
            print("="*100)
            
            # Show sample: first 10 and last 5 rows
            sample_vol = pd.concat([data_vol.head(10), data_vol.tail(5)])
            sample_vol_reset = sample_vol.reset_index()
            
            # Format for display
            print(tabulate(
                sample_vol_reset, 
                headers='keys', 
                tablefmt='grid',
                floatfmt='.6f',
                showindex=False
            ))
        else:
            print(f"✗ EMPTY RESULT - Vol query returned empty or None in {elapsed:.3f}s")
            print(f"  Result object: {result_vol}")
            print(f"  Has data attr: {hasattr(result_vol, 'data') if result_vol else False}")
            print(f"  Has post_processed_data attr: {hasattr(result_vol, 'post_processed_data') if result_vol else False}")
            
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*100)
    print("[STEP 2d] RUN GREEKS QUERY")
    print("="*100)
    try:
        start_time = time.time()
        result_greeks = manager.get_timeseries(
            type_='greeks',
            start=start,
            end=end,
            interval='1d',
            model='bs'
        )
        elapsed = time.time() - start_time
        
        # Check for data in post_processed_data attribute (cached data returns this)
        data_greeks = None
        if result_greeks:
            if hasattr(result_greeks, 'post_processed_data') and result_greeks.post_processed_data is not None:
                data_greeks = result_greeks.post_processed_data
            elif hasattr(result_greeks, 'data') and result_greeks.data is not None:
                data_greeks = result_greeks.data
        
        if data_greeks is not None and not data_greeks.empty:
            print(f"✓ SUCCESS - Greeks query completed in {elapsed:.3f}s")
            print("\nData Summary:")
            print(f"  Rows: {len(data_greeks)}")
            print(f"  Columns: {list(data_greeks.columns)}")
            print(f"  Date range: {data_greeks.index.min()} to {data_greeks.index.max()}")
            
            # Save to CSV
            csv_path = 'greeks_data.csv'
            data_greeks.to_csv(csv_path)
            print(f"  Saved to: {csv_path}")
            
            print("\n" + "="*100)
            print("GREEKS DATA TABLE (Sample - First 10 & Last 5 rows)")
            print("="*100)
            
            # Show sample: first 10 and last 5 rows
            sample_greeks = pd.concat([data_greeks.head(10), data_greeks.tail(5)])
            sample_greeks_reset = sample_greeks.reset_index()
            
            # Format for display - split into two parts due to many columns
            print("\n--- Part 1: Vega, Vanna, Volga, Delta, Gamma, Theta, Rho ---")
            cols_part1 = ['Datetime', 'vega', 'vanna', 'volga', 'delta', 'gamma', 'theta', 'rho']
            available_cols1 = [col for col in cols_part1 if col in sample_greeks_reset.columns]
            print(tabulate(
                sample_greeks_reset[available_cols1], 
                headers='keys', 
                tablefmt='grid',
                floatfmt='.6f',
                showindex=False
            ))
            
            print("\n--- Part 2: Midpoint Greeks (Vega, Vanna, Volga, Delta, Gamma, Theta, Rho) ---")
            cols_part2 = ['Datetime', 'midpoint_vega', 'midpoint_vanna', 'midpoint_volga', 
                         'midpoint_delta', 'midpoint_gamma', 'midpoint_theta', 'midpoint_rho']
            available_cols2 = [col for col in cols_part2 if col in sample_greeks_reset.columns]
            print(tabulate(
                sample_greeks_reset[available_cols2], 
                headers='keys', 
                tablefmt='grid',
                floatfmt='.6f',
                showindex=False
            ))
        else:
            print(f"✗ EMPTY RESULT - Greeks query returned empty or None in {elapsed:.3f}s")
            print(f"  Result object: {result_greeks}")
            print(f"  Has data attr: {hasattr(result_greeks, 'data') if result_greeks else False}")
            print(f"  Has post_processed_data attr: {hasattr(result_greeks, 'post_processed_data') if result_greeks else False}")
            
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*100)
    print("[CACHE TEST] SECOND RUN - Testing Cache Hits")
    print("="*100)
    
    # Run spot again to test cache
    print("\nSPOT (2nd run - should be cached):")
    start_time = time.time()
    result_spot2 = manager.get_timeseries(
        type_='spot',
        start=start,
        end=end,
        interval='1d'
    )
    elapsed = time.time() - start_time
    
    data_spot2 = None
    if result_spot2:
        if hasattr(result_spot2, 'post_processed_data') and result_spot2.post_processed_data is not None:
            data_spot2 = result_spot2.post_processed_data
        elif hasattr(result_spot2, 'data') and result_spot2.data is not None:
            data_spot2 = result_spot2.data
    
    if data_spot2 is not None and not data_spot2.empty:
        print(f"✓ Spot cache hit: {len(data_spot2)} rows in {elapsed:.6f}s (expected < 0.020s for cache)")
    else:
        print(f"✗ Spot cache miss or empty: {elapsed:.3f}s")
    
    # Run vol again
    print("\nVOL (2nd run):")
    start_time = time.time()
    result_vol2 = manager.get_timeseries(
        type_='vol',
        start=start,
        end=end,
        interval='1d'
    )
    elapsed = time.time() - start_time
    
    data_vol2 = None
    if result_vol2:
        if hasattr(result_vol2, 'post_processed_data') and result_vol2.post_processed_data is not None:
            data_vol2 = result_vol2.post_processed_data
        elif hasattr(result_vol2, 'data') and result_vol2.data is not None:
            data_vol2 = result_vol2.data
    
    if data_vol2 is not None and not data_vol2.empty:
        print(f"✓ Vol cache hit: {len(data_vol2)} rows in {elapsed:.6f}s (expected < 0.010s for cache)")
    else:
        print(f"✗ Vol cache miss or empty: {elapsed:.3f}s")
    
    # Run greeks again
    print("\nGREEKS (2nd run):")
    start_time = time.time()
    result_greeks2 = manager.get_timeseries(
        type_='greeks',
        start=start,
        end=end,
        interval='1d',
        model='bs'
    )
    elapsed = time.time() - start_time
    
    data_greeks2 = None
    if result_greeks2:
        if hasattr(result_greeks2, 'post_processed_data') and result_greeks2.post_processed_data is not None:
            data_greeks2 = result_greeks2.post_processed_data
        elif hasattr(result_greeks2, 'data') and result_greeks2.data is not None:
            data_greeks2 = result_greeks2.data
    
    if data_greeks2 is not None and not data_greeks2.empty:
        print(f"✓ Greeks cache hit: {len(data_greeks2)} rows in {elapsed:.6f}s (expected < 0.010s for cache)")
    else:
        print(f"✗ Greeks cache miss or empty: {elapsed:.3f}s")


if __name__ == '__main__':
    test_single_ticker()
