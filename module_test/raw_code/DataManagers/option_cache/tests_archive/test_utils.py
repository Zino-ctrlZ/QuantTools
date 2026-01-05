"""
Test cache utility functions.
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Set WORK_DIR if not already set
if 'WORK_DIR' not in os.environ:
    os.environ['WORK_DIR'] = str(Path(__file__).parent.parent.parent.parent.parent)

from helpers import load_option_data_cache
from utils import (
    filter_today_from_data,
    get_cached_data,
    check_cache_completeness,
    save_to_cache,
    get_cache_for_type_and_model,
    log_cache_stats
)


def test_filter_today_from_data():
    """Test filtering today's data from cacheable data."""
    print("\n=== Test 1: Filter Today From Data ===")
    
    today = pd.Timestamp.now().normalize()
    yesterday = today - pd.Timedelta(days=1)
    two_days_ago = today - pd.Timedelta(days=2)
    
    # Create data from T-2 to T-0 (today)
    dates = [two_days_ago, yesterday, today]
    test_df = pd.DataFrame({
        'close': [100, 101, 102],
        'volume': [1000, 1100, 1200]
    }, index=dates)
    
    # Filter: should get T-2 and T-1 as cacheable, T-0 as today_data
    cacheable, today_data = filter_today_from_data(test_df)
    
    assert len(cacheable) == 2, f"Expected 2 cacheable rows, got {len(cacheable)}"
    assert len(today_data) == 1, f"Expected 1 today row, got {len(today_data)}"
    assert two_days_ago in cacheable.index, "T-2 should be cacheable"
    assert yesterday in cacheable.index, "T-1 should be cacheable"
    assert today in today_data.index, "T-0 should be in today_data"
    
    print(f"✓ Filtered {len(test_df)} rows → {len(cacheable)} cacheable, {len(today_data)} today")
    print(f"✓ Cacheable dates: {[d.date() for d in cacheable.index]}")
    print(f"✓ Today dates: {[d.date() for d in today_data.index]}")
    
    # Test with no today data
    past_only_df = pd.DataFrame({
        'close': [100, 101]
    }, index=[two_days_ago, yesterday])
    
    cacheable2, today_data2 = filter_today_from_data(past_only_df)
    assert len(cacheable2) == 2, "All past data should be cacheable"
    assert len(today_data2) == 0, "No today data"
    print(f"✓ Past-only data: {len(cacheable2)} cacheable, {len(today_data2)} today")


def test_cache_operations():
    """Test get, save, and retrieve operations."""
    print("\n=== Test 2: Cache Operations ===")
    
    caches = load_option_data_cache()
    spot_cache = caches[0]
    
    # Create test data
    dates = pd.date_range('2024-01-01', '2024-01-05', freq='D')
    test_df = pd.DataFrame({
        'close': [100, 101, 102, 103, 104],
        'volume': [1000, 1100, 1200, 1300, 1400]
    }, index=dates)
    
    opttick = 'TEST_C_20241220_100.0'
    interval = '1d'
    
    # Test: get from empty cache
    cached = get_cached_data(spot_cache, interval, opttick)
    assert cached is None, "Should return None for non-existent data"
    print(f"✓ Empty cache returns None")
    
    # Test: save to cache
    save_to_cache(spot_cache, interval, opttick, test_df, merge_with_existing=False)
    print(f"✓ Saved {len(test_df)} rows to cache")
    
    # Test: retrieve from cache
    cached = get_cached_data(spot_cache, interval, opttick)
    assert cached is not None, "Should retrieve saved data"
    assert len(cached) == len(test_df), "Retrieved data length mismatch"
    pd.testing.assert_frame_equal(cached, test_df)
    print(f"✓ Retrieved {len(cached)} rows from cache")
    
    # Test: merge with existing data
    new_dates = pd.date_range('2024-01-06', '2024-01-08', freq='D')
    new_df = pd.DataFrame({
        'close': [105, 106, 107],
        'volume': [1500, 1600, 1700]
    }, index=new_dates)
    
    save_to_cache(spot_cache, interval, opttick, new_df, merge_with_existing=True)
    merged = get_cached_data(spot_cache, interval, opttick)
    assert len(merged) == len(test_df) + len(new_df), "Merged data length incorrect"
    print(f"✓ Merged data: {len(test_df)} + {len(new_df)} = {len(merged)} rows")
    
    # Clean up
    del spot_cache[interval]


def test_cache_completeness():
    """Test checking if cached data covers requested range."""
    print("\n=== Test 3: Cache Completeness Check ===")
    
    # Create test data with gaps
    dates = pd.date_range('2024-01-01', '2024-01-10', freq='D')
    # Remove some dates to create gaps
    dates_with_gaps = dates[[0, 1, 2, 5, 6, 9]]  # Missing Jan 4, 5, 8, 9, 10
    
    test_df = pd.DataFrame({
        'close': np.random.rand(len(dates_with_gaps))
    }, index=dates_with_gaps)
    
    # Test: None/empty data
    is_complete, missing = check_cache_completeness(None, '2024-01-01', '2024-01-10')
    assert not is_complete, "None data should be incomplete"
    assert len(missing) == 2, "Should return start and end dates"
    print(f"✓ None data: incomplete, missing range: {missing[0].date()} to {missing[1].date()}")
    
    # Test: Partial data
    is_complete, missing = check_cache_completeness(test_df, '2024-01-01', '2024-01-10')
    assert not is_complete, "Data with gaps should be incomplete"
    print(f"✓ Partial data: incomplete, {len(missing)} missing dates")
    
    # Test: Complete data
    complete_df = pd.DataFrame({
        'close': np.random.rand(10)
    }, index=dates)
    is_complete, missing = check_cache_completeness(complete_df, '2024-01-01', '2024-01-10')
    assert is_complete, "Complete data should be marked complete"
    assert len(missing) == 0, "Complete data should have no missing dates"
    print(f"✓ Complete data: complete, 0 missing dates")


def test_get_cache_for_type_and_model():
    """Test cache selection based on type and model."""
    print("\n=== Test 4: Cache Selection ===")
    
    caches = load_option_data_cache()
    spot, bs_vol, bs_greeks, binom_vol, binom_greeks = caches
    
    # Test spot (model-agnostic)
    cache = get_cache_for_type_and_model('spot', 'bs', caches)
    assert cache is spot, "Should return spot cache for type='spot'"
    print("✓ type='spot' → OPTION_SPOT_CACHE")
    
    # Test BS vol
    cache = get_cache_for_type_and_model('vol', 'bs', caches)
    assert cache is bs_vol, "Should return BS vol cache"
    print("✓ type='vol', model='bs' → BS_VOL_CACHE")
    
    # Test Binomial vol
    cache = get_cache_for_type_and_model('vol', 'binomial', caches)
    assert cache is binom_vol, "Should return Binomial vol cache"
    print("✓ type='vol', model='binomial' → BINOMIAL_VOL_CACHE")
    
    # Test BS greeks
    cache = get_cache_for_type_and_model('greeks', 'bs', caches)
    assert cache is bs_greeks, "Should return BS greeks cache"
    print("✓ type='greeks', model='bs' → BS_GREEKS_CACHE")
    
    # Test Binomial greeks
    cache = get_cache_for_type_and_model('greeks', 'binomial', caches)
    assert cache is binom_greeks, "Should return Binomial greeks cache"
    print("✓ type='greeks', model='binomial' → BINOMIAL_GREEKS_CACHE")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Cache Utility Functions")
    print("=" * 60)
    
    try:
        test_filter_today_from_data()
        test_cache_operations()
        test_cache_completeness()
        test_get_cache_for_type_and_model()
        
        print("\n" + "=" * 60)
        print("✅ ALL UTILITY TESTS PASSED")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)
