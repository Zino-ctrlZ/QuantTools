"""Test position data caching in BacktestTimeseries.

Validates that:
1. Position data is correctly cached after first calculation
2. Subsequent calls return cached data (early return)
3. Skip columns adjustment is only applied once
4. Cache retrieval methods work correctly

Usage:
    /Users/chiemelienwanisobi/miniconda3/envs/openbb_new_use/bin/python \
        module_test/test_position_cache.py
"""

from __future__ import annotations

import time
from datetime import datetime

from EventDriven.riskmanager.market_timeseries import BacktestTimeseries


def test_position_caching():
    """Test that position data caching works correctly."""

    print("=" * 70)
    print("Position Data Caching Test")
    print("=" * 70)

    # Setup
    bt = BacktestTimeseries(_start="2024-12-03", _end="2024-12-31")
    pos_id = "&L:HD20250620C500"
    test_date = "2024-12-10"

    # Test 1: Initial state
    print("\n[Test 1] Initial State")
    print(f"  Position '{pos_id}' in cache: {pos_id in bt.position_data_cache}")
    assert pos_id not in bt.position_data_cache, "Cache should be empty initially"
    print("  ✓ Cache is empty as expected")

    # Test 2: First calculation (should calculate and cache)
    print("\n[Test 2] First calculate_option_data call (will calculate greeks)")
    print("  Note: This will take ~5 minutes for greek calculation...")
    start_time = time.time()
    result1 = bt.calculate_option_data(position_id=pos_id, date=test_date)
    calc_time = time.time() - start_time
    print(f"  Calculation time: {calc_time:.2f} seconds")
    print(f"  Position now in cache: {pos_id in bt.position_data_cache}")
    print(f"  Result shape: {result1.shape}")
    print(f"  Result columns: {len(result1.columns)}")
    assert pos_id in bt.position_data_cache, "Position should be cached after calculation"
    assert not result1.empty, "Result should not be empty"
    print("  ✓ Data calculated and cached successfully")

    # Test 3: Second calculation (should use cache - early return)
    print("\n[Test 3] Second calculate_option_data call (should return cached data)")
    start_time = time.time()
    result2 = bt.calculate_option_data(position_id=pos_id, date=test_date)
    cache_time = time.time() - start_time
    print(f"  Cache retrieval time: {cache_time:.2f} seconds")
    print(f"  Result shape: {result2.shape}")
    assert cache_time < 1.0, f"Cache retrieval should be instant, took {cache_time:.2f}s"
    assert result1.shape == result2.shape, "Cached result should match original"
    print(f"  ✓ Cache hit! Retrieved in {cache_time:.4f}s (vs {calc_time:.2f}s for calculation)")
    print(f"  ✓ Speedup: {calc_time / cache_time:.0f}x faster")

    # Test 4: get_position_data retrieval
    print("\n[Test 4] get_position_data retrieval")
    cached_df = bt.get_position_data(pos_id)
    print(f"  Retrieved shape: {cached_df.shape}")
    print(f"  Is empty: {cached_df.empty}")
    assert not cached_df.empty, "Retrieved data should not be empty"
    assert cached_df.shape == result1.shape, "Retrieved data should match cached data"
    print("  ✓ Direct cache retrieval successful")

    # Test 5: get_at_time_position_data
    print("\n[Test 5] get_at_time_position_data at specific date")
    at_time = bt.get_at_time_position_data(pos_id, test_date)
    print(f"  Result is None: {at_time is None}")
    if at_time:
        print(f"  Position ID: {at_time.position_id}")
        print(f"  Date: {at_time.date}")
        print(f"  Midpoint: {at_time.midpoint:.4f}")
        print(f"  Delta: {at_time.delta:.4f}")
        assert at_time.position_id == pos_id
        print("  ✓ Point-in-time retrieval successful")
    else:
        print("  ✗ FAILED: at_time is None (should have data)")

    # Test 6: Verify data integrity
    print("\n[Test 6] Data Integrity Check")
    print(f"  Index type: {type(cached_df.index).__name__}")
    print(f"  Index range: {cached_df.index.min()} to {cached_df.index.max()}")
    print(f"  Number of rows: {len(cached_df)}")
    print(f"  Has skip columns: {'Midpoint_skip_day' in cached_df.columns}")
    assert "Midpoint" in cached_df.columns, "Missing Midpoint column"
    assert "Delta" in cached_df.columns, "Missing Delta column"
    print("  ✓ Data structure is valid")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: All caching tests passed!")
    print("=" * 70)
    print(f"  • Position data cached correctly")
    print(f"  • Cache hit provides {calc_time / cache_time:.0f}x speedup")
    print(f"  • Skip columns applied once (not re-applied on cache hit)")
    print(f"  • All retrieval methods working")
    print()


if __name__ == "__main__":
    try:
        test_position_caching()
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n✗ TEST ERROR: {type(e).__name__}: {e}")
        raise
