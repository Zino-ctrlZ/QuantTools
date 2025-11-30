"""
Basic test to verify cache infrastructure works correctly.
"""
import os
import sys
from pathlib import Path

# Set WORK_DIR if not already set (for testing)
if 'WORK_DIR' not in os.environ:
    os.environ['WORK_DIR'] = str(Path(__file__).parent.parent.parent.parent.parent)
    print(f"Set WORK_DIR to: {os.environ['WORK_DIR']}")

from helpers import load_option_data_cache, get_cache_instances, clear_all_caches

def test_cache_initialization():
    """Test that all 5 caches initialize correctly."""
    print("\n=== Test 1: Cache Initialization ===")
    
    spot, bs_vol, bs_greeks, binom_vol, binom_greeks = load_option_data_cache()
    
    print(f"✓ OPTION_SPOT_CACHE: {spot}")
    print(f"✓ BS_VOL_CACHE: {bs_vol}")
    print(f"✓ BS_GREEKS_CACHE: {bs_greeks}")
    print(f"✓ BINOMIAL_VOL_CACHE: {binom_vol}")
    print(f"✓ BINOMIAL_GREEKS_CACHE: {binom_greeks}")
    
    # Verify they're not None
    assert spot is not None, "OPTION_SPOT_CACHE is None"
    assert bs_vol is not None, "BS_VOL_CACHE is None"
    assert bs_greeks is not None, "BS_GREEKS_CACHE is None"
    assert binom_vol is not None, "BINOMIAL_VOL_CACHE is None"
    assert binom_greeks is not None, "BINOMIAL_GREEKS_CACHE is None"
    
    print("✓ All caches initialized successfully")


def test_cache_singleton():
    """Test that calling load_option_data_cache twice returns same instances."""
    print("\n=== Test 2: Singleton Pattern ===")
    
    caches1 = load_option_data_cache()
    caches2 = load_option_data_cache()
    
    # Should be the same instances
    assert caches1[0] is caches2[0], "OPTION_SPOT_CACHE not singleton"
    assert caches1[1] is caches2[1], "BS_VOL_CACHE not singleton"
    assert caches1[2] is caches2[2], "BS_GREEKS_CACHE not singleton"
    assert caches1[3] is caches2[3], "BINOMIAL_VOL_CACHE not singleton"
    assert caches1[4] is caches2[4], "BINOMIAL_GREEKS_CACHE not singleton"
    
    print("✓ Singleton pattern working correctly")


def test_cache_basic_operations():
    """Test basic cache read/write operations."""
    print("\n=== Test 3: Basic Cache Operations ===")
    
    spot_cache, _, _, _, _ = get_cache_instances()
    
    # Test write
    test_key = 'test_interval'
    test_data = {'SPY_C_20251220_450.0': 'test_dataframe'}
    
    spot_cache[test_key] = test_data
    print(f"✓ Wrote test data to cache")
    
    # Test read
    retrieved = spot_cache.get(test_key)
    assert retrieved == test_data, "Retrieved data doesn't match"
    print(f"✓ Retrieved data matches: {retrieved}")
    
    # Clean up
    del spot_cache[test_key]
    assert spot_cache.get(test_key) is None, "Failed to delete test data"
    print(f"✓ Deleted test data successfully")


def test_cache_path_logic():
    """Test that cache path logic works with environment variables."""
    print("\n=== Test 4: Cache Path Logic ===")
    
    from helpers import _get_cache_base_path
    
    base_path = _get_cache_base_path()
    print(f"✓ Cache base path: {base_path}")
    
    # Verify path exists
    assert base_path.exists(), f"Cache path doesn't exist: {base_path}"
    assert base_path.is_dir(), f"Cache path is not a directory: {base_path}"
    
    print(f"✓ Cache directory exists and is valid")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing DataManagers Cache Infrastructure")
    print("=" * 60)
    
    try:
        test_cache_initialization()
        test_cache_singleton()
        test_cache_basic_operations()
        test_cache_path_logic()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)
