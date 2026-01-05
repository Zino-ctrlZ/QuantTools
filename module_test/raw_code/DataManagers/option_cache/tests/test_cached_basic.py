# ruff: noqa
"""
Basic test for CachedOptionDataManager structure.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Set WORK_DIR if not already set
if 'WORK_DIR' not in os.environ:
    os.environ['WORK_DIR'] = str(Path(__file__).parent.parent.parent.parent)
    print(f"Set WORK_DIR to: {os.environ['WORK_DIR']}")

# Add parent directory to path to allow package-style imports
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from DataManagers.DataManagers_cached import CachedOptionDataManager


def test_initialization():
    """Test that CachedOptionDataManager can be initialized."""
    print("\n=== Test 1: Initialization ===")
    
    # Test 1: Initialize with parameters
    dm = CachedOptionDataManager(
        symbol='SPY',
        exp='2025-12-20',
        right='C',
        strike=450.0
    )
    
    print(f"✓ Created instance (params): {dm.opttick}")
    print(f"  - Symbol: {dm.symbol}, Strike: {dm.strike}, Right: {dm.right}")
    print(f"  - Cache enabled: {dm.enable_cache}")
    
    assert dm.symbol == 'SPY', "Symbol mismatch"
    assert dm.strike == 450.0, "Strike mismatch"
    assert dm.right == 'C', "Right mismatch"
    assert dm.enable_cache is True, "Cache should be enabled by default"
    assert dm._CACHES is not None, "Caches should be loaded"
    
    # Test 2: Initialize with opttick string
    dm2 = CachedOptionDataManager(opttick="AAPL20260821C290")
    
    print(f"✓ Created instance (opttick): {dm2.opttick}")
    print(f"  - Symbol: {dm2.symbol}, Strike: {dm2.strike}, Right: {dm2.right}")
    
    assert dm2.symbol == 'AAPL', "Symbol should be AAPL"
    assert dm2.strike == 290.0, "Strike should be 290"
    assert dm2.right == 'C', "Right should be C"


def test_cache_toggle():
    """Test enabling/disabling cache."""
    print("\n=== Test 2: Cache Toggle ===")
    
    # Test instance-level disable
    dm1 = CachedOptionDataManager(
        symbol='SPY',
        exp='2025-12-20',
        right='C',
        strike=450.0,
        enable_cache=False
    )
    assert dm1.enable_cache is False, "Cache should be disabled"
    print(f"✓ Instance-level disable works")
    
    # Test global disable
    CachedOptionDataManager.disable_caching()
    dm2 = CachedOptionDataManager(
        symbol='SPY',
        exp='2025-12-20',
        right='P',
        strike=440.0
    )
    assert dm2.enable_cache is False, "Cache should be disabled globally"
    print(f"✓ Global disable works")
    
    # Re-enable
    CachedOptionDataManager.enable_caching()
    dm3 = CachedOptionDataManager(
        symbol='SPY',
        exp='2025-12-20',
        right='C',
        strike=460.0
    )
    assert dm3.enable_cache is True, "Cache should be re-enabled"
    print(f"✓ Global enable works")


def test_should_use_cache():
    """Test cache decision logic."""
    print("\n=== Test 3: Should Use Cache ===")
    
    dm = CachedOptionDataManager(
        symbol='SPY',
        exp='2025-12-20',
        right='C',
        strike=450.0
    )
    
    # Should cache 1d
    assert dm._should_use_cache('1d') is True, "Should cache 1d interval"
    print(f"✓ interval='1d' → use cache")
    
    # Should NOT cache 5Min
    assert dm._should_use_cache('5Min') is False, "Should NOT cache 5Min"
    print(f"✓ interval='5Min' → skip cache")
    
    # Should NOT cache when disabled
    dm.enable_cache = False
    assert dm._should_use_cache('1d') is False, "Should NOT cache when disabled"
    print(f"✓ Cache disabled → skip cache")


def test_inheritance():
    """Test that it properly inherits from OptionDataManager."""
    print("\n=== Test 4: Inheritance ===")
    
    dm = CachedOptionDataManager(
        symbol='SPY',
        exp='2025-12-20',
        right='C',
        strike=450.0
    )
    
    # Check inherited attributes
    assert hasattr(dm, 'spot_manager'), "Should have spot_manager"
    assert hasattr(dm, 'vol_manager'), "Should have vol_manager"
    assert hasattr(dm, 'greek_manager'), "Should have greek_manager"
    assert hasattr(dm, 'opttick'), "Should have opttick"
    assert hasattr(dm, 'get_timeseries'), "Should have get_timeseries method"
    
    print(f"✓ Has spot_manager: {type(dm.spot_manager).__name__}")
    print(f"✓ Has vol_manager: {type(dm.vol_manager).__name__}")
    print(f"✓ Has greek_manager: {type(dm.greek_manager).__name__}")
    print(f"✓ Has get_timeseries method")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing CachedOptionDataManager Basic Structure")
    print("=" * 60)
    
    try:
        test_initialization()
        test_cache_toggle()
        test_should_use_cache()
        test_inheritance()
        
        print("\n" + "=" * 60)
        print("✅ ALL BASIC TESTS PASSED")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)
