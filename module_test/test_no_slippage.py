"""Test no slippage execution option.

Validates that:
1. ExecutionHandlerConfig accepts 'none' slippage model
2. calculate_slippage_value returns 0.0 when slippage_model='none'
3. All other slippage models still work correctly

Usage:
    /Users/chiemelienwanisobi/miniconda3/envs/openbb_new_use/bin/python \
        module_test/test_no_slippage.py
"""

from __future__ import annotations

from queue import Queue
from datetime import datetime

from EventDriven.configs.core import ExecutionHandlerConfig
from EventDriven.execution import SimulatedExecutionHandler
from EventDriven.event import OrderEvent


def test_no_slippage_config():
    """Test that ExecutionHandlerConfig accepts 'none' slippage model."""
    print("=" * 70)
    print("Testing No Slippage Configuration")
    print("=" * 70)

    # Test 1: Create config with 'none' slippage model
    print("\n[Test 1] Create ExecutionHandlerConfig with slippage_model='none'")
    config = ExecutionHandlerConfig(slippage_model="none")
    print(f"  Config slippage_model: {config.slippage_model}")
    assert config.slippage_model == "none", "Config should accept 'none' slippage model"
    print("  ✓ Config created successfully with slippage_model='none'")

    # Test 2: Create execution handler with no slippage config
    print("\n[Test 2] Create SimulatedExecutionHandler with no slippage")
    events = Queue()
    handler = SimulatedExecutionHandler(events=events, config=config)
    print(f"  Handler config slippage_model: {handler.config.slippage_model}")
    assert handler.config.slippage_model == "none"
    print("  ✓ Handler created successfully")

    # Test 3: Calculate slippage for BUY order with 'none' model
    print("\n[Test 3] Calculate slippage for BUY order (should be 0.0)")
    order_event = OrderEvent(
        datetime=datetime(2024, 12, 10),
        symbol="HD",
        order_type="MKT",
        direction="BUY",
        quantity=10,
        cash=1000.0,
        position={"close": 5.0, "spread": 0.10, "trade_id": "&L:HD20250620C500"},
        signal_id="test_signal_001",
    )
    slippage = handler.calculate_slippage_value(order_event)
    print(f"  Slippage value: {slippage}")
    assert slippage == 0.0, f"Expected 0.0, got {slippage}"
    print("  ✓ Slippage is 0.0 for BUY order")

    # Test 4: Calculate slippage for SELL order with 'none' model
    print("\n[Test 4] Calculate slippage for SELL order (should be 0.0)")
    order_event.direction = "SELL"
    slippage = handler.calculate_slippage_value(order_event)
    print(f"  Slippage value: {slippage}")
    assert slippage == 0.0, f"Expected 0.0, got {slippage}"
    print("  ✓ Slippage is 0.0 for SELL order")

    print("\n" + "=" * 70)
    print("SUMMARY: All no slippage tests passed!")
    print("=" * 70)


def test_other_slippage_models():
    """Test that other slippage models still work correctly."""
    print("\n" + "=" * 70)
    print("Testing Other Slippage Models")
    print("=" * 70)

    events = Queue()

    # Test randomized model
    print("\n[Test 5] Randomized slippage model")
    config_random = ExecutionHandlerConfig(slippage_model="randomized")
    handler_random = SimulatedExecutionHandler(events=events, config=config_random, max_slippage_pct=0.002)
    order_event = OrderEvent(
        datetime=datetime(2024, 12, 10),
        symbol="HD",
        order_type="MKT",
        direction="BUY",
        quantity=10,
        cash=1000.0,
        position={"close": 5.0, "spread": 0.10, "trade_id": "&L:HD20250620C500"},
        signal_id="test_signal_002",
    )
    slippage = handler_random.calculate_slippage_value(order_event)
    print(f"  Randomized slippage: {slippage:.6f}")
    assert slippage != 0.0, "Randomized slippage should not be 0"
    assert slippage > 0, "BUY order should have positive slippage"
    print(f"  ✓ Randomized slippage working: {slippage:.6f}")

    # Test fixed model
    print("\n[Test 6] Fixed slippage model")
    config_fixed = ExecutionHandlerConfig(slippage_model="fixed")
    handler_fixed = SimulatedExecutionHandler(events=events, config=config_fixed, max_slippage_pct=0.002)
    slippage = handler_fixed.calculate_slippage_value(order_event)
    print(f"  Fixed slippage: {slippage:.6f}")
    expected_fixed = 5.0 * 0.002 * 10  # close * max_slippage_pct * quantity
    assert abs(slippage - expected_fixed) < 0.001, f"Expected {expected_fixed}, got {slippage}"
    print(f"  ✓ Fixed slippage working: {slippage:.6f}")

    # Test spread_pct model
    print("\n[Test 7] Spread percentage slippage model")
    config_spread = ExecutionHandlerConfig(slippage_model="spread_pct", pct_alpha=0.25)
    handler_spread = SimulatedExecutionHandler(events=events, config=config_spread)
    slippage = handler_spread.calculate_slippage_value(order_event)
    print(f"  Spread pct slippage: {slippage:.6f}")
    expected_spread = 0.10 * 0.25 * 10  # spread * pct_alpha * quantity
    assert abs(slippage - expected_spread) < 0.001, f"Expected {expected_spread}, got {slippage}"
    print(f"  ✓ Spread pct slippage working: {slippage:.6f}")

    print("\n" + "=" * 70)
    print("SUMMARY: All slippage model tests passed!")
    print("=" * 70)


def test_comparison_with_without_slippage():
    """Compare execution with and without slippage."""
    print("\n" + "=" * 70)
    print("Comparison: With vs Without Slippage")
    print("=" * 70)

    events = Queue()

    order_event = OrderEvent(
        datetime=datetime(2024, 12, 10),
        symbol="HD",
        order_type="MKT",
        direction="BUY",
        quantity=10,
        cash=1000.0,
        position={"close": 5.0, "spread": 0.10, "trade_id": "&L:HD20250620C500"},
        signal_id="test_signal_003",
    )

    # With slippage
    print("\n[Test 8] Execution with fixed slippage (0.2%)")
    config_with = ExecutionHandlerConfig(slippage_model="fixed")
    handler_with = SimulatedExecutionHandler(events=events, config=config_with, max_slippage_pct=0.002)
    slippage_with = handler_with.calculate_slippage_value(order_event)
    print(f"  Slippage amount: ${slippage_with:.4f}")
    print(f"  Price impact: {(slippage_with / (order_event.position['close'] * order_event.quantity)) * 100:.2f}%")

    # Without slippage
    print("\n[Test 9] Execution with NO slippage")
    config_without = ExecutionHandlerConfig(slippage_model="none")
    handler_without = SimulatedExecutionHandler(events=events, config=config_without)
    slippage_without = handler_without.calculate_slippage_value(order_event)
    print(f"  Slippage amount: ${slippage_without:.4f}")
    print(f"  Price impact: {(slippage_without / (order_event.position['close'] * order_event.quantity)) * 100:.2f}%")

    print(f"\n  Difference: ${abs(slippage_with - slippage_without):.4f}")
    print(f"  Cost savings with no slippage: ${slippage_with:.4f}")

    print("\n" + "=" * 70)
    print("SUMMARY: Comparison complete!")
    print("=" * 70)


if __name__ == "__main__":
    try:
        test_no_slippage_config()
        test_other_slippage_models()
        test_comparison_with_without_slippage()
        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED!")
        print("=" * 70)
        print("\nNo slippage option successfully implemented and verified.")
        print("To use: ExecutionHandlerConfig(slippage_model='none')")
        print()
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n✗ TEST ERROR: {type(e).__name__}: {e}")
        raise
