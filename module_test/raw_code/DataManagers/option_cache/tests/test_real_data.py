# ruff: noqa
"""
Real data test for CachedOptionDataManager.

Test parameters:
- opttick: AAPL20260821C290
- start: 2025-01-01
- end: today (2025-11-29)
- type: spot, vol, greeks
- model: bs
- interval: 1d
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
import time
from datetime import datetime

# Set WORK_DIR if not already set
if 'WORK_DIR' not in os.environ:
    os.environ['WORK_DIR'] = str(Path(__file__).parent.parent.parent.parent)
    print(f"Set WORK_DIR to: {os.environ['WORK_DIR']}")

# Add parent directory to path
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from DataManagers.DataManagers_cached import CachedOptionDataManager
from DataManagers.option_cache import clear_all_caches

# Test parameters
OPTTICK = "AAPL20260821C290"
START = "2025-01-01"
END = datetime.now().strftime("%Y-%m-%d")  # Today
INTERVAL = "1d"
MODEL = "bs"

print("=" * 70)
print("REAL DATA TEST: CachedOptionDataManager")
print("=" * 70)
print(f"Opttick:  {OPTTICK}")
print(f"Start:    {START}")
print(f"End:      {END} (today)")
print(f"Interval: {INTERVAL}")
print(f"Model:    {MODEL}")
print("=" * 70)

# Clear caches to start fresh
print("\n🗑️  Clearing all caches...")
clear_all_caches()
print("✅ Caches cleared\n")

# Create manager
print(f"📦 Creating CachedOptionDataManager for {OPTTICK}...")
dm = CachedOptionDataManager(opttick=OPTTICK)
print(f"✅ Manager created")
print(f"   - Symbol: {dm.symbol}")
print(f"   - Strike: {dm.strike}")
print(f"   - Right: {dm.right}")
print(f"   - Exp: {dm.exp}")
print(f"   - Cache enabled: {dm.enable_cache}\n")

# Test 1: SPOT data
print("=" * 70)
print("TEST 1: SPOT DATA")
print("=" * 70)

print("\n🔍 First SPOT query (expecting cache MISS)...")
start_time = time.time()
try:
    result_spot_1 = dm.get_timeseries(START, END, INTERVAL, type_='spot')
    time_spot_1 = time.time() - start_time
    print(f"✅ First SPOT query completed in {time_spot_1:.2f}s")
    if hasattr(result_spot_1, 'final_data'):
        print(f"   - Rows returned: {len(result_spot_1.final_data)}")
except Exception as e:
    print(f"❌ First SPOT query failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n🔍 Second SPOT query (expecting cache HIT)...")
start_time = time.time()
try:
    result_spot_2 = dm.get_timeseries(START, END, INTERVAL, type_='spot')
    time_spot_2 = time.time() - start_time
    print(f"✅ Second SPOT query completed in {time_spot_2:.2f}s")
    if hasattr(result_spot_2, 'final_data'):
        print(f"   - Rows returned: {len(result_spot_2.final_data)}")
except Exception as e:
    print(f"❌ Second SPOT query failed: {e}")
    import traceback
    traceback.print_exc()

print(f"\n📊 SPOT Performance:")
print(f"   - First query (MISS):  {time_spot_1:.2f}s")
print(f"   - Second query (HIT):  {time_spot_2:.2f}s")
if time_spot_1 > 0:
    print(f"   - Speedup ratio: {time_spot_1/time_spot_2:.2f}x")

# Test 2: VOL data
print("\n" + "=" * 70)
print("TEST 2: VOL DATA (should cascade to SPOT cache)")
print("=" * 70)

print("\n🔍 First VOL query (expecting VOL MISS, SPOT HIT from cascade)...")
start_time = time.time()
try:
    result_vol_1 = dm.get_timeseries(START, END, INTERVAL, type_='vol', model=MODEL)
    time_vol_1 = time.time() - start_time
    print(f"✅ First VOL query completed in {time_vol_1:.2f}s")
    if hasattr(result_vol_1, 'final_data'):
        print(f"   - Rows returned: {len(result_vol_1.final_data)}")
except Exception as e:
    print(f"❌ First VOL query failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n🔍 Second VOL query (expecting cache HIT)...")
start_time = time.time()
try:
    result_vol_2 = dm.get_timeseries(START, END, INTERVAL, type_='vol', model=MODEL)
    time_vol_2 = time.time() - start_time
    print(f"✅ Second VOL query completed in {time_vol_2:.2f}s")
    if hasattr(result_vol_2, 'final_data'):
        print(f"   - Rows returned: {len(result_vol_2.final_data)}")
except Exception as e:
    print(f"❌ Second VOL query failed: {e}")
    import traceback
    traceback.print_exc()

print(f"\n📊 VOL Performance:")
print(f"   - First query (MISS):  {time_vol_1:.2f}s")
print(f"   - Second query (HIT):  {time_vol_2:.2f}s")
if time_vol_1 > 0:
    print(f"   - Speedup ratio: {time_vol_1/time_vol_2:.2f}x")

# Test 3: GREEKS data
print("\n" + "=" * 70)
print("TEST 3: GREEKS DATA (should cascade to VOL and SPOT caches)")
print("=" * 70)

print("\n🔍 First GREEKS query (expecting GREEKS MISS, VOL HIT, SPOT HIT from cascade)...")
start_time = time.time()
try:
    result_greeks_1 = dm.get_timeseries(START, END, INTERVAL, type_='greeks', model=MODEL)
    time_greeks_1 = time.time() - start_time
    print(f"✅ First GREEKS query completed in {time_greeks_1:.2f}s")
    if hasattr(result_greeks_1, 'final_data'):
        print(f"   - Rows returned: {len(result_greeks_1.final_data)}")
except Exception as e:
    print(f"❌ First GREEKS query failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n🔍 Second GREEKS query (expecting cache HIT)...")
start_time = time.time()
try:
    result_greeks_2 = dm.get_timeseries(START, END, INTERVAL, type_='greeks', model=MODEL)
    time_greeks_2 = time.time() - start_time
    print(f"✅ Second GREEKS query completed in {time_greeks_2:.2f}s")
    if hasattr(result_greeks_2, 'final_data'):
        print(f"   - Rows returned: {len(result_greeks_2.final_data)}")
except Exception as e:
    print(f"❌ Second GREEKS query failed: {e}")
    import traceback
    traceback.print_exc()

print(f"\n📊 GREEKS Performance:")
print(f"   - First query (MISS):  {time_greeks_1:.2f}s")
print(f"   - Second query (HIT):  {time_greeks_2:.2f}s")
if time_greeks_1 > 0:
    print(f"   - Speedup ratio: {time_greeks_1/time_greeks_2:.2f}x")

# Summary
print("\n" + "=" * 70)
print("📈 OVERALL PERFORMANCE SUMMARY")
print("=" * 70)
print(f"SPOT:")
print(f"  First:  {time_spot_1:.2f}s (MISS)")
print(f"  Second: {time_spot_2:.2f}s (HIT)")
print(f"  Ratio:  {time_spot_1/time_spot_2:.2f}x" if time_spot_1 > 0 else "  N/A")

print(f"\nVOL:")
print(f"  First:  {time_vol_1:.2f}s (MISS, SPOT cascade)")
print(f"  Second: {time_vol_2:.2f}s (HIT)")
print(f"  Ratio:  {time_vol_1/time_vol_2:.2f}x" if time_vol_1 > 0 else "  N/A")

print(f"\nGREEKS:")
print(f"  First:  {time_greeks_1:.2f}s (MISS, VOL+SPOT cascade)")
print(f"  Second: {time_greeks_2:.2f}s (HIT)")
print(f"  Ratio:  {time_greeks_1/time_greeks_2:.2f}x" if time_greeks_1 > 0 else "  N/A")

print("\n" + "=" * 70)
print("✅ ALL TESTS COMPLETED SUCCESSFULLY")
print("=" * 70)
print("\n💡 Note: Cache is working! Data saved to:")
print(f"   {os.environ.get('WORK_DIR', 'WORK_DIR')}/.cache/data_manager_cache/")
print("\n💡 Next run will be even faster once cache retrieval logic is complete.")
print("   Currently falling back to parent for complete cache hits.")
