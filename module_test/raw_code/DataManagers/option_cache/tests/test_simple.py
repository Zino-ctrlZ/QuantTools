#!/usr/bin/env python3
# ruff: noqa
"""
Expanded cache test to verify spot, vol, and greeks caching across multiple options.
Tests 13 different options to ensure cache works correctly for various tickers.
"""
import sys
from pathlib import Path

# Add parent directories to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

import os
WORK_DIR = Path(__file__).parent.parent.parent.parent
os.environ["WORK_DIR"] = str(WORK_DIR)
print(f"Set WORK_DIR to: {WORK_DIR}")

import pandas as pd
import time
from datetime import datetime
from DataManagers.DataManagers_cached import CachedOptionDataManager
from DataManagers.option_cache import clear_all_caches
from DataManagers import set_skip_mysql_query

# Test parameters
TEST_OPTIONS = [
    "AAPL20260821C300",
    "AMD20260618C270",
    "AMD20260618C280",
    "BA20260618C270",
    "BA20260618C275",
    "META20260618C800",
    "META20260618C810",
    "NVDA20260918C200",
    "NVDA20260918C210",
    "TSLA20260717C470",
    "TSLA20260717C480",
    "AMZN20260618C290",
    "AMZN20260618C295",
]

START = pd.Timestamp("2025-10-01")
END = pd.Timestamp("2025-11-28")
INTERVAL = "1d"
MODEL = "bs"

print("\n" + "="*70)
print("EXPANDED CACHE TEST: Multiple Options")
print("="*70)
print(f"Testing {len(TEST_OPTIONS)} options:")
for opt in TEST_OPTIONS:
    print(f"  - {opt}")
print(f"\nDate Range: {START.date()} to {END.date()}")
print(f"Interval:   {INTERVAL}")
print(f"Model:      {MODEL}")
print("="*70 + "\n")

# Skip MySQL queries for better performance
print("[SETUP] Setting skip_mysql_query=True for better performance...")
set_skip_mysql_query(True)

# Clear all caches
print("[SETUP] Clearing all caches...")
clear_all_caches()
print("[SETUP] Caches cleared\n")

def parse_opttick(opttick: str):
    """Parse opttick into symbol, exp, right, strike"""
    # Example: AAPL20260821C300
    # Find where the date starts (first digit)
    i = 0
    while i < len(opttick) and not opttick[i].isdigit():
        i += 1
    
    symbol = opttick[:i]
    rest = opttick[i:]
    
    # Date is next 8 characters: YYYYMMDD
    exp_str = rest[:8]
    exp = f"{exp_str[0:4]}-{exp_str[4:6]}-{exp_str[6:8]}"
    
    # Right is next character (C or P)
    right = rest[8]
    
    # Strike is the rest
    strike = float(rest[9:])
    
    return symbol, exp, right, strike


def test_option(opttick: str, type_name: str, model: str = MODEL):
    """Test a specific option for a specific data type"""
    try:
        # Parse option
        symbol, exp, right, strike = parse_opttick(opttick)
        
        # Create manager
        dm = CachedOptionDataManager(
            symbol=symbol,
            right=right,
            strike=strike,
            exp=exp,
            enable_cache=True
        )
        
        # First query (cache MISS)
        start_time = time.time()
        result_1 = dm.get_timeseries(START, END, INTERVAL, type_=type_name, model=model)
        time_1 = time.time() - start_time
        
        # Check result
        if not hasattr(result_1, 'post_processed_data'):
            return None, None, f"No post_processed_data"
        
        data_1 = result_1.post_processed_data
        if data_1.empty:
            return None, None, "Empty data"
        
        rows_1 = len(data_1)
        
        # Second query (cache HIT)
        start_time = time.time()
        result_2 = dm.get_timeseries(START, END, INTERVAL, type_=type_name, model=model)
        time_2 = time.time() - start_time
        
        # Check result
        if not hasattr(result_2, 'post_processed_data'):
            return None, None, f"No post_processed_data on 2nd query"
        
        data_2 = result_2.post_processed_data
        rows_2 = len(data_2)
        
        # Compare
        speedup = time_1 / time_2 if time_2 > 0 else 1.0
        
        return {
            'time_1': time_1,
            'time_2': time_2,
            'speedup': speedup,
            'rows_1': rows_1,
            'rows_2': rows_2,
        }, None
        
    except Exception as e:
        return None, str(e)


# Run tests
print("[TEST] Testing all options for SPOT, VOL, GREEKS...\n")

all_results = {}
total_tests = len(TEST_OPTIONS) * 3  # 3 types per option
completed = 0

for opttick in TEST_OPTIONS:
    print(f"\n{'='*70}")
    print(f"TESTING: {opttick}")
    print(f"{'='*70}")
    
    all_results[opttick] = {}
    
    for type_name in ['spot', 'vol', 'greeks']:
        print(f"\n  [{type_name.upper()}] ", end='', flush=True)
        
        result, error = test_option(opttick, type_name)
        completed += 1
        
        if error:
            print(f"FAIL - {error}")
            all_results[opttick][type_name] = {'status': 'FAIL', 'error': error}
        elif result is None:
            print(f"FAIL - Unknown error")
            all_results[opttick][type_name] = {'status': 'FAIL', 'error': 'Unknown'}
        else:
            print(f"PASS - {result['rows_1']} rows, {result['speedup']:.2f}x speedup ({result['time_1']:.2f}s -> {result['time_2']:.2f}s)")
            all_results[opttick][type_name] = {'status': 'PASS', **result}
    
    print(f"\n  Progress: {completed}/{total_tests} tests completed")


# Summary
print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)

summary_by_type = {'spot': {'pass': 0, 'fail': 0}, 'vol': {'pass': 0, 'fail': 0}, 'greeks': {'pass': 0, 'fail': 0}}
summary_by_option = {}

for opttick, type_results in all_results.items():
    option_pass = 0
    option_fail = 0
    
    for type_name, result in type_results.items():
        if result['status'] == 'PASS':
            summary_by_type[type_name]['pass'] += 1
            option_pass += 1
        else:
            summary_by_type[type_name]['fail'] += 1
            option_fail += 1
    
    summary_by_option[opttick] = {'pass': option_pass, 'fail': option_fail}

# Print by type
print("\nResults by Type:")
for type_name in ['spot', 'vol', 'greeks']:
    total = summary_by_type[type_name]['pass'] + summary_by_type[type_name]['fail']
    pass_count = summary_by_type[type_name]['pass']
    print(f"  {type_name.upper():8s}: {pass_count}/{total} passed")

# Print by option
print("\nResults by Option:")
for opttick in TEST_OPTIONS:
    stats = summary_by_option[opttick]
    total = stats['pass'] + stats['fail']
    status = "PASS" if stats['fail'] == 0 else "FAIL"
    print(f"  {opttick:20s}: {stats['pass']}/{total} passed [{status}]")

# Calculate averages for passed tests
print("\nPerformance Metrics (for passed tests):")
for type_name in ['spot', 'vol', 'greeks']:
    times_1 = []
    times_2 = []
    speedups = []
    
    for opttick, type_results in all_results.items():
        if type_name in type_results and type_results[type_name]['status'] == 'PASS':
            times_1.append(type_results[type_name]['time_1'])
            times_2.append(type_results[type_name]['time_2'])
            speedups.append(type_results[type_name]['speedup'])
    
    if times_1:
        avg_time_1 = sum(times_1) / len(times_1)
        avg_time_2 = sum(times_2) / len(times_2)
        avg_speedup = sum(speedups) / len(speedups)
        print(f"  {type_name.upper():8s}: {avg_time_1:.3f}s -> {avg_time_2:.3f}s (avg {avg_speedup:.2f}x speedup)")

print("="*70)

# Overall result
total_pass = sum(s['pass'] for s in summary_by_option.values())
total_fail = sum(s['fail'] for s in summary_by_option.values())
success_rate = (total_pass / (total_pass + total_fail)) * 100 if (total_pass + total_fail) > 0 else 0

print(f"\nOVERALL: {total_pass}/{total_pass + total_fail} tests passed ({success_rate:.1f}%)")

if total_fail == 0:
    print("\n[RESULT] ALL TESTS PASSED")
else:
    print(f"\n[RESULT] {total_fail} TESTS FAILED")
    sys.exit(1)
