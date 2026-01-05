"""
Simple focused benchmark for LimitsAndSizingCog._analyze_impl()

Directly creates sample position contexts and benchmarks the analyze method.
Based on the notebook approach in positions_and_limits_focus.ipynb Cell 21.

Usage:
    cd /path/to/QuantTools
    python EventDriven/riskmanager/position/cogs/benchmarks/simple_focused_benchmark.py --baseline --iterations 100
"""

from __future__ import annotations

import time
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import pandas as pd
import sys

# Ensure proper imports
sys.path.insert(0, '/Users/chiemelienwanisobi/cloned_repos/QuantTools')

from EventDriven.riskmanager.position.cogs.limits import LimitsAndSizingCog
from EventDriven.configs.core import LimitsEnabledConfig
from EventDriven.dataclasses.limits import PositionLimits


def create_sample_positions_for_benchmark(num_positions: int = 10):
    """
    Create sample position data for benchmarking.
    Returns a list of mock position objects that _analyze_impl would process.
    """
    from EventDriven.dataclasses.states import PositionState, PortfolioState, PortfolioMetaInfo, PositionAnalysisContext
    from EventDriven.dataclasses.timeseries import AtTimePositionData
    from EventDriven.riskmanager.market_data import AtIndexResult
    import pandas as pd
    from datetime import datetime, timedelta
    
    print(f"Creating {num_positions} sample positions...")
    
    positions = []
    base_date = datetime(2024, 6, 14)
    
    for i in range(num_positions):
        # Create trade_id with proper format
        # Format: &L:SYMBOL_YYYYMMDD_C/P_STRIKE&S:SYMBOL_YYYYMMDD_C/P_STRIKE
        expiry = (base_date + timedelta(days=180+i)).strftime("%Y%m%d")
        strike1 = int(150 + i * 5)
        strike2 = int(strike1 + 10)
        trade_id = f"&L:AAPL{expiry}C{strike1}&S:AAPL{expiry}C{strike2}"
        
        # Create mock position data (AtTimePositionData)
        position_data = AtTimePositionData(
            position_id=trade_id,
            date=base_date + timedelta(days=i),
            close=10.5 + i,
            bid=10.0 + i,
            ask=11.0 + i,
            midpoint=10.5 + i,
            delta=0.45 + (i * 0.05),  # Vary delta
            gamma=0.02,
            theta=-0.15,
            vega=0.25,
        )
        
        # Create mock underlier data
        undl_price = 150.0 + i * 2
        underlier_data = AtIndexResult(
            sym="AAPL",
            date=pd.Timestamp(base_date + timedelta(days=i)),
            spot=pd.Series([undl_price], index=[base_date + timedelta(days=i)]),
            chain_spot=pd.Series({"close": undl_price, "open": undl_price-1, "high": undl_price+1, "low": undl_price-2}),
            rates=pd.Series([0.05], index=[base_date + timedelta(days=i)]),
            dividends=pd.Series([0.0], index=[base_date + timedelta(days=i)])
        )
        
        # Create position
        position = PositionState(
            trade_id=trade_id,
            signal_id=f"SIGNAL_{i}",
            underlier_tick="AAPL",
            quantity=10,
            entry_price=10.0,
            current_position_data=position_data,
            current_underlier_data=underlier_data,
            pnl=100.0 + i * 10,
            last_updated=base_date + timedelta(days=i),
        )
        
        positions.append(position)
    
    # Create portfolio state
    portfolio = PortfolioState(
        cash=100000.0,
        positions=positions,
        pnl=sum([p.pnl for p in positions]),
        total_value=100000.0 + sum([p.pnl for p in positions]),
        last_updated=base_date,
    )
    
    # Create portfolio meta info  
    portfolio_meta = PortfolioMetaInfo(
        start_date=base_date - timedelta(days=30),
        t_plus_n=0,
    )
    
    # Create context
    context = PositionAnalysisContext(
        date=base_date,
        portfolio=portfolio,
        portfolio_meta=portfolio_meta,
    )
    
    return context


def setup_cog_with_limits(context):
    """
    Create a LimitsAndSizingCog and initialize position_limits.
    Similar to notebook Cell 21 approach.
    
    Also patches database lookup functions to avoid errors with mock data.
    """
    print("Setting up LimitsAndSizingCog with position limits...")
    
    # Patch adjust_for_events to skip database lookups
    from EventDriven.riskmanager.position.cogs import analyze_utils
    original_adjust = analyze_utils.adjust_for_events
    
    def mock_adjust_for_events(start, date, option):
        """Mock version that just returns the option unchanged"""
        return option
    
    analyze_utils.adjust_for_events = mock_adjust_for_events
    
    # Create cog
    config = LimitsEnabledConfig()
    cog = LimitsAndSizingCog(config=config)
    
    # Initialize position limits for all positions
    for position in context.portfolio.positions:
        cog.position_limits[position.trade_id] = PositionLimits(
            delta=500.0,  # Sample delta limit
            dte=120,
            moneyness=1.15,
        )
    
    print(f"Cog initialized with {len(cog.position_limits)} position limits")
    return cog


def benchmark_analyze_impl(
    cog,  # LimitsAndSizingCog
    context,  # PositionAnalysisContext
    iterations: int = 100
) -> Dict[str, Any]:
    """
    Benchmark cog._analyze_impl() by calling it repeatedly.
    
    Args:
        cog: Fully initialized LimitsAndSizingCog
        context: PositionAnalysisContext with sample positions
        iterations: Number of times to call _analyze_impl
        
    Returns:
        Dictionary with timing results
    """
    num_positions = len(context.portfolio.positions)
    
    print(f"\n{'='*70}")
    print("FOCUSED BENCHMARK: LimitsAndSizingCog._analyze_impl()")
    print(f"Positions: {num_positions}, Iterations: {iterations}")
    print(f"{'='*70}\n")
    
    timings = []
    
    print("Starting benchmark...")
    
    for i in range(iterations):
        iteration_start = time.perf_counter()
        
        # Run _analyze_impl
        _ = cog._analyze_impl(context)
        
        iteration_time = time.perf_counter() - iteration_start
        timings.append(iteration_time)
        
        # Print progress every 20 iterations
        if (i + 1) % 20 == 0:
            avg_so_far = sum(timings) / len(timings)
            print(f"Iteration {i+1}/{iterations}: {iteration_time:.6f}s "
                  f"(avg so far: {avg_so_far:.6f}s)")
    
    # Calculate statistics
    avg_time = sum(timings) / len(timings)
    min_time = min(timings)
    max_time = max(timings)
    
    # Time per position
    avg_time_per_position_ms = (avg_time / num_positions * 1000) if num_positions > 0 else 0
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'iterations': iterations,
        'num_positions': num_positions,
        'timings_seconds': timings,
        'avg_time_seconds': avg_time,
        'min_time_seconds': min_time,
        'max_time_seconds': max_time,
        'avg_time_per_position_ms': avg_time_per_position_ms,
        'total_time_seconds': sum(timings),
    }
    
    print(f"\n{'='*70}")
    print("RESULTS:")
    print(f"  Average time per iteration:  {avg_time:.6f}s")
    print(f"  Min time:                    {min_time:.6f}s")
    print(f"  Max time:                    {max_time:.6f}s")
    print(f"  Avg time per position:       {avg_time_per_position_ms:.3f}ms")
    print(f"  Total positions analyzed:    {num_positions * iterations:,}")
    print(f"{'='*70}\n")
    
    return results


def compare_with_baseline(current_results: Dict[str, Any], baseline_file: Path):
    """Compare current results with baseline."""
    if not baseline_file.exists():
        print("No baseline file found for comparison")
        return
    
    with open(baseline_file, 'r') as f:
        baseline = json.load(f)
    
    baseline_avg = baseline['avg_time_seconds']
    current_avg = current_results['avg_time_seconds']
    
    time_diff = current_avg - baseline_avg
    pct_change = (time_diff / baseline_avg) * 100
    
    baseline_per_pos = baseline.get('avg_time_per_position_ms', 0)
    current_per_pos = current_results['avg_time_per_position_ms']
    
    print(f"{'='*70}")
    print("COMPARISON WITH BASELINE:")
    print(f"  Baseline avg time:     {baseline_avg:.6f}s")
    print(f"  Current avg time:      {current_avg:.6f}s")
    print(f"  Time difference:       {time_diff:+.6f}s ({pct_change:+.2f}%)")
    print("")
    print(f"  Baseline per position: {baseline_per_pos:.3f}ms")
    print(f"  Current per position:  {current_per_pos:.3f}ms")
    
    if pct_change < 0:
        speedup = -pct_change
        print("")
        print(f"  🚀 SPEEDUP: {speedup:.2f}% faster!")
    elif pct_change > 0:
        print("")
        print(f"  ⚠️  REGRESSION: {pct_change:.2f}% slower")
    else:
        print("")
        print("  No change")
    
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Simple focused benchmark for LimitsAndSizingCog._analyze_impl()'
    )
    parser.add_argument(
        '--baseline',
        action='store_true',
        help='Save results as baseline for future comparisons'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=100,
        help='Number of iterations (default: 100)'
    )
    parser.add_argument(
        '--positions',
        type=int,
        default=10,
        help='Number of sample positions to create (default: 10)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Custom output filename (default: auto-generated)'
    )
    
    args = parser.parse_args()
    
    # Create sample data
    print("="*70)
    print("SETUP PHASE")
    print("="*70)
    
    context = create_sample_positions_for_benchmark(num_positions=args.positions)
    cog = setup_cog_with_limits(context)
    
    # Run benchmark
    results = benchmark_analyze_impl(cog, context, iterations=args.iterations)
    
    # Determine output file
    benchmarks_dir = Path(__file__).resolve().parent
    
    if args.output:
        output_file = benchmarks_dir / args.output
    elif args.baseline:
        output_file = benchmarks_dir / 'simple_focused_baseline.json'
        results['version'] = 'baseline'
        results['description'] = 'Baseline before Task #2 (early returns optimization)'
    else:
        # Auto-generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = benchmarks_dir / f'simple_focused_results_{timestamp}.json'
        results['version'] = 'current'
        results['description'] = 'Current performance measurement'
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_file}")
    
    # Compare with baseline if not creating baseline
    if not args.baseline:
        baseline_file = benchmarks_dir / 'simple_focused_baseline.json'
        compare_with_baseline(results, baseline_file)


if __name__ == '__main__':
    main()
