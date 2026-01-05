"""
Comprehensive benchmark test for 50 option tickers.
Tests Spot, Vol, and Greeks caching performance across diverse expirations.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import time

from DataManagers.DataManagers_cached import CachedOptionDataManager


def parse_opttick(opttick: str) -> Tuple[str, datetime, str, float]:
    """Parse option ticker to extract components."""
    # Find where the date starts (after symbol, before date)
    i = 0
    while not opttick[i].isdigit():
        i += 1
    
    symbol = opttick[:i]
    date_str = opttick[i:i+8]
    option_type = opttick[i+8]
    strike = float(opttick[i+9:])
    
    exp_date = datetime.strptime(date_str, '%Y%m%d')
    
    return symbol, exp_date, option_type, strike


def get_business_days_before(exp_date: datetime, n_days: int = 30) -> Tuple[datetime, datetime]:
    """Get start and end dates for N business days before expiration."""
    # Use pandas for business day calculation
    end_date = exp_date - timedelta(days=1)  # Day before expiration
    
    # Generate business days
    bdays = pd.bdate_range(end=end_date, periods=n_days, freq='B')
    start_date = bdays[0].to_pydatetime()
    end_date = bdays[-1].to_pydatetime()
    
    return start_date, end_date


def benchmark_query(optticks: List[str], query_type: str, run_number: int) -> Dict:
    """Benchmark a specific query type across all tickers."""
    print(f"\n{'='*80}")
    print(f"Run #{run_number} - {query_type.upper()} Query")
    print(f"{'='*80}")
    
    results = {
        'opttick': [],
        'symbol': [],
        'expiration': [],
        'strike': [],
        'option_type': [],
        'n_days': [],
        'query_time': [],
        'success': [],
        'error': []
    }
    
    total_time = 0
    success_count = 0
    
    for idx, opttick in enumerate(optticks, 1):
        manager = None
        try:
            # Parse ticker
            symbol, exp_date, opt_type, strike = parse_opttick(opttick)
            start_date, end_date = get_business_days_before(exp_date, n_days=30)
            
            print(f"\n[{idx}/{len(optticks)}] {opttick}")
            print(f"  {symbol} | Exp: {exp_date.date()} | Strike: {strike} | Type: {opt_type}")
            print(f"  Query Period: {start_date.date()} to {end_date.date()}")
            
            # Initialize manager for this specific option
            manager = CachedOptionDataManager(
                symbol=symbol,
                exp=exp_date,
                right=opt_type,
                strike=strike
            )
            
            # Execute query with timing
            start_time = time.time()
            
            if query_type == 'spot':
                result = manager.get_timeseries(
                    type_='spot',
                    start=start_date,
                    end=end_date,
                    interval='1d'
                )
            elif query_type == 'vol':
                result = manager.get_timeseries(
                    type_='vol',
                    start=start_date,
                    end=end_date,
                    interval='1d'
                )
            elif query_type == 'greeks':
                result = manager.get_timeseries(
                    type_='greeks',
                    start=start_date,
                    end=end_date,
                    interval='1d',
                    model='bs'
                )
            
            elapsed_time = time.time() - start_time
            
            # Check result
            if result and hasattr(result, 'data') and result.data is not None and not result.data.empty:
                n_days = len(result.data)
                print(f"  ✓ Success: {n_days} days | Time: {elapsed_time:.3f}s")
                success_count += 1
                success = True
                error_msg = None
            else:
                print(f"  ✗ Empty result | Time: {elapsed_time:.3f}s")
                n_days = 0
                success = False
                error_msg = "Empty result"
            
            total_time += elapsed_time
            
            # Record results
            results['opttick'].append(opttick)
            results['symbol'].append(symbol)
            results['expiration'].append(exp_date)
            results['strike'].append(strike)
            results['option_type'].append(opt_type)
            results['n_days'].append(n_days)
            results['query_time'].append(elapsed_time)
            results['success'].append(success)
            results['error'].append(error_msg)
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"  ✗ ERROR: {str(e)} | Time: {elapsed_time:.3f}s")
            
            results['opttick'].append(opttick)
            results['symbol'].append(symbol if 'symbol' in locals() else None)
            results['expiration'].append(exp_date if 'exp_date' in locals() else None)
            results['strike'].append(strike if 'strike' in locals() else None)
            results['option_type'].append(opt_type if 'opt_type' in locals() else None)
            results['n_days'].append(0)
            results['query_time'].append(elapsed_time)
            results['success'].append(False)
            results['error'].append(str(e))
            
            total_time += elapsed_time
    
    # Summary
    print(f"\n{'='*80}")
    print(f"Run #{run_number} - {query_type.upper()} Summary")
    print(f"{'='*80}")
    print(f"Total queries: {len(optticks)}")
    print(f"Successful: {success_count} ({success_count/len(optticks)*100:.1f}%)")
    print(f"Failed: {len(optticks) - success_count}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time: {total_time/len(optticks):.3f}s")
    print(f"Min time: {min(results['query_time']):.3f}s")
    print(f"Max time: {max(results['query_time']):.3f}s")
    
    return results


def calculate_cache_performance(run1_df: pd.DataFrame, run2_df: pd.DataFrame, 
                                query_type: str) -> Dict:
    """Calculate cache performance metrics comparing two runs."""
    # Filter successful queries from both runs
    successful_both = run1_df[run1_df['success'] & run2_df['success']].copy()
    
    if len(successful_both) == 0:
        return None
    
    # Get matching times
    run1_times = successful_both['query_time'].values
    run2_times = run2_df.loc[successful_both.index, 'query_time'].values
    
    # Calculate metrics
    total_time_run1 = run1_times.sum()
    total_time_run2 = run2_times.sum()
    
    avg_time_run1 = run1_times.mean()
    avg_time_run2 = run2_times.mean()
    
    speedup_factor = avg_time_run1 / avg_time_run2 if avg_time_run2 > 0 else 0
    time_saved = total_time_run1 - total_time_run2
    percent_faster = ((avg_time_run1 - avg_time_run2) / avg_time_run1 * 100) if avg_time_run1 > 0 else 0
    
    return {
        'query_type': query_type,
        'n_queries': len(successful_both),
        'run1_total_time': total_time_run1,
        'run2_total_time': total_time_run2,
        'run1_avg_time': avg_time_run1,
        'run2_avg_time': avg_time_run2,
        'speedup_factor': speedup_factor,
        'time_saved': time_saved,
        'percent_faster': percent_faster
    }


def print_comprehensive_statistics(results_dict: Dict[str, List[Dict]]):
    """Print comprehensive statistics across all query types."""
    print("\n" + "="*100)
    print("COMPREHENSIVE BENCHMARK STATISTICS")
    print("="*100)
    
    # Overall summary table
    print("\n" + "-"*100)
    print("CACHE PERFORMANCE SUMMARY")
    print("-"*100)
    print(f"{'Query Type':<12} {'Queries':<10} {'Run 1 Avg':<12} {'Run 2 Avg':<12} {'Speedup':<10} {'% Faster':<12} {'Time Saved':<12}")
    print("-"*100)
    
    for query_type in ['spot', 'vol', 'greeks']:
        if query_type in results_dict:
            run1_df = pd.DataFrame(results_dict[query_type][0])
            run2_df = pd.DataFrame(results_dict[query_type][1])
            
            perf = calculate_cache_performance(run1_df, run2_df, query_type)
            
            if perf:
                print(f"{perf['query_type'].upper():<12} "
                      f"{perf['n_queries']:<10} "
                      f"{perf['run1_avg_time']:<12.3f} "
                      f"{perf['run2_avg_time']:<12.3f} "
                      f"{perf['speedup_factor']:<10.1f}x "
                      f"{perf['percent_faster']:<12.1f}% "
                      f"{perf['time_saved']:<12.2f}s")
    
    # Detailed breakdown per query type
    for query_type in ['spot', 'vol', 'greeks']:
        if query_type not in results_dict:
            continue
            
        run1_df = pd.DataFrame(results_dict[query_type][0])
        run2_df = pd.DataFrame(results_dict[query_type][1])
        
        print(f"\n{'='*100}")
        print(f"{query_type.upper()} QUERY DETAILED STATISTICS")
        print(f"{'='*100}")
        
        # Run 1 stats
        print(f"\nRun 1 (Cold Cache):")
        print(f"  Total queries: {len(run1_df)}")
        print(f"  Successful: {run1_df['success'].sum()} ({run1_df['success'].sum()/len(run1_df)*100:.1f}%)")
        print(f"  Total time: {run1_df['query_time'].sum():.2f}s")
        print(f"  Average time: {run1_df['query_time'].mean():.3f}s")
        print(f"  Median time: {run1_df['query_time'].median():.3f}s")
        print(f"  Std dev: {run1_df['query_time'].std():.3f}s")
        print(f"  Min time: {run1_df['query_time'].min():.3f}s")
        print(f"  Max time: {run1_df['query_time'].max():.3f}s")
        
        # Run 2 stats
        print(f"\nRun 2 (Warm Cache):")
        print(f"  Total queries: {len(run2_df)}")
        print(f"  Successful: {run2_df['success'].sum()} ({run2_df['success'].sum()/len(run2_df)*100:.1f}%)")
        print(f"  Total time: {run2_df['query_time'].sum():.2f}s")
        print(f"  Average time: {run2_df['query_time'].mean():.3f}s")
        print(f"  Median time: {run2_df['query_time'].median():.3f}s")
        print(f"  Std dev: {run2_df['query_time'].std():.3f}s")
        print(f"  Min time: {run2_df['query_time'].min():.3f}s")
        print(f"  Max time: {run2_df['query_time'].max():.3f}s")
        
        # Performance comparison
        perf = calculate_cache_performance(run1_df, run2_df, query_type)
        if perf:
            print(f"\nCache Performance:")
            print(f"  Speedup factor: {perf['speedup_factor']:.1f}x")
            print(f"  Percent faster: {perf['percent_faster']:.1f}%")
            print(f"  Time saved: {perf['time_saved']:.2f}s")
            print(f"  Time saved per query: {perf['time_saved']/perf['n_queries']:.3f}s")
        
        # Symbol breakdown
        print(f"\nBreakdown by Symbol (Run 2 - Cache Hits):")
        symbol_stats = run2_df[run2_df['success']].groupby('symbol').agg({
            'query_time': ['count', 'mean', 'min', 'max', 'sum']
        }).round(3)
        symbol_stats.columns = ['Count', 'Avg Time', 'Min Time', 'Max Time', 'Total Time']
        print(symbol_stats.to_string())
        
        # Slowest queries in Run 2 (potential cache misses)
        print(f"\nSlowest 10 Queries in Run 2 (Potential partial cache misses):")
        slowest = run2_df.nlargest(10, 'query_time')[['opttick', 'symbol', 'query_time', 'success']]
        print(slowest.to_string(index=False))


def main():
    """Main benchmark execution."""
    optticks = [
        'COST20250620C1080', 'COST20250620C1060',
        'AMZN20240920C195', 'AMZN20240920C185',
        'AMZN20250117C195', 'AMZN20250117C200',
        'AMD20250117C165',
        'AAPL20241220C215',
        'AAPL20250620C250', 'AAPL20250620C260',
        'TSLA20250321C265', 'TSLA20250321C260',
        'AMD20250321C210', 'AMD20250321C200',
        'META20250620C590', 'META20250620C580',
        'SBUX20250321C95', 'SBUX20250321C90',
        'TSLA20250620C250', 'TSLA20250620C240',
        'META20250620C660', 'META20250620C670',
        'AMD20250620C220', 'AMD20250620C210',
        'TSLA20250815C320', 'TSLA20250815C330',
        'AMZN20250620C220', 'AMZN20250620C215',
        'SBUX20250718C105', 'SBUX20250718C100',
        'TSLA20250919C490', 'TSLA20250919C480',
        'BA20250919C250', 'BA20250919C260',
        'BA20250919C200', 'BA20250919C210',
        'AMD20260618C270', 'AMD20260618C280',
        'AAPL20260821C290', 'AAPL20260821C300',
        'BA20260618C275', 'BA20260618C270',
        'META20260618C810', 'META20260618C800',
        'NVDA20260918C210', 'NVDA20260918C200',
        'TSLA20260717C470', 'TSLA20260717C480',
        'AMZN20260618C295', 'AMZN20260618C290'
    ]
    
    print("="*100)
    print("MASSIVE BENCHMARK TEST - 50 OPTION TICKERS")
    print("="*100)
    print(f"Total tickers: {len(optticks)}")
    print("Query period: 30 business days before each expiration")
    print("Query types: Spot, Vol, Greeks")
    print("Runs per type: 2 (cold cache vs warm cache)")
    print("="*100)
    
    # Store all results
    results_dict = {}
    
    # Run benchmarks for each query type (2 runs each)
    for query_type in ['spot', 'vol', 'greeks']:
        results_dict[query_type] = []
        
        # Run 1: Cold cache (or mixed cache state)
        results_run1 = benchmark_query(optticks, query_type, run_number=1)
        results_dict[query_type].append(results_run1)
        
        # Brief pause
        time.sleep(2)
        
        # Run 2: Warm cache (should hit cache for most queries)
        results_run2 = benchmark_query(optticks, query_type, run_number=2)
        results_dict[query_type].append(results_run2)
        
        # Brief pause between query types
        time.sleep(2)
    
    # Print comprehensive statistics
    print_comprehensive_statistics(results_dict)
    
    # Save detailed results to CSV
    print("\n" + "="*100)
    print("SAVING DETAILED RESULTS")
    print("="*100)
    
    for query_type in ['spot', 'vol', 'greeks']:
        for run_num, results in enumerate(results_dict[query_type], 1):
            df = pd.DataFrame(results)
            filename = f"benchmark_results_{query_type}_run{run_num}.csv"
            df.to_csv(filename, index=False)
            print(f"Saved: {filename}")
    
    print("\n" + "="*100)
    print("BENCHMARK COMPLETE")
    print("="*100)


if __name__ == '__main__':
    main()
