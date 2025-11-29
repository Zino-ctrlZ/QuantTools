"""
Benchmark for Portfolio.analyze_positions() and _create_ctx()
"""

import pandas as pd
import numpy as np
import time
import json
from pathlib import Path
from statistics import mean, stdev

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from EventDriven.data import HistoricTradeDataHandler
from EventDriven.new_portfolio import OptionSignalPortfolio
from EventDriven.riskmanager.new_base import RiskManager
from EventDriven.eventScheduler import EventScheduler
from EventDriven.types import OrderData


def setup_portfolio(trades_path: str, weights_path: str, num_positions: int = 10):
    """Quick portfolio setup"""
    # Load data
    with open(weights_path, "r") as f:
        weights = json.load(f)
    
    trades_df = pd.read_csv(trades_path).iloc[:, 1:]
    trades_df["Duration"] = trades_df.Duration.apply(lambda x: int(x.split(" ")[0]))
    trades = trades_df.iloc[-2:]
    
    # Setup components
    symbol_list = list(weights.keys())
    start_date = pd.to_datetime(trades.EntryTime).min()
    end_date = pd.to_datetime(trades.ExitTime).max()
    
    scheduler = EventScheduler(start_date, end_date)
    bars = HistoricTradeDataHandler(scheduler, trades, symbol_list, finalize_trades=False)
    
    risk_manager = RiskManager(
        symbol_list=bars.symbol_list,
        bkt_start=start_date,
        bkt_end=end_date,
        initial_capital=20000,
    )
    
    portfolio = OptionSignalPortfolio(
        bars, scheduler, risk_manager=risk_manager, initial_capital=20000.0
    )
    
    portfolio.config.weights_haircut = 0.0
    portfolio.weight_map = weights
    
    # Get current date from scheduler
    current_date = pd.to_datetime(scheduler.current_date)
    
    # Insert the sample position (using real trade data that exists in market data)
    # Adjusted date to match the position date
    position_date = pd.Timestamp('2024-01-11')
    portfolio.current_positions['META'] = {
        'META20240104LONG': {
            'position': OrderData(
                trade_id="&L:META20240920C450&S:META20240920C460",
                long=["META20240920C450"],
                short=["META20240920C460"],
                close=np.float64(3.0),
                quantity=12
            ),
            'entry_price': np.float64(2144.767092345258),
            'quantity': 10,
            'market_value': np.float64(2299.9999999999973),
            'signal_id': 'META20240104LONG'
        }
    }
    
    return portfolio, position_date
    
    return portfolio, current_date


def benchmark_create_ctx(portfolio, current_date, iterations=200):
    """Benchmark _create_ctx() method"""
    print(f"\n_create_ctx() - {iterations} iterations:")
    
    timings = []
    for i in range(iterations):
        start = time.perf_counter()
        ctx = portfolio._create_ctx(current_date)
        end = time.perf_counter()
        timings.append((end - start) * 1000)
        
        if i == 0:
            assert ctx is not None and len(ctx.portfolio.positions) > 0
    
    avg = mean(timings)
    std = stdev(timings) if len(timings) > 1 else 0
    cv = (std / avg * 100) if avg > 0 else 0
    
    print(f"  Avg: {avg:.4f} ms | Std: {std:.4f} ms | CV: {cv:.2f}%")
    print(f"  Range: {min(timings):.4f} - {max(timings):.4f} ms")
    
    return {"avg_ms": avg, "std_ms": std, "cv_pct": cv, "min_ms": min(timings), "max_ms": max(timings)}


def benchmark_analyze_positions(portfolio, position_date, iterations=200):
    """Benchmark full analyze_positions() call"""
    print(f"\nanalyze_positions() - {iterations} iterations:")
    
    # Set the scheduler date to match our position date
    original_date = portfolio.eventScheduler.current_date
    portfolio.eventScheduler.current_date = position_date
    
    timings = []
    for i in range(iterations):
        start = time.perf_counter()
        meta = portfolio.analyze_positions()
        end = time.perf_counter()
        timings.append((end - start) * 1000)
        
        if i == 0:
            assert meta is not None
    
    # Restore original date
    portfolio.eventScheduler.current_date = original_date
    
    avg = mean(timings)
    std = stdev(timings) if len(timings) > 1 else 0
    cv = (std / avg * 100) if avg > 0 else 0
    
    print(f"  Avg: {avg:.4f} ms | Std: {std:.4f} ms | CV: {cv:.2f}%")
    print(f"  Range: {min(timings):.4f} - {max(timings):.4f} ms")
    
    return {"avg_ms": avg, "std_ms": std, "cv_pct": cv, "min_ms": min(timings), "max_ms": max(timings)}


def main():
    """Run benchmark"""
    print("=" * 70)
    print("PORTFOLIO ANALYZE_POSITIONS BENCHMARK")
    print("=" * 70)
    
    base_path = Path(__file__).parent.parent.parent
    trades_path = base_path / "input/profitable_trades_11.csv"
    weights_path = base_path / "input/profitable_weights_11.json"
    
    # Setup with 10 positions
    portfolio, current_date = setup_portfolio(str(trades_path), str(weights_path), num_positions=10)
    
    positions_count = sum(len(pos) for pos in portfolio.current_positions.values())
    print(f"✓ Initialized with {positions_count} positions")
    
    # Run benchmark for _create_ctx (our optimization target)
    ctx_results = benchmark_create_ctx(portfolio, current_date, iterations=200)
    
    # Note: analyze_positions() requires full position limits setup
    # Since we're optimizing up to risk_manager.analyze_position(), _create_ctx is our focus
    
    # Save results
    results = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "positions_count": positions_count,
        "create_ctx": ctx_results,
        "note": "Baseline for _create_ctx optimization. Focus: market data lookups and PositionState creation"
    }
    
    output_path = Path(__file__).parent / "baseline_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {output_path}")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    main()
