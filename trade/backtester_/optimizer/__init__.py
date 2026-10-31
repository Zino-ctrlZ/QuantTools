"""Grid-search optimizer for strategy ``bt_params`` on the shared-capital simulator.

Tests every settings mix of one strategy on a book of tickers, in forked worker
processes, and picks the best mix for the whole book. Per-asset tuning is a loop
that calls it once per ticker (``optimize_per_ticker``, also used by
``MultiAssetStrategy.optimize`` and ``SystematicMultiAsset.optimize(per_asset=True)``).

Core Classes:
    ParamOptimizer: Runs the grid and picks the winner.

Core Dataclasses:
    OptimizationResult: Trials table, best settings, and full re-run results.

Core Functions:
    optimize_per_ticker: One optimizer run per ticker with solo sizing defaults.

Usage:
    >>> from trade.backtester_.optimizer import ParamOptimizer, optimize_per_ticker
    >>> opt = ParamOptimizer(MyStrategy, data=d, start_trading_date="2016-01-01",
    ...                      space={"ma_length": [10, 20, 30]},
    ...                      simulator_kwargs={"capital": 100_000, "percent_of_capital": 0.25})
"""

from trade.backtester_.optimizer.core import OptimizationResult
from trade.backtester_.optimizer.engine import (
    SOLO_SIMULATOR_KWARGS,
    ParamOptimizer,
    optimize_per_ticker,
)

__all__ = ["OptimizationResult", "ParamOptimizer", "SOLO_SIMULATOR_KWARGS", "optimize_per_ticker"]
