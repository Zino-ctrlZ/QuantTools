"""Define the shared types the parameter optimizer passes around.

Like ``backtesting.py``, each settings mix is just a ``{name: value}`` dict, and
results are filed in a table indexed by the mix's values.

Core Dataclasses:
    OptimizationResult: Trials table, winning settings, and re-run results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from trade.backtester_.simulator import SimulationResult

## ticker -> {parameter name: value}: which settings each ticker runs with in one simulation.
TickerParams = Dict[str, Dict[str, Any]]


@dataclass(frozen=True)
class OptimizationResult:
    """Everything one optimizer run produced.

    Attributes:
        trials: One row per tested mix, indexed by the mix's values (one index level
            per setting name), like ``backtesting.py``'s heatmap. Columns are ``score``,
            ``feasible``, ``error``, then every ``full_stats`` column.
        best_params: The winning mix, e.g. ``{"ma_length": 10}``, applied to every
            ticker in the book. Empty when nothing passed.
        best_score: Objective value of the winning mix (NaN when there is none).
        best_result: Full result for ``best_params``, re-run in the parent.
        top_results: Row label (as in ``trials.index``) to full result for the best
            few passing mixes.
    """

    trials: pd.DataFrame
    best_params: Dict[str, Any]
    best_score: float
    best_result: Optional[SimulationResult]
    top_results: Dict[Tuple[Any, ...], SimulationResult] = field(default_factory=dict)

    def __repr__(self) -> str:
        """Summarize the run in one line."""
        n_ok = int(self.trials["feasible"].sum()) if "feasible" in self.trials else 0
        return f"OptimizationResult({len(self.trials)} trials, {n_ok} feasible, best_score={self.best_score:.4f})"
