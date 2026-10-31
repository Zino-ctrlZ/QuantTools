"""Namespace over the pure statistic builders in ``stats.py``.

``StatsCalculator`` does not hold a ``DayRecorder``. ``ResultBuilder`` pulls
frames from the recorder (and trade tables) and passes series into these
methods. Keeping this layer thin preserves the existing ``eq_stats`` /
``trade_stats`` parity with single-strategy legs.

Core Classes:
    StatsCalculator: Thin façade over module-level stats functions.
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd

from trade.backtester_.simulator import stats as stats_mod
from trade.helpers.Logging import setup_logger

logger = setup_logger(__name__)


class StatsCalculator:
    """Call the pure metric builders used when assembling a SimulationResult."""

    @staticmethod
    def eq_stats(eq: pd.Series) -> pd.Series:
        """Equity-curve return, risk, and drawdown statistics."""
        return stats_mod.eq_stats(eq)

    @staticmethod
    def trade_stats(trades: pd.DataFrame) -> pd.Series:
        """Round-trip trade statistics for one trades table."""
        return stats_mod.trade_stats(trades)

    @staticmethod
    def cash_stats(cash: pd.Series, equity: pd.Series, gross_exposure: pd.Series) -> pd.Series:
        """Idle capital and funding-pressure summary."""
        return stats_mod.compute_cash_stats(cash, equity, gross_exposure)

    @staticmethod
    def exposure_stats(
        equity: pd.Series,
        gross_exposure: pd.Series,
        net_exposure: pd.Series,
        long_exposure: pd.Series,
        short_exposure: pd.Series,
        leverage: pd.Series,
        n_open: pd.Series,
        strategy_exposure: pd.DataFrame,
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """Portfolio and per-strategy exposure summaries."""
        return stats_mod.compute_exposure_stats(
            equity,
            gross_exposure,
            net_exposure,
            long_exposure,
            short_exposure,
            leverage,
            n_open,
            strategy_exposure,
        )

    @staticmethod
    def extended_stats(
        eq: pd.Series,
        trades: pd.DataFrame,
        *,
        risk_free_rate: float = 0.0,
        buy_hold_return: float = float("nan"),
    ) -> pd.Series:
        """Aggregator-style extras (Sharpe with RF, streaks, …)."""
        return stats_mod.extended_stats(
            eq,
            trades,
            risk_free_rate=risk_free_rate,
            buy_hold_return=buy_hold_return,
        )

    @staticmethod
    def combine(equity_stats: pd.Series, trades_all: pd.Series, extended: pd.Series) -> pd.Series:
        """One flat series for optimizer scoring."""
        return stats_mod.combine_stats(equity_stats, trades_all, extended)
