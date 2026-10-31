"""Assemble ``SimulationResult`` from the daily log and trade books.

``ResultBuilder`` is the only place that talks to both ``DayRecorder`` and
``StatsCalculator``. It owns conversion of ``TradeRecord`` lists into
``trades_df``. It does not run the day loop or mutate cash.

Comment density: orchestration

Core Classes:
    ResultBuilder: Lean/full result assembly after ``Simulator.run``.

Processing Flow:
    1. Flatten ``TradeRecord`` lists into ``trades_df`` (simulator-owned converter).
    2. Turn the recorder into a daily frame; derive strategy PnL columns.
    3. Expand ``SkipEvent`` rows with ``asdict`` into ``skipped``.
    4. ``lean``: equity + extended stats only.
    5. ``full``: trade, cash, and optional exposure summaries via StatsCalculator.
"""

from __future__ import annotations

from dataclasses import asdict, fields
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

import pandas as pd

from trade.backtester_.simulator.core import SimulationResult, SkipEvent, StrategyKey, TradeRecord
from trade.backtester_.simulator.recorder import DayRecorder
from trade.backtester_.simulator.stats_calculator import StatsCalculator
from trade.helpers.Logging import setup_logger

if TYPE_CHECKING:
    from trade.backtester_._strategy import StrategyBase

logger = setup_logger(__name__)

## Scalar TradeRecord fields that become columns; entry_info / exit_info are flattened.
_TRADE_SCALAR_SKIP = frozenset({"entry_info", "exit_info"})


class ResultBuilder:
    """Build the object ``Simulator.run`` returns.

    Args:
        strategies: Strategy instances used for column labels.
        recorder: Daily snapshots from the run.
        trades: Per-key ``TradeRecord`` lists from the broker.
        skipped: Capital-limit and book-veto ``SkipEvent`` rows.
        stats_level: ``full`` or ``lean``.
        track_exposure: Whether exposure columns exist on the daily frame.
        risk_free_rate: Passed into extended stats.
        buy_and_hold_return: Optional baseline for extended stats.
    """

    def __init__(
        self,
        strategies: Dict[StrategyKey, StrategyBase],
        recorder: DayRecorder,
        trades: Dict[StrategyKey, List[TradeRecord]],
        skipped: List[SkipEvent],
        *,
        stats_level: Literal["full", "lean"] = "full",
        track_exposure: bool = True,
        risk_free_rate: float = 0.0,
        buy_and_hold_return: Optional[float] = None,
    ) -> None:
        """Store inputs; call ``build`` after the fill loop finishes."""
        self.strategies = strategies
        self.recorder = recorder
        self.trades = trades
        self.skipped = skipped
        self.stats_level = stats_level
        self.track_exposure = track_exposure
        self.risk_free_rate = risk_free_rate
        self.buy_and_hold_return = buy_and_hold_return
        self.stats = StatsCalculator()

    def _skipped_frame(self) -> pd.DataFrame:
        """Expand frozen skip events into the result table (``asdict`` once per row)."""
        if not self.skipped:
            return pd.DataFrame()
        return pd.DataFrame([asdict(event) for event in self.skipped])

    @staticmethod
    def _record_to_row(record: TradeRecord) -> Dict[str, Any]:
        """Flatten one ``TradeRecord`` into a trades_df row dict."""
        row: Dict[str, Any] = {
            f.name: getattr(record, f.name) for f in fields(record) if f.name not in _TRADE_SCALAR_SKIP
        }
        for key, value in record.entry_info.items():
            row[f"entry_{key}"] = value
        for key, value in record.exit_info.items():
            row[f"exit_{key}"] = value
        if record.entry_date is not None:
            row["year"] = pd.Timestamp(record.entry_date).year
            row["month"] = pd.Timestamp(record.entry_date).month
        return row

    def build_trades_df(self) -> pd.DataFrame:
        """One row per ``TradeRecord`` (open-only rows keep null exit fields)."""
        rows: List[Dict[str, Any]] = []
        for records in self.trades.values():
            for record in records:
                rows.append(self._record_to_row(record))
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)

    def _column_labels(self) -> Dict[str, str]:
        """Friendly names for result columns: ticker when unique, else the full key."""
        tickers = [s.ticker for s in self.strategies.values()]
        if len(set(tickers)) == len(tickers):
            return {k: s.ticker for k, s in self.strategies.items()}
        return {k: k for k in self.strategies}

    def build(self) -> SimulationResult:
        """Turn the daily log and trade list into a ``SimulationResult``."""
        daily = self.recorder.to_frame()
        labels = self._column_labels()
        keys = list(self.strategies)
        strategy_pnl = daily[[f"pnl::{k}" for k in keys]]
        strategy_pnl.columns = [labels[k] for k in keys]

        eq = daily["equity"]
        trades_df = self.build_trades_df()
        empty_series = pd.Series(dtype=float)
        empty_frame = pd.DataFrame()
        buy_hold = float(self.buy_and_hold_return) if self.buy_and_hold_return is not None else float("nan")

        if self.stats_level == "lean":
            ## Equity-only ranking path: skip trade_stats, cash_stats, and exposure.
            logger.debug("ResultBuilder lean path rows=%s trades=%s", len(daily), len(trades_df))
            equity_stats = self.stats.eq_stats(eq)
            trades_all = empty_series
            extended = self.stats.extended_stats(
                eq,
                empty_frame,
                risk_free_rate=self.risk_free_rate,
                buy_hold_return=buy_hold,
            )
            return SimulationResult(
                num_trades=len(trades_df),
                eq_start_date=eq.index[0],
                eq_end_date=eq.index[-1],
                eq=eq,
                trades_df=trades_df,
                equity_stats=equity_stats,
                equity_stats_by_ticker=None,
                trades_all=trades_all,
                trades_stats_by_ticker=empty_frame,
                cash=daily["cash"],
                cash_utilization=empty_series,
                gross_exposure=daily["gross_exposure"],
                net_exposure=empty_series,
                long_exposure=empty_series,
                short_exposure=empty_series,
                leverage=empty_series,
                n_open=empty_series,
                strategy_pnl=strategy_pnl,
                strategy_exposure=empty_frame,
                skipped=self._skipped_frame(),
                cash_stats=empty_series,
                exposure_stats=empty_series,
                exposure_stats_by_strategy=empty_frame,
                full_stats=self.stats.combine(equity_stats, trades_all, extended),
                trades=self.trades,
            )

        ## With finalize=False the last trades have no exit yet; stats only cover round trips.
        closed = trades_df[trades_df["exit_date"].notna()] if "exit_date" in trades_df else trades_df.iloc[0:0]
        trades_stats_by_ticker = (
            closed.groupby("ticker").apply(self.stats.trade_stats).T if not closed.empty else empty_frame
        )
        equity_stats = self.stats.eq_stats(eq)
        trades_all = self.stats.trade_stats(closed)
        extended = self.stats.extended_stats(
            eq,
            closed,
            risk_free_rate=self.risk_free_rate,
            buy_hold_return=buy_hold,
        )

        if self.track_exposure:
            strategy_exposure = daily[[f"exp::{k}" for k in keys]]
            strategy_exposure.columns = [labels[k] for k in keys]
            net_exposure = daily["net_exposure"]
            long_exposure = daily["long_exposure"]
            short_exposure = daily["short_exposure"]
            leverage = daily["leverage"]
            n_open = daily["n_open"]
            exposure_stats, exposure_stats_by_strategy = self.stats.exposure_stats(
                eq,
                daily["gross_exposure"],
                net_exposure,
                long_exposure,
                short_exposure,
                leverage,
                n_open,
                strategy_exposure,
            )
        else:
            strategy_exposure = empty_frame
            net_exposure = long_exposure = short_exposure = leverage = n_open = empty_series
            exposure_stats, exposure_stats_by_strategy = empty_series, empty_frame

        return SimulationResult(
            num_trades=len(trades_df),
            eq_start_date=eq.index[0],
            eq_end_date=eq.index[-1],
            eq=eq,
            trades_df=trades_df,
            equity_stats=equity_stats,
            equity_stats_by_ticker=None,
            trades_all=trades_all,
            trades_stats_by_ticker=trades_stats_by_ticker,
            cash=daily["cash"],
            ## Masked where equity <= 0 so wipeout days don't produce inf or sign-flipped ratios.
            cash_utilization=(1.0 - daily["cash"] / daily["equity"]).where(daily["equity"] > 0),
            gross_exposure=daily["gross_exposure"],
            net_exposure=net_exposure,
            long_exposure=long_exposure,
            short_exposure=short_exposure,
            leverage=leverage,
            n_open=n_open,
            strategy_pnl=strategy_pnl,
            strategy_exposure=strategy_exposure,
            skipped=self._skipped_frame(),
            cash_stats=self.stats.cash_stats(daily["cash"], daily["equity"], daily["gross_exposure"]),
            exposure_stats=exposure_stats,
            exposure_stats_by_strategy=exposure_stats_by_strategy,
            full_stats=self.stats.combine(equity_stats, trades_all, extended),
            trades=self.trades,
        )
