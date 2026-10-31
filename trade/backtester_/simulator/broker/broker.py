"""Façade over Account, Portfolio, and ExecutionEngine for the day loop.

``Broker`` is what ``Simulator`` talks to for money and positions. Capital-limit
skips live on the account; book vetoes append to the same ``skipped`` list via
``skip_by_book`` so ``SimulationResult.skipped`` stays one table.

Comment density: orchestration

Core Classes:
    Broker: Shared-cash trading façade used by the simulator day loop.

Processing Flow:
    1. ``reset`` rebuilds trade books and resets account + portfolio.
    2. ``mark_to_market`` updates every open position that has a bar today.
    3. ``fill_open`` / ``fill_close`` / ``close_all`` delegate to ExecutionEngine.
    4. ``context`` builds the read-only snapshot book hooks need.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
import pandas as pd

from trade.backtester_.simulator.broker.account import Account
from trade.backtester_.simulator.broker.execution import ExecutionEngine
from trade.backtester_.simulator.broker.portfolio import Portfolio
from trade.backtester_.simulator.core import (
    BarIndexFn,
    BuyingPowerPolicy,
    PendingOrder,
    PortfolioContext,
    SimPosition,
    SkipEvent,
    StrategyKey,
    TradeRecord,
)
from trade.helpers.Logging import setup_logger

if TYPE_CHECKING:
    from trade.backtester_._strategy import StrategyBase

logger = setup_logger(__name__)


class Broker:
    """Shared-cash account that sizes and fills opens and closes.

    Args:
        strategies: Strategy instances keyed for the run.
        capital: Starting cash.
        trade_size: Fixed dollar size mode, or None.
        percent_of_capital: Percent-of-equity size mode, or None.
        max_gross_leverage: Gross leverage cap.
        min_cash: Cash floor after longs.
        buying_power_policy: Skip or scale when a limit binds.
        commission_pct: Fraction of notional charged as commission per fill.
        slippage_pct: Fraction of notional charged as slippage cost per fill.
        bar_index: ``(key, date) -> bar index | None`` for marks and fills.
    """

    def __init__(
        self,
        strategies: Dict[StrategyKey, StrategyBase],
        capital: float,
        *,
        trade_size: Optional[float] = None,
        percent_of_capital: Optional[float] = None,
        max_gross_leverage: float = np.inf,
        min_cash: Optional[float] = 0.0,
        buying_power_policy: BuyingPowerPolicy = BuyingPowerPolicy.SKIP,
        commission_pct: float = 0.0,
        slippage_pct: float = 0.0,
        bar_index: BarIndexFn,
    ) -> None:
        """Build account, portfolio, and execution engine for this strategy set."""
        self.strategies: Dict[StrategyKey, StrategyBase] = {
            StrategyKey(k): s for k, s in strategies.items()
        }
        self._bar_index = bar_index
        self.account = Account(
            capital,
            trade_size=trade_size,
            percent_of_capital=percent_of_capital,
            max_gross_leverage=max_gross_leverage,
            min_cash=min_cash,
            buying_power_policy=buying_power_policy,
            commission_pct=commission_pct,
            slippage_pct=slippage_pct,
        )
        self.portfolio = Portfolio(self.strategies.keys())
        self.trades: Dict[StrategyKey, List[TradeRecord]] = {k: [] for k in self.strategies}
        self.execution = ExecutionEngine(
            self.account,
            self.portfolio,
            self.strategies,
            self.trades,
            bar_index,
        )
        logger.info(
            "Broker init strategies=%s capital=%s",
            len(strategies),
            self.account.initial_capital,
        )

    def reset(self) -> None:
        """Clear cash, positions, trade logs, and capital-limit skips for a new run."""
        self.account.reset()
        self.portfolio.reset()
        self.trades = {k: [] for k in self.strategies}
        ## ExecutionEngine holds references; refresh them after trade-book rebuild.
        self.execution.trades = self.trades
        self.execution.strategies = self.strategies

    @property
    def cash(self) -> float:
        """Current cash balance."""
        return self.account.cash

    @property
    def positions(self) -> Dict[StrategyKey, SimPosition]:
        """Open positions by strategy key."""
        return self.portfolio.positions

    @property
    def realized_pnl(self) -> Dict[StrategyKey, float]:
        """Cumulative realized profit and loss by strategy key."""
        return self.portfolio.realized_pnl

    @property
    def skipped(self) -> List[SkipEvent]:
        """Capital-limit and book-veto skip rows (shared list)."""
        return self.account.skipped

    @property
    def initial_capital(self) -> float:
        """Starting cash for the pool."""
        return self.account.initial_capital

    @property
    def equity(self) -> float:
        """Cash plus open position market value."""
        return self.execution.equity()

    @property
    def gross_exposure(self) -> float:
        """Total absolute position size."""
        return self.portfolio.gross_exposure

    @property
    def net_exposure(self) -> float:
        """Signed net position size."""
        return self.portfolio.net_exposure

    @property
    def long_exposure(self) -> float:
        """Long-only absolute size."""
        return self.portfolio.long_exposure

    @property
    def short_exposure(self) -> float:
        """Short-only absolute size."""
        return self.portfolio.short_exposure

    def mark_to_market(self, ts: pd.Timestamp) -> None:
        """Refresh each open position that has a bar on ``ts``."""
        for key in list(self.portfolio.positions):
            i = self._bar_index(key, ts)
            if i is None:
                continue
            price = float(self.strategies[key]._close[i])
            self.portfolio.mark(key, ts, price=price, index=i)

    def fill_open(self, order: PendingOrder, ts: pd.Timestamp) -> None:
        """Open a position through the execution engine."""
        self.execution.fill_open(order, ts)

    def fill_close(self, key: StrategyKey, ts: pd.Timestamp, *, reason: str) -> None:
        """Close a position through the execution engine."""
        self.execution.fill_close(key, ts, reason=reason)

    def close_all(self, ts: pd.Timestamp, *, reason: str) -> None:
        """Flatten every open position."""
        self.execution.close_all(ts, reason=reason)

    def context(self, ts: pd.Timestamp) -> PortfolioContext:
        """Read-only portfolio snapshot for book hooks."""
        by_ticker: Dict[str, List[StrategyKey]] = {}
        for key, pos in self.portfolio.positions.items():
            by_ticker.setdefault(pos.ticker, []).append(key)
        return PortfolioContext(
            date=ts,
            equity=self.equity,
            cash=self.cash,
            gross_exposure=self.gross_exposure,
            open_keys=tuple(self.portfolio.positions),
            open_by_ticker={t: tuple(k) for t, k in by_ticker.items()},
        )

    def skip_by_book(self, ts: pd.Timestamp, key: StrategyKey, reason: str) -> None:
        """Append a book rejection next to capital-limit skips."""
        self.account.record_skip(
            ts,
            StrategyKey(key),
            requested_notional=np.nan,
            filled_notional=0.0,
            reason=f"book: {reason}",
        )
