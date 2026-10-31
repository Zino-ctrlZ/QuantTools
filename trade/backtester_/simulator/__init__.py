"""Portfolio simulator: run many strategies against one shared capital pool.

``Simulator`` sequences the day loop. ``Broker`` owns cash and fill execution;
``FillRouter`` (direct or book) gates batches; ``OrderManager`` holds delayed
orders; ``RunJournal`` points at daily, trade, skip, and mark logs;
``ResultBuilder`` / ``StatsCalculator`` assemble the result.

Core Classes:
    Simulator: The portfolio day loop.
    Broker: Shared-cash façade for marks and fills.
    DirectFillRouter, BookFillRouter: Fill policy chosen once at init.
    OrderManager: Delayed order lifecycle (``tplusn``).
    RunJournal: Directory of run logs (daily + views into broker books).
    DayRecorder: Daily portfolio snapshots (owned by RunJournal).
    ResultBuilder: Assembles ``SimulationResult``.
    StatsCalculator: Pure metric builders used by ResultBuilder.
    BuyingPowerPolicy: Skip or shrink opens that break capital limits.

Core Dataclasses:
    SimulationResult: Equity, trades, statistics, and cash series.
    DailySnapshot: Frozen end-of-day portfolio photo.
    MarkPoint: One daily mark on an open position's return path.
    SkipEvent: One capital-limit or book-veto skip.
    TradeRecord: One portfolio round-trip owned by the simulator.
    PathExtremes: MAE/MFE summary from a mark path.
    StrategyKey: Branded string for a portfolio child id.
    SimPosition: One open position.
    PendingOrder: A t+n scheduled open or close.
    OpenCandidate, PortfolioContext, Veto: What a book's hooks see and return.

Core Functions:
    eq_stats, trade_stats, compute_cash_stats, compute_exposure_stats,
    extended_stats: Result statistics builders.
    drawdown_stats, yearly_returns: Building blocks of extended_stats.
    combine_stats: Builds ``SimulationResult.full_stats``.

Usage:
    >>> from trade.backtester_.simulator import Simulator, BuyingPowerPolicy
    >>> result = Simulator(strategies, capital=1, percent_of_capital=0.25, start_date="2016-01-01").run()
"""

from trade.backtester_.simulator.broker import Account, Broker, ExecutionEngine, Portfolio
from trade.backtester_.simulator.core import (
    BarIndexFn,
    BuyingPowerPolicy,
    DailySnapshot,
    MarkPoint,
    OpenCandidate,
    PathExtremes,
    PendingOrder,
    BookHookError,
    PortfolioContext,
    SimPosition,
    SimulationResult,
    SkipEvent,
    StrategyKey,
    TradeRecord,
    Veto,
    extremes_from_mark_path,
    trade_records_to_legacy_events,
)
from trade.backtester_.simulator.engine import Simulator
from trade.backtester_.simulator.fill_router import (
    BookFillRouter,
    DirectFillRouter,
    FillRouter,
    make_fill_router,
)
from trade.backtester_.simulator.journal import RunJournal
from trade.backtester_.simulator.order_manager import OrderManager
from trade.backtester_.simulator.recorder import DayRecorder
from trade.backtester_.simulator.result_builder import ResultBuilder
from trade.backtester_.simulator.stats import (
    combine_stats,
    compute_cash_stats,
    compute_exposure_stats,
    drawdown_stats,
    eq_stats,
    extended_stats,
    trade_stats,
    yearly_returns,
)
from trade.backtester_.simulator.stats_calculator import StatsCalculator

__all__ = [
    "Account",
    "Broker",
    "ExecutionEngine",
    "Portfolio",
    "OrderManager",
    "RunJournal",
    "DailySnapshot",
    "DayRecorder",
    "ResultBuilder",
    "StatsCalculator",
    "BarIndexFn",
    "FillRouter",
    "DirectFillRouter",
    "BookFillRouter",
    "make_fill_router",
    "combine_stats",
    "drawdown_stats",
    "extended_stats",
    "yearly_returns",
    "Simulator",
    "SimulationResult",
    "BuyingPowerPolicy",
    "MarkPoint",
    "SkipEvent",
    "TradeRecord",
    "PathExtremes",
    "StrategyKey",
    "extremes_from_mark_path",
    "trade_records_to_legacy_events",
    "SimPosition",
    "PendingOrder",
    "OpenCandidate",
    "PortfolioContext",
    "Veto",
    "BookHookError",
    "eq_stats",
    "trade_stats",
    "compute_cash_stats",
    "compute_exposure_stats",
]
