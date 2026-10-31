"""Define the shared types for the portfolio simulator.

These are the records ``Simulator`` passes between its loop steps: the
buying-power policy enum, open positions, t+n pending orders, and the final
result container whose analytics fields mirror ``StrategyLegResult``.

It also holds the records a ``PortfolioBook`` sees and returns at fill time.
They live here rather than in the book module because the engine needs them
at runtime and the book module imports the engine.

Core Classes:
    BuyingPowerPolicy: Whether a capped open is skipped or shrunk to fit.
    BookHookError: A book hook returned the wrong type.

Core Dataclasses:
    MarkPoint: One daily mark on an open position's return path (frozen).
    DailySnapshot: One end-of-day portfolio photo (frozen).
    SkipEvent: One capital-limit or book-veto skip (frozen).
    PathExtremes: MAE/MFE summary derived from a mark path (frozen).
    TradeRecord: One portfolio round-trip owned by the simulator (frozen).
    SimPosition: One open position owned by one strategy.
    PendingOrder: An open or close scheduled for a later bar.
    SimulationResult: Equity, trades, statistics, and cash series from one run.
    OpenCandidate: A child's open waiting for book approval.
    PortfolioContext: Read-only view of portfolio state handed to book hooks.
    Veto: A book hook's refusal, recorded in ``SimulationResult.skipped``.

Core Types:
    StrategyKey: Branded string for a portfolio child id (usually ``ClassName:TICKER``).
    BarIndexFn: ``(strategy_key, date) -> bar index | None`` shared by broker and fill routers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd

from EventDriven.types import PositionEffect, SignalID

## _strategy imports this package at module load, so these are annotation-only.
if TYPE_CHECKING:
    from trade.backtester_._strategy import StrategyBase, TradeDecision


class StrategyKey(str):
    """Portfolio child id — usually ``f\"{ClassName}:{ticker}\"``.

    A thin ``str`` subclass so maps and records can advertise that a value is a
    strategy key, not a bare ticker or free-form label. No parsing: books may
    use other naming schemes. ``__slots__`` keeps instances attribute-free.
    """

    __slots__ = ()


## Lookup: strategy key + date -> bar index, or None when that strategy has no bar.
BarIndexFn = Callable[[StrategyKey, pd.Timestamp], Optional[int]]


class BuyingPowerPolicy(str, Enum):
    """What to do when an open would break the gross leverage cap or the minimum cash floor."""

    SKIP = "skip"
    SCALE_DOWN = "scale_down"


@dataclass(frozen=True, slots=True)
class MarkPoint:
    """One daily mark on an open position's unlevered return path.

    Appended by ``Portfolio.mark`` (and seeded at open). Immutable so history
    points are not rewritten after the fact. ``__getitem__`` supports
    ``p["cum_return_pct"]``-style reads for callers that still index by key.

    Attributes:
        date: Mark date.
        index: Strategy bar index for that mark (MAE/MFE day counts).
        cum_return_pct: Unlevered cumulative return since entry times side.
    """

    date: pd.Timestamp
    index: int
    cum_return_pct: float

    def __getitem__(self, key: str) -> Union[pd.Timestamp, int, float]:
        """Allow ``point["date"]``-style reads."""
        try:
            return getattr(self, key)
        except AttributeError as exc:
            raise KeyError(key) from exc


@dataclass(frozen=True, slots=True)
class DailySnapshot:
    """One end-of-day portfolio photo.

    Built by ``DayRecorder.snapshot`` from live broker values. Exposure fields
    are ``None`` when tracking is off (lean / ``track_exposure=False``). Nested
    ``strategy_pnl`` / ``strategy_exposure`` dicts are copied at construction;
    the outer record is frozen so the day loop replaces rows instead of mutating
    them.

    Attributes:
        date: Trading day for this snapshot.
        equity: Cash plus open position market value.
        cash: Ledger cash (may be negative).
        gross_exposure: Sum of absolute position sizes.
        strategy_pnl: Running realized plus unrealized dollars by strategy key.
        net_exposure: Signed net position size, or None if not tracking.
        long_exposure: Long-only absolute size, or None if not tracking.
        short_exposure: Short-only absolute size, or None if not tracking.
        n_open: Number of open positions, or None if not tracking.
        leverage: Gross over equity when equity is positive, else NaN; None if not tracking.
        strategy_exposure: Signed market value by strategy key, or None if not tracking.
    """

    date: pd.Timestamp
    equity: float
    cash: float
    gross_exposure: float
    strategy_pnl: Dict[str, float]
    net_exposure: Optional[float] = None
    long_exposure: Optional[float] = None
    short_exposure: Optional[float] = None
    n_open: Optional[int] = None
    leverage: Optional[float] = None
    strategy_exposure: Optional[Dict[str, float]] = None


@dataclass(frozen=True, slots=True)
class SkipEvent:
    """One open that was blocked or scaled by capital limits or the book.

    Appended via ``Account.record_skip``. Capital skips and book vetoes share
    this schema so ``ResultBuilder`` can ``asdict`` the list into one
    ``SimulationResult.skipped`` frame.

    Attributes:
        date: Decision date.
        strategy_key: Strategy that tried to open.
        requested_notional: Size asked before the cap (``nan`` for book vetoes).
        filled_notional: Size allowed after the policy (0 when skipped entirely).
        available_cash: Ledger cash at the decision.
        reason: Human-readable limit or ``book: ...`` veto text.
    """

    date: pd.Timestamp
    strategy_key: StrategyKey
    requested_notional: float
    filled_notional: float
    available_cash: float
    reason: str


@dataclass(frozen=True, slots=True)
class PathExtremes:
    """MAE/MFE and path series derived from a position's ``MarkPoint`` list.

    Attributes:
        mae: Worst cumulative unlevered return on the path.
        mfe: Best cumulative unlevered return on the path.
        days_to_mae: Bars from entry index to the MAE mark.
        days_to_mfe: Bars from entry index to the MFE mark.
        mae_date: Date of the MAE mark.
        mfe_date: Date of the MFE mark.
        trade_returns: Per-mark cumulative returns (tuple for immutability).
        trade_return_dates: Per-mark dates aligned with ``trade_returns``.
    """

    mae: Optional[float]
    mfe: Optional[float]
    days_to_mae: Optional[int]
    days_to_mfe: Optional[int]
    mae_date: Optional[pd.Timestamp]
    mfe_date: Optional[pd.Timestamp]
    trade_returns: Optional[Tuple[float, ...]]
    trade_return_dates: Optional[Tuple[pd.Timestamp, ...]]


def extremes_from_mark_path(return_path: Sequence[MarkPoint], entry_index: int) -> PathExtremes:
    """Compute MAE/MFE stats from a mark path for a closing ``TradeRecord``.

    Args:
        return_path: Daily marks since entry (may be empty).
        entry_index: Strategy bar index at open (for day counts).

    Returns:
        Frozen extremes; all fields ``None`` when ``return_path`` is empty.
    """
    if not return_path:
        return PathExtremes(None, None, None, None, None, None, None, None)
    returns = tuple(float(point.cum_return_pct) for point in return_path)
    dates = tuple(pd.Timestamp(point.date) for point in return_path)
    mae_idx = min(range(len(returns)), key=lambda i: returns[i])
    mfe_idx = max(range(len(returns)), key=lambda i: returns[i])
    return PathExtremes(
        mae=float(returns[mae_idx]),
        mfe=float(returns[mfe_idx]),
        days_to_mae=int(return_path[mae_idx].index - entry_index),
        days_to_mfe=int(return_path[mfe_idx].index - entry_index),
        mae_date=dates[mae_idx],
        mfe_date=dates[mfe_idx],
        trade_returns=returns,
        trade_return_dates=dates,
    )


@dataclass(frozen=True, slots=True)
class TradeRecord:
    """One portfolio round-trip (or still-open trade) owned by the simulator.

    ``ExecutionEngine`` appends on open and replaces the last record on close.
    Strategy ``additional_on_entry_info`` / ``additional_on_exit_info`` land in
    ``entry_info`` / ``exit_info``; StrategyBase does not write this log.

    Attributes:
        strategy_key: Portfolio key for the child strategy.
        ticker: Traded ticker.
        side: +1 long, -1 short.
        signal_id: Signal that opened the trade.
        entry_date: Fill date at open.
        entry_price: Fill price at open.
        entry_index: Strategy bar index at open.
        entry_equity: Portfolio equity / initial capital just after the open.
        units: Position size in units.
        notional: Dollar size at open.
        multiplier: Signal leverage multiplier.
        entry_commission: Commission cash charged at open (percent of notional).
        entry_slippage: Slippage cash charged at open (percent of notional).
        entry_info: Strategy extras from ``additional_on_entry_info``.
        exit_date: Fill date at close, or None while open.
        exit_price: Exit mark price, or None while open.
        exit_index: Strategy bar index at close, or None while open.
        exit_equity: Portfolio equity / initial capital just after the close.
        return_pct: Unlevered raw entry-to-exit price return times side (no fees).
        pnl: Dollar profit and loss after entry and exit fees.
        levered_return_pct: Fee-adjusted dollar PnL over entry notional times multiplier.
        close_reason: Why the close happened (signal, finalize, wipeout).
        mae: Worst raw mark on the path (price only).
        mfe: Best raw mark on the path (price only).
        days_to_mae: Bars from entry to MAE.
        days_to_mfe: Bars from entry to MFE.
        mae_date: Date of MAE.
        mfe_date: Date of MFE.
        trade_returns: Cumulative mark returns along the path.
        trade_return_dates: Dates aligned with ``trade_returns``.
        exit_commission: Commission cash charged at close, or None while open.
        exit_slippage: Slippage cash charged at close, or None while open.
        exit_info: Strategy extras from ``additional_on_exit_info``.
    """

    strategy_key: StrategyKey
    ticker: str
    side: int
    signal_id: Optional[SignalID]
    entry_date: pd.Timestamp
    entry_price: float
    entry_index: int
    entry_equity: float
    units: float
    notional: float
    multiplier: float
    entry_commission: float = 0.0
    entry_slippage: float = 0.0
    entry_info: Dict[str, Any] = field(default_factory=dict)
    exit_date: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_index: Optional[int] = None
    exit_equity: Optional[float] = None
    return_pct: Optional[float] = None
    pnl: Optional[float] = None
    levered_return_pct: Optional[float] = None
    close_reason: Optional[str] = None
    mae: Optional[float] = None
    mfe: Optional[float] = None
    days_to_mae: Optional[int] = None
    days_to_mfe: Optional[int] = None
    mae_date: Optional[pd.Timestamp] = None
    mfe_date: Optional[pd.Timestamp] = None
    trade_returns: Optional[Tuple[float, ...]] = None
    trade_return_dates: Optional[Tuple[pd.Timestamp, ...]] = None
    exit_commission: Optional[float] = None
    exit_slippage: Optional[float] = None
    exit_info: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_closed(self) -> bool:
        """True when exit fields have been filled."""
        return self.exit_date is not None

    def to_legacy_events(self) -> List[Dict[str, Any]]:
        """Emit the old open/close event-dict pair shape for ``StrategyBase.simulate``.

        Open always; close only when ``is_closed``. ``position_info`` is omitted
        (``None``) because the simulator no longer snapshots it on the trade book.
        """
        open_event: Dict[str, Any] = {
            "date": self.entry_date,
            "action": "open",
            "price": self.entry_price,
            "equity": self.entry_equity,
            "signal_id": self.signal_id,
            "commission": self.entry_commission,
            "slippage": self.entry_slippage,
            **self.entry_info,
        }
        events = [open_event]
        if not self.is_closed:
            return events
        events.append(
            {
                "date": self.exit_date,
                "action": "close",
                "price": self.exit_price,
                "equity": self.exit_equity,
                "return_pct": self.return_pct,
                "entry_price": self.entry_price,
                "side": self.side,
                "position_info": None,
                "signal_id": self.signal_id,
                "mae": self.mae,
                "mfe": self.mfe,
                "days_to_mae": self.days_to_mae,
                "days_to_mfe": self.days_to_mfe,
                "mae_date": self.mae_date,
                "mfe_date": self.mfe_date,
                "trade_returns": list(self.trade_returns) if self.trade_returns is not None else None,
                "trade_return_dates": (
                    list(self.trade_return_dates) if self.trade_return_dates is not None else None
                ),
                "commission": self.exit_commission,
                "slippage": self.exit_slippage,
                **self.exit_info,
            }
        )
        return events


def trade_records_to_legacy_events(records: Sequence[TradeRecord]) -> List[Dict[str, Any]]:
    """Flatten ``TradeRecord`` lists into the legacy open/close event-dict log.

    Args:
        records: Simulator-owned round-trips for one strategy.

    Returns:
        Chronological open/close dicts matching the pre-``TradeRecord`` ``simulate`` shape.
    """
    events: List[Dict[str, Any]] = []
    for record in records:
        events.extend(record.to_legacy_events())
    return events


@dataclass
class SimPosition:
    """One open position owned by one strategy.

    Attributes:
        strategy_key: Unique key of the owning strategy.
        ticker: Ticker traded by the strategy.
        side: +1 for long, -1 for short.
        units: Number of units held (always positive).
        entry_price: Fill price at open.
        entry_date: Fill date at open.
        signal_id: Signal that produced the open.
        multiplier: Leverage multiplier applied at open.
        last_price: Most recent mark price.
        entry_index: Strategy bar index at open (for days_to_mae / days_to_mfe).
        last_index: Strategy bar index of the most recent mark.
        return_path: Per-bar ``MarkPoint`` marks since entry (MAE/MFE at close).
    """

    strategy_key: StrategyKey
    ticker: str
    side: int
    units: float
    entry_price: float
    entry_date: pd.Timestamp
    signal_id: Optional[SignalID]
    multiplier: float
    last_price: float
    entry_index: int
    last_index: int
    return_path: List[MarkPoint] = field(default_factory=list)

    @property
    def market_value(self) -> float:
        """Return the signed market value (negative for shorts)."""
        return self.side * self.units * self.last_price

    @property
    def gross_value(self) -> float:
        """Return the absolute market value used for exposure limits."""
        return self.units * self.last_price

    @property
    def unrealized_pnl(self) -> float:
        """Return profit and loss since entry at the last mark."""
        return self.side * self.units * (self.last_price - self.entry_price)


@dataclass
class PendingOrder:
    """An open or close scheduled for a later bar because of a strategy's t+n delay.

    Attributes:
        strategy_key: Strategy that produced the decision.
        pos_effect: OPEN or CLOSE.
        signal_date: Date the decision was made.
        side: Side from the decision (opens only).
        signal_id: Signal from the decision.
        multiplier: Multiplier captured at signal time (opens only).
    """

    strategy_key: StrategyKey
    pos_effect: PositionEffect
    signal_date: pd.Timestamp
    side: int = 0
    signal_id: Optional[SignalID] = None
    multiplier: float = 1.0


@dataclass(frozen=True)
class SimulationResult:
    """Everything ``Simulator.run`` hands back after a shared-cash backtest.

    Equity, trades, and the main stats use the same names as a single-strategy
    leg result so notebooks can compare them. Cash, exposure, and per-strategy
    P&L are portfolio-only. If you turned ``track_exposure`` off, the exposure
    curves and summaries below are empty, but equity and trades are still filled.

    Attributes:
        num_trades: How many round trips completed.
        eq_start_date: First day on the equity curve.
        eq_end_date: Last day on the equity curve.
        eq: Portfolio worth each day, in dollars.
        trades_df: One row per completed trade (entry, exit, return, size, and why it closed).
        equity_stats: Performance numbers on the equity curve (return, drawdown, Sharpe, …).
        equity_stats_by_ticker: Not filled yet (always None for shared-cash runs).
        trades_all: Win rate, payoff, and similar stats across every trade.
        trades_stats_by_ticker: Those same trade stats, split by ticker.
        cash: Cash held each day (negative means borrowed).
        cash_utilization: Share of equity that is not sitting in cash. Above 1 means
            borrowing; below 0 can happen when short proceeds swell cash.
        gross_exposure: Total dollars in positions each day (longs and shorts both count).
        net_exposure: Long dollars minus short dollars each day (empty if not tracking).
        long_exposure: Dollars in longs only (empty if not tracking).
        short_exposure: Dollars in shorts only (empty if not tracking).
        leverage: Total position size divided by equity that day. Missing when equity
            is zero or below, or when tracking is off.
        n_open: How many positions are open each day (empty if not tracking).
        strategy_pnl: Running profit and loss for each strategy, in dollars.
        strategy_exposure: How much each strategy holds each day (long positive, short
            negative). Empty if not tracking.
        skipped: Opens that were blocked or shrunk by cash, leverage, or the book.
            Columns include ``date``, ``strategy_key``, ``requested_notional``,
            ``filled_notional``, ``available_cash`` (cash at the moment of the skip),
            and ``reason``.
        cash_stats: Short summary of how idle or tight cash was over the run.
        exposure_stats: Short summary of portfolio leverage and how often the book was flat.
        exposure_stats_by_strategy: For each strategy/sleeve (for example ``mom`` and
            ``mr``), average size, share of the book, and how often it was held.
            Child keys such as ``mom:AAPL`` are rolled up into the sleeve name.
            Daily ``strategy_exposure`` still keeps the per-child columns.
        full_stats: One flat series of equity, trade, and extra stats for easy scoring.
            Trade fields that clash with equity names get a ``trade_`` prefix.
        trades: Per-strategy list of ``TradeRecord`` round-trips (simulator-owned).
    """

    num_trades: int
    eq_start_date: pd.Timestamp
    eq_end_date: pd.Timestamp
    eq: pd.Series
    trades_df: pd.DataFrame
    equity_stats: pd.Series
    equity_stats_by_ticker: Optional[pd.DataFrame]
    trades_all: pd.Series
    trades_stats_by_ticker: pd.DataFrame
    cash: pd.Series
    cash_utilization: pd.Series
    gross_exposure: pd.Series
    net_exposure: pd.Series
    long_exposure: pd.Series
    short_exposure: pd.Series
    leverage: pd.Series
    n_open: pd.Series
    strategy_pnl: pd.DataFrame
    strategy_exposure: pd.DataFrame
    skipped: pd.DataFrame
    cash_stats: pd.Series
    exposure_stats: pd.Series
    exposure_stats_by_strategy: pd.DataFrame
    full_stats: pd.Series
    trades: Dict[StrategyKey, List[TradeRecord]] = field(default_factory=dict)

    @property
    def equity(self) -> pd.Series:
        """Portfolio worth each day (same series as ``eq``)."""
        return self.eq

    def __repr__(self) -> str:
        """One-line peek: strategy count, trades, CAGR, and max drawdown."""
        cagr = self.equity_stats.get("cagr", float("nan"))
        max_drawdown = self.equity_stats.get("max_drawdown", float("nan"))
        n_strategies = len(self.strategy_pnl.columns)
        return (
            f"SimulationResult({n_strategies} strategies, {self.num_trades} trades, "
            f"{len(self.skipped)} capped, cagr={cagr:.1%}, max_dd={max_drawdown:.1%})"
        )


@dataclass(frozen=True)
class OpenCandidate:
    """A child's open waiting for book approval on its fill date.

    Attributes:
        key: Strategy key as the book assigned it; its format belongs to the book.
        ticker: Ticker the child trades.
        decision: The child's open decision (side, signal id, multiplier).
        strategy: The child, so hooks can read its indicators.
        index: The child's bar index on the fill date.
    """

    key: StrategyKey
    ticker: str
    decision: TradeDecision
    strategy: StrategyBase
    index: int


@dataclass(frozen=True)
class PortfolioContext:
    """Read-only view of simulator state handed to book hooks.

    Plain values only, so a live loader could build one from broker positions.

    Attributes:
        date: Bar being processed.
        equity: Cash plus signed market value.
        cash: Current cash.
        gross_exposure: Sum of absolute position values.
        open_keys: Keys holding a position right now (pending t+n opens excluded).
        open_by_ticker: Ticker to the keys holding it.
    """

    date: pd.Timestamp
    equity: float
    cash: float
    gross_exposure: float
    open_keys: Tuple[StrategyKey, ...]
    open_by_ticker: Mapping[str, Tuple[StrategyKey, ...]]

    @property
    def n_open(self) -> int:
        """Return how many positions are open."""
        return len(self.open_keys)


@dataclass(frozen=True)
class Veto:
    """A book hook's refusal.

    Attributes:
        reason: Written as ``"category (details)"`` so ``skipped["reason"]`` groups by category.
    """

    reason: str


class BookHookError(RuntimeError):
    """A book hook returned the wrong type or a decision for the wrong effect."""

    def __init__(
        self, book: str, hook: str, *, date: Any, key: Optional[StrategyKey], problem: str
    ) -> None:
        """Name the hook, date, and key so the failing bar can be replayed.

        Args:
            book: Book class name.
            hook: Hook name, e.g. ``"should_open"``.
            date: Bar being processed.
            key: Strategy key the hook was deciding on, if any.
            problem: What went wrong.
        """
        self.hook, self.date, self.key = hook, date, key
        target = f" for {key!r}" if key else ""
        super().__init__(f"{book}.{hook}{target} on {date:%Y-%m-%d}: {problem}")
