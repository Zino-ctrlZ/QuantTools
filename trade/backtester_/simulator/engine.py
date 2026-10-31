"""Shared-cash backtester day loop for one or many strategies.

Each strategy says when to open or close. ``Broker`` owns cash, positions, and
fill execution; ``FillRouter`` chooses direct vs book-gated batches;
``OrderManager`` holds delayed orders; ``RunJournal`` (daily + views into
trades/skips/marks) and ``ResultBuilder`` build the result. This module
sequences the calendar and strategy decisions.

Comment density: orchestration

Core Classes:
    Simulator: Runs the day loop over every strategy's dates.

Processing Flow:
    1. Build the trading calendar from every strategy, trimmed to the start date.
    2. Seed one prior flat business day (cash = capital, no positions) so the
       first equity point matches initial capital even when day-1 opens fill.
    3. Each day, mark open positions to that day's close.
    4. If equity hits the wipeout floor, close everything and stop.
    5. Fill delayed orders due today via the fill router (closes before opens).
    6. Ask each strategy for new decisions; fill now or schedule for later.
    7. Record worth, cash, and P&L via RunJournal.record_day.
    8. On the last day, optionally close leftovers and build the result.
       ``stats_level="lean"`` skips trade/cash/exposure summaries so optimizer
       trials only pay for equity and extended stats used to rank mixes.

Book routing:
    Pass a book such as ``SystematicMultiAsset`` and the simulator builds its
    children and installs a ``BookFillRouter``. A plain list of strategies uses
    ``DirectFillRouter``. Rejections show up in ``skipped``.

Risk/Assumptions:
    Fills are at the close of the bar. Optional ``commission_pct`` /
    ``slippage_pct`` are percent-of-notional cash costs (default zero).
    Cash can go negative when limits allow it (borrowed money, no interest).
    Closes always exit the full position.
    ``StrategyBase.simulate`` is this simulator with one strategy, fully
    invested, no cash floor, and no wipeout stop.
    Book checks happen at fill time, so a delayed open does not reserve a slot
    until it actually fills. End-of-run and wipeout closes skip the book.

Usage:
    >>> from trade.backtester_.simulator import Simulator
    >>> sim = Simulator([strat_a, strat_b], capital=100_000, percent_of_capital=0.5)
    >>> result = sim.run()
    >>> result.equity_stats["cagr"]
    >>> Simulator(book, capital=100_000, percent_of_capital=0.25).run()
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd

from EventDriven.types import PositionEffect
from trade.backtester_.simulator.broker import Broker
from trade.backtester_.simulator.core import BuyingPowerPolicy, PendingOrder, SimulationResult, StrategyKey
from trade.backtester_.simulator.fill_router import FillRouter, make_fill_router
from trade.backtester_.simulator.journal import RunJournal
from trade.backtester_.simulator.order_manager import OrderManager
from trade.backtester_.simulator.recorder import DayRecorder
from trade.backtester_.simulator.result_builder import ResultBuilder
from trade.helpers.helper import to_datetime
from trade.helpers.Logging import setup_logger

## StrategyBase.simulate delegates to Simulator, so a runtime import here would be circular.
if TYPE_CHECKING:
    from trade.backtester_._strategy import StrategyBase, TradeDecision
    from trade.backtester_.systematic.book import PortfolioBook

logger = setup_logger(__name__)


class Simulator:
    """Backtest several strategies that share one pile of cash.

    Pass a list of strategies, or a book such as ``SystematicMultiAsset``.
    Every open is sized from either a fixed dollar amount (``trade_size``) or a
    fraction of current equity (``percent_of_capital``). A strategy can still
    ask for more or less size on a given signal through its multiplier.

    After ``run()``, look at equity and trades for performance, and at the
    exposure fields to see how loaded the book was and which strategies used
    the capital.
    """

    def __init__(
        self,
        asset_strategies: Union[PortfolioBook, List[StrategyBase]],
        capital: float,
        *,
        trade_size: Optional[float] = None,
        percent_of_capital: Optional[float] = None,
        max_gross_leverage: float = np.inf,
        min_cash: Optional[float] = 0.0,
        buying_power_policy: BuyingPowerPolicy = BuyingPowerPolicy.SKIP,
        commission_pct: float = 0.0,
        slippage_pct: float = 0.0,
        wipeout_floor: float = 0.0,
        strategy_keys: Optional[List[str]] = None,
        start_date: Optional[Union[datetime, str]] = None,
        risk_free_rate: float = 0.0,
        finalize: bool = True,
        enforce_open_on_signal: bool = False,
        enforce_close_on_signal: bool = False,
        track_exposure: bool = True,
        stats_level: Literal["full", "lean"] = "full",
    ) -> None:
        """Create a simulator with starting cash and trading rules.

        You must pass exactly one of ``trade_size`` or ``percent_of_capital``.
        Capital limits (leverage cap and cash floor) only affect new opens;
        closes always exit the full position.

        Args:
            asset_strategies: The strategies to run together, or a portfolio book.
                A book builds its own children and can block or reorder opens.
            capital: How much cash the portfolio starts with.
            trade_size: Dollar size of a typical open (before any multiplier).
            percent_of_capital: Open size as a share of current equity, for example
                ``0.25`` means about a quarter of the book per open.
            max_gross_leverage: How many times equity you may have in total
                position size. ``np.inf`` means no cap.
            min_cash: Cash that must stay after an open. ``0.0`` means no
                borrowing. ``None`` means no floor. A negative number is a credit
                line (for example ``-50_000`` allows up to $50k borrowed).
            buying_power_policy: What to do when an open would break the leverage
                or cash limit: skip it, or shrink it to what still fits.
            commission_pct: Fraction of each fill's notional charged as commission
                (default ``0.0``). Cash only; fill prices stay the bar close.
            slippage_pct: Fraction of each fill's notional charged as slippage cost
                (default ``0.0``). Cash only; MAE/MFE stay raw price returns.
            wipeout_floor: If equity falls to this level or below, close everything
                and stop the run.
            strategy_keys: Optional names for each strategy in the results. By
                default each is ``ClassName:TICKER``. Do not pass this with a book;
                the book already names its children.
            start_date: First date to trade. ``None`` uses the earliest strategy
                start date (or the first bar of data if none set one). Equity and
                other daily series are seeded one business day earlier at flat
                cash so ``equity.iloc[0]`` equals initial capital. Each strategy
                can still refuse to trade before its own start.
            risk_free_rate: Yearly rate used only when scoring risk-adjusted
                returns in ``full_stats``.
            finalize: If True, close every leftover position on the last day so
                every trade has an exit. If False, open trades stay marked to market.
            enforce_open_on_signal: Only matters when a strategy uses ``tplusn > 0``
                (signal on one bar, fill ``tplusn`` bars later). When False (default),
                a pending open fills on its due date even if the open signal has
                already flipped off. When True, the simulator calls
                ``should_open`` again on the fill bar and drops the pending open
                unless that decision is still ``ok``. With ``tplusn=0`` (fill on
                the signal bar), this flag has no effect because there is no delay.
            enforce_close_on_signal: Same delay/re-check idea for closes. Only
                matters when ``tplusn > 0``. When False (default), a pending close
                fills on its due date even if the close signal is gone. When True,
                ``should_close`` is checked again on the fill bar and the pending
                close is dropped unless it is still ``ok``. Independent of
                ``enforce_open_on_signal``; you can enforce one side and not the other.
            track_exposure: If True (default), keep daily how much is invested,
                long vs short, leverage, and how much each strategy holds, plus
                summary tables on the result. Turn off when you only care about
                the equity score (for example in a large optimizer grid). Total
                position size is still watched for the leverage cap either way.
                Forced off when ``stats_level="lean"``.
            stats_level: How much of the result to build after the fill loop.
                ``"full"`` (default) builds trade, cash, and exposure summaries.
                ``"lean"`` keeps equity, ``eq_stats``, and ``extended_stats`` (so
                keys like ``sharpe_ratio`` still land in ``full_stats``) and
                skips the heavier summary tables. The fill loop is the same either way.

        Raises:
            ValueError: Bad capital, sizing, leverage, cash floor, stats level, or duplicate names.
            TypeError: A book mixed with ``strategy_keys``, or a book stuffed inside a list.
            BookConfigError: The book's own config is invalid.
        """
        asset_strategies, strategy_keys, self.book = self._route(asset_strategies, strategy_keys)
        self._validate_inputs(asset_strategies, capital, trade_size, percent_of_capital, max_gross_leverage, min_cash)
        if stats_level not in ("full", "lean"):
            raise ValueError(f"stats_level must be 'full' or 'lean', got {stats_level!r}")
        raw_keys = strategy_keys or [f"{type(s).__name__}:{s.ticker}" for s in asset_strategies]
        keys = [StrategyKey(k) for k in raw_keys]
        if len(keys) != len(asset_strategies) or len(set(keys)) != len(keys):
            raise ValueError(f"Strategy keys must be unique, one per strategy. Got {keys}")
        ## Dict preserves insertion order, which doubles as same-day open priority.
        self.strategies: Dict[StrategyKey, StrategyBase] = dict(zip(keys, asset_strategies))
        self.wipeout_floor = float(wipeout_floor)
        self.start_date: Optional[pd.Timestamp] = pd.Timestamp(to_datetime(start_date)) if start_date else None
        self.risk_free_rate = float(risk_free_rate)
        self.finalize = finalize
        self.enforce_open_on_signal = enforce_open_on_signal
        self.enforce_close_on_signal = enforce_close_on_signal
        self.stats_level: Literal["full", "lean"] = stats_level
        ## Lean never needs daily exposure frames; force that off so snapshots stay cheap.
        self.track_exposure = bool(track_exposure) and stats_level == "full"

        self.broker = Broker(
            self.strategies,
            capital,
            trade_size=trade_size,
            percent_of_capital=percent_of_capital,
            max_gross_leverage=max_gross_leverage,
            min_cash=min_cash,
            buying_power_policy=buying_power_policy,
            commission_pct=commission_pct,
            slippage_pct=slippage_pct,
            bar_index=self._bar_index,
        )
        self.commission_pct = float(commission_pct)
        self.slippage_pct = float(slippage_pct)
        ## Chosen once: no ``if self.book`` on the fill hot path.
        self.fill_router: FillRouter = make_fill_router(
            self.broker, self.book, self.strategies, self._bar_index
        )
        self.orders = OrderManager()
        self.journal = RunJournal(
            self.broker,
            DayRecorder(list(self.strategies), track_exposure=self.track_exposure),
        )
        ## Compatibility alias for older call sites that used ``sim.recorder``.
        self.recorder = self.journal.daily

        ## Compatibility aliases used by older call sites / notebooks.
        self.initial_capital = self.broker.initial_capital
        self.trade_size = trade_size
        self.percent_of_capital = percent_of_capital
        self.max_gross_leverage = float(max_gross_leverage)
        self.min_cash = None if min_cash is None else float(min_cash)
        self.buying_power_policy = BuyingPowerPolicy(buying_power_policy)

        logger.info(
            "Simulator init strategies=%s book=%s capital=%s stats_level=%s track_exposure=%s",
            len(self.strategies),
            type(self.book).__name__ if self.book is not None else None,
            self.initial_capital,
            self.stats_level,
            self.track_exposure,
        )

    # ------------------------------------------------------------------ #
    # Compatibility views (delegate to Broker)
    # ------------------------------------------------------------------ #
    @property
    def cash(self) -> float:
        """Current cash balance."""
        return self.broker.cash

    @property
    def positions(self) -> Dict:
        """Open positions by strategy key."""
        return self.broker.positions

    @property
    def realized_pnl(self) -> Dict[StrategyKey, float]:
        """Cumulative realized profit and loss by strategy key."""
        return self.broker.realized_pnl

    @property
    def equity(self) -> float:
        """What the portfolio is worth right now: cash plus open positions."""
        return self.broker.equity

    @property
    def gross_exposure(self) -> float:
        """Total dollars tied up in positions, ignoring long vs short direction."""
        return self.broker.gross_exposure

    @property
    def net_exposure(self) -> float:
        """Net bet size: longs add, shorts subtract."""
        return self.broker.net_exposure

    @property
    def long_exposure(self) -> float:
        """Dollars in long positions only."""
        return self.broker.long_exposure

    @property
    def short_exposure(self) -> float:
        """Dollars in short positions only (as a positive size)."""
        return self.broker.short_exposure

    @staticmethod
    def _route(
        asset_strategies: Union[PortfolioBook, List[StrategyBase]],
        strategy_keys: Optional[List[str]],
    ) -> Tuple[List[StrategyBase], Optional[List[str]], Optional[PortfolioBook]]:
        """Decide whether the caller passed a book or a plain list of strategies.

        A book is expanded into its child strategies and kept so fills can ask it
        for approval. A list is left alone and uses the simple fill path.

        Returns:
            The strategies to run, their names (or None to invent defaults), and
            the book object when one was passed (otherwise None).

        Raises:
            TypeError: Book plus custom keys, or a book hidden inside a list.
        """
        ## Imported here because the book module imports _strategy, which imports this package.
        from trade.backtester_.systematic.book import PortfolioBook

        if isinstance(asset_strategies, PortfolioBook):
            if strategy_keys is not None:
                raise TypeError("Simulator(book) sets strategy_keys from the book; remove strategy_keys")
            ## build() returns new children every call, so rules never stack across runs.
            children = asset_strategies.build()
            return list(children.values()), list(children), asset_strategies
        if any(isinstance(s, PortfolioBook) for s in asset_strategies):
            raise TypeError("Pass a book on its own, Simulator(book, ...), not inside a list of strategies")
        return asset_strategies, strategy_keys, None

    @staticmethod
    def _validate_inputs(
        asset_strategies: List[StrategyBase],
        capital: float,
        trade_size: Optional[float],
        percent_of_capital: Optional[float],
        max_gross_leverage: float,
        min_cash: Optional[float],
    ) -> None:
        """Fail early on settings that cannot run a sensible backtest.

        Raises:
            ValueError: When a setting is missing, conflicting, or out of range.
        """
        if not asset_strategies:
            raise ValueError("asset_strategies must contain at least one strategy")
        if capital <= 0:
            raise ValueError("capital must be positive")
        if (trade_size is None) == (percent_of_capital is None):
            raise ValueError("Provide exactly one of trade_size or percent_of_capital")
        if trade_size is not None and trade_size <= 0:
            raise ValueError("trade_size must be positive")
        if percent_of_capital is not None and not 0 < percent_of_capital <= 1:
            raise ValueError("percent_of_capital must be in (0, 1]; use multiplier for leverage")
        if max_gross_leverage <= 0:
            raise ValueError("max_gross_leverage must be positive (use np.inf for no cap)")
        ## A floor above starting capital means no long could ever open.
        if min_cash is not None and min_cash > capital:
            raise ValueError(f"min_cash ({min_cash}) cannot exceed capital ({capital})")

    def _reset_state(self) -> None:
        """Wipe broker, pending orders, and the daily log for a clean run."""
        self.broker.reset()
        self.orders.reset()
        self.journal.reset()

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def run(self) -> SimulationResult:
        """Walk day by day, trade, and return the full backtest result.

        Returns:
            Everything you need to judge the run: curves, trades, and stats.

        Raises:
            ValueError: If ``start_date`` is after every strategy's last date.
        """
        self._reset_state()
        calendar = self._build_calendar()
        logger.info(
            "Simulator.run start days=%s first=%s last=%s",
            len(calendar),
            calendar[0].date() if len(calendar) else None,
            calendar[-1].date() if len(calendar) else None,
        )
        ## Flat seed before any fills so day-1 opens do not rewrite starting equity.
        self._seed_prior_flat_day(calendar)
        wiped = False
        for ts in calendar:
            self.broker.mark_to_market(ts)
            ## Leverage can push equity to zero or below. Past that point the
            ## numbers stop meaning anything, so liquidate and stop.
            if self.broker.equity <= self.wipeout_floor:
                logger.warning(
                    "Simulator wipeout equity=%s floor=%s date=%s",
                    self.broker.equity,
                    self.wipeout_floor,
                    ts.date(),
                )
                self.broker.close_all(ts, reason="wipeout")
                self.journal.record_day(ts)
                wiped = True
                break
            self._execute_pending(ts)
            self._process_decisions(ts)
            self.journal.record_day(ts)
        else:
            ## Only finalize when the loop ran to the end (not after a wipeout).
            ## record_day already ran for the last bar with positions still open;
            ## close_all then flattens, so replace_last_day rewrites that row.
            if self.finalize:
                self.broker.close_all(calendar[-1], reason="finalize")
                self.journal.replace_last_day(calendar[-1])

        for strategy in self.strategies.values():
            strategy.reset_strategy_state()
        result = self._build_result()
        logger.info(
            "Simulator.run done wiped=%s trades=%s skipped=%s stats_level=%s equity_end=%s",
            wiped,
            result.num_trades,
            len(result.skipped),
            self.stats_level,
            float(result.eq.iloc[-1]) if len(result.eq) else None,
        )
        return result

    def _effective_start_date(self) -> Optional[pd.Timestamp]:
        """First date the portfolio should trade (series seed one bus day earlier)."""
        if self.start_date is not None:
            return self.start_date
        ## Strategies load warmup bars for their indicators but cannot trade before
        ## their own start_trading_date, so those bars would only add flat equity.
        strategy_starts = [s.start_date for s in self.strategies.values() if s.start_date is not None]
        return min(strategy_starts) if strategy_starts else None

    def _build_calendar(self) -> pd.DatetimeIndex:
        """All trading days any strategy can see, trimmed to the start date.

        Raises:
            ValueError: If no dates remain after applying the start date.
        """
        calendar = pd.DatetimeIndex([])
        for strategy in self.strategies.values():
            calendar = calendar.union(strategy._index)
        ## Trimming the calendar (not just gating decisions) keeps warmup bars out of
        ## every series, so CAGR, drawdown and cash stats are measured from the start.
        start = self._effective_start_date()
        if start is not None:
            calendar = calendar[calendar >= start]
        if calendar.empty:
            raise ValueError(f"No trading dates on or after start date {start}")
        return calendar

    def _seed_prior_flat_day(self, calendar: pd.DatetimeIndex) -> None:
        """Append one cash-only snapshot the business day before trading starts.

        Args:
            calendar: Trade calendar (must be non-empty). Broker must still be flat.
        """
        seed_ts = pd.Timestamp(calendar[0] - pd.tseries.offsets.BDay(1))
        self.journal.record_day(seed_ts)
        logger.debug(
            "Simulator seeded prior flat day %s equity=%s cash=%s",
            seed_ts.date(),
            self.broker.equity,
            self.broker.cash,
        )

    def _bar_index(self, key: StrategyKey, ts: pd.Timestamp) -> Optional[int]:
        """Where this date sits in one strategy's price history, if it has a bar that day."""
        return self.strategies[key]._dates_map.get(ts)

    # ------------------------------------------------------------------ #
    # Decisions and scheduling
    # ------------------------------------------------------------------ #
    def _process_decisions(self, ts: pd.Timestamp) -> None:
        """Ask each strategy what to do today, then fill or schedule the order.

        Same-day opens beat closes for one strategy. Immediate (t+0) orders fill
        now; delayed orders wait for a later bar on that strategy's calendar.
        """
        immediate: List[PendingOrder] = []
        for key, strategy in self.strategies.items():
            i = self._bar_index(key, ts)
            if i is None:
                continue
            order = self._decision_to_order(
                key=key,
                ts=ts,
                open_decision=strategy.should_open(index=i),
                close_decision=strategy.should_close(index=i),
            )
            if order is None:
                continue
            tn = int(strategy.tplusn) if strategy.tplusn is not None else 0
            if tn == 0:
                immediate.append(order)
            else:
                ## Delay is counted in the strategy's own bars, clamped to its last bar.
                exec_ts = strategy._index[min(i + tn, strategy._n - 1)]
                self.orders.schedule(order, exec_ts)
        self.fill_router.fill(immediate, ts)

    @staticmethod
    def _decision_to_order(
        key: StrategyKey,
        ts: pd.Timestamp,
        open_decision: TradeDecision,
        close_decision: TradeDecision,
    ) -> Optional[PendingOrder]:
        """Pick at most one open or close from today's signals for one strategy."""
        if open_decision.ok:
            return PendingOrder(
                strategy_key=StrategyKey(key),
                pos_effect=PositionEffect.OPEN,
                signal_date=ts,
                side=open_decision.side,
                signal_id=open_decision.signal_id,
                ## TradeDecision.multiplier is Optional; treat None as unlevered.
                multiplier=open_decision.multiplier if open_decision.multiplier is not None else 1.0,
            )
        if close_decision.ok:
            return PendingOrder(
                strategy_key=StrategyKey(key), pos_effect=PositionEffect.CLOSE, signal_date=ts
            )
        return None

    def _execute_pending(self, ts: pd.Timestamp) -> None:
        """Fill orders that were delayed from an earlier signal and come due today."""
        due = self.orders.due(ts, self._still_valid)
        self.fill_router.fill(due, ts)

    def _still_valid(self, order: PendingOrder, ts: pd.Timestamp) -> bool:
        """Return whether a delayed order should still fill on ``ts``."""
        enforce = self.enforce_open_on_signal if order.pos_effect == PositionEffect.OPEN else self.enforce_close_on_signal
        if not enforce:
            return True
        strategy = self.strategies[order.strategy_key]
        i = self._bar_index(order.strategy_key, ts)
        if i is None:
            return False
        decision = strategy.should_open(index=i) if order.pos_effect == PositionEffect.OPEN else strategy.should_close(index=i)
        return decision.ok

    # ------------------------------------------------------------------ #
    # Result
    # ------------------------------------------------------------------ #
    def _buy_and_hold_return(self, start: pd.Timestamp, end: pd.Timestamp) -> float:
        """Simple equal-weight buy-and-hold return of the tickers in this run."""
        ticker_returns = []
        for strategy in self.strategies.values():
            closes = pd.Series(strategy._close, index=strategy._index).loc[start:end]
            if len(closes) >= 2 and closes.iloc[0] > 0:
                ticker_returns.append(closes.iloc[-1] / closes.iloc[0] - 1)
        return float(np.mean(ticker_returns)) if ticker_returns else np.nan

    def _build_result(self) -> SimulationResult:
        """Turn the daily log and trade list into the object ``run`` returns."""
        daily = self.journal.daily_frame()
        buy_hold = self._buy_and_hold_return(daily.index[0], daily.index[-1]) if len(daily) else np.nan
        return ResultBuilder(
            self.strategies,
            self.journal.daily,
            self.journal.trades,
            self.journal.skipped,
            stats_level=self.stats_level,
            track_exposure=self.track_exposure,
            risk_free_rate=self.risk_free_rate,
            buy_and_hold_return=buy_hold,
        ).build()
