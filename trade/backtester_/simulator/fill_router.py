"""Fill policy at the broker: direct closes/opens vs book-gated fills.

The day loop collects and schedules orders; these routers decide how a batch
reaches ``Broker``. Pick one at ``Simulator`` init so the hot path has no
``if book`` branch.

Comment density: orchestration

Core Classes:
    FillRouter: Abstract base; subclasses must implement ``fill``.
    DirectFillRouter: Closes first, then opens, no book hooks.
    BookFillRouter: ``should_close`` / ``rank_opens`` / ``should_open`` then broker.

Core Functions:
    make_fill_router: Build the direct or book router from an optional book.

Processing Flow (book):
    1. Approve and fill closes so freed cash/slots are visible.
    2. Build open candidates; book ranks; drop unranked with a skip row.
    3. For each ranked key, ``should_open`` may veto or resize, then broker fills.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import pandas as pd

from EventDriven.types import PositionEffect
from trade.backtester_.simulator.broker import Broker
from trade.backtester_.simulator.core import (
    BarIndexFn,
    BookHookError,
    OpenCandidate,
    PendingOrder,
    StrategyKey,
    Veto,
)
from trade.helpers.Logging import setup_logger

if TYPE_CHECKING:
    from trade.backtester_._strategy import StrategyBase, TradeDecision
    from trade.backtester_.systematic.book import PortfolioBook

logger = setup_logger(__name__)


class FillRouter(ABC):
    """How a day's order batch reaches the broker.

    Subclasses must implement ``fill``. Instantiating a subclass that omits it
    fails at construction time.
    """

    @abstractmethod
    def fill(self, orders: List[PendingOrder], ts: pd.Timestamp) -> None:
        """Execute ``orders`` for ``ts`` (closes before opens when both appear).

        Args:
            orders: Pending opens and closes due today (immediate or delayed).
            ts: Fill date.
        """


class DirectFillRouter(FillRouter):
    """Fill closes then opens with no book policy.

    Args:
        broker: Shared-cash broker that mutates cash and positions.
    """

    def __init__(self, broker: Broker) -> None:
        """Store the broker used for every fill in this run."""
        self.broker = broker

    def fill(self, orders: List[PendingOrder], ts: pd.Timestamp) -> None:
        """Close first so same-day opens can use freed cash and slots.

        Args:
            orders: Pending opens and closes due today (immediate or delayed).
            ts: Fill date.
        """
        for order in orders:
            if order.pos_effect == PositionEffect.CLOSE:
                self.broker.fill_close(order.strategy_key, ts, reason="signal")
        for order in orders:
            if order.pos_effect == PositionEffect.OPEN:
                self.broker.fill_open(order, ts)


class BookFillRouter(FillRouter):
    """Gate every fill through a ``PortfolioBook`` before calling the broker.

    After each accepted fill the book sees an updated portfolio context, so slot
    and cash caps apply as positions change through the day.

    Args:
        broker: Shared-cash broker.
        book: Portfolio book with ``should_close``, ``rank_opens``, ``should_open``.
        strategies: Child strategies keyed like the broker.
        bar_index: ``(strategy_key, date) -> bar index | None``.
    """

    def __init__(
        self,
        broker: Broker,
        book: PortfolioBook,
        strategies: Dict[StrategyKey, StrategyBase],
        bar_index: BarIndexFn,
    ) -> None:
        """Wire broker, book, and strategy lookups used at fill time."""
        self.broker = broker
        self.book = book
        self.strategies = strategies
        self._bar_index = bar_index

    def fill(self, orders: List[PendingOrder], ts: pd.Timestamp) -> None:
        """Approve closes, rank opens, then fill survivors through the broker.

        Args:
            orders: Pending opens and closes due today.
            ts: Fill date.
        """
        self._fill_closes(orders, ts)
        self._fill_opens(orders, ts)

    def _verdict(
        self,
        hook: str,
        ts: pd.Timestamp,
        key: str,
        out: Any,
        effect: PositionEffect,
    ) -> Union[TradeDecision, Veto]:
        """Normalize a book's answer into either a trade decision or a veto.

        Raises:
            BookHookError: Wrong return type, or a decision that does not match open/close.
        """
        from trade.backtester_._strategy import TradeDecision

        book_name = type(self.book).__name__
        if isinstance(out, Veto):
            return out
        if not isinstance(out, TradeDecision):
            raise BookHookError(
                book_name,
                hook,
                date=ts,
                key=key,
                problem=f"returned {type(out).__name__} {out!r}; expected TradeDecision or Veto(reason)",
            )
        if not out.ok:
            return Veto(f"{hook} returned ok=False (return Veto(reason) to say why)")
        if out.pos_effect != effect:
            raise BookHookError(
                book_name,
                hook,
                date=ts,
                key=key,
                problem=f"returned pos_effect={out.pos_effect}; expected {effect}",
            )
        return out

    def _fill_closes(self, orders: List[PendingOrder], ts: pd.Timestamp) -> None:
        """Ask the book to approve each close, then fill accepted ones."""
        from trade.backtester_._strategy import TradeDecision

        for order in orders:
            key = order.strategy_key
            if order.pos_effect != PositionEffect.CLOSE or key not in self.broker.positions:
                continue
            pos = self.broker.positions[key]
            proposed = TradeDecision(
                ok=True,
                side=int(pos.side),
                pos_effect=PositionEffect.CLOSE,
                signal_id=pos.signal_id if pos.signal_id is not None else "N/A",
            )
            out = self.book.should_close(key, proposed, self.broker.context(ts))
            verdict = self._verdict("should_close", ts, key, out, PositionEffect.CLOSE)
            if isinstance(verdict, Veto):
                self.broker.skip_by_book(ts, key, verdict.reason)
                continue
            self.broker.fill_close(key, ts, reason="signal")

    def _fill_opens(self, orders: List[PendingOrder], ts: pd.Timestamp) -> None:
        """Build candidates, let the book rank and approve, then fill."""
        from trade.backtester_._strategy import TradeDecision

        opens: Dict[StrategyKey, PendingOrder] = {}
        for order in orders:
            if order.pos_effect == PositionEffect.OPEN:
                opens.setdefault(order.strategy_key, order)

        candidates: List[OpenCandidate] = []
        for key, order in opens.items():
            i = self._bar_index(key, ts)
            ## Same skip rule as broker fill_open: already held or no bar today.
            if key in self.broker.positions or i is None:
                continue
            strategy = self.strategies[key]
            decision = TradeDecision(
                ok=True,
                side=order.side,
                pos_effect=PositionEffect.OPEN,
                signal_id=order.signal_id,
                multiplier=order.multiplier,
            )
            candidates.append(
                OpenCandidate(
                    key=StrategyKey(key),
                    ticker=strategy.ticker,
                    decision=decision,
                    strategy=strategy,
                    index=i,
                )
            )
        if not candidates:
            return

        ranked = self.book.rank_opens(list(candidates), self.broker.context(ts))
        by_key = {c.key: c for c in candidates}
        ranked_keys = [getattr(c, "key", None) for c in ranked] if isinstance(ranked, list) else None
        ## A ranking that invents or repeats candidates would open the same child twice.
        if ranked_keys is None or any(k not in by_key for k in ranked_keys) or len(set(ranked_keys)) != len(ranked_keys):
            raise BookHookError(
                type(self.book).__name__,
                "rank_opens",
                date=ts,
                key=None,
                problem=f"must return a list of the given candidates without duplicates; got {ranked!r}",
            )
        ## Walk candidates in their original order so skipped rows are identical across runs.
        kept = set(ranked_keys)
        for candidate in candidates:
            if candidate.key not in kept:
                self.broker.skip_by_book(ts, candidate.key, "dropped by rank_opens")

        ## A fresh context per candidate, so each approved fill counts toward caps for the next.
        for key in ranked_keys:
            out = self.book.should_open(by_key[key], self.broker.context(ts))
            verdict = self._verdict("should_open", ts, key, out, PositionEffect.OPEN)
            if isinstance(verdict, Veto):
                self.broker.skip_by_book(ts, key, verdict.reason)
                continue
            ## The hook may resize (multiplier) or relabel the open; side and signal come from its decision.
            order = replace(
                opens[key],
                side=verdict.side,
                signal_id=verdict.signal_id,
                multiplier=verdict.multiplier if verdict.multiplier is not None else 1.0,
            )
            self.broker.fill_open(order, ts)


def make_fill_router(
    broker: Broker,
    book: Optional[PortfolioBook],
    strategies: Dict[StrategyKey, StrategyBase],
    bar_index: BarIndexFn,
) -> FillRouter:
    """Return a direct or book fill router for this run.

    Args:
        broker: Shared-cash broker.
        book: Portfolio book when the caller passed one; otherwise None.
        strategies: Child strategies for open candidates.
        bar_index: Bar lookup used when building book open candidates.

    Returns:
        ``BookFillRouter`` when ``book`` is set, else ``DirectFillRouter``.
    """
    if book is None:
        logger.info("FillRouter=DirectFillRouter")
        return DirectFillRouter(broker)
    logger.info("FillRouter=BookFillRouter book=%s", type(book).__name__)
    return BookFillRouter(broker, book, strategies, bar_index)
