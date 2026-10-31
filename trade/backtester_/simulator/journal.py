"""Run-scoped directory of logs: daily curves, trades, skips, and mark paths.

``RunJournal`` does not own fill or mark logic. It holds the ``DayRecorder`` and
exposes the broker's ``TradeRecord`` book, skip list, and open positions (each
position carries its ``return_path``). The day loop and ``ResultBuilder`` can
reach every timed log through one object.

Comment density: orchestration

Core Classes:
    RunJournal: Façade over daily, trade, skip, and position mark logs.

Processing Flow:
    1. ``Simulator`` may call ``record_day`` once before the loop for a prior flat seed.
    2. ``record_day`` / ``replace_last_day`` delegate to ``DayRecorder``.
    3. Properties expose ``trades``, ``skipped``, and ``positions`` from the broker.
    4. ``ResultBuilder`` reads the same handles at the end of the run.
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from trade.backtester_.simulator.broker import Broker
from trade.backtester_.simulator.core import SimPosition, SkipEvent, StrategyKey, TradeRecord
from trade.backtester_.simulator.recorder import DayRecorder
from trade.helpers.Logging import setup_logger

logger = setup_logger(__name__)


class RunJournal:
    """One place to find every timed log for a simulator run.

    Daily portfolio rows are written here via ``DayRecorder``. Trade records and
    skip rows stay on ``Broker`` (filled by ExecutionEngine / Account); open
    positions (and their mark ``return_path``) stay on ``Portfolio``. This class
    only points at them.

    Args:
        broker: Live cash, positions, trades, skipped.
        daily: DayRecorder for end-of-day portfolio snapshots.
    """

    def __init__(self, broker: Broker, daily: DayRecorder) -> None:
        """Wire the daily log to the broker that supplies snapshot scalars."""
        self.broker = broker
        self.daily = daily

    # --- views into broker-owned logs ---

    @property
    def trades(self) -> Dict[StrategyKey, List[TradeRecord]]:
        """Per-strategy ``TradeRecord`` round-trips written by ExecutionEngine."""
        return self.broker.trades

    @property
    def skipped(self) -> List[SkipEvent]:
        """Capital-limit and book-veto rows (date, key, notionals, reason)."""
        return self.broker.skipped

    @property
    def positions(self) -> Dict[StrategyKey, SimPosition]:
        """Open positions; each ``SimPosition.return_path`` is the daily mark log."""
        return self.broker.positions

    # --- day-loop API ---

    def reset(self) -> None:
        """Clear the daily series only (``broker.reset`` clears trades/skips/cash)."""
        self.daily.reset()

    def record_day(self, ts: pd.Timestamp) -> None:
        """Append today's portfolio snapshot from the broker.

        Args:
            ts: Trading day just finished (marks and fills already applied).
        """
        self.daily.record(ts, self.broker)

    def replace_last_day(self, ts: pd.Timestamp) -> None:
        """Rewrite the last daily row after the book was changed again that day.

        Used when ``finalize=True``: the loop already recorded the last calendar
        day with positions still open, then ``close_all`` flattens everything.
        Without a rewrite, the final equity/cash/exposure point would still show
        the pre-close state. This replaces that row with a fresh snapshot at
        ``ts`` (typically all cash, zero exposure).

        Args:
            ts: Date of the row to replace (usually the last calendar day).
        """
        self.daily.replace_last(ts, self.broker)

    def daily_frame(self) -> pd.DataFrame:
        """Date-indexed daily portfolio log for result construction."""
        return self.daily.to_frame()
