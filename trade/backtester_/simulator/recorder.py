"""Daily portfolio snapshots for the simulator result curves.

``DayRecorder`` lives with the day loop, not inside ``Broker``. Each bar the
simulator asks the broker for current values and appends a frozen
``DailySnapshot`` (defined in ``core``). Exposure fields stay ``None`` when
tracking is off so lean runs stay cheap. ``to_frame`` expands per-strategy maps
into the wide ``pnl::`` / ``exp::`` columns ``ResultBuilder`` expects.

Comment density: orchestration

Core Classes:
    DayRecorder: Append and rewrite daily snapshots; build the result frame.

Processing Flow:
    1. ``snapshot`` reads broker scalars and builds a ``DailySnapshot``.
    2. ``record`` appends; ``replace_last`` swaps the final row after finalize.
    3. ``to_frame`` expands maps into a date-indexed DataFrame.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import pandas as pd

from trade.backtester_.simulator.broker import Broker
from trade.backtester_.simulator.core import DailySnapshot, StrategyKey
from trade.helpers.Logging import setup_logger

logger = setup_logger(__name__)


class DayRecorder:
    """Collect one ``DailySnapshot`` per trading day.

    Args:
        strategy_keys: Keys used for strategy PnL / exposure maps.
        track_exposure: When True, fill exposure fields on each snapshot.
    """

    def __init__(self, strategy_keys: Sequence[StrategyKey], *, track_exposure: bool = True) -> None:
        """Start an empty daily log."""
        self.strategy_keys = [StrategyKey(k) for k in strategy_keys]
        self.track_exposure = bool(track_exposure)
        self.rows: List[DailySnapshot] = []

    def reset(self) -> None:
        """Clear all recorded days."""
        self.rows = []

    def snapshot(self, ts: pd.Timestamp, broker: Broker) -> DailySnapshot:
        """Build one day's snapshot by reading the broker ledger directly.

        The broker stays a live ledger (cash, positions, exposures). This method
        owns photo schema: adds ``date``, applies ``track_exposure``, and returns
        a frozen ``DailySnapshot``.

        Args:
            ts: Date for the row.
            broker: Live cash and positions.

        Returns:
            Frozen snapshot ready to append or to replace the last row.
        """
        equity = float(broker.equity)
        gross = float(broker.gross_exposure)
        strategy_pnl = {key: float(broker.portfolio.running_pnl(key)) for key in self.strategy_keys}
        if not self.track_exposure:
            return DailySnapshot(
                date=ts,
                equity=equity,
                cash=float(broker.cash),
                gross_exposure=gross,
                strategy_pnl=strategy_pnl,
            )
        positions = broker.positions
        strategy_exposure = {
            key: float(positions[key].market_value) if key in positions else 0.0
            for key in self.strategy_keys
        }
        return DailySnapshot(
            date=ts,
            equity=equity,
            cash=float(broker.cash),
            gross_exposure=gross,
            strategy_pnl=strategy_pnl,
            net_exposure=float(broker.net_exposure),
            long_exposure=float(broker.long_exposure),
            short_exposure=float(broker.short_exposure),
            n_open=len(positions),
            leverage=(gross / equity) if equity > 0 else float("nan"),
            strategy_exposure=strategy_exposure,
        )

    def record(self, ts: pd.Timestamp, broker: Broker) -> None:
        """Append today's snapshot."""
        self.rows.append(self.snapshot(ts, broker))

    def replace_last(self, ts: pd.Timestamp, broker: Broker) -> None:
        """Rewrite the last row after finalize flattens the book.

        Replaces the whole frozen snapshot; fields are never mutated in place.
        """
        if not self.rows:
            self.record(ts, broker)
            return
        self.rows[-1] = self.snapshot(ts, broker)

    def to_frame(self) -> pd.DataFrame:
        """Expand snapshots into the wide date-indexed frame ResultBuilder expects.

        Fixed columns are equity, cash, and gross exposure. Each strategy key
        becomes ``pnl::{key}`` and, when tracking, ``exp::{key}`` plus net/long/
        short/n_open/leverage columns.
        """
        if not self.rows:
            return pd.DataFrame()

        records: List[Dict] = []
        for snap in self.rows:
            row: Dict = {
                "date": snap.date,
                "equity": snap.equity,
                "cash": snap.cash,
                "gross_exposure": snap.gross_exposure,
            }
            for key, pnl in snap.strategy_pnl.items():
                row[f"pnl::{key}"] = pnl
            if snap.strategy_exposure is not None:
                row["net_exposure"] = snap.net_exposure
                row["long_exposure"] = snap.long_exposure
                row["short_exposure"] = snap.short_exposure
                row["n_open"] = snap.n_open
                row["leverage"] = snap.leverage
                for key, exp in snap.strategy_exposure.items():
                    row[f"exp::{key}"] = exp
            records.append(row)
        return pd.DataFrame(records).set_index("date")
