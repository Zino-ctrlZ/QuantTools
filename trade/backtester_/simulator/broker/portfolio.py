"""Open positions, marks, and exposure views for the shared-cash simulator.

``Portfolio`` owns the position map and per-strategy realized profit and loss.
It does not hold cash; equity is ``cash + sum(market_value)`` with cash supplied
by the caller (usually ``Account``).

Comment density: domain policy

Core Classes:
    Portfolio: Positions, marks, and exposure aggregates.

Processing Flow:
    1. ``reset`` clears positions and zeros realized profit and loss for the fixed key set.
    2. ``mark`` refreshes one position's last price and appends a ``MarkPoint``.
    3. Exposure properties sum signed or absolute market values for leverage caps
       and book context.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import pandas as pd

from trade.backtester_.simulator.core import MarkPoint, SimPosition, StrategyKey
from trade.helpers.Logging import setup_logger

logger = setup_logger(__name__)


class Portfolio:
    """Open positions and realized profit and loss by strategy key.

    Strategy keys are fixed at construction. A new ``Simulator`` / ``Broker`` is
    created when the child set changes; ``reset`` only clears run state.

    Args:
        strategy_keys: Keys that may hold positions; used to init realized PnL.
    """

    def __init__(self, strategy_keys: Iterable[StrategyKey]) -> None:
        """Create an empty book for the given strategy keys."""
        self._keys: List[StrategyKey] = [StrategyKey(k) for k in strategy_keys]
        self.positions: Dict[StrategyKey, SimPosition] = {}
        self.realized_pnl: Dict[StrategyKey, float] = {k: 0.0 for k in self._keys}

    def reset(self) -> None:
        """Clear positions and zero realized profit and loss for the fixed key set."""
        self.positions = {}
        self.realized_pnl = {k: 0.0 for k in self._keys}

    def market_value_total(self) -> float:
        """Sum of signed market values across open positions."""
        return sum(p.market_value for p in self.positions.values())

    @property
    def gross_exposure(self) -> float:
        """Total dollars tied up in positions, ignoring long vs short direction."""
        return sum(p.gross_value for p in self.positions.values())

    @property
    def net_exposure(self) -> float:
        """Net bet size: longs add, shorts subtract."""
        return sum(p.market_value for p in self.positions.values())

    @property
    def long_exposure(self) -> float:
        """Dollars in long positions only."""
        return sum(p.gross_value for p in self.positions.values() if p.side > 0)

    @property
    def short_exposure(self) -> float:
        """Dollars in short positions only (as a positive size)."""
        return sum(p.gross_value for p in self.positions.values() if p.side < 0)

    def mark(self, key: StrategyKey, ts: pd.Timestamp, *, price: float, index: int) -> None:
        """Refresh one open position to today's close and append its return path.

        Args:
            key: Strategy key of the position.
            ts: Mark date.
            price: Close price for this bar.
            index: Strategy bar index for MAE/MFE path alignment.
        """
        pos = self.positions.get(key)
        if pos is None:
            return
        pos.last_price = float(price)
        pos.last_index = index
        ## Same unlevered cumulative return StrategyBase tracks, so MAE/MFE match single-strategy runs.
        pos.return_path.append(
            MarkPoint(
                date=ts,
                index=index,
                cum_return_pct=(pos.last_price - pos.entry_price) / pos.entry_price * pos.side,
            )
        )

    def add_position(self, position: SimPosition) -> None:
        """Register a newly opened position under its strategy key."""
        self.positions[position.strategy_key] = position

    def pop_position(self, key: StrategyKey) -> Optional[SimPosition]:
        """Remove and return a position, or None if flat."""
        return self.positions.pop(key, None)

    def running_pnl(self, key: StrategyKey) -> float:
        """Realized plus unrealized dollars for one strategy key."""
        unrealized = self.positions[key].unrealized_pnl if key in self.positions else 0.0
        return self.realized_pnl[key] + unrealized
