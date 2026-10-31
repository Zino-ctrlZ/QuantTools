"""Delayed order lifecycle for ``tplusn`` fills.

``OrderManager`` stores pending opens and closes until their fill date. The
simulator creates orders from strategy decisions; this class only schedules,
releases due orders, and optionally cancels them when signal enforcement fails.

Comment density: orchestration

Core Classes:
    OrderManager: Create/schedule, track, cancel, and release delayed orders.

Processing Flow:
    1. ``schedule`` parks an order under its execution timestamp.
    2. ``due`` pops that day's orders and keeps those that pass ``still_valid``.
    3. ``reset`` clears the book for a new ``Simulator.run``.

Risk/Assumptions:
    Immediate (``tplusn=0``) orders never enter this queue; the sim fills them
    directly. Enforcement callbacks belong to the simulator (it owns strategies).
"""

from __future__ import annotations

from typing import Callable, Dict, List

import pandas as pd

from trade.backtester_.simulator.core import PendingOrder
from trade.helpers.Logging import setup_logger

logger = setup_logger(__name__)

## Caller decides whether a delayed order is still live on the fill bar.
StillValidFn = Callable[[PendingOrder, pd.Timestamp], bool]


class OrderManager:
    """Track pending opens/closes until their fill date.

    Owns scheduling and release. Creation from ``TradeDecision`` stays on the
    simulator so signal priority (open beats close) stays next to strategy calls.
    Updates to sitting pending orders are not supported; book hooks may
    ``replace`` fields only at fill time.
    """

    def __init__(self) -> None:
        """Start with an empty pending book."""
        self._pending: Dict[pd.Timestamp, List[PendingOrder]] = {}

    def reset(self) -> None:
        """Drop every pending order."""
        self._pending = {}

    def schedule(self, order: PendingOrder, exec_ts: pd.Timestamp) -> None:
        """Park an order to fill on ``exec_ts``.

        Args:
            order: Open or close produced on the signal bar.
            exec_ts: Strategy calendar date when the fill is due.
        """
        self._pending.setdefault(exec_ts, []).append(order)

    def due(self, ts: pd.Timestamp, still_valid: StillValidFn) -> List[PendingOrder]:
        """Pop and return orders due on ``ts`` that still pass validation.

        Args:
            ts: Fill date being processed.
            still_valid: Returns False to cancel (e.g. enforce_*_on_signal).

        Returns:
            Orders to attempt filling today, in schedule order.
        """
        return [order for order in self._pending.pop(ts, []) if still_valid(order, ts)]

    def cancel_all(self) -> None:
        """Cancel every pending order (same as ``reset``; named for clarity)."""
        self.reset()
