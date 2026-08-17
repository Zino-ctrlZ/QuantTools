"""Corrected position PnL helpers for cog analysis.

Portfolio open-book ``PositionState.pnl`` / ``entry_price`` can be wrong after a
partial SELL because book ``entry_price`` stays the original *total* while
``quantity`` shrinks. Cogs that need a stable return versus initial cost should
prefer the trade ledger via ``correct_position_pnl``.

Comment density: domain policy.

Core Dataclasses:
    CorrectedPositionPnL: Ledger-backed (or fallback) dollar PnL and ratio.

Core Functions:
    correct_position_pnl: Derive corrected PnL fields from ``PositionState``.

Usage:
    >>> from EventDriven.riskmanager.position.cogs.pnl_utils import correct_position_pnl
    >>> corrected = correct_position_pnl(pos_state)
    >>> if corrected.pnl_pct >= threshold:
    ...     ...
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from EventDriven.dataclasses.states import PositionState


@dataclass(frozen=True, slots=True)
class CorrectedPositionPnL:
    """Ledger-backed (or open-book fallback) PnL snapshot for cog decisions.

    Attributes:
        pnl: Dollar PnL. Ledger path uses ``trade.total_pnl`` (realized +
            unrealized). Fallback uses ``pos_state.pnl``.
        entry_price: True ``$/contract`` entry. Ledger path uses
            ``buy_ledger.avg_price``; fallback uses ``pos_state.entry_price``.
        initial_quantity: Contracts used for the initial notional. Ledger path
            uses ``buy_ledger.quantity``; fallback uses ``fallback_initial_qty``
            or ``pos_state.quantity``.
        initial_cost: ``entry_price * initial_quantity`` (scaled dollars).
        pnl_pct: ``pnl / initial_cost`` when cost is nonzero, else ``0.0``.
        used_ledger: ``True`` when values came from the trade buy ledger.
    """

    pnl: float
    entry_price: float
    initial_quantity: int
    initial_cost: float
    pnl_pct: float
    used_ledger: bool


def correct_position_pnl(
    pos_state: PositionState,
    *,
    fallback_initial_qty: Optional[int] = None,
) -> CorrectedPositionPnL:
    """Return PnL versus frozen initial cost, preferring the trade ledger.

    Prefer ``trade.total_pnl / (buy_ledger.avg_price * buy_ledger.quantity)``.
    Both sides are ×100-scaled dollars, so the ratio is unitless and stable
    across partial closes (the buy ledger quantity does not shrink on SELL).

    When no usable trade ledger is attached (e.g. synthetic test states), fall
    back to ``pos_state.pnl / (entry_price * qty)``. That fallback is only
    reliable pre-trim because portfolio SELL accounting inflates
    ``PositionState.entry_price``.

    Args:
        pos_state: Open position from a portfolio analysis context.
        fallback_initial_qty: Optional quantity override for the fallback
            denominator (e.g. sizing metadata ``initial_quantity``). Ignored
            when the ledger path succeeds.

    Returns:
        Corrected dollar PnL, entry, initial cost, and percentage.
    """
    trade = getattr(pos_state, "trades", None)
    if trade is not None:
        buy_ledger = getattr(trade, "buy_ledger", None)
        total_pnl = getattr(trade, "total_pnl", None)
        if buy_ledger is not None and total_pnl is not None:
            entry_price = float(buy_ledger.avg_price)
            initial_quantity = int(buy_ledger.quantity)
            ## Initial notional is total bought for this trade_id; stable across sells.
            initial_cost = entry_price * float(initial_quantity)
            if initial_cost != 0 and initial_quantity > 0:
                pnl = float(total_pnl)
                return CorrectedPositionPnL(
                    pnl=pnl,
                    entry_price=entry_price,
                    initial_quantity=initial_quantity,
                    initial_cost=initial_cost,
                    pnl_pct=pnl / initial_cost,
                    used_ledger=True,
                )

    ## Fallback: open-book fields. Only reliable while quantity == initial buy qty.
    entry_price = float(pos_state.entry_price or 0.0)
    if fallback_initial_qty is not None and int(fallback_initial_qty) > 0:
        initial_quantity = int(fallback_initial_qty)
    else:
        initial_quantity = int(pos_state.quantity or 0)
    initial_cost = entry_price * float(initial_quantity)
    pnl = float(pos_state.pnl or 0.0)
    pnl_pct = pnl / initial_cost if initial_cost != 0 else 0.0
    return CorrectedPositionPnL(
        pnl=pnl,
        entry_price=entry_price,
        initial_quantity=initial_quantity,
        initial_cost=initial_cost,
        pnl_pct=pnl_pct,
        used_ledger=False,
    )
