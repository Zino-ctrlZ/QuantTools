"""Regression checks for Trade and TradeLedger PnL consistency.

Validates:
1. total_pnl stays synchronized with closed_pnl + unrealized_pnl
2. market price updates refresh total_pnl
3. ledger rejects zero or negative quantity

Usage:
    /Users/chiemelienwanisobi/miniconda3/envs/openbb_new_use/bin/python \
        module_test/test_trade_pnl_consistency.py
"""

from __future__ import annotations

from datetime import datetime
from math import isclose

from EventDriven.trade import Trade
from EventDriven.tradeLedger import TradeLedger


def test_trade_pnl_consistency() -> None:
    trade = Trade(trade_id="T1", symbol="AAPL", signal_id="S1")

    # Buy 2 contracts at 100 each.
    trade._update_kw(
        entry_time=datetime(2026, 1, 2),
        direction="BUY",
        fill_cost=200.0,
        quantity=2,
        symbol="AAPL",
        commission=0.0,
        market_value=200.0,
        slippage=0.0,
        normalize=False,
    )
    trade.update_current_price(120.0)

    assert isclose(trade.closed_pnl, 0.0)
    assert isclose(trade.unrealized_pnl, 40.0)
    assert isclose(trade.total_pnl, 40.0)
    assert isclose(trade.total_pnl, trade.closed_pnl + trade.unrealized_pnl)

    # Close 1 contract at 130.
    trade._update_kw(
        entry_time=datetime(2026, 1, 3),
        direction="SELL",
        fill_cost=130.0,
        quantity=1,
        symbol="AAPL",
        commission=0.0,
        market_value=130.0,
        slippage=0.0,
        normalize=False,
    )

    assert isclose(trade.closed_pnl, 30.0)
    assert isclose(trade.unrealized_pnl, 20.0)
    assert isclose(trade.total_pnl, 50.0)
    assert isclose(trade.total_pnl, trade.closed_pnl + trade.unrealized_pnl)

    # Mark remaining open position down to 90.
    trade.update_current_price(90.0)

    assert isclose(trade.closed_pnl, 30.0)
    assert isclose(trade.unrealized_pnl, -10.0)
    assert isclose(trade.total_pnl, 20.0)
    assert isclose(trade.total_pnl, trade.closed_pnl + trade.unrealized_pnl)


def test_ledger_positive_quantity_guard() -> None:
    ledger = TradeLedger("guard_test")

    raised = False
    try:
        ledger._add_entry_kw(
            entry_time=datetime(2026, 1, 2),
            trade_id="T2",
            signal_id="S2",
            fill_cost=0.0,
            quantity=0,
            symbol="AAPL",
            commission=0.0,
            market_value=0.0,
            slippage=0.0,
            direction="BUY",
            normalize=False,
        )
    except ValueError:
        raised = True

    assert raised, "TradeLedger must reject non-positive quantity entries"


if __name__ == "__main__":
    test_trade_pnl_consistency()
    test_ledger_positive_quantity_guard()
    print("All trade consistency tests passed.")
