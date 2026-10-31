"""Fill opens and closes against an Account and Portfolio.

``ExecutionEngine`` is the only place that turns a pending order into cash and
position changes. It also owns the per-strategy ``TradeRecord`` log: one frozen
round-trip per open, replaced on close. Strategy ``open_action`` / ``close_action``
keep live state in sync; StrategyBase does not write the trade book.

Commission and slippage are percent-of-notional cash costs charged on each fill.
Fill prices stay the market close so MAE/MFE and ``return_pct`` remain raw price
paths; dollar ``pnl`` subtracts the fees.

Comment density: orchestration

Core Classes:
    ExecutionEngine: Size, apply buying power, mutate account and portfolio.

Processing Flow:
    1. ``fill_open`` resolves notional, applies buying power, debits/credits cash,
       charges open costs, inserts a ``SimPosition``, runs ``open_action``,
       appends a ``TradeRecord``.
    2. ``fill_close`` removes the position, books cash, charges exit costs, books
       fee-adjusted realized PnL, replaces the last ``TradeRecord``, then
       ``close_action``.
    3. ``close_all`` force-closes every open key (finalize / wipeout).

Risk/Assumptions:
    Fills are at the bar close (no price slippage model); slippage is a cash cost.
    Closes always exit the full position; buying power is not re-checked.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Dict, List

import pandas as pd

from trade.backtester_.simulator.broker.account import Account
from trade.backtester_.simulator.broker.portfolio import Portfolio
from trade.backtester_.simulator.core import (
    BarIndexFn,
    MarkPoint,
    PendingOrder,
    SimPosition,
    StrategyKey,
    TradeRecord,
    extremes_from_mark_path,
)
from trade.helpers.Logging import setup_logger

if TYPE_CHECKING:
    from trade.backtester_._strategy import StrategyBase

logger = setup_logger(__name__)


class ExecutionEngine:
    """Execute opens and closes against shared account and portfolio state.

    Args:
        account: Cash and capital limits.
        portfolio: Positions and realized profit and loss.
        strategies: Live strategy instances keyed like the portfolio.
        trades: Per-key ``TradeRecord`` lists owned by the broker.
        bar_index: Maps ``(strategy_key, date)`` to a bar index when present.
    """

    def __init__(
        self,
        account: Account,
        portfolio: Portfolio,
        strategies: Dict[StrategyKey, StrategyBase],
        trades: Dict[StrategyKey, List[TradeRecord]],
        bar_index: BarIndexFn,
    ) -> None:
        """Wire ledger objects and the simulator trade book used at fill time."""
        self.account = account
        self.portfolio = portfolio
        self.strategies = strategies
        self.trades = trades
        self._bar_index = bar_index

    def equity(self) -> float:
        """Cash plus signed market value of open positions."""
        return self.account.cash + self.portfolio.market_value_total()

    def fill_open(self, order: PendingOrder, ts: pd.Timestamp) -> None:
        """Buy or short for one strategy at today's close, sized to capital rules.

        Args:
            order: Pending open (side, multiplier, signal id).
            ts: Fill date.
        """
        key = StrategyKey(order.strategy_key)
        i = self._bar_index(key, ts)
        ## Skip if the strategy already holds a position or has no bar today.
        if key in self.portfolio.positions or i is None:
            return
        strategy = self.strategies[key]
        price = float(strategy._close[i])
        notional = self.account.apply_buying_power(
            order,
            ts,
            self.account.resolve_notional(self.equity(), order.multiplier),
            equity=self.equity(),
            gross_exposure=self.portfolio.gross_exposure,
        )
        if notional <= 0 or price <= 0:
            return
        units = notional / price
        ## One cash formula covers both sides: longs spend cash, shorts receive it.
        self.account.debit(order.side * units * price)
        ## Fees are cash costs on notional; prices stay unslipped for the mark path.
        entry_commission, entry_slippage = self.account.charge_costs(notional)
        self.portfolio.add_position(
            SimPosition(
                strategy_key=key,
                ticker=strategy.ticker,
                side=order.side,
                units=units,
                entry_price=price,
                entry_date=ts,
                signal_id=order.signal_id,
                multiplier=order.multiplier,
                last_price=price,
                entry_index=i,
                last_index=i,
                return_path=[MarkPoint(date=ts, index=i, cum_return_pct=0.0)],
            )
        )
        ## Strategy state first so stops/filters see the open; then our trade book.
        strategy.open_action(index=i, side=order.side, signal_id=order.signal_id, entry_price=price)
        self.trades[key].append(
            TradeRecord(
                strategy_key=key,
                ticker=strategy.ticker,
                side=order.side,
                signal_id=order.signal_id,
                entry_date=ts,
                entry_price=price,
                entry_index=i,
                entry_equity=self.equity() / self.account.initial_capital,
                units=units,
                notional=notional,
                multiplier=order.multiplier,
                entry_commission=entry_commission,
                entry_slippage=entry_slippage,
                entry_info=dict(strategy.additional_on_entry_info(index=i)),
            )
        )

    def fill_close(self, key: StrategyKey, ts: pd.Timestamp, *, reason: str) -> None:
        """Exit one strategy's whole position at the latest mark and book the cash.

        Args:
            key: Strategy key to flatten.
            ts: Close date.
            reason: Why the close happened (signal, finalize, wipeout).
        """
        key = StrategyKey(key)
        pos = self.portfolio.pop_position(key)
        if pos is None:
            return
        exit_notional = pos.units * pos.last_price
        self.account.credit(pos.side * pos.units * pos.last_price)
        exit_commission, exit_slippage = self.account.charge_costs(exit_notional)
        opened = self.trades[key][-1]
        ## Dollar PnL nets fees; return_pct / MAE / MFE stay raw price economics.
        pnl = (
            pos.unrealized_pnl
            - opened.entry_commission
            - opened.entry_slippage
            - exit_commission
            - exit_slippage
        )
        self.portfolio.realized_pnl[key] += pnl
        ## Strategies without a bar today (finalize / wipeout) close on their last marked bar.
        i = self._bar_index(key, ts)
        close_index = i if i is not None else pos.last_index
        entry_notional = pos.units * pos.entry_price
        return_pct = (
            (pos.last_price - pos.entry_price) / pos.entry_price * pos.side if pos.entry_price else 0.0
        )
        extremes = extremes_from_mark_path(pos.return_path, pos.entry_index)
        strategy = self.strategies[key]
        self.trades[key][-1] = replace(
            opened,
            exit_date=ts,
            exit_price=pos.last_price,
            exit_index=close_index,
            exit_equity=self.equity() / self.account.initial_capital,
            return_pct=return_pct,
            pnl=pnl,
            levered_return_pct=(pnl / entry_notional * pos.multiplier) if entry_notional else 0.0,
            close_reason=reason,
            mae=extremes.mae,
            mfe=extremes.mfe,
            days_to_mae=extremes.days_to_mae,
            days_to_mfe=extremes.days_to_mfe,
            mae_date=extremes.mae_date,
            mfe_date=extremes.mfe_date,
            trade_returns=extremes.trade_returns,
            trade_return_dates=extremes.trade_return_dates,
            exit_commission=exit_commission,
            exit_slippage=exit_slippage,
            exit_info=dict(strategy.additional_on_exit_info(index=close_index)),
        )
        ## Match former _record_close_trade order: book the close, then clear strategy state.
        strategy.close_action(index=close_index)

    def close_all(self, ts: pd.Timestamp, *, reason: str) -> None:
        """Force-close every open position (end of run, or after a wipeout)."""
        for key in list(self.portfolio.positions):
            self.fill_close(key, ts, reason=reason)
