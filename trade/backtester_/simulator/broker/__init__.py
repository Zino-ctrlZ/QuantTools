"""Shared-cash broker: account, portfolio, and fill execution.

Core Classes:
    Broker: Façade the simulator day loop talks to.
    Account: Cash and capital limits.
    Portfolio: Positions and marks.
    ExecutionEngine: Open/close fills.
"""

from trade.backtester_.simulator.broker.account import Account
from trade.backtester_.simulator.broker.broker import Broker
from trade.backtester_.simulator.broker.execution import ExecutionEngine
from trade.backtester_.simulator.broker.portfolio import Portfolio

__all__ = ["Account", "Broker", "ExecutionEngine", "Portfolio"]
