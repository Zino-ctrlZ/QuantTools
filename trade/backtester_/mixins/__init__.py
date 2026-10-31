"""Strategy mixins: optional extensions plus the strategy and book foundations.

Core Classes:
    MissedExitRecoveryMixin: Optional live missed-exit recovery for strategies.
    ParamContractMixin: ``bt_params`` contract shared by StrategyBase and books.
    MarketDataMixin: Data, ``_resolve``, indicators and rules for StrategyBase.
"""

from trade.backtester_.mixins.contract import ParamContractMixin
from trade.backtester_.mixins.market_data import Bar, Indicator, MarketDataMixin, Rule, RuleFilter
from trade.backtester_.mixins.missed_exit_recovery import MissedExitRecoveryMixin

__all__ = [
    "Bar",
    "Indicator",
    "MarketDataMixin",
    "MissedExitRecoveryMixin",
    "ParamContractMixin",
    "Rule",
    "RuleFilter",
]
