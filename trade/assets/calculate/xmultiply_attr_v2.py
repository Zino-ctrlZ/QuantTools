"""Compatibility re-export of the TimeseriesDataManager PnL attribution loader.

Prefer ``trade.assets.calculate.xmultiply_attr``. This module exists so existing
``xmultiply_attr_v2`` imports keep working during the cutover; delete after callers
import the canonical module only.
"""

from trade.assets.calculate.data_classes import OptionPnlPayload
from trade.assets.calculate.xmultiply_attr import (
    add_dod_change,
    calculate_pnl_decomposition,
    get_symbol_timeseries,
    load_option_pnl_data,
    load_rate_payload,
    load_symbol_payload,
)

__all__ = [
    "OptionPnlPayload",
    "add_dod_change",
    "calculate_pnl_decomposition",
    "get_symbol_timeseries",
    "load_option_pnl_data",
    "load_rate_payload",
    "load_symbol_payload",
]
