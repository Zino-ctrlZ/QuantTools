"""Vendor symbol maps and shared lookback defaults for feature inputs."""

from __future__ import annotations

from typing import Dict

import pandas as pd

## Full-history lower bound for vendor fetches / feature panels.
HISTORY_START: pd.Timestamp = pd.Timestamp("1970-01-01")

## Logical market key → vendor ticker (yfinance / MarketTimeseries).
MARKET_SYMBOLS: Dict[str, str] = {
    "SPY": "SPY",
    "VIX": "^VIX",
    "VIX3M": "^VIX3M",
    "XLU": "XLU",
    "XLY": "XLY",
    "XLP": "XLP",
    "XLE": "XLE",
    "IWM": "IWM",
    "QQQ": "QQQ",
    "MTUM": "MTUM",
    "QUAL": "QUAL",
    "USMV": "USMV",
    "RSP": "RSP",
    "EEM": "EEM",
    "FXI": "FXI",
    "GLD": "GLD",
    "SLV": "SLV",
    "DBC": "DBC",
    "USO": "USO",
    "UNG": "UNG",
    "CPER": "CPER",
    "DBA": "DBA",
    ## Positive USDCNH returns ⇒ yuan depreciation (more CNY per USD).
    "USDCNH": "CNY=X",
    "UUP": "UUP",
}

## Logical IV key → FRED series id (percentage-point index levels).
FRED_IV_SERIES: Dict[str, str] = {
    "EEM_IV": "VXEEMCLS",
    "FXI_IV": "VXFXICLS",
}

## Default rolling windows used by market formulas.
RATIO_ZSCORE_LOOKBACK: int = 63
XLU_SPY_ZSCORE_LOOKBACK: int = RATIO_ZSCORE_LOOKBACK  # legacy alias
IV_ZSCORE_LOOKBACK: int = 252
IV_PERCENTILE_LOOKBACK: int = 252
IV_MAX_STALENESS_DAYS: int = 5
