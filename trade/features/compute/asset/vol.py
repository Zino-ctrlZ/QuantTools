"""Asset volatility / ATR / high-distance features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trade.backtester_.indicators import create_atr_dataframe
from trade.features.compute.registry import feature
from trade.features.compute.transforms import annualized_realized_vol


@feature(scope="asset", formula_id="asset_realized_vol_v1", inputs=("close",), window=20)
def realized_vol_20d(ctx) -> pd.Series:
    return annualized_realized_vol(ctx.asset_close(), 20)


@feature(scope="asset", formula_id="asset_vol_ratio_v1", inputs=("close",), short_window=20, long_window=126)
def vol_ratio_20d_vs_126d(ctx) -> pd.Series:
    daily = ctx.asset_close().pct_change()
    short = daily.rolling(20).std() * np.sqrt(252)
    long = daily.rolling(126).std() * np.sqrt(252)
    return short / long.replace(0, np.nan)


@feature(scope="asset", formula_id="asset_atr_pct_v1", inputs=("ohlcv",), window=20)
def atr_20d_pct(ctx) -> pd.Series:
    atr = create_atr_dataframe(ctx.ohlcv, length=20)["atr"]
    return atr / ctx.asset_close()


@feature(scope="asset", formula_id="asset_dist_from_high_v1", inputs=("close",), window=60)
def dist_from_60d_high(ctx) -> pd.Series:
    close = ctx.asset_close()
    return close / close.rolling(60).max() - 1.0


@feature(scope="asset", formula_id="asset_drawdown_from_high_v1", inputs=("close",), window=252)
def drawdown_from_252d_high(ctx) -> pd.Series:
    close = ctx.asset_close()
    return close / close.rolling(252).max() - 1.0


@feature(scope="asset", formula_id="asset_vs_spy_ret_v1", inputs=("close", "SPY"), horizon=60)
def asset_vs_spy_ret_60d(ctx) -> pd.Series:
    close = ctx.asset_close()
    spy = ctx.close("SPY").reindex(close.index)
    return (close / close.shift(60)) / (spy / spy.shift(60)) - 1.0
