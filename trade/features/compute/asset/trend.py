"""Asset trend-position features (SMA distance / slopes / flags)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trade.features.compute.registry import feature


@feature(scope="asset", formula_id="asset_close_vs_sma_v1", inputs=("close",), sma_window=50)
def close_vs_sma_50(ctx) -> pd.Series:
    close = ctx.asset_close()
    return close / close.rolling(50).mean() - 1.0


@feature(scope="asset", formula_id="asset_close_vs_sma_v1", inputs=("close",), sma_window=200)
def close_vs_sma_200(ctx) -> pd.Series:
    close = ctx.asset_close()
    return close / close.rolling(200).mean() - 1.0


@feature(scope="asset", formula_id="asset_above_sma_v1", inputs=("close",), sma_window=200)
def above_sma_200(ctx) -> pd.Series:
    close = ctx.asset_close()
    return (close > close.rolling(200).mean()).astype(int)


@feature(scope="asset", formula_id="asset_log_sma_slope_v1", inputs=("close",), sma_window=20, slope_horizon=20)
def sma_20_slope_20d(ctx) -> pd.Series:
    sma = ctx.asset_close().rolling(20).mean()
    return np.log(sma / sma.shift(20)) / 20


@feature(scope="asset", formula_id="asset_log_sma_slope_v1", inputs=("close",), sma_window=50, slope_horizon=20)
def sma_50_slope_20d(ctx) -> pd.Series:
    sma = ctx.asset_close().rolling(50).mean()
    return np.log(sma / sma.shift(20)) / 20


@feature(scope="asset", formula_id="asset_log_sma_slope_v1", inputs=("close",), sma_window=200, slope_horizon=20)
def sma_200_slope_20d(ctx) -> pd.Series:
    sma = ctx.asset_close().rolling(200).mean()
    return np.log(sma / sma.shift(20)) / 20
