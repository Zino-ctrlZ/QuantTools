"""SPY market features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trade.features.compute.registry import feature
from trade.features.compute.transforms import annualized_realized_vol, efficiency_ratio, sanitize_ratio


@feature(scope="market", formula_id="close_return_v1", inputs=("SPY",), horizon=20)
def spy_ret_20d(ctx) -> pd.Series:
    return ctx.close("SPY").pct_change(20)


@feature(scope="market", formula_id="close_return_v1", inputs=("SPY",), horizon=60)
def spy_ret_60d(ctx) -> pd.Series:
    return ctx.close("SPY").pct_change(60)


@feature(scope="market", formula_id="close_return_v1", inputs=("SPY",), horizon=120)
def spy_ret_120d(ctx) -> pd.Series:
    return ctx.close("SPY").pct_change(120)


@feature(scope="market", formula_id="kaufman_er_v1", inputs=("SPY",), window=20)
def spy_kaufman_er_20d(ctx) -> pd.Series:
    return efficiency_ratio(ctx.close("SPY"), 20)


@feature(scope="market", formula_id="kaufman_er_v1", inputs=("SPY",), window=60)
def spy_kaufman_er_60d(ctx) -> pd.Series:
    return efficiency_ratio(ctx.close("SPY"), 60)


@feature(scope="market", formula_id="kaufman_er_v1", inputs=("SPY",), window=120)
def spy_kaufman_er_120d(ctx) -> pd.Series:
    return efficiency_ratio(ctx.close("SPY"), 120)


@feature(scope="market", formula_id="momentum_quality_v1", inputs=("SPY",), horizon=20)
def spy_momentum_quality_20d(ctx) -> pd.Series:
    close = ctx.close("SPY")
    return close.pct_change(20) * efficiency_ratio(close, 20)


@feature(scope="market", formula_id="momentum_quality_v1", inputs=("SPY",), horizon=60)
def spy_momentum_quality_60d(ctx) -> pd.Series:
    close = ctx.close("SPY")
    return close.pct_change(60) * efficiency_ratio(close, 60)


@feature(scope="market", formula_id="momentum_quality_v1", inputs=("SPY",), horizon=120)
def spy_momentum_quality_120d(ctx) -> pd.Series:
    close = ctx.close("SPY")
    return close.pct_change(120) * efficiency_ratio(close, 120)


@feature(scope="market", formula_id="realized_vol_v1", inputs=("SPY",), window=20)
def spy_realized_vol_20d(ctx) -> pd.Series:
    return annualized_realized_vol(ctx.close("SPY"), 20)


@feature(scope="market", formula_id="above_sma_v1", inputs=("SPY",), sma_window=200)
def spy_above_sma_200(ctx) -> pd.Series:
    close = ctx.close("SPY")
    return (close > close.rolling(200).mean()).astype(int)


@feature(scope="market", formula_id="close_vs_sma_v1", inputs=("SPY",), sma_window=200)
def spy_close_vs_sma_200(ctx) -> pd.Series:
    close = ctx.close("SPY")
    return sanitize_ratio(close / close.rolling(200).mean() - 1.0)


@feature(scope="market", formula_id="log_sma_slope_v1", inputs=("SPY",), sma_window=20, horizon=20)
def spy_sma_20_slope_20d(ctx) -> pd.Series:
    sma = ctx.close("SPY").rolling(20).mean()
    return sanitize_ratio(np.log(sma / sma.shift(20)) / 20)


@feature(scope="market", formula_id="log_sma_slope_v1", inputs=("SPY",), sma_window=50, horizon=20)
def spy_sma_50_slope_20d(ctx) -> pd.Series:
    sma = ctx.close("SPY").rolling(50).mean()
    return sanitize_ratio(np.log(sma / sma.shift(20)) / 20)


@feature(scope="market", formula_id="log_sma_slope_v1", inputs=("SPY",), sma_window=200, horizon=20)
def spy_sma_200_slope_20d(ctx) -> pd.Series:
    sma = ctx.close("SPY").rolling(200).mean()
    return sanitize_ratio(np.log(sma / sma.shift(20)) / 20)


@feature(scope="market", formula_id="drawdown_from_high_v1", inputs=("SPY",), window=252)
def spy_drawdown_from_252d_high(ctx) -> pd.Series:
    close = ctx.close("SPY")
    return sanitize_ratio(close / close.rolling(252).max() - 1.0)
