"""UUP dollar-strength regime features."""

from __future__ import annotations

import pandas as pd

from trade.features.compute.registry import feature
from trade.features.compute.transforms import annualized_realized_vol, sanitize_ratio


@feature(scope="market", formula_id="close_return_v1", inputs=("UUP",), horizon=20)
def uup_ret_20d(ctx) -> pd.Series:
    return ctx.close("UUP").pct_change(20)


@feature(scope="market", formula_id="close_return_v1", inputs=("UUP",), horizon=60)
def uup_ret_60d(ctx) -> pd.Series:
    return ctx.close("UUP").pct_change(60)


@feature(scope="market", formula_id="close_vs_sma_v1", inputs=("UUP",), sma_window=200)
def uup_close_vs_sma_200(ctx) -> pd.Series:
    close = ctx.close("UUP")
    return sanitize_ratio(close / close.rolling(200).mean() - 1.0)


@feature(scope="market", formula_id="sma_ratio_slope_v1", inputs=("UUP",), sma_window=50, horizon=20)
def uup_sma_50_slope_20d(ctx) -> pd.Series:
    sma = ctx.close("UUP").rolling(50).mean()
    return sanitize_ratio(sma / sma.shift(20) - 1.0)


@feature(scope="market", formula_id="realized_vol_v1", inputs=("UUP",), window=20)
def uup_realized_vol_20d(ctx) -> pd.Series:
    return annualized_realized_vol(ctx.close("UUP"), 20)


@feature(scope="market", formula_id="drawdown_from_high_v1", inputs=("UUP",), window=252)
def uup_drawdown_from_252d_high(ctx) -> pd.Series:
    close = ctx.close("UUP")
    return sanitize_ratio(close / close.rolling(252).max() - 1.0)
