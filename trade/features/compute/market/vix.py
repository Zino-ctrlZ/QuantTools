"""VIX market features."""

from __future__ import annotations

import pandas as pd

from trade.features.compute.registry import feature
from trade.features.compute.transforms import annualized_realized_vol, rolling_pctile, sanitize_ratio


@feature(scope="market", formula_id="level_v1", inputs=("VIX",))
def vix_level(ctx) -> pd.Series:
    return ctx.close("VIX")


@feature(scope="market", formula_id="above_sma_bool_v1", inputs=("VIX",), sma_window=200)
def vix_above_200ma(ctx) -> pd.Series:
    close = ctx.close("VIX")
    return (close > close.rolling(200).mean()).astype(bool)


@feature(scope="market", formula_id="close_vs_sma_v1", inputs=("VIX",), sma_window=200)
def vix_pct_200ma(ctx) -> pd.Series:
    close = ctx.close("VIX")
    return sanitize_ratio(close / close.rolling(200).mean() - 1.0)


@feature(scope="market", formula_id="close_return_v1", inputs=("VIX",), horizon=20)
def vix_20_day_return(ctx) -> pd.Series:
    return ctx.close("VIX").pct_change(20)


@feature(scope="market", formula_id="rolling_pctile_v1", inputs=("VIX",), lookback=252)
def vix_pctile_252(ctx) -> pd.Series:
    return rolling_pctile(ctx.close("VIX"), 252)


@feature(scope="market", formula_id="vix_vs_spy_rvol_v1", inputs=("VIX", "SPY"), window=20)
def vix_spy_rvol_20_ratio(ctx) -> pd.Series:
    spy_rvol = annualized_realized_vol(ctx.close("SPY"), 20)
    return sanitize_ratio((ctx.close("VIX") / 100.0) / spy_rvol.replace(0, float("nan")))


@feature(scope="market", formula_id="close_ratio_minus_one_v1", inputs=("VIX", "VIX3M"))
def vix_vs_vix3m(ctx) -> pd.Series:
    return sanitize_ratio(ctx.close("VIX") / ctx.close("VIX3M").replace(0, float("nan")) - 1.0)
