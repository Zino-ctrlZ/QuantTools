"""Hybrid features: asset realized vol vs FXI/EEM IV."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trade.features.compute.registry import feature
from trade.features.compute.transforms import annualized_realized_vol, rolling_zscore, sanitize_ratio


@feature(scope="hybrid", formula_id="asset_iv_rvol_v1", inputs=("close", "FXI_IV"), window=20)
def fxi_iv_to_asset_rvol_20d_ratio(ctx) -> pd.Series:
    rvol = annualized_realized_vol(ctx.asset_close(), 20).replace(0, np.nan)
    return sanitize_ratio(ctx.iv("FXI_IV") / rvol)


@feature(scope="hybrid", formula_id="asset_iv_rvol_v1", inputs=("close", "FXI_IV"), window=20)
def fxi_iv_minus_asset_rvol_20d(ctx) -> pd.Series:
    rvol = annualized_realized_vol(ctx.asset_close(), 20).replace(0, np.nan)
    return sanitize_ratio(ctx.iv("FXI_IV") - rvol)


@feature(scope="hybrid", formula_id="asset_iv_rvol_v1", inputs=("close", "FXI_IV"), window=20, lookback=252)
def fxi_iv_to_asset_rvol_20d_ratio_zscore(ctx) -> pd.Series:
    rvol = annualized_realized_vol(ctx.asset_close(), 20).replace(0, np.nan)
    ratio = sanitize_ratio(ctx.iv("FXI_IV") / rvol)
    return rolling_zscore(ratio, 252)


@feature(scope="hybrid", formula_id="asset_iv_rvol_v1", inputs=("close", "EEM_IV"), window=20)
def eem_iv_to_asset_rvol_20d_ratio(ctx) -> pd.Series:
    rvol = annualized_realized_vol(ctx.asset_close(), 20).replace(0, np.nan)
    return sanitize_ratio(ctx.iv("EEM_IV") / rvol)


@feature(scope="hybrid", formula_id="asset_iv_rvol_v1", inputs=("close", "EEM_IV"), window=20)
def eem_iv_minus_asset_rvol_20d(ctx) -> pd.Series:
    rvol = annualized_realized_vol(ctx.asset_close(), 20).replace(0, np.nan)
    return sanitize_ratio(ctx.iv("EEM_IV") - rvol)


@feature(scope="hybrid", formula_id="asset_iv_rvol_v1", inputs=("close", "EEM_IV"), window=20, lookback=252)
def eem_iv_to_asset_rvol_20d_ratio_zscore(ctx) -> pd.Series:
    rvol = annualized_realized_vol(ctx.asset_close(), 20).replace(0, np.nan)
    ratio = sanitize_ratio(ctx.iv("EEM_IV") / rvol)
    return rolling_zscore(ratio, 252)
