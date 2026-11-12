"""Asset path-structure features (ADX / Aroon / choppiness)."""

from __future__ import annotations

import pandas as pd

from trade.features.compute.asset import indicators as ind
from trade.features.compute.registry import feature


@feature(scope="asset", formula_id="asset_ohlcv_v1", inputs=("ohlcv",))
def adx_20d(ctx) -> pd.Series:
    return ind._adx_20d(ctx.asset_col("high"), ctx.asset_col("low"), ctx.asset_close())


@feature(scope="asset", formula_id="asset_ohlcv_v1", inputs=("ohlcv",))
def aroon_oscillator_20d(ctx) -> pd.Series:
    return ind._aroon_oscillator_20d(ctx.asset_col("high"), ctx.asset_col("low"))


@feature(scope="asset", formula_id="asset_ohlcv_v1", inputs=("ohlcv",))
def choppiness_index_20d(ctx) -> pd.Series:
    return ind._choppiness_index_20d(ctx.asset_col("high"), ctx.asset_col("low"), ctx.asset_close())
