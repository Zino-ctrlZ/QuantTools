"""Asset range / compression features."""

from __future__ import annotations

import pandas as pd

from trade.features.compute.asset import indicators as ind
from trade.features.compute.registry import feature


@feature(scope="asset", formula_id="asset_ohlcv_v1", inputs=("ohlcv",))
def bollinger_band_width_20d(ctx) -> pd.Series:
    return ind._bollinger_band_width_20d(ctx.asset_close())


@feature(scope="asset", formula_id="asset_ohlcv_v1", inputs=("ohlcv",))
def donchian_width_20d(ctx) -> pd.Series:
    return ind._donchian_width_20d(ctx.asset_col("high"), ctx.asset_col("low"), ctx.asset_close())


@feature(scope="asset", formula_id="asset_ohlcv_v1", inputs=("ohlcv",))
def range_position_20d(ctx) -> pd.Series:
    return ind._range_position_20d(ctx.asset_col("high"), ctx.asset_col("low"), ctx.asset_close())


@feature(scope="asset", formula_id="asset_ohlcv_v1", inputs=("ohlcv",))
def range_compression_percentile_126d(ctx) -> pd.Series:
    width = ind._donchian_width_20d(ctx.asset_col("high"), ctx.asset_col("low"), ctx.asset_close())
    return ind._range_compression_percentile_126d(width)
