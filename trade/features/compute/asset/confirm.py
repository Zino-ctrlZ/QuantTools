"""Asset signal-confirmation features."""

from __future__ import annotations

import pandas as pd

from trade.features.compute.asset import indicators as ind
from trade.features.compute.registry import feature


@feature(scope="asset", formula_id="asset_ohlcv_v1", inputs=("ohlcv",))
def relative_volume_1d(ctx) -> pd.Series:
    return ind._relative_volume_1d(ctx.asset_col("volume"))


@feature(scope="asset", formula_id="asset_ohlcv_v1", inputs=("ohlcv",))
def close_location_value_1d(ctx) -> pd.Series:
    return ind._close_location_value_1d(ctx.asset_col("high"), ctx.asset_col("low"), ctx.asset_close())


@feature(scope="asset", formula_id="asset_ohlcv_v1", inputs=("ohlcv",))
def directional_close_location_1d(ctx) -> pd.Series:
    return ind._directional_close_location_1d(
        ctx.asset_col("high"), ctx.asset_col("low"), ctx.asset_close()
    )


@feature(scope="asset", formula_id="asset_ohlcv_v1", inputs=("ohlcv",))
def true_range_vs_atr_20d(ctx) -> pd.Series:
    return ind._true_range_vs_atr_20d(ctx.asset_col("high"), ctx.asset_col("low"), ctx.asset_close())


@feature(scope="asset", formula_id="asset_ohlcv_v1", inputs=("ohlcv",), deprecated=True)
def signed_volume_1d(ctx) -> pd.Series:
    return ind._signed_volume_1d(ctx.asset_close(), ctx.asset_col("volume"))


@feature(scope="asset", formula_id="asset_ohlcv_v1", inputs=("ohlcv",))
def directional_volume_confirmation_1d(ctx) -> pd.Series:
    return ind._directional_volume_confirmation_1d(ctx.asset_close(), ctx.asset_col("volume"))


@feature(scope="asset", formula_id="asset_ohlcv_v1", inputs=("close",))
def followthrough_return_1d(ctx) -> pd.Series:
    return ind._followthrough_return_1d(ctx.asset_close())


@feature(scope="asset", formula_id="asset_ohlcv_v1", inputs=("close",))
def followthrough_return_3d(ctx) -> pd.Series:
    return ind._followthrough_return_3d(ctx.asset_close())
