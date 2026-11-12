"""Asset volume-participation features."""

from __future__ import annotations

import pandas as pd

from trade.features.compute.asset import indicators as ind
from trade.features.compute.registry import feature


@feature(scope="asset", formula_id="asset_ohlcv_v1", inputs=("ohlcv",))
def relative_volume_20d(ctx) -> pd.Series:
    return ind._relative_volume_20d(ctx.asset_col("volume"))


@feature(scope="asset", formula_id="asset_ohlcv_v1", inputs=("ohlcv",))
def volume_percentile_252d(ctx) -> pd.Series:
    return ind._volume_percentile_vs_prior(ctx.asset_col("volume"))


@feature(scope="asset", formula_id="asset_ohlcv_v1", inputs=("ohlcv",))
def dollar_volume_20d(ctx) -> pd.Series:
    return ind._dollar_volume_20d(ctx.asset_close(), ctx.asset_col("volume"))


@feature(scope="asset", formula_id="asset_ohlcv_v1", inputs=("ohlcv",))
def dollar_volume_change_20d(ctx) -> pd.Series:
    return ind._dollar_volume_change_20d(ctx.asset_close(), ctx.asset_col("volume"))


@feature(scope="asset", formula_id="asset_ohlcv_v1", inputs=("ohlcv",))
def signed_volume_pressure_20d(ctx) -> pd.Series:
    return ind._signed_volume_pressure_20d(ctx.asset_close(), ctx.asset_col("volume"))


@feature(scope="asset", formula_id="asset_ohlcv_v1", inputs=("ohlcv",))
def chaikin_money_flow_20d(ctx) -> pd.Series:
    return ind._chaikin_money_flow_20d(
        ctx.asset_col("high"), ctx.asset_col("low"), ctx.asset_close(), ctx.asset_col("volume")
    )


@feature(scope="asset", formula_id="asset_ohlcv_v1", inputs=("ohlcv",))
def price_volume_correlation_60d(ctx) -> pd.Series:
    return ind._price_volume_correlation_60d(ctx.asset_close(), ctx.asset_col("volume"))
