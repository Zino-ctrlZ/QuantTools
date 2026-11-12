"""Asset overnight gap-behavior features."""

from __future__ import annotations

import pandas as pd

from trade.features.compute.asset import indicators as ind
from trade.features.compute.registry import feature


def _gap(ctx) -> pd.Series:
    return ind._overnight_gap_1d(ctx.asset_col("open"), ctx.asset_close())


@feature(scope="asset", formula_id="asset_ohlcv_v1", inputs=("ohlcv",))
def overnight_gap_1d(ctx) -> pd.Series:
    return _gap(ctx)


@feature(scope="asset", formula_id="asset_ohlcv_v1", inputs=("ohlcv",))
def absolute_overnight_gap_1d(ctx) -> pd.Series:
    return ind._absolute_overnight_gap_1d(_gap(ctx))


@feature(scope="asset", formula_id="asset_ohlcv_v1", inputs=("ohlcv",))
def avg_abs_overnight_gap_20d(ctx) -> pd.Series:
    return ind._avg_abs_overnight_gap_20d(_gap(ctx))


@feature(scope="asset", formula_id="asset_ohlcv_v1", inputs=("ohlcv",))
def overnight_gap_volatility_20d(ctx) -> pd.Series:
    return ind._overnight_gap_volatility_20d(_gap(ctx))


@feature(scope="asset", formula_id="asset_ohlcv_v1", inputs=("ohlcv",))
def negative_gap_frequency_60d(ctx) -> pd.Series:
    return ind._gap_frequency_60d(_gap(ctx), negative=True)


@feature(scope="asset", formula_id="asset_ohlcv_v1", inputs=("ohlcv",))
def positive_gap_frequency_60d(ctx) -> pd.Series:
    return ind._gap_frequency_60d(_gap(ctx), negative=False)


@feature(scope="asset", formula_id="asset_ohlcv_v1", inputs=("ohlcv",))
def gap_vs_atr_20d(ctx) -> pd.Series:
    return ind._gap_vs_atr_20d(
        ctx.asset_col("open"), ctx.asset_col("high"), ctx.asset_col("low"), ctx.asset_close()
    )


@feature(scope="asset", formula_id="asset_ohlcv_v1", inputs=("ohlcv",))
def overnight_return_share_20d(ctx) -> pd.Series:
    return ind._overnight_return_share_20d(ctx.asset_col("open"), ctx.asset_close())
