"""Asset downside-path and return-distribution features."""

from __future__ import annotations

import pandas as pd

from trade.features.compute.asset import indicators as ind
from trade.features.compute.registry import feature


@feature(scope="asset", formula_id="asset_ohlcv_v1", inputs=("close",))
def ulcer_index_20d(ctx) -> pd.Series:
    return ind._ulcer_index_20d(ctx.asset_close())


@feature(scope="asset", formula_id="asset_ohlcv_v1", inputs=("close",))
def drawdown_duration_252d(ctx) -> pd.Series:
    return ind._drawdown_duration_252d(ctx.asset_close())


@feature(scope="asset", formula_id="asset_ohlcv_v1", inputs=("close",))
def time_since_60d_high(ctx) -> pd.Series:
    return ind._time_since_60d_extreme(ctx.asset_close(), find_max=True)


@feature(scope="asset", formula_id="asset_ohlcv_v1", inputs=("close",))
def time_since_60d_low(ctx) -> pd.Series:
    return ind._time_since_60d_extreme(ctx.asset_close(), find_max=False)


@feature(scope="asset", formula_id="asset_ohlcv_v1", inputs=("close",))
def realized_skew_60d(ctx) -> pd.Series:
    return ind._sanitize(ctx.asset_close().pct_change().rolling(60, min_periods=60).skew())


@feature(scope="asset", formula_id="asset_ohlcv_v1", inputs=("close",))
def realized_kurtosis_60d(ctx) -> pd.Series:
    return ind._sanitize(ctx.asset_close().pct_change().rolling(60, min_periods=60).kurt())
