"""Asset momentum / Kaufman ER / momentum-quality features."""

from __future__ import annotations

import pandas as pd

from trade.features.compute.registry import feature
from trade.features.compute.transforms import (
    efficiency_ratio,
    return_to_vol,
    vol_adjusted_return,
)


@feature(scope="asset", formula_id="asset_close_return_v1", inputs=("close",), horizon=20)
def ret_20d(ctx) -> pd.Series:
    return ctx.asset_close().pct_change(20)


@feature(
    scope="asset",
    formula_id="return_vol_adj_v1",
    inputs=("close",),
    horizon=20,
    vol_window=60,
)
def ret_vol_adj_20d(ctx) -> pd.Series:
    """Asset 20d return scaled by 60d daily-return vol."""
    return vol_adjusted_return(ctx.asset_close(), 20, 60)


@feature(scope="asset", formula_id="asset_close_return_v1", inputs=("close",), horizon=60)
def ret_60d(ctx) -> pd.Series:
    return ctx.asset_close().pct_change(60)


@feature(scope="asset", formula_id="asset_close_return_v1", inputs=("close",), horizon=120)
def ret_120d(ctx) -> pd.Series:
    return ctx.asset_close().pct_change(120)


@feature(scope="asset", formula_id="asset_kaufman_er_v1", inputs=("close",), window=20)
def kaufman_er_20d(ctx) -> pd.Series:
    return efficiency_ratio(ctx.asset_close(), 20)


@feature(scope="asset", formula_id="asset_kaufman_er_v1", inputs=("close",), window=60)
def kaufman_er_60d(ctx) -> pd.Series:
    return efficiency_ratio(ctx.asset_close(), 60)


@feature(scope="asset", formula_id="asset_kaufman_er_v1", inputs=("close",), window=120)
def kaufman_er_120d(ctx) -> pd.Series:
    return efficiency_ratio(ctx.asset_close(), 120)


@feature(scope="asset", formula_id="asset_momentum_quality_v1", inputs=("close",), horizon=20)
def momentum_quality_20d(ctx) -> pd.Series:
    close = ctx.asset_close()
    return close.pct_change(20) * efficiency_ratio(close, 20)


@feature(scope="asset", formula_id="asset_momentum_quality_v1", inputs=("close",), horizon=60)
def momentum_quality_60d(ctx) -> pd.Series:
    close = ctx.asset_close()
    return close.pct_change(60) * efficiency_ratio(close, 60)


@feature(scope="asset", formula_id="asset_momentum_quality_v1", inputs=("close",), horizon=120)
def momentum_quality_120d(ctx) -> pd.Series:
    close = ctx.asset_close()
    return close.pct_change(120) * efficiency_ratio(close, 120)


@feature(scope="asset", formula_id="asset_return_to_vol_v1", inputs=("close",), horizon=5)
def return_to_vol_5d(ctx) -> pd.Series:
    """5-day close return divided by 5-day realized volatility."""
    return return_to_vol(ctx.asset_close(), 5)


@feature(scope="asset", formula_id="asset_return_to_vol_v1", inputs=("close",), horizon=60)
def return_to_vol_60d(ctx) -> pd.Series:
    """60-day close return divided by 60-day realized volatility."""
    return return_to_vol(ctx.asset_close(), 60)
