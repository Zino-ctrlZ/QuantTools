"""Asset 126d rolling regime-context features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trade.features.compute.registry import feature
from trade.features.compute.transforms import down_day_fraction, efficiency_ratio

_WINDOW = 126


@feature(
    scope="asset",
    formula_id="asset_rolling_context_v1",
    inputs=("close",),
    vol_window=20,
    smooth_window=126,
    deprecated=True,
)
def rolling_avg_20d_vol_126d(ctx) -> pd.Series:
    vol = ctx.asset_close().pct_change().rolling(20).std() * np.sqrt(252)
    return vol.rolling(_WINDOW).mean()


@feature(
    scope="asset",
    formula_id="asset_rolling_context_v1",
    inputs=("close",),
    ret_horizon=20,
    smooth_window=126,
    deprecated=True,
)
def rolling_avg_20d_ret_126d(ctx) -> pd.Series:
    return ctx.asset_close().pct_change(20).rolling(_WINDOW).mean()


@feature(scope="asset", formula_id="asset_rolling_context_v1", inputs=("close",), ret_horizon=20, smooth_window=126)
def avg_decline_20d_126d(ctx) -> pd.Series:
    ret = ctx.asset_close().pct_change(20)
    return ret.where(ret < 0).rolling(_WINDOW, min_periods=20).mean()


@feature(scope="asset", formula_id="asset_rolling_context_v1", inputs=("close",), ret_horizon=20, smooth_window=126)
def decline_freq_20d_126d(ctx) -> pd.Series:
    ret = ctx.asset_close().pct_change(20)
    return (ret < 0).rolling(_WINDOW, min_periods=20).mean()


@feature(
    scope="asset",
    formula_id="asset_rolling_context_v1",
    inputs=("close",),
    ret_horizon=20,
    smooth_window=126,
    deprecated=True,
)
def count_large_20d_declines_126d(ctx) -> pd.Series:
    ret = ctx.asset_close().pct_change(20)
    return (ret < -0.10).rolling(_WINDOW).sum()


@feature(scope="asset", formula_id="asset_down_day_frac_v1", inputs=("close",), window=10)
def down_day_frac_10d(ctx) -> pd.Series:
    """Fraction of negative daily closes over the prior 10 sessions."""
    return down_day_fraction(ctx.asset_close(), 10)


@feature(scope="asset", formula_id="asset_rolling_context_v1", inputs=("close",), prior_horizon=10, smooth_window=126)
def p_down_given_down_10d_126d(ctx) -> pd.Series:
    down = ctx.asset_close().pct_change() < 0
    prior = down.shift(10)
    joint = (down & prior).rolling(_WINDOW).sum()
    conditioning = prior.rolling(_WINDOW).sum()
    return joint / conditioning.replace(0, np.nan)


@feature(scope="asset", formula_id="asset_rolling_context_v1", inputs=("close",), window=50, smooth_window=126)
def rolling_er_50_126d(ctx) -> pd.Series:
    return efficiency_ratio(ctx.asset_close(), 50).rolling(_WINDOW).mean()
