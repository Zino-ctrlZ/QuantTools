"""FRED implied-vol index features (EEM / FXI)."""

from __future__ import annotations

import pandas as pd

from trade.features.compute.registry import feature
from trade.features.compute.transforms import rolling_pctile, rolling_zscore


@feature(scope="market", formula_id="iv_level_v1", inputs=("EEM_IV",))
def eem_iv_level(ctx) -> pd.Series:
    return ctx.iv("EEM_IV")


@feature(scope="market", formula_id="iv_zscore_v1", inputs=("EEM_IV",), lookback=252)
def eem_iv_zscore(ctx) -> pd.Series:
    return rolling_zscore(ctx.iv("EEM_IV"), 252)


@feature(scope="market", formula_id="iv_percentile_v1", inputs=("EEM_IV",), lookback=252)
def eem_iv_percentile_252d(ctx) -> pd.Series:
    return rolling_pctile(ctx.iv("EEM_IV"), 252)


@feature(scope="market", formula_id="iv_level_v1", inputs=("FXI_IV",), deprecated=True)
def fxi_iv_level(ctx) -> pd.Series:
    return ctx.iv("FXI_IV")


@feature(scope="market", formula_id="iv_zscore_v1", inputs=("FXI_IV",), lookback=252, deprecated=True)
def fxi_iv_zscore(ctx) -> pd.Series:
    return rolling_zscore(ctx.iv("FXI_IV"), 252)


@feature(
    scope="market",
    formula_id="iv_percentile_v1",
    inputs=("FXI_IV",),
    lookback=252,
    deprecated=True,
)
def fxi_iv_percentile_252d(ctx) -> pd.Series:
    return rolling_pctile(ctx.iv("FXI_IV"), 252)
