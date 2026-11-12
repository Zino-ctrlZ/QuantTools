"""USD/CNH China FX regime features."""

from __future__ import annotations

import pandas as pd

from trade.features.compute.registry import feature
from trade.features.compute.transforms import sanitize_ratio


@feature(scope="market", formula_id="close_return_v1", inputs=("USDCNH",), horizon=20)
def usdcnh_ret_20d(ctx) -> pd.Series:
    return ctx.close("USDCNH").pct_change(20)


@feature(scope="market", formula_id="close_return_v1", inputs=("USDCNH",), horizon=60)
def usdcnh_ret_60d(ctx) -> pd.Series:
    return ctx.close("USDCNH").pct_change(60)


@feature(scope="market", formula_id="close_vs_sma_v1", inputs=("USDCNH",), sma_window=200)
def usdcnh_close_vs_sma_200(ctx) -> pd.Series:
    close = ctx.close("USDCNH")
    return sanitize_ratio(close / close.rolling(200).mean() - 1.0)
