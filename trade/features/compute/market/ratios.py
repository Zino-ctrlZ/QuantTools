"""Cross-asset ratio z-scores and relative momentum.

Vol-adjusted relative momentum is the primary registered catalog. Raw
``relative_momentum`` features remain registered under the deprecated theme
for strategy backward compatibility.
"""

from __future__ import annotations

import pandas as pd

from trade.features.compute.registry import feature
from trade.features.compute.transforms import (
    ratio_zscore,
    relative_momentum,
    vol_adjusted_relative_momentum,
)


@feature(scope="market", formula_id="ratio_zscore_v2", inputs=("XLU", "SPY"), lookback=63)
def xlu_spy_zscore(ctx) -> pd.Series:
    return ratio_zscore(ctx.close("XLU"), ctx.close("SPY"), lookback=126)


@feature(scope="market", formula_id="ratio_zscore_v1", inputs=("XLY", "XLP"), lookback=63)
def xly_xlp_ratio_zscore(ctx) -> pd.Series:
    return ratio_zscore(ctx.close("XLY"), ctx.close("XLP"), lookback=63)


@feature(scope="market", formula_id="ratio_zscore_v1", inputs=("IWM", "SPY"), lookback=63)
def iwm_spy_ratio_zscore(ctx) -> pd.Series:
    return ratio_zscore(ctx.close("IWM"), ctx.close("SPY"), lookback=63)


@feature(scope="market", formula_id="ratio_zscore_v1", inputs=("QQQ", "SPY"), lookback=63)
def qqq_spy_ratio_zscore(ctx) -> pd.Series:
    return ratio_zscore(ctx.close("QQQ"), ctx.close("SPY"), lookback=63)


@feature(scope="market", formula_id="ratio_zscore_v1", inputs=("MTUM", "SPY"), lookback=63)
def mtum_spy_ratio_zscore(ctx) -> pd.Series:
    return ratio_zscore(ctx.close("MTUM"), ctx.close("SPY"), lookback=63)


@feature(scope="market", formula_id="ratio_zscore_v1", inputs=("QUAL", "SPY"), lookback=63)
def qual_spy_ratio_zscore(ctx) -> pd.Series:
    return ratio_zscore(ctx.close("QUAL"), ctx.close("SPY"), lookback=63)


@feature(scope="market", formula_id="ratio_zscore_v1", inputs=("USMV", "SPY"), lookback=63)
def usmv_spy_ratio_zscore(ctx) -> pd.Series:
    return ratio_zscore(ctx.close("USMV"), ctx.close("SPY"), lookback=63)


@feature(scope="market", formula_id="ratio_zscore_v2", inputs=("RSP", "SPY"), lookback=63)
def rsp_spy_ratio_zscore(ctx) -> pd.Series:
    return ratio_zscore(ctx.close("RSP"), ctx.close("SPY"), lookback=126)


@feature(scope="market", formula_id="ratio_zscore_v1", inputs=("EEM", "SPY"), lookback=63)
def eem_spy_ratio_zscore(ctx) -> pd.Series:
    return ratio_zscore(ctx.close("EEM"), ctx.close("SPY"), lookback=63)


@feature(scope="market", formula_id="ratio_zscore_v1", inputs=("FXI", "SPY"), lookback=63)
def fxi_spy_ratio_zscore(ctx) -> pd.Series:
    return ratio_zscore(ctx.close("FXI"), ctx.close("SPY"), lookback=63)


@feature(scope="market", formula_id="ratio_zscore_v1", inputs=("GLD", "SPY"), lookback=63)
def gld_spy_ratio_zscore(ctx) -> pd.Series:
    return ratio_zscore(ctx.close("GLD"), ctx.close("SPY"), lookback=63)


@feature(scope="market", formula_id="ratio_zscore_v1", inputs=("SLV", "SPY"), lookback=63)
def slv_spy_ratio_zscore(ctx) -> pd.Series:
    return ratio_zscore(ctx.close("SLV"), ctx.close("SPY"), lookback=63)


@feature(scope="market", formula_id="ratio_zscore_v1", inputs=("GLD", "SLV"), lookback=63)
def gld_slv_ratio_zscore(ctx) -> pd.Series:
    return ratio_zscore(ctx.close("GLD"), ctx.close("SLV"), lookback=63)


@feature(scope="market", formula_id="ratio_zscore_v1", inputs=("DBC", "SPY"), lookback=63)
def dbc_spy_ratio_zscore(ctx) -> pd.Series:
    return ratio_zscore(ctx.close("DBC"), ctx.close("SPY"), lookback=63)


@feature(scope="market", formula_id="ratio_zscore_v1", inputs=("USO", "SPY"), lookback=63)
def uso_spy_ratio_zscore(ctx) -> pd.Series:
    return ratio_zscore(ctx.close("USO"), ctx.close("SPY"), lookback=63)


@feature(scope="market", formula_id="ratio_zscore_v1", inputs=("UNG", "SPY"), lookback=63)
def ung_spy_ratio_zscore(ctx) -> pd.Series:
    return ratio_zscore(ctx.close("UNG"), ctx.close("SPY"), lookback=63)


@feature(scope="market", formula_id="ratio_zscore_v1", inputs=("CPER", "SPY"), lookback=63)
def cper_spy_ratio_zscore(ctx) -> pd.Series:
    return ratio_zscore(ctx.close("CPER"), ctx.close("SPY"), lookback=63)


@feature(scope="market", formula_id="ratio_zscore_v1", inputs=("DBA", "SPY"), lookback=63)
def dba_spy_ratio_zscore(ctx) -> pd.Series:
    return ratio_zscore(ctx.close("DBA"), ctx.close("SPY"), lookback=63)


@feature(scope="market", formula_id="ratio_zscore_v2", inputs=("XLE", "XLU"), lookback=63)
def xle_xlu_ratio_zscore(ctx) -> pd.Series:
    return ratio_zscore(ctx.close("XLE"), ctx.close("XLU"), lookback=126)


## ---------------------------------------------------------------------------
## Vol-adjusted relative momentum (primary catalog)
## ---------------------------------------------------------------------------


@feature(
    scope="market",
    formula_id="relative_momentum_vol_adj_v1",
    inputs=("XLY", "XLP"),
    horizon=20,
    vol_window=60,
)
def xly_xlp_relative_momentum_vol_adj_20d(ctx) -> pd.Series:
    """XLY vs XLP 20d relative momentum scaled by 60d relative-return vol."""
    return vol_adjusted_relative_momentum(ctx.close("XLY"), ctx.close("XLP"), 20, 60)


@feature(
    scope="market",
    formula_id="relative_momentum_vol_adj_v1",
    inputs=("IWM", "SPY"),
    horizon=20,
    vol_window=60,
)
def iwm_spy_relative_momentum_vol_adj_20d(ctx) -> pd.Series:
    """IWM vs SPY 20d relative momentum scaled by 60d relative-return vol."""
    return vol_adjusted_relative_momentum(ctx.close("IWM"), ctx.close("SPY"), 20, 60)


@feature(
    scope="market",
    formula_id="relative_momentum_vol_adj_v1",
    inputs=("QQQ", "SPY"),
    horizon=20,
    vol_window=60,
)
def qqq_spy_relative_momentum_vol_adj_20d(ctx) -> pd.Series:
    """QQQ vs SPY 20d relative momentum scaled by 60d relative-return vol."""
    return vol_adjusted_relative_momentum(ctx.close("QQQ"), ctx.close("SPY"), 20, 60)


@feature(
    scope="market",
    formula_id="relative_momentum_vol_adj_v1",
    inputs=("MTUM", "SPY"),
    horizon=20,
    vol_window=60,
)
def mtum_spy_relative_momentum_vol_adj_20d(ctx) -> pd.Series:
    """MTUM vs SPY 20d relative momentum scaled by 60d relative-return vol."""
    return vol_adjusted_relative_momentum(ctx.close("MTUM"), ctx.close("SPY"), 20, 60)


@feature(
    scope="market",
    formula_id="relative_momentum_vol_adj_v1",
    inputs=("EEM", "SPY"),
    horizon=20,
    vol_window=60,
)
def eem_spy_relative_momentum_vol_adj_20d(ctx) -> pd.Series:
    """EEM vs SPY 20d relative momentum scaled by 60d relative-return vol."""
    return vol_adjusted_relative_momentum(ctx.close("EEM"), ctx.close("SPY"), 20, 60)


@feature(
    scope="market",
    formula_id="relative_momentum_vol_adj_v1",
    inputs=("FXI", "SPY"),
    horizon=20,
    vol_window=60,
)
def fxi_spy_relative_momentum_vol_adj_20d(ctx) -> pd.Series:
    """FXI vs SPY 20d relative momentum scaled by 60d relative-return vol."""
    return vol_adjusted_relative_momentum(ctx.close("FXI"), ctx.close("SPY"), 20, 60)


@feature(
    scope="market",
    formula_id="relative_momentum_vol_adj_v1",
    inputs=("GLD", "SPY"),
    horizon=20,
    vol_window=60,
)
def gld_spy_relative_momentum_vol_adj_20d(ctx) -> pd.Series:
    """GLD vs SPY 20d relative momentum scaled by 60d relative-return vol."""
    return vol_adjusted_relative_momentum(ctx.close("GLD"), ctx.close("SPY"), 20, 60)


@feature(
    scope="market",
    formula_id="relative_momentum_vol_adj_v1",
    inputs=("SLV", "SPY"),
    horizon=20,
    vol_window=60,
)
def slv_spy_relative_momentum_vol_adj_20d(ctx) -> pd.Series:
    """SLV vs SPY 20d relative momentum scaled by 60d relative-return vol."""
    return vol_adjusted_relative_momentum(ctx.close("SLV"), ctx.close("SPY"), 20, 60)


@feature(
    scope="market",
    formula_id="relative_momentum_vol_adj_v1",
    inputs=("DBC", "SPY"),
    horizon=20,
    vol_window=60,
)
def dbc_spy_relative_momentum_vol_adj_20d(ctx) -> pd.Series:
    """DBC vs SPY 20d relative momentum scaled by 60d relative-return vol."""
    return vol_adjusted_relative_momentum(ctx.close("DBC"), ctx.close("SPY"), 20, 60)


@feature(
    scope="market",
    formula_id="relative_momentum_vol_adj_v1",
    inputs=("USO", "SPY"),
    horizon=20,
    vol_window=60,
)
def uso_spy_relative_momentum_vol_adj_20d(ctx) -> pd.Series:
    """USO vs SPY 20d relative momentum scaled by 60d relative-return vol."""
    return vol_adjusted_relative_momentum(ctx.close("USO"), ctx.close("SPY"), 20, 60)


@feature(
    scope="market",
    formula_id="relative_momentum_vol_adj_v1",
    inputs=("UNG", "SPY"),
    horizon=20,
    vol_window=60,
)
def ung_spy_relative_momentum_vol_adj_20d(ctx) -> pd.Series:
    """UNG vs SPY 20d relative momentum scaled by 60d relative-return vol."""
    return vol_adjusted_relative_momentum(ctx.close("UNG"), ctx.close("SPY"), 20, 60)


@feature(
    scope="market",
    formula_id="relative_momentum_vol_adj_v1",
    inputs=("CPER", "SPY"),
    horizon=20,
    vol_window=60,
)
def cper_spy_relative_momentum_vol_adj_20d(ctx) -> pd.Series:
    """CPER vs SPY 20d relative momentum scaled by 60d relative-return vol."""
    return vol_adjusted_relative_momentum(ctx.close("CPER"), ctx.close("SPY"), 20, 60)


@feature(
    scope="market",
    formula_id="relative_momentum_vol_adj_v1",
    inputs=("DBA", "SPY"),
    horizon=20,
    vol_window=60,
)
def dba_spy_relative_momentum_vol_adj_20d(ctx) -> pd.Series:
    """DBA vs SPY 20d relative momentum scaled by 60d relative-return vol."""
    return vol_adjusted_relative_momentum(ctx.close("DBA"), ctx.close("SPY"), 20, 60)


## ---------------------------------------------------------------------------
## Raw relative momentum (soft-retired; keep for strategy consumers)
## ---------------------------------------------------------------------------


@feature(
    scope="market",
    formula_id="relative_momentum_v1",
    inputs=("XLY", "XLP"),
    horizon=20,
    deprecated=True,
)
def xly_xlp_relative_momentum_20d(ctx) -> pd.Series:
    """Deprecated raw XLY−XLP 20d relative momentum."""
    return relative_momentum(ctx.close("XLY"), ctx.close("XLP"), 20)


@feature(
    scope="market",
    formula_id="relative_momentum_v1",
    inputs=("XLY", "XLP"),
    horizon=60,
    deprecated=True,
)
def xly_xlp_relative_momentum_60d(ctx) -> pd.Series:
    """Deprecated raw XLY−XLP 60d relative momentum."""
    return relative_momentum(ctx.close("XLY"), ctx.close("XLP"), 60)


@feature(
    scope="market",
    formula_id="relative_momentum_v1",
    inputs=("IWM", "SPY"),
    horizon=20,
    deprecated=True,
)
def iwm_spy_relative_momentum_20d(ctx) -> pd.Series:
    """Deprecated raw IWM−SPY 20d relative momentum."""
    return relative_momentum(ctx.close("IWM"), ctx.close("SPY"), 20)


@feature(
    scope="market",
    formula_id="relative_momentum_v1",
    inputs=("IWM", "SPY"),
    horizon=60,
    deprecated=True,
)
def iwm_spy_relative_momentum_60d(ctx) -> pd.Series:
    """Deprecated raw IWM−SPY 60d relative momentum."""
    return relative_momentum(ctx.close("IWM"), ctx.close("SPY"), 60)


@feature(
    scope="market",
    formula_id="relative_momentum_v1",
    inputs=("QQQ", "SPY"),
    horizon=20,
    deprecated=True,
)
def qqq_spy_relative_momentum_20d(ctx) -> pd.Series:
    """Deprecated raw QQQ−SPY 20d relative momentum."""
    return relative_momentum(ctx.close("QQQ"), ctx.close("SPY"), 20)


@feature(
    scope="market",
    formula_id="relative_momentum_v1",
    inputs=("QQQ", "SPY"),
    horizon=60,
    deprecated=True,
)
def qqq_spy_relative_momentum_60d(ctx) -> pd.Series:
    """Deprecated raw QQQ−SPY 60d relative momentum."""
    return relative_momentum(ctx.close("QQQ"), ctx.close("SPY"), 60)


@feature(
    scope="market",
    formula_id="relative_momentum_v1",
    inputs=("MTUM", "SPY"),
    horizon=20,
    deprecated=True,
)
def mtum_spy_relative_momentum_20d(ctx) -> pd.Series:
    """Deprecated raw MTUM−SPY 20d relative momentum."""
    return relative_momentum(ctx.close("MTUM"), ctx.close("SPY"), 20)


@feature(
    scope="market",
    formula_id="relative_momentum_v1",
    inputs=("MTUM", "SPY"),
    horizon=60,
    deprecated=True,
)
def mtum_spy_relative_momentum_60d(ctx) -> pd.Series:
    """Deprecated raw MTUM−SPY 60d relative momentum."""
    return relative_momentum(ctx.close("MTUM"), ctx.close("SPY"), 60)


@feature(
    scope="market",
    formula_id="relative_momentum_v1",
    inputs=("EEM", "SPY"),
    horizon=20,
    deprecated=True,
)
def eem_spy_relative_momentum_20d(ctx) -> pd.Series:
    """Deprecated raw EEM−SPY 20d relative momentum."""
    return relative_momentum(ctx.close("EEM"), ctx.close("SPY"), 20)


@feature(
    scope="market",
    formula_id="relative_momentum_v1",
    inputs=("EEM", "SPY"),
    horizon=60,
    deprecated=True,
)
def eem_spy_relative_momentum_60d(ctx) -> pd.Series:
    """Deprecated raw EEM−SPY 60d relative momentum."""
    return relative_momentum(ctx.close("EEM"), ctx.close("SPY"), 60)


@feature(
    scope="market",
    formula_id="relative_momentum_v1",
    inputs=("FXI", "SPY"),
    horizon=20,
    deprecated=True,
)
def fxi_spy_relative_momentum_20d(ctx) -> pd.Series:
    """Deprecated raw FXI−SPY 20d relative momentum."""
    return relative_momentum(ctx.close("FXI"), ctx.close("SPY"), 20)


@feature(
    scope="market",
    formula_id="relative_momentum_v1",
    inputs=("FXI", "SPY"),
    horizon=60,
    deprecated=True,
)
def fxi_spy_relative_momentum_60d(ctx) -> pd.Series:
    """Deprecated raw FXI−SPY 60d relative momentum."""
    return relative_momentum(ctx.close("FXI"), ctx.close("SPY"), 60)


@feature(
    scope="market",
    formula_id="relative_momentum_v1",
    inputs=("GLD", "SPY"),
    horizon=20,
    deprecated=True,
)
def gld_spy_relative_momentum_20d(ctx) -> pd.Series:
    """Deprecated raw GLD−SPY 20d relative momentum."""
    return relative_momentum(ctx.close("GLD"), ctx.close("SPY"), 20)


@feature(
    scope="market",
    formula_id="relative_momentum_v1",
    inputs=("GLD", "SPY"),
    horizon=60,
    deprecated=True,
)
def gld_spy_relative_momentum_60d(ctx) -> pd.Series:
    """Deprecated raw GLD−SPY 60d relative momentum."""
    return relative_momentum(ctx.close("GLD"), ctx.close("SPY"), 60)


@feature(
    scope="market",
    formula_id="relative_momentum_v1",
    inputs=("SLV", "SPY"),
    horizon=20,
    deprecated=True,
)
def slv_spy_relative_momentum_20d(ctx) -> pd.Series:
    """Deprecated raw SLV−SPY 20d relative momentum."""
    return relative_momentum(ctx.close("SLV"), ctx.close("SPY"), 20)


@feature(
    scope="market",
    formula_id="relative_momentum_v1",
    inputs=("SLV", "SPY"),
    horizon=60,
    deprecated=True,
)
def slv_spy_relative_momentum_60d(ctx) -> pd.Series:
    """Deprecated raw SLV−SPY 60d relative momentum."""
    return relative_momentum(ctx.close("SLV"), ctx.close("SPY"), 60)


@feature(
    scope="market",
    formula_id="relative_momentum_v1",
    inputs=("DBC", "SPY"),
    horizon=20,
    deprecated=True,
)
def dbc_spy_relative_momentum_20d(ctx) -> pd.Series:
    """Deprecated raw DBC−SPY 20d relative momentum."""
    return relative_momentum(ctx.close("DBC"), ctx.close("SPY"), 20)


@feature(
    scope="market",
    formula_id="relative_momentum_v1",
    inputs=("DBC", "SPY"),
    horizon=60,
    deprecated=True,
)
def dbc_spy_relative_momentum_60d(ctx) -> pd.Series:
    """Deprecated raw DBC−SPY 60d relative momentum."""
    return relative_momentum(ctx.close("DBC"), ctx.close("SPY"), 60)


@feature(
    scope="market",
    formula_id="relative_momentum_v1",
    inputs=("USO", "SPY"),
    horizon=20,
    deprecated=True,
)
def uso_spy_relative_momentum_20d(ctx) -> pd.Series:
    """Deprecated raw USO−SPY 20d relative momentum."""
    return relative_momentum(ctx.close("USO"), ctx.close("SPY"), 20)


@feature(
    scope="market",
    formula_id="relative_momentum_v1",
    inputs=("USO", "SPY"),
    horizon=60,
    deprecated=True,
)
def uso_spy_relative_momentum_60d(ctx) -> pd.Series:
    """Deprecated raw USO−SPY 60d relative momentum."""
    return relative_momentum(ctx.close("USO"), ctx.close("SPY"), 60)


@feature(
    scope="market",
    formula_id="relative_momentum_v1",
    inputs=("UNG", "SPY"),
    horizon=20,
    deprecated=True,
)
def ung_spy_relative_momentum_20d(ctx) -> pd.Series:
    """Deprecated raw UNG−SPY 20d relative momentum."""
    return relative_momentum(ctx.close("UNG"), ctx.close("SPY"), 20)


@feature(
    scope="market",
    formula_id="relative_momentum_v1",
    inputs=("UNG", "SPY"),
    horizon=60,
    deprecated=True,
)
def ung_spy_relative_momentum_60d(ctx) -> pd.Series:
    """Deprecated raw UNG−SPY 60d relative momentum."""
    return relative_momentum(ctx.close("UNG"), ctx.close("SPY"), 60)


@feature(
    scope="market",
    formula_id="relative_momentum_v1",
    inputs=("CPER", "SPY"),
    horizon=20,
    deprecated=True,
)
def cper_spy_relative_momentum_20d(ctx) -> pd.Series:
    """Deprecated raw CPER−SPY 20d relative momentum."""
    return relative_momentum(ctx.close("CPER"), ctx.close("SPY"), 20)


@feature(
    scope="market",
    formula_id="relative_momentum_v1",
    inputs=("CPER", "SPY"),
    horizon=60,
    deprecated=True,
)
def cper_spy_relative_momentum_60d(ctx) -> pd.Series:
    """Deprecated raw CPER−SPY 60d relative momentum."""
    return relative_momentum(ctx.close("CPER"), ctx.close("SPY"), 60)


@feature(
    scope="market",
    formula_id="relative_momentum_v1",
    inputs=("DBA", "SPY"),
    horizon=20,
    deprecated=True,
)
def dba_spy_relative_momentum_20d(ctx) -> pd.Series:
    """Deprecated raw DBA−SPY 20d relative momentum."""
    return relative_momentum(ctx.close("DBA"), ctx.close("SPY"), 20)


@feature(
    scope="market",
    formula_id="relative_momentum_v1",
    inputs=("DBA", "SPY"),
    horizon=60,
    deprecated=True,
)
def dba_spy_relative_momentum_60d(ctx) -> pd.Series:
    """Deprecated raw DBA−SPY 60d relative momentum."""
    return relative_momentum(ctx.close("DBA"), ctx.close("SPY"), 60)

