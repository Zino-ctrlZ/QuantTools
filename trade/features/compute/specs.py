"""Durable registry storage for feature specs and starter packs.

``FEATURE_SPECS`` lives in this module on purpose: Jupyter / ``%autoreload``
often re-executes ``registry.py`` and family modules, but ``load_families``
is gated after the first import. Keeping the dict here preserves registered
rows across those reloads. ``Feature`` sits beside the dict so instance and
class identity stay aligned when only the decorator module reloads.

Comment density: domain policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Mapping, Tuple

import pandas as pd

from trade.features.compute.context import FeatureContext
from trade.features.themes import FeatureTheme

ComputeFn = Callable[[FeatureContext], pd.Series]


@dataclass(frozen=True, slots=True)
class Feature:
    """One named feature: metadata plus the compute callable.

    Attributes:
        name: Public feature name (function name / cache key / caller API).
        compute: ``(FeatureContext) -> Series`` implementing the formula.
        inputs: Declared inputs (``SPY``, ``close``, ``FXI_IV``, ``ohlcv``, …).
        formula_id: Cache-bust label; bump when the math changes.
        scope: ``market`` | ``hybrid`` | ``asset``.
        theme: Research theme from ``FEATURE_THEMES`` (``unmapped`` /
            ``deprecated`` sentinels allowed).
        deprecated: True when the feature lives under the deprecated theme
            (soft-retired; excluded from default catalog pulls).
        params: Optional knobs recorded for introspection (horizon, lookback, …).
    """

    name: str
    compute: ComputeFn
    inputs: Tuple[str, ...]
    formula_id: str
    scope: str
    theme: FeatureTheme
    deprecated: bool = False
    params: Mapping[str, int] = field(default_factory=dict)


## Populated by ``@feature`` when family modules are imported.
## Do not rebind this name elsewhere — mutate in place only.
FEATURE_SPECS: Dict[str, Feature] = {}

## Set by ``load_families``; lives here so a reload of ``families.py`` does
## not forget that registrations already landed in ``FEATURE_SPECS``.
FAMILIES_LOADED: bool = False

BASE_ASSET_FEATURE_SET = [
    ## Asset-level starter pack
    "ret_20d",
    "close_vs_sma_200",
    "above_sma_200",
    "adx_20d",
    "kaufman_er_20d",
    "aroon_oscillator_20d",
    "realized_vol_20d",
    "vol_ratio_20d_vs_126d",
    "atr_20d_pct",
    "range_position_20d",
    "relative_volume_20d",
    "drawdown_from_252d_high",
    "asset_vs_spy_ret_60d",
    "p_down_given_down_10d_126d",
]

BASE_MARKET_FEATURE_SET = [
    "spy_above_sma_200",
    "spy_ret_20d",
    "vix_level",
    "vix_pctile_252",
    "vix_vs_vix3m",
    "vix_spy_rvol_20_ratio",

    # Equity and defensive rotation
    "xly_xlp_relative_momentum_20d",
    "iwm_spy_relative_momentum_20d",
    "qqq_spy_relative_momentum_20d",
    "rsp_spy_ratio_zscore",
    "xle_xlu_ratio_zscore",
    "xlu_spy_zscore",
    "eem_spy_relative_momentum_20d",
    
    # Commodity rotation
    "gld_spy_relative_momentum_20d",
    "dbc_spy_relative_momentum_20d",
    
    # Dollar regime
    "uup_ret_20d",
    "uup_close_vs_sma_200",
]
