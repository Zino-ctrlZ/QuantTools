"""Derived feature-name tuples over the ``@feature`` registry.

Formulas live in ``market/``, ``asset/``, and ``hybrid/`` family modules.
This module only exposes convenient name lists after families are loaded.
"""

from __future__ import annotations

from typing import Tuple

from trade.features.compute.families import load_families
from trade.features.compute.registry import FEATURE_SPECS, Feature, FeatureSpec, features_for_scope
from trade.features.themes import FeatureTheme

load_families()

MARKET_SETUP_FEATURES: Tuple[str, ...] = features_for_scope("market")
ASSET_SETUP_FEATURES: Tuple[str, ...] = features_for_scope("asset")
ASSET_IV_RVOL_FEATURES: Tuple[str, ...] = features_for_scope("hybrid")

IV_MARKET_FEATURES: Tuple[str, ...] = tuple(
    name
    for name in (
        "eem_iv_level",
        "fxi_iv_level",
        "eem_iv_zscore",
        "fxi_iv_zscore",
        "eem_iv_percentile_252d",
        "fxi_iv_percentile_252d",
    )
    if name in FEATURE_SPECS
)

CHINA_MARKET_FEATURES: Tuple[str, ...] = tuple(
    name
    for name, spec in FEATURE_SPECS.items()
    if spec.theme == FeatureTheme.CHINA_MARKET_REGIME and spec.scope == "market"
)
DOLLAR_REGIME_FEATURES: Tuple[str, ...] = tuple(
    name
    for name, spec in FEATURE_SPECS.items()
    if spec.theme == FeatureTheme.DOLLAR_REGIME and spec.scope == "market"
)
SPY_MOMENTUM_QUALITY_FEATURES: Tuple[str, ...] = tuple(
    name
    for name in FEATURE_SPECS
    if name.startswith("spy_")
    and (
        name.startswith("spy_ret_")
        or name.startswith("spy_kaufman_er_")
        or name.startswith("spy_momentum_quality_")
    )
)

__all__ = [
    "ASSET_IV_RVOL_FEATURES",
    "ASSET_SETUP_FEATURES",
    "CHINA_MARKET_FEATURES",
    "DOLLAR_REGIME_FEATURES",
    "FEATURE_SPECS",
    "Feature",
    "FeatureSpec",
    "IV_MARKET_FEATURES",
    "MARKET_SETUP_FEATURES",
    "SPY_MOMENTUM_QUALITY_FEATURES",
]
