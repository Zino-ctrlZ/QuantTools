"""Derived feature Series with L1 memory and CustomCache L2 persistence.

Core loop: request feature names → load market history (default 1970 → last
NY business day) via datamanager → compute → cache (L1/L2) → slice to index.

Core Classes:
    FeatureDataManager: L1 → L2 → compute orchestration.
    FeatureResult: ``series`` dict return container.
    Feature: Registry row (name, compute, formula_id, scope, theme).
    FeatureTheme: Closed set of research theme ids (``str`` enum).
    validate_registry: Load-time wiring check (themes, inputs, scope).

Usage:
    >>> from trade.features import FeatureDataManager
    >>> mgr = FeatureDataManager()
    >>> market = mgr.get_market_features(index=df.index)
    >>> vendor = mgr.get_features_for_ticker("AAPL", features=["ret_20d"])
    >>> custom = mgr.get_features_from_ohlcv("AAPL", ohlcv, features=["ret_20d"])
"""

from trade.features.catalog import (
    FEATURE_SPECS,
    Feature,
    get_spec,
    list_features,
    list_features_by_theme,
)
from trade.features.keys import MARKET_SYMBOL, feature_cache_key
from trade.features.l1 import FeatureL1Cache
from trade.features.manager import (
    FeatureDataManager,
    get_feature_data_manager,
    reset_feature_data_manager,
)
from trade.features.result import FeatureResult
from trade.features.compute.validate import validate_registry
from trade.features.themes import (
    ASSET_FEATURE_THEMES,
    FEATURE_THEMES,
    FeatureTheme,
    REGIME_FEATURE_THEMES,
    coerce_theme,
    map_entry_prefixed_feature_to_theme,
    map_features_to_themes,
)

__all__ = [
    "ASSET_FEATURE_THEMES",
    "FEATURE_SPECS",
    "FEATURE_THEMES",
    "Feature",
    "FeatureDataManager",
    "FeatureL1Cache",
    "FeatureResult",
    "FeatureTheme",
    "MARKET_SYMBOL",
    "REGIME_FEATURE_THEMES",
    "coerce_theme",
    "feature_cache_key",
    "get_feature_data_manager",
    "get_spec",
    "list_features",
    "list_features_by_theme",
    "map_entry_prefixed_feature_to_theme",
    "map_features_to_themes",
    "reset_feature_data_manager",
    "validate_registry",
]
