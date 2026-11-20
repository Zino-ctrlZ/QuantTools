"""Unit tests for FeatureDataManager L1/L2 (seeded inputs; no vendors)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trade.features.catalog import get_spec, list_features, resolve_features
from trade.features.compute.definitions import (
    ASSET_IV_RVOL_FEATURES,
    ASSET_SETUP_FEATURES,
    MARKET_SETUP_FEATURES,
)
from trade.features.keys import CUSTOM_BARS_TAG, feature_cache_key
from trade.features.manager import FeatureDataManager, reset_feature_data_manager
from trade.features.result import FeatureResult
from trade.features.themes import FeatureTheme


@pytest.fixture(autouse=True)
def _reset_manager():
    """Isolate each test from singleton / disk state."""
    reset_feature_data_manager()
    yield
    reset_feature_data_manager()


def _synth_ohlc(start: str, periods: int, seed: int = 0) -> pd.DataFrame:
    """Build a simple OHLC frame for seeding."""
    idx = pd.bdate_range(start, periods=periods)
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, size=periods))
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
        },
        index=idx,
    )


def test_catalog_market_features_registered():
    """Market names resolve with theme and a bound compute callable."""
    assert "spy_ret_20d" in list_features("market")
    spec = get_spec("spy_ret_20d")
    assert spec.scope == "market"
    assert spec.theme == FeatureTheme.EQUITY_MARKET_MOMENTUM
    assert spec.formula_id == "close_return_v1"
    assert callable(spec.compute)


def test_registry_validation_passes_and_theme_lookup_is_strict():
    """Load-time validator is clean; unlisted names no longer default to unmapped."""
    from trade.features.compute.validate import validate_registry
    from trade.features.themes import map_features_to_themes

    validate_registry()
    with pytest.raises(KeyError, match="FEATURE_THEMES"):
        map_features_to_themes("not_a_real_feature")


def test_get_features_defaults_to_all_market():
    """Omitting features resolves to the full market catalog."""
    assert resolve_features(None) == list(MARKET_SETUP_FEATURES)
    with pytest.raises(ValueError, match="non-empty"):
        resolve_features([])


def test_resolve_features_include_hybrid():
    """With include_hybrid, default set is market plus hybrid IV−rvol names."""
    names = resolve_features(None, include_hybrid=True)
    assert names[: len(MARKET_SETUP_FEATURES)] == list(MARKET_SETUP_FEATURES)
    assert set(ASSET_IV_RVOL_FEATURES).issubset(names)


def test_resolve_features_include_asset():
    """With include_asset, default set also includes asset-local names."""
    names = resolve_features(None, include_hybrid=True, include_asset=True)
    assert set(ASSET_IV_RVOL_FEATURES).issubset(names)
    assert set(ASSET_SETUP_FEATURES).issubset(names)
    assert "ret_20d" in list_features("asset")
    assert get_spec("ret_20d").scope == "asset"


def test_deprecated_theme_bucket_exists_and_defaults_exclude_flagged():
    """Deprecated theme is wired; soft-retired names drop out of defaults."""
    from dataclasses import replace

    from trade.features.compute.specs import FEATURE_SPECS
    from trade.features.compute.validate import validate_registry

    assert FeatureTheme.DEPRECATED.value in {
        theme.value for theme in FeatureTheme
    }
    validate_registry()
    assert all(not get_spec(name).deprecated for name in resolve_features(None))

    original = FEATURE_SPECS["spy_ret_20d"]
    FEATURE_SPECS["spy_ret_20d"] = replace(
        original,
        theme=FeatureTheme.DEPRECATED,
        deprecated=True,
    )
    try:
        assert "spy_ret_20d" not in resolve_features(None)
        with pytest.warns(DeprecationWarning, match="deprecated"):
            resolved = resolve_features(["spy_ret_20d"])
        assert resolved == ["spy_ret_20d"]
    finally:
        FEATURE_SPECS["spy_ret_20d"] = original


def test_deprecated_flag_must_match_theme_at_validation():
    """Validator rejects deprecated=True when the theme is not deprecated."""
    from dataclasses import replace

    from trade.features.compute.specs import FEATURE_SPECS
    from trade.features.compute.validate import validate_feature

    original = FEATURE_SPECS["spy_ret_20d"]
    bad = replace(original, deprecated=True)
    problems = validate_feature(bad)
    assert any("deprecated=" in msg for msg in problems)


def test_get_market_features_rejects_asset_names():
    """Market endpoint refuses non-market feature names."""
    mgr = FeatureDataManager()
    with pytest.raises(ValueError, match="market-only"):
        mgr.get_market_features(features=["ret_20d"])


def test_from_ohlcv_requires_both_ticker_and_frame():
    """Caller-OHLCV endpoint enforces ticker and ohlcv together."""
    mgr = FeatureDataManager()
    spy = _synth_ohlc("2019-01-01", 50, seed=2)
    with pytest.raises(ValueError, match="requires ohlcv"):
        mgr.get_features_from_ohlcv("AAPL", None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ticker"):
        mgr.get_features_from_ohlcv("", spy)
    with pytest.raises(TypeError, match="DataFrame"):
        mgr.get_features_from_ohlcv("AAPL", "not-a-frame")  # type: ignore[arg-type]


def test_compute_market_from_seeded_ohlc():
    """Seeded SPY OHLC computes features; second call reuses L1/L2."""
    spy = _synth_ohlc("2019-01-01", 400, seed=1)
    start, end = spy.index.min(), spy.index.max()
    idx = spy.index[-100:]

    mgr = FeatureDataManager()
    mgr.inputs.seed_ohlc({"SPY": spy})

    result = mgr.get_market_features(
        ["spy_ret_20d", "spy_realized_vol_20d"],
        index=idx,
        start=start,
        end=end,
        recompute=True,
    )
    assert isinstance(result, FeatureResult)
    assert set(result.series) == {"spy_ret_20d", "spy_realized_vol_20d"}
    assert result.series["spy_ret_20d"].notna().sum() > 0

    result2 = mgr.get_market_features(
        ["spy_ret_20d", "spy_realized_vol_20d"],
        index=idx,
        start=start,
        end=end,
    )
    pd.testing.assert_series_equal(
        result.series["spy_ret_20d"],
        result2.series["spy_ret_20d"],
        check_names=False,
    )


def test_start_end_clip_returned_series_without_index():
    """start/end bound the returned series even when index is omitted."""
    spy = _synth_ohlc("2019-01-01", 400, seed=7)
    mgr = FeatureDataManager()
    mgr.inputs.seed_ohlc({"SPY": spy})

    ## Warm a durable full-span cache first.
    mgr.get_market_features(
        ["spy_ret_20d"],
        end=spy.index.max(),
        recompute=True,
    )

    window_start = spy.index[100]
    window_end = spy.index[200]
    clipped = mgr.get_market_features(
        ["spy_ret_20d"],
        start=window_start,
        end=window_end,
    ).series["spy_ret_20d"]

    assert clipped.index.min() >= pd.Timestamp(window_start).normalize()
    assert clipped.index.max() <= pd.Timestamp(window_end).normalize()
    assert len(clipped) < 400


def test_ticker_start_clips_return_but_durable_keeps_listing_history():
    """Caller start only slices the return; L1 still holds pre-window history."""
    asset = _synth_ohlc("2015-01-01", 800, seed=8)
    spy = _synth_ohlc("2015-01-01", 800, seed=9)
    mgr = FeatureDataManager()
    mgr.inputs.seed_ohlc({"SPY": spy})
    mgr.inputs.seed_asset_ohlc({"AAPL": asset})

    window_start = pd.Timestamp("2018-01-01")
    result = mgr.get_features_for_ticker(
        "AAPL",
        ["ret_20d"],
        start=window_start,
        end=asset.index.max(),
        recompute=True,
    )
    returned = result.series["ret_20d"]
    assert returned.index.min() >= window_start.normalize()
    ## No leading warmup NaNs inside the requested return window.
    assert returned.notna().iloc[0]

    key = feature_cache_key(
        mgr,
        symbol="AAPL",
        feature="ret_20d",
        formula_id=get_spec("ret_20d").formula_id,
    )
    durable = mgr._l1.get(key)
    assert durable is not None
    assert durable.index.min() < window_start


def test_get_features_for_ticker_self_queries_seeded_asset():
    """Ticker endpoint pulls seeded asset spot and persists durable L1/L2."""
    asset = _synth_ohlc("2019-01-01", 400, seed=3)
    spy = _synth_ohlc("2019-01-01", 400, seed=4)
    start, end = asset.index.min(), asset.index.max()
    idx = asset.index[-80:]

    mgr = FeatureDataManager()
    mgr.inputs.seed_ohlc({"SPY": spy})
    mgr.inputs.seed_asset_ohlc({"AAPL": asset})

    result = mgr.get_features_for_ticker(
        "AAPL",
        ["ret_20d", "realized_vol_20d", "asset_vs_spy_ret_60d"],
        index=idx,
        start=start,
        end=end,
        recompute=True,
    )
    assert set(result.series) == {
        "ret_20d",
        "realized_vol_20d",
        "asset_vs_spy_ret_60d",
    }
    assert result.series["ret_20d"].notna().sum() > 0

    ## Cached durable series must not keep leading warmup NaNs.
    key = feature_cache_key(
        mgr,
        symbol="AAPL",
        feature="ret_20d",
        formula_id=get_spec("ret_20d").formula_id,
    )
    cached = mgr._l1.get(key)
    assert cached is not None
    assert cached.notna().iloc[0]
    assert cached.notna().iloc[-1]

    result2 = mgr.get_features_for_ticker(
        "AAPL",
        ["ret_20d"],
        index=idx,
        start=start,
        end=end,
    )
    pd.testing.assert_series_equal(
        result.series["ret_20d"],
        result2.series["ret_20d"],
        check_names=False,
    )


def test_get_features_from_ohlcv_is_l1_only_custom():
    """Caller OHLCV path stores L1 under bars=custom and never writes L2."""
    asset = _synth_ohlc("2019-01-01", 400, seed=5)
    spy = _synth_ohlc("2019-01-01", 400, seed=6)
    start, end = asset.index.min(), asset.index.max()

    mgr = FeatureDataManager()
    mgr.inputs.seed_ohlc({"SPY": spy})

    writes = []
    orig = mgr.cache_it

    def _track(key, value, *, expire=None):
        writes.append(key)
        return orig(key, value, expire=expire)

    mgr.cache_it = _track  # type: ignore[method-assign]

    result = mgr.get_features_from_ohlcv(
        "AAPL",
        asset,
        ["ret_20d"],
        start=start,
        end=end,
        recompute=True,
    )
    assert result.series["ret_20d"].notna().sum() > 0
    assert writes == []

    custom_key = feature_cache_key(
        mgr,
        symbol="AAPL",
        feature="ret_20d",
        formula_id=get_spec("ret_20d").formula_id,
        custom_bars=True,
    )
    hist_key = feature_cache_key(
        mgr,
        symbol="AAPL",
        feature="ret_20d",
        formula_id=get_spec("ret_20d").formula_id,
        custom_bars=False,
    )
    ## construct_cache_key uppercases parts, so bars:CUSTOM appears in the key.
    assert f"bars:{CUSTOM_BARS_TAG.upper()}" in custom_key
    assert mgr._l1.get(custom_key) is not None
    assert mgr._l1.get(hist_key) is None
    assert mgr.get(custom_key, default=None) is None
