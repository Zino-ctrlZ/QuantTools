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


def test_ticker_and_ohlcv_must_be_paired():
    """Passing only ticker or only ohlcv raises."""
    mgr = FeatureDataManager()
    spy = _synth_ohlc("2019-01-01", 50, seed=2)
    with pytest.raises(ValueError, match="together"):
        mgr.get_features(ticker="AAPL")
    with pytest.raises(ValueError, match="together"):
        mgr.get_features(ohlcv=spy)


def test_compute_market_from_seeded_ohlc():
    """Seeded SPY OHLC computes features; second call reuses L1/L2."""
    spy = _synth_ohlc("2019-01-01", 400, seed=1)
    start, end = spy.index.min(), spy.index.max()
    idx = spy.index[-100:]

    mgr = FeatureDataManager()
    mgr.inputs.seed_ohlc({"SPY": spy})

    result = mgr.get_features(
        ["spy_ret_20d", "spy_realized_vol_20d"],
        index=idx,
        start=start,
        end=end,
        recompute=True,
    )
    assert isinstance(result, FeatureResult)
    assert set(result.series) == {"spy_ret_20d", "spy_realized_vol_20d"}
    assert result.series["spy_ret_20d"].notna().sum() > 0

    result2 = mgr.get_features(
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


def test_compute_asset_local_from_ohlcv():
    """Asset-local CORE features compute from ticker+ohlcv (seeded SPY for vs-SPY)."""
    asset = _synth_ohlc("2019-01-01", 400, seed=3)
    spy = _synth_ohlc("2019-01-01", 400, seed=4)
    start, end = asset.index.min(), asset.index.max()
    idx = asset.index[-80:]

    mgr = FeatureDataManager()
    mgr.inputs.seed_ohlc({"SPY": spy})

    result = mgr.get_features(
        ["ret_20d", "realized_vol_20d", "asset_vs_spy_ret_60d", "adx_20d"],
        index=idx,
        start=start,
        end=end,
        ticker="AAPL",
        ohlcv=asset,
        recompute=True,
    )
    assert set(result.series) == {
        "ret_20d",
        "realized_vol_20d",
        "asset_vs_spy_ret_60d",
        "adx_20d",
    }
    assert result.series["ret_20d"].notna().sum() > 0
    assert result.series["adx_20d"].notna().sum() > 0
