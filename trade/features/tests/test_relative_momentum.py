"""Tests for raw and vol-adjusted relative momentum transforms and catalog."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trade.features.catalog import get_spec, list_features, resolve_features
from trade.features.compute.specs import BASE_ASSET_FEATURE_SET, BASE_MARKET_FEATURE_SET
from trade.features.compute.transforms import (
    relative_momentum,
    vol_adjusted_relative_momentum,
    vol_adjusted_return,
)
from trade.features.themes import FeatureTheme


def _price_path(start: float, shocks: np.ndarray) -> pd.Series:
    """Build a close series from additive shocks on a business-day index."""
    idx = pd.bdate_range("2020-01-01", periods=len(shocks))
    return pd.Series(start + np.cumsum(shocks), index=idx, name="close")


def test_vol_adjusted_relative_momentum_matches_hand_formula() -> None:
    """Vol-adj equals raw relative momentum over relative-return vol * sqrt(h)."""
    rng = np.random.default_rng(0)
    n = 120
    num = _price_path(100.0, rng.normal(0.05, 1.0, size=n))
    den = _price_path(100.0, rng.normal(0.02, 0.8, size=n))
    horizon, vol_window = 20, 60

    got = vol_adjusted_relative_momentum(num, den, horizon=horizon, vol_window=vol_window)
    raw = relative_momentum(num, den, horizon)
    daily_rel = num.pct_change() - den.pct_change()
    vol = daily_rel.rolling(vol_window).std() * np.sqrt(horizon)
    expected = raw / vol.replace(0, np.nan)

    pd.testing.assert_series_equal(got, expected, check_names=False)
    assert got.iloc[: max(horizon, vol_window) - 1].isna().all()


def test_vol_adjusted_relative_momentum_zero_vol_is_nan() -> None:
    """Constant relative path yields NaN after vol window fills."""
    idx = pd.bdate_range("2020-01-01", periods=80)
    ## Identical closes → zero daily relative returns → zero rolling vol.
    num = pd.Series(np.linspace(100.0, 120.0, len(idx)), index=idx)
    den = num.copy()
    out = vol_adjusted_relative_momentum(num, den, horizon=20, vol_window=60)
    assert out.iloc[60:].isna().all()


def test_vol_adjusted_relative_momentum_no_lookahead() -> None:
    """Truncating the series at t does not change the value at t."""
    rng = np.random.default_rng(1)
    n = 100
    num = _price_path(50.0, rng.normal(0, 1.0, size=n))
    den = _price_path(50.0, rng.normal(0, 1.0, size=n))
    full = vol_adjusted_relative_momentum(num, den, horizon=20, vol_window=60)
    t = 75
    truncated = vol_adjusted_relative_momentum(
        num.iloc[: t + 1], den.iloc[: t + 1], horizon=20, vol_window=60
    )
    assert np.isclose(full.iloc[t], truncated.iloc[t], equal_nan=True)


def test_vol_adj_features_registered_with_metadata() -> None:
    """Primary vol-adj names are market-scoped with expected formula_id."""
    name = "iwm_spy_relative_momentum_vol_adj_20d"
    spec = get_spec(name)
    assert spec.scope == "market"
    assert spec.theme == FeatureTheme.EQUITY_MARKET_ROTATION_MOMENTUM
    assert spec.formula_id == "relative_momentum_vol_adj_v1"
    assert spec.params.get("horizon") == 20
    assert spec.params.get("vol_window") == 60
    assert not spec.deprecated
    assert name in list_features("market")


def test_raw_relative_momentum_soft_retired_but_still_resolvable() -> None:
    """Raw names stay registered under deprecated for strategy consumers."""
    name = "iwm_spy_relative_momentum_20d"
    spec = get_spec(name)
    assert spec.deprecated
    assert spec.theme == FeatureTheme.DEPRECATED
    assert name not in resolve_features(None)
    with pytest.warns(DeprecationWarning, match="deprecated"):
        assert resolve_features([name]) == [name]


def test_base_market_pack_uses_vol_adj_not_raw() -> None:
    """Starter pack lists vol-adj momentum and omits raw relative momentum."""
    assert "iwm_spy_relative_momentum_vol_adj_20d" in BASE_MARKET_FEATURE_SET
    assert "xly_xlp_relative_momentum_vol_adj_20d" in BASE_MARKET_FEATURE_SET
    assert "gld_spy_relative_momentum_vol_adj_20d" in BASE_MARKET_FEATURE_SET
    for name in BASE_MARKET_FEATURE_SET:
        if "relative_momentum" in name:
            assert "vol_adj" in name
            assert not get_spec(name).deprecated


def test_asset_spy_vol_adj_registered_and_replaces_raw() -> None:
    """Asset vs SPY vol-adj is primary; raw 60d relative return is soft-retired."""
    name = "asset_spy_relative_momentum_vol_adj_20d"
    spec = get_spec(name)
    assert spec.scope == "asset"
    assert spec.theme == FeatureTheme.ASSET_RELATIVE_STRENGTH
    assert spec.formula_id == "relative_momentum_vol_adj_v1"
    assert spec.params.get("horizon") == 20
    assert spec.params.get("vol_window") == 60
    assert not spec.deprecated
    assert name in list_features("asset")
    assert name in BASE_ASSET_FEATURE_SET

    retired = "asset_vs_spy_ret_60d"
    retired_spec = get_spec(retired)
    assert retired_spec.deprecated
    assert retired_spec.theme == FeatureTheme.DEPRECATED
    assert retired not in BASE_ASSET_FEATURE_SET
    assert retired not in resolve_features(None)
    with pytest.warns(DeprecationWarning, match="deprecated"):
        assert resolve_features([retired]) == [retired]


def test_vol_adjusted_return_matches_hand_formula() -> None:
    """Vol-adj return equals horizon return over daily-return vol * sqrt(h)."""
    rng = np.random.default_rng(2)
    close = _price_path(100.0, rng.normal(0.05, 1.0, size=120))
    horizon, vol_window = 20, 60

    got = vol_adjusted_return(close, horizon=horizon, vol_window=vol_window)
    ret = close.pct_change(horizon)
    vol = close.pct_change().rolling(vol_window).std() * np.sqrt(horizon)
    expected = ret / vol.replace(0, np.nan)

    pd.testing.assert_series_equal(got, expected, check_names=False)


def test_ret_vol_adj_features_in_base_packs_raw_kept() -> None:
    """Base packs use vol-adj 20d returns; raw ret_20d family stays registered."""
    assert "ret_vol_adj_20d" in BASE_ASSET_FEATURE_SET
    assert "ret_20d" not in BASE_ASSET_FEATURE_SET
    assert not get_spec("ret_20d").deprecated
    assert get_spec("ret_vol_adj_20d").formula_id == "return_vol_adj_v1"
    assert get_spec("ret_vol_adj_20d").params.get("vol_window") == 60

    assert "spy_ret_vol_adj_20d" in BASE_MARKET_FEATURE_SET
    assert "spy_ret_20d" not in BASE_MARKET_FEATURE_SET
    assert not get_spec("spy_ret_20d").deprecated

    assert "uup_ret_vol_adj_20d" in BASE_MARKET_FEATURE_SET
    assert "uup_ret_20d" not in BASE_MARKET_FEATURE_SET
    assert not get_spec("uup_ret_20d").deprecated
