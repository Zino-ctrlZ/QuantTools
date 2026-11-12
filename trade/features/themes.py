"""Research theme groupings for feature catalogs and notebooks.

``FeatureTheme`` is the closed set of theme ids. ``FEATURE_THEMES`` maps each
theme value (string) to feature names so notebook lookups like
``FEATURE_THEMES["asset_momentum"]`` keep working. Every registered feature
must appear in exactly one bucket; intentional no-theme names go under
``FEATURE_THEMES["unmapped"]`` (no silent default). ``ASSET_FEATURE_THEMES`` /
``REGIME_FEATURE_THEMES`` are legacy notebook split views (string keys; may
include placeholders not in the compute catalog).
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Union


class FeatureTheme(str, Enum):
    """Canonical research theme for a feature.

    ``str`` subclass so notebook / DataFrame code can treat values as plain text.
    """

    ## Asset state
    ASSET_MOMENTUM = "asset_momentum"
    ASSET_TREND = "asset_trend"
    ASSET_PRICE_LOCATION = "asset_price_location"
    ASSET_VOLATILITY = "asset_volatility"
    ASSET_DOWNSIDE_BEHAVIOR = "asset_downside_behavior"
    ASSET_TREND_QUALITY = "asset_trend_quality"
    ASSET_RELATIVE_STRENGTH = "asset_relative_strength"
    ASSET_PATH_STRUCTURE = "asset_path_structure"
    ASSET_RANGE_STRUCTURE = "asset_range_structure"
    VOLUME_PARTICIPATION = "volume_participation"
    ASSET_DOWNSIDE_PATH = "asset_downside_path"
    ASSET_RETURN_DISTRIBUTION = "asset_return_distribution"
    ASSET_GAP_BEHAVIOR = "asset_gap_behavior"
    ASSET_SIGNAL_CONFIRMATION = "asset_signal_confirmation"
    ## Market regime
    EQUITY_MARKET_MOMENTUM = "equity_market_momentum"
    EQUITY_MARKET_TREND = "equity_market_trend"
    EQUITY_MARKET_TREND_QUALITY = "equity_market_trend_quality"
    EQUITY_MARKET_VOLATILITY = "equity_market_volatility"
    EQUITY_MARKET_DEFENSIVE_RISK = "equity_market_defensive_risk"
    EQUITY_MARKET_ROTATION = "equity_market_rotation"
    EQUITY_MARKET_ROTATION_MOMENTUM = "equity_market_rotation_momentum"
    COMMODITY_ROTATION = "commodity_rotation"
    COMMODITY_ROTATION_MOMENTUM = "commodity_rotation_momentum"
    IMPLIED_VOLATILITY_LEVEL = "implied_volatility_level"
    IMPLIED_VOLATILITY_MOMENTUM = "implied_volatility_momentum"
    VOLATILITY_RISK_PREMIUM = "volatility_risk_premium"
    VOLATILITY_TERM_STRUCTURE = "volatility_term_structure"
    CHINA_MARKET_REGIME = "china_market_regime"
    DOLLAR_REGIME = "dollar_regime"
    ## Sentinel
    UNMAPPED = "unmapped"


FEATURE_THEMES: Dict[str, List[str]] = {
    FeatureTheme.ASSET_MOMENTUM.value: [
        "ret_20d",
        "ret_60d",
        "ret_120d",
        "rolling_avg_20d_ret_126d",
    ],
    FeatureTheme.ASSET_TREND.value: [
        "close_vs_sma_50",
        "close_vs_sma_200",
        "above_sma_200",
        "sma_20_slope_20d",
        "sma_50_slope_20d",
        "sma_200_slope_20d",
    ],
    FeatureTheme.ASSET_PRICE_LOCATION.value: [
        "dist_from_60d_high",
        "drawdown_from_252d_high",
    ],
    FeatureTheme.ASSET_VOLATILITY.value: [
        "realized_vol_20d",
        "atr_20d_pct",
        "vol_ratio_20d_vs_126d",
        "rolling_avg_20d_vol_126d",
    ],
    FeatureTheme.ASSET_DOWNSIDE_BEHAVIOR.value: [
        "avg_decline_20d_126d",
        "decline_freq_20d_126d",
        "count_large_20d_declines_126d",
        "p_down_given_down_10d_126d",
    ],
    FeatureTheme.ASSET_TREND_QUALITY.value: [
        "return_to_vol_60d",
        "rolling_er_50_126d",
        "kaufman_er_20d",
        "kaufman_er_60d",
        "kaufman_er_120d",
        "momentum_quality_20d",
        "momentum_quality_60d",
        "momentum_quality_120d",
    ],
    FeatureTheme.ASSET_RELATIVE_STRENGTH.value: [
        "asset_vs_spy_ret_60d",
    ],
    FeatureTheme.ASSET_PATH_STRUCTURE.value: [
        "adx_20d",
        "aroon_oscillator_20d",
    ],
    FeatureTheme.ASSET_RANGE_STRUCTURE.value: [
        "bollinger_band_width_20d",
        "donchian_width_20d",
        "range_position_20d",
        "range_compression_percentile_126d",
    ],
    FeatureTheme.VOLUME_PARTICIPATION.value: [
        "relative_volume_20d",
        "volume_percentile_252d",
        "dollar_volume_20d",
        "dollar_volume_change_20d",
        "signed_volume_pressure_20d",
        "chaikin_money_flow_20d",
        "price_volume_correlation_60d",
    ],
    FeatureTheme.ASSET_DOWNSIDE_PATH.value: [
        "ulcer_index_20d",
        "drawdown_duration_252d",
        "time_since_60d_high",
        "time_since_60d_low",
    ],
    FeatureTheme.ASSET_RETURN_DISTRIBUTION.value: [
        "realized_skew_60d",
        "realized_kurtosis_60d",
    ],
    FeatureTheme.ASSET_GAP_BEHAVIOR.value: [
        "overnight_gap_1d",
        "absolute_overnight_gap_1d",
        "avg_abs_overnight_gap_20d",
        "overnight_gap_volatility_20d",
        "negative_gap_frequency_60d",
        "positive_gap_frequency_60d",
        "gap_vs_atr_20d",
        "overnight_return_share_20d",
    ],
    FeatureTheme.ASSET_SIGNAL_CONFIRMATION.value: [
        "relative_volume_1d",
        "close_location_value_1d",
        "directional_close_location_1d",
        "true_range_vs_atr_20d",
        "signed_volume_1d",
        "directional_volume_confirmation_1d",
        "followthrough_return_1d",
        "followthrough_return_3d",
        "choppiness_index_20d",
    ],
    FeatureTheme.EQUITY_MARKET_MOMENTUM.value: [
        "spy_ret_20d",
        "spy_ret_60d",
        "spy_ret_120d",
    ],
    FeatureTheme.EQUITY_MARKET_TREND.value: [
        "spy_above_sma_200",
        "spy_close_vs_sma_200",
        "spy_sma_20_slope_20d",
        "spy_sma_50_slope_20d",
        "spy_sma_200_slope_20d",
        "spy_drawdown_from_252d_high",
    ],
    FeatureTheme.EQUITY_MARKET_TREND_QUALITY.value: [
        "spy_kaufman_er_20d",
        "spy_kaufman_er_60d",
        "spy_kaufman_er_120d",
        "spy_momentum_quality_20d",
        "spy_momentum_quality_60d",
        "spy_momentum_quality_120d",
    ],
    FeatureTheme.EQUITY_MARKET_VOLATILITY.value: [
        "spy_realized_vol_20d",
    ],
    FeatureTheme.EQUITY_MARKET_DEFENSIVE_RISK.value: [
        "xlu_spy_zscore",
        "xly_xlp_ratio_zscore",
        "qual_spy_ratio_zscore",
        "usmv_spy_ratio_zscore",
    ],
    FeatureTheme.EQUITY_MARKET_ROTATION.value: [
        "iwm_spy_ratio_zscore",
        "qqq_spy_ratio_zscore",
        "mtum_spy_ratio_zscore",
        "rsp_spy_ratio_zscore",
        "eem_spy_ratio_zscore",
        "fxi_spy_ratio_zscore",
    ],
    FeatureTheme.EQUITY_MARKET_ROTATION_MOMENTUM.value: [
        "xly_xlp_relative_momentum_20d",
        "xly_xlp_relative_momentum_60d",
        "iwm_spy_relative_momentum_20d",
        "iwm_spy_relative_momentum_60d",
        "qqq_spy_relative_momentum_20d",
        "qqq_spy_relative_momentum_60d",
        "mtum_spy_relative_momentum_20d",
        "mtum_spy_relative_momentum_60d",
        "eem_spy_relative_momentum_20d",
        "eem_spy_relative_momentum_60d",
        "fxi_spy_relative_momentum_20d",
        "fxi_spy_relative_momentum_60d",
    ],
    FeatureTheme.COMMODITY_ROTATION.value: [
        "gld_spy_ratio_zscore",
        "slv_spy_ratio_zscore",
        "gld_slv_ratio_zscore",
        "dbc_spy_ratio_zscore",
        "uso_spy_ratio_zscore",
        "ung_spy_ratio_zscore",
        "cper_spy_ratio_zscore",
        "dba_spy_ratio_zscore",
        "xle_xlu_ratio_zscore",
    ],
    FeatureTheme.COMMODITY_ROTATION_MOMENTUM.value: [
        "gld_spy_relative_momentum_20d",
        "gld_spy_relative_momentum_60d",
        "slv_spy_relative_momentum_20d",
        "slv_spy_relative_momentum_60d",
        "dbc_spy_relative_momentum_20d",
        "dbc_spy_relative_momentum_60d",
        "uso_spy_relative_momentum_20d",
        "uso_spy_relative_momentum_60d",
        "ung_spy_relative_momentum_20d",
        "ung_spy_relative_momentum_60d",
        "cper_spy_relative_momentum_20d",
        "cper_spy_relative_momentum_60d",
        "dba_spy_relative_momentum_20d",
        "dba_spy_relative_momentum_60d",
    ],
    FeatureTheme.IMPLIED_VOLATILITY_LEVEL.value: [
        "vix_level",
        "vix_above_200ma",
        "vix_pct_200ma",
        "vix_pctile_252",
        "fxi_iv_level",
        "eem_iv_level",
        "fxi_iv_zscore",
        "eem_iv_zscore",
        "fxi_iv_percentile_252d",
        "eem_iv_percentile_252d",
    ],
    FeatureTheme.IMPLIED_VOLATILITY_MOMENTUM.value: [
        "vix_20_day_return",
    ],
    FeatureTheme.VOLATILITY_RISK_PREMIUM.value: [
        "vix_spy_rvol_20_ratio",
        "fxi_iv_to_asset_rvol_20d_ratio",
        "eem_iv_to_asset_rvol_20d_ratio",
        "fxi_iv_minus_asset_rvol_20d",
        "eem_iv_minus_asset_rvol_20d",
        "fxi_iv_to_asset_rvol_20d_ratio_zscore",
        "eem_iv_to_asset_rvol_20d_ratio_zscore",
    ],
    FeatureTheme.VOLATILITY_TERM_STRUCTURE.value: [
        "vix_vs_vix3m",
    ],
    FeatureTheme.CHINA_MARKET_REGIME.value: [
        "usdcnh_ret_20d",
        "usdcnh_ret_60d",
        "usdcnh_close_vs_sma_200",
    ],
    FeatureTheme.DOLLAR_REGIME.value: [
        "uup_ret_20d",
        "uup_ret_60d",
        "uup_close_vs_sma_200",
        "uup_sma_50_slope_20d",
        "uup_realized_vol_20d",
        "uup_drawdown_from_252d_high",
    ],
    ## Intentional no-theme features only. Never rely on a silent default.
    FeatureTheme.UNMAPPED.value: [],
}

## feature name → theme (built once; O(1) lookup).
_FEATURE_TO_THEME: Dict[str, FeatureTheme] = {
    name: FeatureTheme(theme) for theme, names in FEATURE_THEMES.items() for name in names
}

## ---------------------------------------------------------------------------
## Legacy notebook split views (string keys; may include placeholders)
## ---------------------------------------------------------------------------

ASSET_FEATURE_THEMES: Dict[str, List[str]] = {
    "momentum": [
        "ret_20d",
        "ret_60d",
        "ret_120d",
    ],
    "trend_position": [
        "close_vs_sma_50",
        "close_vs_sma_200",
        "above_sma_200",
        "sma_20_slope_20d",
        "sma_50_slope_20d",
        "sma_200_slope_20d",
    ],
    "price_location": [
        "breakdown_score",
        "dist_from_60d_high",
        "drawdown_from_252d_high",
    ],
    "asset_volatility": [
        "rvol",
        "realized_vol_20d",
        "atr_20d_pct",
        "atr_loss",
        "rolling_avg_20d_vol_126d",
        "vol_ratio_20d_vs_126d",
    ],
    "downside_persistence": [
        "avg_decline_20d_126d",
        "decline_freq_20d_126d",
        "count_large_20d_declines_126d",
        "p_down_given_down_10d_126d",
    ],
    "trend_quality": [
        "rolling_er_50_126d",
        "return_to_vol_60d",
        "kaufman_er_20d",
        "kaufman_er_60d",
        "kaufman_er_120d",
        "momentum_quality_20d",
        "momentum_quality_60d",
        "momentum_quality_120d",
    ],
    "relative_strength": [
        "asset_vs_spy_ret_60d",
    ],
}

REGIME_FEATURE_THEMES: Dict[str, List[str]] = {
    "equity_market_trend": [
        "spy_above_sma_200",
        "spy_close_vs_sma_200",
        "spy_sma_20_slope_20d",
        "spy_sma_50_slope_20d",
        "spy_sma_200_slope_20d",
        "spy_ret_20d",
        "spy_ret_60d",
        "spy_ret_120d",
    ],
    "equity_market_trend_quality": [
        "spy_kaufman_er_20d",
        "spy_kaufman_er_60d",
        "spy_kaufman_er_120d",
        "spy_momentum_quality_20d",
        "spy_momentum_quality_60d",
        "spy_momentum_quality_120d",
    ],
    "equity_market_volatility": [
        "spy_realized_vol_20d",
        "spy_vol_ratio_20d_vs_126d",
    ],
    "equity_market_defensive_risk": [
        "xlu_spy_zscore",
        "xly_xlp_ratio_zscore",
        "qual_spy_ratio_zscore",
        "usmv_spy_ratio_zscore",
    ],
    "equity_market_rotation": [
        "iwm_spy_ratio_zscore",
        "qqq_spy_ratio_zscore",
        "mtum_spy_ratio_zscore",
        "rsp_spy_ratio_zscore",
        "eem_spy_ratio_zscore",
        "fxi_spy_ratio_zscore",
    ],
    "equity_market_rotation_momentum": [
        "xly_xlp_relative_momentum_20d",
        "xly_xlp_relative_momentum_60d",
        "iwm_spy_relative_momentum_20d",
        "iwm_spy_relative_momentum_60d",
        "qqq_spy_relative_momentum_20d",
        "qqq_spy_relative_momentum_60d",
        "mtum_spy_relative_momentum_20d",
        "mtum_spy_relative_momentum_60d",
        "eem_spy_relative_momentum_20d",
        "eem_spy_relative_momentum_60d",
        "fxi_spy_relative_momentum_20d",
        "fxi_spy_relative_momentum_60d",
    ],
    "commodity_rotation": [
        "gld_spy_ratio_zscore",
        "slv_spy_ratio_zscore",
        "gld_slv_ratio_zscore",
        "dbc_spy_ratio_zscore",
        "uso_spy_ratio_zscore",
        "ung_spy_ratio_zscore",
        "cper_spy_ratio_zscore",
        "dba_spy_ratio_zscore",
        "xle_xlu_ratio_zscore",
    ],
    "commodity_rotation_momentum": [
        "gld_spy_relative_momentum_20d",
        "gld_spy_relative_momentum_60d",
        "slv_spy_relative_momentum_20d",
        "slv_spy_relative_momentum_60d",
        "dbc_spy_relative_momentum_20d",
        "dbc_spy_relative_momentum_60d",
        "uso_spy_relative_momentum_20d",
        "uso_spy_relative_momentum_60d",
        "ung_spy_relative_momentum_20d",
        "ung_spy_relative_momentum_60d",
        "cper_spy_relative_momentum_20d",
        "cper_spy_relative_momentum_60d",
        "dba_spy_relative_momentum_20d",
        "dba_spy_relative_momentum_60d",
    ],
    "implied_volatility_regime": [
        "vix_level",
        "vix_above_200ma",
        "vix_pct_200ma",
        "vix_20_day_return",
        "vix_pctile_252",
        "vix_vs_vix3m",
        "fxi_iv_level",
        "eem_iv_level",
        "fxi_iv_zscore",
        "eem_iv_zscore",
        "fxi_iv_percentile_252d",
        "eem_iv_percentile_252d",
    ],
    "volatility_risk_premium": [
        "vix_spy_rvol_20_ratio",
        "fxi_iv_to_asset_rvol_20d_ratio",
        "eem_iv_to_asset_rvol_20d_ratio",
        "fxi_iv_minus_asset_rvol_20d",
        "eem_iv_minus_asset_rvol_20d",
        "fxi_iv_to_asset_rvol_20d_ratio_zscore",
        "eem_iv_to_asset_rvol_20d_ratio_zscore",
    ],
    "market_breadth": [
        "pct_stocks_above_sma_50",
        "pct_stocks_above_sma_200",
    ],
    "market_stress": [
        "spy_drawdown_from_252d_high",
        "average_correlation_60d",
    ],
    "china_market_regime": [
        "usdcnh_ret_20d",
        "usdcnh_ret_60d",
        "usdcnh_close_vs_sma_200",
    ],
    "dollar_regime": [
        "uup_ret_20d",
        "uup_ret_60d",
        "uup_close_vs_sma_200",
        "uup_sma_50_slope_20d",
        "uup_realized_vol_20d",
        "uup_drawdown_from_252d_high",
    ],
}


def coerce_theme(theme: Union[FeatureTheme, str]) -> FeatureTheme:
    """Accept a ``FeatureTheme`` or its string value."""
    if isinstance(theme, FeatureTheme):
        return theme
    return FeatureTheme(theme)


def map_features_to_themes(feature: str) -> FeatureTheme:
    """Return the unified ``FeatureTheme`` that owns ``feature``.

    Args:
        feature: Feature name (without ``entry_`` / ``exit_`` prefix).

    Returns:
        Matching theme from ``FEATURE_THEMES`` (including explicit ``unmapped``).

    Raises:
        KeyError: If ``feature`` is not listed in any ``FEATURE_THEMES`` bucket.
    """
    try:
        return _FEATURE_TO_THEME[feature]
    except KeyError as exc:
        raise KeyError(
            f"Feature {feature!r} is not in FEATURE_THEMES. "
            "Add it to a theme bucket (or FEATURE_THEMES['unmapped'])."
        ) from exc


def map_entry_prefixed_feature_to_theme(feature: str) -> FeatureTheme:
    """Map an ``entry_``-prefixed feature name onto a unified theme.

    Args:
        feature: Feature name that may include an ``entry_`` prefix.

    Returns:
        Theme from ``map_features_to_themes``.
    """
    return map_features_to_themes(feature.replace("entry_", ""))
