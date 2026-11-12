"""Shared causal transforms used by market / asset / hybrid feature formulas."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trade.features.compute.symbols import (
    IV_PERCENTILE_LOOKBACK,
    RATIO_ZSCORE_LOOKBACK,
)


def sanitize_ratio(series: pd.Series) -> pd.Series:
    """Replace non-finite values with NaN after ratio / z-score arithmetic."""
    return series.replace([np.inf, -np.inf], np.nan)


def rolling_zscore(series: pd.Series, lookback: int) -> pd.Series:
    """Causal rolling z-score; zero rolling std becomes NaN."""
    rolling_mean = series.rolling(lookback).mean()
    rolling_std = series.rolling(lookback).std().replace(0, np.nan)
    return sanitize_ratio((series - rolling_mean) / rolling_std)


def ratio_zscore(
    numerator: pd.Series,
    denominator: pd.Series,
    lookback: int = RATIO_ZSCORE_LOOKBACK,
) -> pd.Series:
    """Rolling z-score of ``numerator / denominator`` (causal, no lookahead)."""
    ratio = numerator / denominator.replace(0, np.nan)
    return rolling_zscore(ratio, lookback)


def relative_momentum(
    numerator: pd.Series,
    denominator: pd.Series,
    horizon: int,
) -> pd.Series:
    """Horizon return of numerator minus horizon return of denominator."""
    return sanitize_ratio(numerator.pct_change(horizon) - denominator.pct_change(horizon))


def annualized_realized_vol(close: pd.Series, window: int = 20) -> pd.Series:
    """Annualized realized volatility from daily close returns."""
    return close.pct_change().rolling(window).std() * np.sqrt(252)


def efficiency_ratio(close: pd.Series, window: int) -> pd.Series:
    """Return Kaufman efficiency ratio over ``window`` bars.

    Args:
        close: Price series (typically adjusted close).
        window: Lookback length in bars.

    Returns:
        Series of ER values in ``[0, 1]`` where defined; leading bars are NaN.
    """
    net_change = close.diff(window).abs()
    path_length = close.diff().abs().rolling(window).sum()
    return net_change / path_length


def rolling_pctile(series: pd.Series, lookback: int = IV_PERCENTILE_LOOKBACK) -> pd.Series:
    """Return rolling percentile rank of the latest value on a 0–1 scale."""

    def _pctile(window: np.ndarray) -> float:
        if len(window) == 0:
            return np.nan
        current = window[-1]
        return float(np.mean(window <= current))

    return series.rolling(lookback).apply(_pctile, raw=True)
