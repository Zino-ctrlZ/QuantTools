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


def vol_adjusted_relative_momentum(
    numerator: pd.Series,
    denominator: pd.Series,
    horizon: int = 20,
    vol_window: int = 60,
) -> pd.Series:
    """Vol-scaled relative momentum: raw horizon spread over relative-return vol.

    Computes

    ``(R_num^(h) - R_den^(h)) / (σ_{num-den}^(v) * sqrt(h))``

    where daily relative returns are ``num.pct_change() - den.pct_change()``,
    ``σ`` is their ``vol_window`` rolling standard deviation, and ``h`` is
    ``horizon``. Zero vol and non-finite values become NaN. All rolling ops
    are causal (no lookahead).

    Args:
        numerator: Asset / long-leg close series.
        denominator: Benchmark / short-leg close series.
        horizon: Momentum lookback in bars (default 20).
        vol_window: Rolling window for daily relative-return std (default 60).

    Returns:
        Vol-adjusted relative momentum series.
    """
    raw = relative_momentum(numerator, denominator, horizon)
    daily_rel = numerator.pct_change() - denominator.pct_change()
    vol = daily_rel.rolling(vol_window).std() * np.sqrt(horizon)
    return sanitize_ratio(raw / vol.replace(0, np.nan))


def annualized_realized_vol(close: pd.Series, window: int = 20) -> pd.Series:
    """Annualized realized volatility from daily close returns."""
    return close.pct_change().rolling(window).std() * np.sqrt(252)


def return_to_vol(close: pd.Series, horizon: int) -> pd.Series:
    """Horizon return divided by same-horizon realized volatility.

    Volatility is the rolling standard deviation of daily returns over
    ``horizon`` bars, scaled by ``sqrt(horizon)`` so the ratio is unitless
    (return per one-sigma move over the window).

    Args:
        close: Price series (typically adjusted close).
        horizon: Lookback length in bars for both return and volatility.

    Returns:
        Vol-normalized return series; zeros in vol become NaN.
    """
    ret = close.pct_change(horizon)
    vol = close.pct_change().rolling(horizon).std() * np.sqrt(horizon)
    return sanitize_ratio(ret / vol.replace(0, np.nan))


def vol_adjusted_return(
    close: pd.Series,
    horizon: int = 20,
    vol_window: int = 60,
) -> pd.Series:
    """Horizon return scaled by a separately windowed return volatility.

    Computes

    ``R^(h) / (σ^(v) * sqrt(h))``

    where ``R^(h)`` is ``close.pct_change(horizon)`` and ``σ^(v)`` is the
    ``vol_window`` rolling standard deviation of daily returns. Zero vol and
    non-finite values become NaN. All rolling ops are causal (no lookahead).

    Args:
        close: Price series (typically adjusted close).
        horizon: Return lookback in bars (default 20).
        vol_window: Rolling window for daily-return std (default 60).

    Returns:
        Vol-adjusted return series.
    """
    ret = close.pct_change(horizon)
    vol = close.pct_change().rolling(vol_window).std() * np.sqrt(horizon)
    return sanitize_ratio(ret / vol.replace(0, np.nan))


def down_day_fraction(close: pd.Series, window: int) -> pd.Series:
    """Fraction of down days (negative close-to-close return) over ``window``.

    Args:
        close: Price series (typically adjusted close).
        window: Rolling lookback in bars.

    Returns:
        Series in ``[0, 1]`` where defined; leading bars are NaN.
    """
    return (close.pct_change() < 0).rolling(window).mean()


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
