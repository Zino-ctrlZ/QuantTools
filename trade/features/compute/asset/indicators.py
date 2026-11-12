"""Low-level OHLCV indicator helpers used by asset family features.

These are not features themselves. Feature functions in ``asset/*.py`` call them
so the feature file stays short while the math stays named and findable.
"""

from __future__ import annotations

from typing import Dict, Literal, Optional

import numpy as np
import pandas as pd
from trade.backtester_.indicators import create_atr_dataframe

FEATURE_AVAILABILITY: Dict[str, Literal["open", "close"]] = {
    "overnight_gap_1d": "open",
    "absolute_overnight_gap_1d": "open",
    "avg_abs_overnight_gap_20d": "open",
    "overnight_gap_volatility_20d": "open",
    "negative_gap_frequency_60d": "open",
    "positive_gap_frequency_60d": "open",
    "gap_vs_atr_20d": "open",
    "overnight_return_share_20d": "close",
}

## --- OHLCV formula helpers ----------------------------------------------

_FLOAT_EQ_TOL = 1e-9


def _sanitize(series: pd.Series) -> pd.Series:
    """Replace infinite values with NaN and coerce to float64."""
    out = series.astype(np.float64, copy=False)
    return out.replace([np.inf, -np.inf], np.nan)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Element-wise division that maps zero denominators and infinities to NaN.

    Args:
        numerator: Series of numerators.
        denominator: Series of denominators; zeros become NaN before division.

    Returns:
        Quotient with infinite values and zero-denom edges replaced by NaN.
    """
    denom = denominator.replace(0, np.nan)
    return _sanitize(numerator / denom)


def _clip_minor_unit_interval(series: pd.Series) -> pd.Series:
    """Clip values only within a tiny float tolerance of [0, 1].

    Material out-of-range values (invalid OHLC) are left unclipped so they are
    not concealed as legitimate close-location scores.

    Args:
        series: Candidate unit-interval scores (e.g. close location value).

    Returns:
        Series with only near-bound floating-point noise clipped to [0, 1].
    """
    out = series.astype(np.float64, copy=True)
    near_low = (out >= -_FLOAT_EQ_TOL) & (out < 0.0)
    near_high = (out > 1.0) & (out <= 1.0 + _FLOAT_EQ_TOL)
    out = out.where(~near_low, 0.0)
    out = out.where(~near_high, 1.0)
    return out


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Wilder true range from daily OHLC."""
    prev_close = close.shift(1)
    return pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def _wilder_smooth_sum_init(series: pd.Series, period: int) -> pd.Series:
    """Wilder smoothing with initial value equal to sum of first ``period`` observations.

    Subsequent values follow ``smoothed_t = smoothed_{t-1} - smoothed_{t-1}/period + x_t``.
    """
    values = series.to_numpy(dtype=np.float64)
    n = len(values)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return pd.Series(out, index=series.index)

    out[period - 1] = np.nansum(values[:period])
    for i in range(period, n):
        out[i] = out[i - 1] - out[i - 1] / period + values[i]
    return pd.Series(out, index=series.index)


def _wilder_smooth_mean_init(series: pd.Series, period: int) -> pd.Series:
    """Wilder smoothing with initial ADX value equal to mean of first ``period`` inputs."""
    values = series.to_numpy(dtype=np.float64)
    n = len(values)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < 2 * period - 1:
        return pd.Series(out, index=series.index)

    first_block = values[period - 1 : 2 * period - 1]
    if np.isnan(first_block).any():
        return pd.Series(out, index=series.index)

    out[2 * period - 2] = float(np.mean(first_block))
    for i in range(2 * period - 1, n):
        if np.isnan(values[i]):
            continue
        out[i] = (out[i - 1] * (period - 1) + values[i]) / period
    return pd.Series(out, index=series.index)


def _periods_since_rolling_extreme(window: np.ndarray, *, find_max: bool) -> float:
    """Return bars since the most recent rolling extreme inside ``window``.

    Indexing convention: the last element of ``window`` is the current bar (offset 0).
    A new extreme on the current bar yields ``0``; one period ago yields ``1``.
    """
    if len(window) == 0:
        return np.nan
    if find_max:
        extreme = np.nanmax(window)
        hit_indices = np.flatnonzero(window == extreme)
    else:
        extreme = np.nanmin(window)
        hit_indices = np.flatnonzero(window == extreme)
    if hit_indices.size == 0:
        return np.nan
    last_hit = int(hit_indices[-1])
    return float(len(window) - 1 - last_hit)


def _rolling_periods_since_extreme(series: pd.Series, period: int, *, find_max: bool) -> pd.Series:
    """Vectorized ``rolling(period, min_periods=period).apply(_periods_since_rolling_extreme)``.

    Same output as the rolling apply, without a Python call per window. Bars since the
    most recent max (or min) inside each trailing ``period`` window; the latest bar is
    offset 0 and ties resolve to the most recent hit.

    Args:
        series: Price series to scan.
        period: Window length in bars.
        find_max: True for the rolling high, False for the rolling low.

    Returns:
        Float series aligned to ``series``. NaN for the first ``period - 1`` bars and for
        any window containing a NaN or ±inf, matching ``min_periods=period``.
    """
    values = series.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    if len(values) >= period:
        windows = np.lib.stride_tricks.sliding_window_view(values, period)
        ## Reverse each window so argmax/argmin return the most recent hit, which is
        ## already the distance in bars from the window's last element.
        reversed_windows = windows[:, ::-1]
        offsets = reversed_windows.argmax(axis=1) if find_max else reversed_windows.argmin(axis=1)
        ## min_periods=period means any NaN in the window yields NaN in the rolling apply.
        ## pandas rolling also treats ±inf as NaN, so non-finite values void the window too.
        complete = np.isfinite(windows).all(axis=1)
        out[period - 1 :] = np.where(complete, offsets, np.nan)
    return pd.Series(out, index=series.index, name=series.name)


def _rolling_percentile_rank(window: np.ndarray) -> float:
    """Percentile rank of the latest observation within ``window`` on [0, 1]."""
    if len(window) == 0:
        return np.nan
    valid = window[~np.isnan(window)]
    if len(valid) == 0:
        return np.nan
    current = valid[-1]
    below = np.sum(valid < current)
    equal = np.sum(valid == current)
    return float((below + 0.5 * equal) / len(valid))


def _volume_percentile_vs_prior(
    volume: pd.Series,
    *,
    lookback: int = 252,
) -> pd.Series:
    """Rank current volume against the prior ``lookback`` sessions (current excluded)."""
    historical = volume.shift(1)
    hist_vals = historical.to_numpy(dtype=np.float64)
    curr_vals = volume.to_numpy(dtype=np.float64)
    n = len(volume)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(lookback - 1, n):
        window = hist_vals[i - lookback + 1 : i + 1]
        if np.isnan(window).any():
            continue
        current = curr_vals[i]
        below = np.sum(window < current)
        equal = np.sum(window == current)
        out[i] = (below + 0.5 * equal) / lookback
    return pd.Series(out, index=volume.index)


def _choppiness_index_20d(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Dreiss choppiness index over 20 sessions."""
    period = 20
    tr = _true_range(high, low, close)
    tr_sum = tr.rolling(period, min_periods=period).sum()
    highest_high = high.rolling(period, min_periods=period).max()
    lowest_low = low.rolling(period, min_periods=period).min()
    price_range = highest_high - lowest_low
    ratio = tr_sum / price_range
    chop = 100.0 * np.log10(ratio) / np.log10(float(period))
    invalid = (price_range <= 0) | (tr_sum <= 0) | ratio.isna()
    chop = chop.where(~invalid, np.nan)
    return _sanitize(chop)


def _adx_20d(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Wilder ADX(20) from daily OHLC."""
    period = 20
    tr = _true_range(high, low, close)

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=high.index,
        dtype=np.float64,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=low.index,
        dtype=np.float64,
    )

    smoothed_tr = _wilder_smooth_sum_init(tr, period)
    smoothed_plus = _wilder_smooth_sum_init(plus_dm, period)
    smoothed_minus = _wilder_smooth_sum_init(minus_dm, period)

    plus_di = 100.0 * smoothed_plus / smoothed_tr
    minus_di = 100.0 * smoothed_minus / smoothed_tr
    di_sum = plus_di + minus_di
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum
    dx = dx.where((smoothed_tr > 0) & (di_sum > 0), np.nan)

    adx = _wilder_smooth_mean_init(dx, period)
    return _sanitize(adx)


def _aroon_oscillator_20d(high: pd.Series, low: pd.Series) -> pd.Series:
    """Aroon oscillator (Up − Down) over 20 sessions."""
    period = 20
    since_high = _rolling_periods_since_extreme(high, period, find_max=True)
    since_low = _rolling_periods_since_extreme(low, period, find_max=False)
    aroon_up = 100.0 * (period - since_high) / period
    aroon_down = 100.0 * (period - since_low) / period
    return _sanitize(aroon_up - aroon_down)


def _bollinger_band_width_20d(close: pd.Series) -> pd.Series:
    """Bollinger BandWidth with 20-day SMA and population std (ddof=0)."""
    period = 20
    middle = close.rolling(period, min_periods=period).mean()
    rolling_std = close.rolling(period, min_periods=period).std(ddof=0)
    width = 4.0 * rolling_std / middle
    return _sanitize(width.where(middle > 0, np.nan))


def _donchian_width_20d(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """20-day Donchian channel width normalized by close."""
    period = 20
    donchian_high = high.rolling(period, min_periods=period).max()
    donchian_low = low.rolling(period, min_periods=period).min()
    width = (donchian_high - donchian_low) / close
    return _sanitize(width.where(close > 0, np.nan))


def _range_position_20d(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Close location inside the 20-day high-low range."""
    period = 20
    highest_high = high.rolling(period, min_periods=period).max()
    lowest_low = low.rolling(period, min_periods=period).min()
    range_width = highest_high - lowest_low
    position = (close - lowest_low) / range_width
    return _sanitize(position.where(range_width > 0, np.nan))


def _range_compression_percentile_126d(donchian_width: pd.Series) -> pd.Series:
    """126-day percentile rank of ``donchian_width_20d``."""
    lookback = 126
    return _sanitize(
        donchian_width.rolling(lookback, min_periods=lookback).apply(
            _rolling_percentile_rank,
            raw=True,
        )
    )


def _relative_volume_20d(volume: pd.Series) -> pd.Series:
    """Current volume divided by the prior 20-day average (today excluded from benchmark)."""
    period = 20
    prior_avg = volume.shift(1).rolling(period, min_periods=period).mean()
    rel = volume / prior_avg
    return _sanitize(rel.where(prior_avg > 0, np.nan))


def _dollar_volume_20d(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Log of the 20-day mean daily dollar volume."""
    period = 20
    daily_dollar = close * volume
    avg = daily_dollar.rolling(period, min_periods=period).mean()
    return _sanitize(np.log(avg.where(avg > 0, np.nan)))


def _dollar_volume_change_20d(close: pd.Series, volume: pd.Series) -> pd.Series:
    """20-day average dollar-volume growth versus one month earlier."""
    period = 20
    daily_dollar = close * volume
    avg = daily_dollar.rolling(period, min_periods=period).mean()
    lagged = avg.shift(period)
    change = avg / lagged - 1.0
    return _sanitize(change.where(lagged > 0, np.nan))


def _signed_volume_pressure_20d(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Rolling signed-volume share over 20 sessions."""
    period = 20
    direction = np.sign(close.diff())
    signed = direction * volume
    numer = signed.rolling(period, min_periods=period).sum()
    denom = volume.rolling(period, min_periods=period).sum()
    return _sanitize(numer / denom.where(denom > 0, np.nan))


def _chaikin_money_flow_20d(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """Chaikin Money Flow over 20 sessions."""
    period = 20
    daily_range = high - low
    mfm = (2.0 * close - high - low) / daily_range
    mfm = mfm.where(daily_range > 0, 0.0)
    mfv = mfm * volume
    numer = mfv.rolling(period, min_periods=period).sum()
    denom = volume.rolling(period, min_periods=period).sum()
    return _sanitize(numer / denom.where(denom > 0, np.nan))


def _price_volume_correlation_60d(close: pd.Series, volume: pd.Series) -> pd.Series:
    """60-day rolling correlation between daily returns and log-volume changes."""
    period = 60
    returns = close.pct_change()
    log_vol_change = np.log(volume.replace(0, np.nan)).diff()
    corr = returns.rolling(period, min_periods=period).corr(log_vol_change)
    return _sanitize(corr)


def _ulcer_index_20d(close: pd.Series) -> pd.Series:
    """Ulcer Index: RMS of percentage drawdowns from a 20-day rolling peak."""
    period = 20
    rolling_peak = close.rolling(period, min_periods=period).max()
    pct_dd = close / rolling_peak - 1.0
    return _sanitize(np.sqrt(pct_dd.pow(2).rolling(period, min_periods=period).mean()))


def _drawdown_duration_252d(close: pd.Series) -> pd.Series:
    """Consecutive sessions below the rolling 252-day high."""
    period = 252
    rolling_high = close.rolling(period, min_periods=period).max()
    at_high = close >= (rolling_high - _FLOAT_EQ_TOL)

    values = close.to_numpy(dtype=np.float64) # noqa: F841
    highs = rolling_high.to_numpy(dtype=np.float64)
    at_high_arr = at_high.to_numpy(dtype=bool)
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        if np.isnan(highs[i]):
            continue
        if at_high_arr[i]:
            out[i] = 0.0
        elif i == 0 or np.isnan(out[i - 1]):
            out[i] = 1.0
        else:
            out[i] = out[i - 1] + 1.0
    return pd.Series(out, index=close.index)


def _time_since_60d_extreme(close: pd.Series, *, find_max: bool) -> pd.Series:
    """Bars since the most recent rolling close extreme inside a 60-day window."""
    return _rolling_periods_since_extreme(close, 60, find_max=find_max)


def _overnight_gap_1d(open_: pd.Series, close: pd.Series) -> pd.Series:
    """Signed overnight gap return: open_t / close_{t-1} - 1.

    Available at the current session open. NaN when previous close is missing
    or non-positive.
    """
    previous_close = close.shift(1)
    gap = open_ / previous_close - 1.0
    return _sanitize(gap.where(previous_close > 0, np.nan))


def _absolute_overnight_gap_1d(overnight_gap: pd.Series) -> pd.Series:
    """Absolute overnight gap magnitude (open-available)."""
    return _sanitize(overnight_gap.abs())


def _avg_abs_overnight_gap_20d(overnight_gap: pd.Series) -> pd.Series:
    """20-day mean absolute overnight gap (open-available; includes today's gap)."""
    return _sanitize(overnight_gap.abs().rolling(20, min_periods=20).mean())


def _overnight_gap_volatility_20d(overnight_gap: pd.Series) -> pd.Series:
    """20-day sample std of signed overnight gaps (open-available)."""
    return _sanitize(overnight_gap.rolling(20, min_periods=20).std(ddof=1))


def _gap_frequency_60d(overnight_gap: pd.Series, *, negative: bool) -> pd.Series:
    """60-day frequency of signed overnight gaps; missing gaps stay NaN (not zero)."""
    valid_gap = overnight_gap.notna()
    if negative:
        condition = overnight_gap < 0
    else:
        condition = overnight_gap > 0
    indicator = pd.Series(
        np.where(valid_gap, condition.astype(float), np.nan),
        index=overnight_gap.index,
        dtype=np.float64,
    )
    return _sanitize(indicator.rolling(60, min_periods=60).mean())


def _gap_vs_atr_20d(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """Signed overnight gap in points divided by prior-session Wilder ATR(20).

    Uses ``create_atr_dataframe`` (same ATR as ``atr_20d_pct``) and deliberately
    lags ATR by one bar so date-*t* high/low never enter the open-time feature.
    Available at the current session open.
    """
    atr_frame = create_atr_dataframe(
        pd.DataFrame({"high": high, "low": low, "close": close}),
        length=20,
    )
    atr_20d = atr_frame["atr"].astype(np.float64)
    prior_atr = atr_20d.shift(1)
    gap_points = open_ - close.shift(1)
    ratio = gap_points / prior_atr
    return _sanitize(ratio.where(prior_atr > 0, np.nan))


def _overnight_return_share_20d(open_: pd.Series, close: pd.Series) -> pd.Series:
    """Share of recent absolute overnight+intraday movement that occurred overnight.

    Requires the current-session close (close-available). Uses sums of absolute
    component returns so opposite-signed moves do not cancel.
    """
    previous_close = close.shift(1)
    overnight = open_ / previous_close - 1.0
    overnight = overnight.where(previous_close > 0, np.nan)
    intraday = close / open_ - 1.0
    intraday = intraday.where(open_ > 0, np.nan)
    rolling_abs_overnight = overnight.abs().rolling(20, min_periods=20).sum()
    rolling_abs_total = (overnight.abs() + intraday.abs()).rolling(20, min_periods=20).sum()
    share = rolling_abs_overnight / rolling_abs_total
    return _sanitize(share.where(rolling_abs_total > 0, np.nan))


def _relative_volume_1d(volume: pd.Series) -> pd.Series:
    """Current volume over trailing 20-day median volume (today excluded).

    Greater than 1 means above-median participation. Median baseline reduces
    sensitivity to isolated volume spikes versus a mean-based relative volume.
    """
    period = 20
    baseline = volume.shift(1).rolling(period, min_periods=period).median()
    return _safe_divide(volume, baseline)


def _daily_return(close: pd.Series) -> pd.Series:
    """One-day close-to-close return without forward-filling gaps."""
    return close.pct_change(fill_method=None)


def _close_location_value_1d(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Where the session close sits in the high-low range, on approximately [0, 1].

    Near 0: close near the session low. Near 1: close near the session high.
    Near 0.5: midpoint close. Zero-range sessions (high == low) yield NaN.
    Only tiny floating-point bound violations are clipped; material OHLC errors
    remain out of range rather than being hidden.
    """
    clv = _safe_divide(close - low, high - low)
    return _sanitize(_clip_minor_unit_interval(clv))


def _directional_close_location_1d(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """Direction-aware intraday conviction from daily return and close location.

    Positive when close location agrees with the daily return direction (e.g.
    down day closing near the low). Negative when they conflict (e.g. up day
    fading to the low). Magnitude near 1 is strong conviction; flat days (zero
    return) map to 0 even if CLV is undefined.
    """
    daily_return = _daily_return(close)
    daily_direction = pd.Series(np.sign(daily_return), index=close.index, dtype=np.float64)
    clv = _close_location_value_1d(high, low, close)
    ## Map CLV from [0, 1] onto [-1, 1], then sign by daily return direction.
    raw = daily_direction * (2.0 * clv - 1.0)
    ## Flat return: convention is zero confirmation rather than NaN from CLV.
    out = raw.where(daily_return != 0.0, 0.0)
    out = out.where(daily_return.notna(), np.nan)
    return _sanitize(out)


def _true_range_vs_atr_20d(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Current true range divided by the prior 20-day mean true range.

    Values above 1 indicate an unusually large session relative to the recent
    volatility regime. Today's true range is excluded from the ATR denominator
    so the feature is not self-referential.
    """
    tr = _true_range(high, low, close)
    atr_prior = tr.shift(1).rolling(20, min_periods=20).mean()
    return _safe_divide(tr, atr_prior)


def _signed_volume_1d(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Raw volume signed by the daily return direction.

    Positive on advances, negative on declines, zero when the close is unchanged
    or volume is zero. Scale-dependent across assets; better for within-asset
    analysis than as a primary cross-sectional confirmation variable.
    """
    daily_direction = pd.Series(np.sign(_daily_return(close)), index=close.index, dtype=np.float64)
    return _sanitize(daily_direction * volume)


def _directional_volume_confirmation_1d(
    close: pd.Series,
    volume: pd.Series,
    *,
    relative_volume: Optional[pd.Series] = None,
) -> pd.Series:
    """Daily return direction times relative volume (retains sign).

    Large positive: high-volume advance. Large negative: high-volume decline.
    Near zero: weak direction, low participation, or a flat session. For
    downside strategies, strongly negative values are bearish volume confirmation.
    """
    rel_vol = relative_volume if relative_volume is not None else _relative_volume_1d(volume)
    daily_direction = pd.Series(np.sign(_daily_return(close)), index=close.index, dtype=np.float64)
    return _sanitize(daily_direction * rel_vol)


def _followthrough_return_1d(close: pd.Series) -> pd.Series:
    """Current 1-day return signed by the prior session's return direction.

    Positive when today's return continues yesterday's direction; negative on
    reversals. Computed as of close_t with only prices through t (no shift of
    the feature onto the prior signal date).
    """
    daily_return = _daily_return(close)
    prior_direction = pd.Series(
        np.sign(daily_return.shift(1)),
        index=close.index,
        dtype=np.float64,
    )
    return _sanitize(prior_direction * daily_return)


def _followthrough_return_3d(close: pd.Series) -> pd.Series:
    """Trailing 3-day return signed by the 1-day return direction at t-3.

    Positive when the last three sessions continued the direction established
    immediately before that window. Available only as of date t; must not be
    backdated onto t-3 (that would look ahead from the signal date).
    """
    daily_return = _daily_return(close)
    trailing_3d = close.pct_change(3, fill_method=None)
    prior_direction = pd.Series(
        np.sign(daily_return.shift(3)),
        index=close.index,
        dtype=np.float64,
    )
    return _sanitize(prior_direction * trailing_3d)


