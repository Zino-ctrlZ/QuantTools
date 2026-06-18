"""Point-in-time value resolution from a short certified lookback window.

Managers fetch the last N business days through the requested date, certify at L1,
then resolve a single row per ``RealTimeFallbackOption``. Each manager supplies
its own timeseries fetcher; this module owns lookback bounds and fallback selection.

Comment density: orchestration

Processing Flow:
    1. Compute lookback window ``[as_of - lookback_bdays, as_of]`` (business days).
    2. ``fetch_timeseries(start, end)`` — manager pulls and certifies at L1.
    3. ``extract_timeseries`` — Series or DataFrame from the manager result.
    4. ``_resolve_from_window`` — exact date, else fallback within the window.

Core Functions:
    resolve_value_at_date: Fetch lookback window and return one indexed row + metadata.

Usage:
    >>> row, meta = resolve_value_at_date(
    ...     "2025-01-15",
    ...     fetch_timeseries=lambda s, e: mgr.get_risk_free_rate_timeseries(
    ...         s, e, certification_level=CertificationLevel.L1
    ...     ),
    ...     extract_timeseries=lambda r: r.timeseries,
    ...     fallback_option=RealTimeFallbackOption.USE_LAST_AVAILABLE,
    ... )
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Optional, Union

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

from trade.datamanager._enums import CertificationLevel, RealTimeFallbackOption
from trade.datamanager.utils.date import is_available_on_date
from trade.helpers.helper_types import DATE_HINT


@dataclass(frozen=True, slots=True)
class PointInTimeMeta:
    """Metadata for how a point-in-time value was resolved.

    Attributes:
        requested_date: Date the caller asked for (normalized).
        resolved_date: Index date the value was taken from (may differ under fallback).
        fallback_option: Fallback policy applied.
        used_fallback: True when value came from a date other than requested.
        window_start: Lookback window start (YYYY-MM-DD).
        window_end: Lookback window end (YYYY-MM-DD), equal to requested date.
        certification_level: Level used on the window fetch (default L1).
        source_result: Manager result from the window fetch (metadata for callers).
    """

    requested_date: datetime
    resolved_date: datetime
    fallback_option: RealTimeFallbackOption
    used_fallback: bool
    window_start: str
    window_end: str
    certification_level: CertificationLevel = CertificationLevel.L1
    source_result: Any = None


def _lookback_window_bounds(
    as_of: DATE_HINT,
    *,
    lookback_bdays: int = 10,
) -> tuple[str, str, datetime]:
    """Return ``(start_str, end_str, requested_dt)`` for the last N B-days through ``as_of``.

    Args:
        as_of: Requested valuation date.
        lookback_bdays: Business days to look back from ``as_of`` (inclusive end).

    Returns:
        Start string, end string (``as_of``), and normalized requested datetime.
    """
    requested_dt = pd.to_datetime(as_of).normalize()
    start_str = (requested_dt - BDay(lookback_bdays)).strftime("%Y-%m-%d")
    end_str = requested_dt.strftime("%Y-%m-%d")
    return start_str, end_str, requested_dt


def _normalize_timeseries_index(
    timeseries: Union[pd.Series, pd.DataFrame],
) -> Union[pd.Series, pd.DataFrame]:
    """Normalize index to midnight and sort for ``.loc`` lookups.

    Args:
        timeseries: Input series or frame.

    Returns:
        Copy with normalized, sorted DatetimeIndex.
    """
    out = timeseries.copy()
    out.index = pd.to_datetime(out.index).normalize()
    return out.sort_index()


def _row_is_valid(
    row: Union[pd.Series, pd.DataFrame],
    *,
    column: Optional[str] = None,
) -> bool:
    """Return True when the candidate row has a non-NaN value.

    Args:
        row: Single-row slice.
        column: Optional DataFrame column to check.

    Returns:
        True if at least one non-NaN value is present.
    """
    if row.empty:
        return False
    if isinstance(row, pd.DataFrame):
        frame = row if column is None else row[[column]]
        return bool(frame.notna().any().any())
    if column is not None and column in row.index:
        return bool(pd.notna(row[column]))
    return bool(pd.notna(row).any())


def _take_row(
    timeseries: Union[pd.Series, pd.DataFrame],
    resolved_dt: datetime,
    *,
    column: Optional[str] = None,
) -> Union[pd.Series, pd.DataFrame]:
    """Extract one row from the window timeseries.

    Args:
        timeseries: Normalized window data.
        resolved_dt: Index date to extract.
        column: Optional DataFrame column subset.

    Returns:
        Single-row Series or DataFrame.
    """
    if isinstance(timeseries, pd.DataFrame):
        row = timeseries.loc[[resolved_dt]]
        return row if column is None else row[[column]]
    return timeseries.loc[[resolved_dt]]


def _reindex_to_requested(
    row: Union[pd.Series, pd.DataFrame],
    requested_dt: datetime,
) -> Union[pd.Series, pd.DataFrame]:
    """Reindex the row to the requested date for downstream alignment.

    Args:
        row: Resolved value row.
        requested_dt: Caller-requested date.

    Returns:
        Same values with index ``[requested_dt]``.
    """
    out = row.copy()
    out.index = [requested_dt]
    return out


def _build_sentinel(
    requested_dt: datetime,
    fallback_option: RealTimeFallbackOption,
    *,
    column: Optional[str] = None,
    as_dataframe: bool = False,
) -> Union[pd.Series, pd.DataFrame]:
    """Build a single-row NaN or 0.0 placeholder for NAN / ZEROED fallbacks.

    Args:
        requested_dt: Caller-requested date.
        fallback_option: ``NAN`` or ``ZEROED``.
        column: Optional DataFrame column name.
        as_dataframe: When True, return a one-row DataFrame.

    Returns:
        Sentinel row indexed at ``requested_dt``.
    """
    fill = np.nan if fallback_option == RealTimeFallbackOption.NAN else 0.0
    if as_dataframe:
        col = column if column is not None else "value"
        return pd.DataFrame({col: [fill]}, index=[requested_dt])
    return pd.Series([fill], index=[requested_dt], dtype=float)


def _resolve_from_window(
    timeseries: Union[pd.Series, pd.DataFrame],
    requested_dt: datetime,
    fallback_option: RealTimeFallbackOption,
    *,
    column: Optional[str] = None,
    trading_day_required: bool = True,
) -> tuple[Union[pd.Series, pd.DataFrame], datetime, bool]:
    """Select one row from the lookback window per ``fallback_option``.

    Args:
        timeseries: Certified window payload.
        requested_dt: Caller-requested date (normalized).
        fallback_option: Policy when exact date is missing or non-trading.
        column: Optional DataFrame column for scalar extraction.
        trading_day_required: When True, non-trading days use fallback policy.

    Returns:
        ``(single_row, resolved_date, used_fallback)``.

    Raises:
        ValueError: When ``RAISE_ERROR`` and no valid value is available.
    """
    ts = _normalize_timeseries_index(timeseries)
    as_dataframe = isinstance(ts, pd.DataFrame)

    ## Non-trading / pre-market today — no recursive refetch; search backward in window.
    if trading_day_required and not is_available_on_date(requested_dt.date()):
        if fallback_option == RealTimeFallbackOption.RAISE_ERROR:
            raise ValueError(f"Date {requested_dt.date()} is not available for market data.")
        if fallback_option in (RealTimeFallbackOption.NAN, RealTimeFallbackOption.ZEROED):
            return (
                _build_sentinel(requested_dt, fallback_option, column=column, as_dataframe=as_dataframe),
                requested_dt,
                True,
            )

    ## Exact hit on requested date.
    if requested_dt in ts.index and _row_is_valid(ts.loc[[requested_dt]], column=column):
        return _take_row(ts, requested_dt, column=column), requested_dt, False

    ## Exact date missing or NaN — apply fallback within the lookback window.
    if fallback_option == RealTimeFallbackOption.RAISE_ERROR:
        raise ValueError(f"No data found for date {requested_dt.date()}.")

    if fallback_option in (RealTimeFallbackOption.NAN, RealTimeFallbackOption.ZEROED):
        return (
            _build_sentinel(requested_dt, fallback_option, column=column, as_dataframe=as_dataframe),
            requested_dt,
            True,
        )

    ## USE_LAST_AVAILABLE: last valid row on or before requested date.
    on_or_before = ts[ts.index <= requested_dt]
    if on_or_before.empty:
        raise ValueError(f"No prior data in lookback window for date {requested_dt.date()}.")

    if isinstance(on_or_before, pd.DataFrame):
        valid_mask = on_or_before.notna().any(axis=1) if column is None else on_or_before[column].notna()
    else:
        valid_mask = on_or_before.notna()

    candidates = on_or_before[valid_mask]
    if candidates.empty:
        raise ValueError(f"No prior non-NaN data in lookback window for date {requested_dt.date()}.")

    resolved_dt = candidates.index[-1]
    row = _take_row(ts, resolved_dt, column=column)
    return _reindex_to_requested(row, requested_dt), resolved_dt, True


def resolve_value_at_date(
    as_of: DATE_HINT,
    *,
    fetch_timeseries: Callable[[str, str], Any],
    extract_timeseries: Callable[[Any], Union[pd.Series, pd.DataFrame]],
    fallback_option: RealTimeFallbackOption,
    column: Optional[str] = None,
    lookback_bdays: int = 10,
    certification_level: CertificationLevel = CertificationLevel.L1,
    trading_day_required: bool = True,
) -> tuple[Union[pd.Series, pd.DataFrame], PointInTimeMeta]:
    """Fetch a lookback window at L1 and resolve one row for ``as_of``.

    Args:
        as_of: Requested valuation date.
        fetch_timeseries: Manager callback ``(start_str, end_str) -> result``.
            Should pass ``certification_level`` through to the timeseries getter.
        extract_timeseries: Pull Series/DataFrame from the manager result.
        fallback_option: How to behave when exact date is missing or non-trading.
        column: Optional DataFrame column for scalar greeks or OHLC.
        lookback_bdays: Business days to look back from ``as_of`` (default 10).
        certification_level: Level for the window fetch (default L1).
        trading_day_required: When True, non-trading days trigger fallback policy.

    Returns:
        ``(single_row, meta)`` — row is indexed at **requested** date even when
        the value came from an earlier date (``USE_LAST_AVAILABLE``).

    Raises:
        ValueError: When fallback policy cannot resolve a value from the window.
    """
    start_str, end_str, requested_dt = _lookback_window_bounds(as_of, lookback_bdays=lookback_bdays)

    result = fetch_timeseries(start_str, end_str)
    window_ts = extract_timeseries(result)

    row, resolved_dt, used_fallback = _resolve_from_window(
        window_ts,
        requested_dt,
        fallback_option,
        column=column,
        trading_day_required=trading_day_required,
    )

    meta = PointInTimeMeta(
        requested_date=requested_dt,
        resolved_date=resolved_dt,
        fallback_option=fallback_option,
        used_fallback=used_fallback,
        window_start=start_str,
        window_end=end_str,
        certification_level=certification_level,
        source_result=result,
    )
    return row, meta
