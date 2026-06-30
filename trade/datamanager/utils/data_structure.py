"""Structural timeseries sanitization for datamanager cache and return paths.

Sanitize normalizes index/column shape and clips to the requested window. It does
not dedupe, reindex to a business-day grid, or forward-fill — those are L3
certification repairs.

Comment density: domain policy

Core Functions:
    _data_structure_sanitize: Structural-only cleanup before cache write or return.
"""

from __future__ import annotations

from datetime import datetime
from typing import Union

import pandas as pd

from trade import HOLIDAY_SET
from trade.datamanager.exceptions import EmptyDataException
from trade.helpers.helper import to_datetime
from trade.helpers.Logging import setup_logger
from trade.datamanager.utils.logging import get_logging_level, UTILS_LOGGER_NAME

logger = setup_logger(UTILS_LOGGER_NAME, stream_log_level=get_logging_level())

PANDAS_DATA_HINT = Union[pd.Series, pd.DataFrame]


def _data_structure_sanitize(
    df: PANDAS_DATA_HINT,
    start: Union[datetime, str],
    end: Union[datetime, str],
    source_name: str = "",
) -> PANDAS_DATA_HINT:
    """Apply structural-only cleanup to a timeseries before cache or return.

    Does not dedupe, reindex to a B-day grid, or forward-fill. Duplicate indices,
    calendar gaps, and unexplained NaNs are handled by the certification pipeline
    (L2 raise / L3 fix).

    Args:
        df: Input timeseries indexed by date/datetime.
        start: Inclusive window start for clipping.
        end: Inclusive window end for clipping.
        source_name: Label for logging and ``EmptyDataException`` context.

    Returns:
        Sanitized copy of ``df`` clipped to ``[start, end]`` without holiday rows.

    Raises:
        TypeError: When input is not a Series/DataFrame or index cannot be coerced.
        EmptyDataException: When no rows remain after clipping.
    """
    logger.info(f"Sanitizing data from {start} to {end}...")
    if not isinstance(df, (pd.Series, pd.DataFrame)):
        raise TypeError(f"Expected pd.Series or pd.DataFrame for sanitization, got {type(df)}")

    df = df.copy()

    # Ensure DatetimeIndex. If not, attempt conversion
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.DatetimeIndex(to_datetime(df.index))
        except Exception as exc:
            raise TypeError("Expected DatetimeIndex for sanitization of timeseries data.") from exc

    # Strip tz awareness so index comparisons stay consistent across sources
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)

    # Dedupe, B-day reindex, and ffill moved to L3 certification — not sanitize

    # Sort the index
    df = df.sort_index()

    # if dataframe, lower case columns
    if isinstance(df, pd.DataFrame):
        df.columns = df.columns.str.lower()

    # Filter by start and end dates
    df = df[(df.index.date >= to_datetime(start).date()) & (df.index.date <= to_datetime(end).date())]

    if df.empty:
        raise EmptyDataException(
            f"No data available after sanitization between {start} and {end}. Source: {source_name}"
        )

    # Re-sort after filtering
    df = df.sort_index()

    # Index name=datetime
    df.index.name = "datetime"

    # Filter out holidays
    df = df[~df.index.strftime("%Y-%m-%d").isin(HOLIDAY_SET)]

    return df
