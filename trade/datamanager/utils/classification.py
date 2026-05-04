"""Option spot date classification for gap-aware caching.

Classifies fetched option date slices into observed rows versus confirmed-missing
dates so the cache layer knows which dates to never re-request from the API.

Core Dataclasses:
    DateClassification: Output container for the three date buckets.

Core Functions:
    classify_option_spot_dates: Splits a fetched DataFrame into observed and
        checked-missing dates given the requested valid window.

Processing Flow:
    1. Build the full set of expected business dates from the valid window.
    2. Compare against what the API actually returned.
    3. Dates absent from the response are classified as checked-missing.
    4. Dates present with at least one non-NaN price value are observed.

Usage:
    >>> from trade.datamanager.utils.classification import classify_option_spot_dates
    >>> result = classify_option_spot_dates(
    ...     fetched=df,
    ...     valid_start="2026-01-05",
    ...     valid_end="2026-01-09",
    ... )
    >>> result.observed_dates
    DatetimeIndex(['2026-01-05', '2026-01-06', '2026-01-07'], dtype='datetime64[ns]', freq=None)
    >>> result.checked_missing_dates
    [datetime.date(2026, 1, 8), datetime.date(2026, 1, 9)]
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Union

import pandas as pd

from trade.helpers.helper import get_missing_dates, to_datetime
from trade import HOLIDAY_SET


@dataclass
class DateClassification:
    """Output of the option spot date classifier.

    Attributes:
        observed_dates: Index of dates where the API returned at least one
            non-NaN price value.
        checked_missing_dates: Business dates inside the valid window that the
            API was asked about but returned no usable data. These should be
            stored in the cache so they are never re-requested.
    """

    observed_dates: pd.DatetimeIndex = field(default_factory=pd.DatetimeIndex)
    checked_missing_dates: List[datetime] = field(default_factory=list)


def classify_option_spot_dates(
    fetched: pd.DataFrame,
    valid_start: Union[datetime, str],
    valid_end: Union[datetime, str],
) -> DateClassification:
    """Classifies fetched option dates into observed rows and confirmed-missing dates.

    Compares the dates present in the API response against the full set of
    expected business days in the valid window. Dates absent from the response
    or present with all-NaN values are marked as checked-missing so they are
    never re-fetched.

    Args:
        fetched: DataFrame returned by the API, indexed by DatetimeIndex.
            May be empty or contain all-NaN rows for some dates.
        valid_start: First date of the valid query window (already synced by
            _sync_date).
        valid_end: Last date of the valid query window (already synced by
            _sync_date).

    Returns:
        DateClassification with observed_dates and checked_missing_dates populated.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> idx = pd.bdate_range("2026-01-05", "2026-01-09")
        >>> df = pd.DataFrame({"close": [10.0, np.nan, 12.0, np.nan, np.nan]}, index=idx)
        >>> result = classify_option_spot_dates(df, "2026-01-05", "2026-01-09")
        >>> list(result.observed_dates.strftime("%Y-%m-%d"))
        ['2026-01-05', '2026-01-07']
        >>> sorted(str(d) for d in result.checked_missing_dates)
        ['2026-01-06', '2026-01-08', '2026-01-09']
    """
    valid_start_dt = to_datetime(valid_start)
    valid_end_dt = to_datetime(valid_end)

    # Full set of expected business days in the valid window, excluding holidays.
    expected_bus_days = pd.bdate_range(start=valid_start_dt, end=valid_end_dt)
    expected_bus_days = pd.DatetimeIndex([d for d in expected_bus_days if d.strftime("%Y-%m-%d") not in HOLIDAY_SET])

    if fetched is None or fetched.empty:
        return DateClassification(
            observed_dates=pd.DatetimeIndex([]),
            checked_missing_dates=list(expected_bus_days.date),
        )

    if not isinstance(fetched.index, pd.DatetimeIndex):
        fetched = fetched.copy()
        fetched.index = to_datetime(fetched.index)

    # Dates where at least one column has a real value.
    has_data_mask = ~fetched.isna().all(axis=1)
    observed_idx = fetched.index[has_data_mask]

    # All expected dates not in the observed set are checked-missing.
    observed_set = set(observed_idx.normalize())
    checked_missing = [d.date() for d in expected_bus_days if d.normalize() not in observed_set]

    return DateClassification(
        observed_dates=observed_idx,
        checked_missing_dates=checked_missing,
    )
