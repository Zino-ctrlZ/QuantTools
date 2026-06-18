"""Option spot date classification for gap-aware caching.

Classifies fetched option date slices into observed rows versus confirmed-missing
dates so the cache layer knows which dates to never re-request from the API.

Core Dataclasses:
    DateClassification: Output container for the three date buckets.

Core Functions:
    classify_option_spot_dates: Splits a fetched DataFrame into observed and
        checked-missing dates given the requested valid window.
    resolve_checked_missing_dates_for_option_contract: Vendor-calendar gaps for a
        contract window without loading spot timeseries (vol/greeks/certifier).

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

from trade.helpers.helper import to_datetime, generate_option_tick_new
from trade import HOLIDAY_SET
from .date import LIST_DATE_CACHE
from ..vars import get_enable_caching
from dbase.DataAPI.ThetaData import list_dates

def get_option_dates(
    ticker: str,
    strike: float,
    right: str,
    expiration: datetime,
) -> List[datetime]:
    """Fetches the list of dates for which the API has option spot data for the given parameters.

    Args:
        ticker: The underlying symbol.
        strike: The option strike price.
        right: The option right ('call' or 'put').
        expiration: The option expiration date.
    Returns:
        A list of datetime objects representing the dates for which option spot data is available.
    """

    option_has_expired = expiration < datetime.now()
    opttick = generate_option_tick_new(
        symbol=ticker,
        strike=strike,
        right=right,
        exp=expiration,
    )

    ## Only use cache if option has expired, otherwise always fetch from source to capture new data availability
    if opttick in LIST_DATE_CACHE and option_has_expired:
        return LIST_DATE_CACHE[opttick]["range"]

    available_dates = list_dates(
        symbol=ticker,
        strike=strike,
        right=right,
        exp=expiration,
    )
    

    ## Only cache if option has expired
    if get_enable_caching() and option_has_expired:
        LIST_DATE_CACHE[opttick] = {
            "range": available_dates,
            "last_updated": datetime.now(),
            "min_date": min(available_dates) if available_dates else None,
            "max_date": max(available_dates) if available_dates else None,
        }
    return available_dates

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
    symbol: str,
    strike: float,
    right: str,
    expiration: datetime,
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
        symbol: The underlying symbol.
        strike: The option strike price.
        right: The option right ('call' or 'put').
        expiration: The option expiration date.
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
        >>> result = classify_option_spot_dates(df, "AAPL", 150.0, "C", datetime(2026, 1, 9), "2026-01-05", "2026-01-09")
        >>> list(result.observed_dates.strftime("%Y-%m-%d"))
        ['2026-01-05', '2026-01-07']
        >>> sorted(str(d) for d in result.checked_missing_dates)
        ['2026-01-06', '2026-01-08', '2026-01-09']
    """
    valid_start_dt = to_datetime(valid_start)
    valid_end_dt = to_datetime(valid_end)

    # Full set of expected business days in the valid window, excluding holidays.
    trading_days = to_datetime(get_option_dates(symbol, strike, right, expiration))
    expected_bus_days = pd.bdate_range(start=valid_start_dt, end=valid_end_dt)
    expected_bus_days = pd.DatetimeIndex([d for d in expected_bus_days if d.strftime("%Y-%m-%d") not in HOLIDAY_SET])

    # Checked missing dates are those expected business days that are not present in the fetched data.
    # This means we asked the API for these dates but got no usable data back, so we should cache them as missing.
    # FYI: This isn't intended to be actual missing dates in the data.
    # It's intention is to identify dates that the API DOESN'T have data. There are instances where v2 data did not exist
    # But v3 data does exist. So we want to make sure we don't cache those dates as missing if they are just missing in v2 but
    # do exist in v3. Eg: NVDA20220121C375
    checked_missing = list(set(expected_bus_days.date) - set(trading_days.date))

    if fetched is None or fetched.empty:
        ## No spot rows loaded — vendor calendar still defines which dates are absent for
        ## the whole contract (spot, vol, greeks share the same list_dates coverage).
        return DateClassification(
            observed_dates=pd.DatetimeIndex([]),
            checked_missing_dates=checked_missing,
        )

    if not isinstance(fetched.index, pd.DatetimeIndex):
        fetched = fetched.copy()
        fetched.index = to_datetime(fetched.index)

    # Dates where at least one column has a real value.
    has_data_mask = ~fetched.isna().all(axis=1)
    observed_idx = fetched.index[has_data_mask]

    # # All expected dates not in the observed set are checked-missing.
    # observed_set = set(observed_idx.normalize())
    # checked_missing = [d.date() for d in expected_bus_days if d.normalize() not in observed_set]

    return DateClassification(
        observed_dates=observed_idx,
        checked_missing_dates=checked_missing,
    )


def resolve_checked_missing_dates_for_option_contract(
    symbol: str,
    strike: float,
    right: str,
    expiration: datetime,
    valid_start: Union[datetime, str],
    valid_end: Union[datetime, str],
) -> List[datetime]:
    """Resolve vendor checked-missing dates for an option contract window.

    Uses ``get_option_dates`` (vendor list_dates) against the sync'd B-day window.
    Spot, vol, and greeks share the same vendor gaps — no spot timeseries load required.

    Args:
        symbol: Underlying ticker.
        strike: Option strike.
        right: Option right (call/put).
        expiration: Option expiration.
        valid_start: Sync'd window start (from ``_sync_date``).
        valid_end: Sync'd window end (from ``_sync_date``).

    Returns:
        Business dates in the window that the vendor does not list for this contract.

    Examples:
        >>> gaps = resolve_checked_missing_dates_for_option_contract(
        ...     "AAPL", 150.0, "C", datetime(2026, 6, 20), "2026-01-05", "2026-01-09"
        ... )
    """
    classification = classify_option_spot_dates(
        fetched=pd.DataFrame(),
        symbol=symbol,
        strike=strike,
        right=right,
        expiration=expiration,
        valid_start=valid_start,
        valid_end=valid_end,
    )
    return list(classification.checked_missing_dates)
