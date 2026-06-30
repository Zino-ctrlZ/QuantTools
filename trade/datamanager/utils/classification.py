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
    2. Fetch vendor ``list_dates`` calendar for the contract (live ThetaData API
       for non-expired options; ``LIST_DATE_CACHE`` only for expired contracts).
    3. Mark dates with at least one non-NaN price column in ``fetched`` as observed.
    4. ``checked_missing`` = expected − vendor_listed − observed (see Meaning below).

Meaning (checked_missing policy):
    A day is ``checked_missing`` only when all three hold:
    - it is an expected business day in the window,
    - vendor ``list_dates`` does not list it, **and**
    - the EOD/quote fetch did not return a usable price row.

    If ``retrieve_eod_ohlc`` returned a real price, the day is **not** missing —
    even when ``list_dates`` intermittently omits it. If ``list_dates`` lists a day
    but the price fetch is blank, that is **not** allowed missing (cert should fail).

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

    For non-expired contracts this always calls live ThetaData ``list_dates`` (no
    ``LIST_DATE_CACHE`` read). Only expired contracts may return a cached range.

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
        observed_dates: Index of dates where the price fetch returned at least one
            non-NaN value (EOD or quote endpoint).
        checked_missing_dates: Expected business days that are neither listed by
            vendor ``list_dates`` nor present with usable prices in ``fetched``.
            These may be cached as intentional gaps so they are not re-requested.
            A day with a real fetched price is never checked-missing, even if
            ``list_dates`` omits it.
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

    Compares expected business days against vendor ``list_dates`` and the actual
    price rows in ``fetched``. ``checked_missing`` is the reconciled diff:

        checked_missing = expected − vendor_listed − observed_price_days

    Meaning:
        - Real fetched price → not missing (even if ``list_dates`` forgot the day).
        - ``list_dates`` lists the day but fetch is blank → not checked-missing;
          certification should treat that as unexplained NaN.
        - Neither listed nor fetched → checked-missing (vendor-confirmed gap).

    Args:
        fetched: DataFrame returned by the price API, indexed by DatetimeIndex.
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
    expected_dates = set(expected_bus_days.date)
    vendor_listed_dates = set(pd.DatetimeIndex(trading_days).date)

    observed_date_set: set = set()
    observed_idx = pd.DatetimeIndex([])
    if fetched is not None and not fetched.empty:
        if not isinstance(fetched.index, pd.DatetimeIndex):
            fetched = fetched.copy()
            fetched.index = to_datetime(fetched.index)

        ## Dates where at least one column has a real value.
        has_data_mask = ~fetched.isna().all(axis=1)
        observed_idx = fetched.index[has_data_mask]
        observed_date_set = set(pd.DatetimeIndex(observed_idx).normalize().date)

    ## Reconciled diff: expected days that are neither vendor-listed nor price-observed.
    checked_missing = list(expected_dates - vendor_listed_dates - observed_date_set)

    if fetched is None or fetched.empty:
        ## No spot rows loaded — vendor calendar minus any observed dates (empty here)
        ## defines gaps for vol/greeks cert when spot was not fetched in this call.
        return DateClassification(
            observed_dates=pd.DatetimeIndex([]),
            checked_missing_dates=checked_missing,
        )

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

    Uses ``get_option_dates`` (live vendor ``list_dates`` for non-expired contracts)
    against the sync'd B-day window. When ``fetched`` is empty (this helper's path),
    observed prices are not available so gaps follow ``expected − vendor_listed``.

    Spot, vol, and greeks share the same vendor gaps — no spot timeseries load required.

    Args:
        symbol: Underlying ticker.
        strike: Option strike.
        right: Option right (call/put).
        expiration: Option expiration.
        valid_start: Sync'd window start (from ``_sync_date``).
        valid_end: Sync'd window end (from ``_sync_date``).

    Returns:
        Business dates in the window that the vendor does not list for this contract
        and for which no price rows were supplied to the classifier.

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
