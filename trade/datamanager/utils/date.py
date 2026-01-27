import pandas as pd
from dataclasses import dataclass
from datetime import datetime, date
from pandas.tseries.offsets import BDay
from trade.helpers.helper import to_datetime, is_busday, is_USholiday
from trade.helpers.helper import ny_now
from trade.optionlib.assets.dividend import SECONDS_IN_DAY, SECONDS_IN_YEAR
from trade.datamanager.vars import TODAY_RELOAD_CUTOFF
from trade.helpers.helper import CustomCache, generate_option_tick_new
from trade.datamanager._enums import OptionSpotEndpointSource
from trade.helpers.helper import is_market_hours_today
from trade.helpers.Logging import setup_logger
from dbase.DataAPI.ThetaData import list_dates
from pathlib import Path
import os
from typing import Tuple, List, Optional, Union
logger = setup_logger("trade.datamanager.utils", stream_log_level="INFO")

DATE_HINT = Union[datetime, str]
PATH = Path(os.environ["GEN_CACHE_PATH"]) / "dm_gen_cache"

## This cache will be used to save the min trading date for each option tick
## This is to avoid calling API all the time
LIST_DATE_CACHE = CustomCache(
    location=PATH.as_posix(),
    fname="list_date_cache",
    clear_on_exit=False,
    expire_days=365,
)

def sync_date_index(*args) -> List[Union[pd.Series, pd.DataFrame]]:
    """Synchronizes the date indices of multiple time series."""
    for i, ts in enumerate(args):
        if ts is None:
            raise ValueError("All time series must be provided and not None. Found None at position {}".format(i))
        if not isinstance(ts, (pd.Series, pd.DataFrame)):
            raise TypeError(
                "All inputs must be pandas Series or DataFrame. Found {} at position {}".format(type(ts), i)
            )
    date_indices = [set(ts.index) for ts in args if ts is not None]
    common_dates = list(set.intersection(*date_indices))
    synced_series = [ts.loc[common_dates] if ts is not None else None for ts in args]
    synced_series = [ts.sort_index() if ts is not None else None for ts in synced_series]
    return synced_series


def time_distance_helper(start: datetime, end: datetime) -> float:
    """Calculates time distance in years between two dates."""
    delta = (to_datetime(end) - to_datetime(start)).days * SECONDS_IN_DAY
    return delta / SECONDS_IN_YEAR


def _sync_date(
    symbol: str,
    start_date: DATE_HINT,
    end_date: DATE_HINT,
    strike: Optional[float] = None,
    expiration: Optional[Union[datetime, str]] = None,
    right: Optional[str] = None,
    endpoint_source: Optional[OptionSpotEndpointSource] = OptionSpotEndpointSource.EOD,
) -> Tuple[datetime, datetime]:
    """Synchronizes requested dates with available data range from Thetadata.

    Queries Thetadata for available dates for the specified option contract and
    adjusts the requested date range to fit within available data bounds.

    Args:
        start_date: Requested start date.
        end_date: Requested end date.
        strike: Option strike price.
        expiration: Option expiration date.
        right: Option type ("C" for call, "P" for put).
        endpoint_source: Source of option spot data.

    Returns:
        Tuple of (adjusted_start_date, adjusted_end_date) constrained to
        available data range.

    Examples:
        >>> opt_mgr = OptionSpotDataManager("AAPL")
        >>> start, end = opt_mgr._sync_date(
        ...     start_date="2025-01-01",
        ...     end_date="2025-12-31",
        ...     strike=150.0,
        ...     expiration="2025-06-20",
        ...     right="C"
        ... )

    Notes:
        - Constrains start_date to max(requested_start, min_available_date)
        - Constrains end_date to min(requested_end, max_available_date)
        - Prevents requesting dates outside available data range
    """

    ## Process Note:
    ## list of dates is only important for min date
    ## Once we have min date, all dates after that are available until expiration or today
    ## For end date, we just need to compare requested end date with expiration/today.
    ## There is added logic for EOD source since data is only available after market close
    opttick = generate_option_tick_new(
        symbol=symbol,
        exp=expiration,
        right=right,
        strike=strike,
    )

    def _get_max_date(requested_end: datetime) -> datetime:
        """
        Determines the maximum allowable end date based on requested end date,
        option expiration, and data source constraints.

        Note: We don't really need list of dates. min_date is < requested_date, all dates in between are available

        Args:
            requested_end: The originally requested end date.
        """
        
        if to_datetime(requested_end) <= to_datetime(expiration):
            ## EOD report is produced after 6pm,
            ## so max date is prev bus day as long as it is trading hours
            if endpoint_source == OptionSpotEndpointSource.EOD:
                max_allowed = prev_busday if is_market_hrs else today
            else:
                max_allowed = today

            ## Get max date within allowed range
            max_date = to_datetime(min(max_allowed.date(), to_datetime(requested_end).date()))

        ## Else, max date is expiration
        else:
            max_date = to_datetime(expiration)

        return max_date

    is_market_hrs = is_market_hours_today()
    today = ny_now()
    prev_busday = (today - BDay(1)).to_pydatetime()
    start_date = to_datetime(start_date)
    end_date = to_datetime(end_date)

    if opttick in LIST_DATE_CACHE.keys():
        logger.info(f"Using cached date range for {start_date} - {end_date} and option tick {opttick}")
        cached_dates = LIST_DATE_CACHE.get(key=opttick)
        min_date = cached_dates["min_date"]
        max_date = _get_max_date(end_date)

        start_date = max(min_date, start_date)
        end_date = min(max_date, end_date)
        return min(start_date, end_date), max(start_date, end_date)

    logger.info(f"Fetching date range from Thetadata for {opttick}")
    dates = list_dates(
        symbol=symbol,
        exp=expiration,
        right=right,
        strike=strike,
    )

    if not dates:
        raise ValueError(f"No trading dates found for {opttick}")

    dates = to_datetime(dates)

    ## Adjust start date to min
    min_date = min(dates)
    start_date = max(min_date, start_date)
    end_date = _get_max_date(end_date)

    LIST_DATE_CACHE.set(key=opttick, value={"min_date": min_date}, expire=None)

    return min(start_date, end_date), max(start_date, end_date)


@dataclass(slots=True)
class DateRangePacket:
    """
    Simple container for start/end date ranges with both datetime and string formats.
    """

    start_date: DATE_HINT
    end_date: DATE_HINT
    start_str: Optional[str] = None
    end_str: Optional[str] = None
    maturity_date: Optional[DATE_HINT] = None
    maturity_str: Optional[str] = None

    def __post_init__(self):
        self.start_date = to_datetime(self.start_date)
        self.end_date = to_datetime(self.end_date)
        if self.maturity_date is not None:
            self.maturity_date = to_datetime(self.maturity_date)

        self.start_str = self.start_str or self.start_date.strftime("%Y-%m-%d")
        self.end_str = self.end_str or self.end_date.strftime("%Y-%m-%d")
        if self.maturity_date is not None:
            self.maturity_str = self.maturity_str or self.maturity_date.strftime("%Y-%m-%d")
        else:
            self.maturity_str = None


def _should_save_today(max_date: date) -> bool:
    """
    Determines if data should be saved today based on the max_date and current time in New York.
    """
    today = date.today()
    current_time = ny_now().time()
    return max_date >= today and current_time >= TODAY_RELOAD_CUTOFF


def is_available_on_date(date: date) -> bool:
    """
    Returns True if the given date is a business day and not a US holiday, False otherwise.
    """
    date = to_datetime(date).strftime("%Y-%m-%d")
    return is_busday(date) and not is_USholiday(date)
