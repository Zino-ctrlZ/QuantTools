import pandas as pd
from dataclasses import dataclass
from datetime import datetime, date
from pandas.tseries.offsets import BDay
from trade.helpers.helper import to_datetime, is_busday, is_USholiday, is_pre_market_hours, is_post_market_hours
from trade.helpers.helper import ny_now
from trade.optionlib.assets.dividend import SECONDS_IN_DAY, SECONDS_IN_YEAR  # noqa
from trade.datamanager.vars import TODAY_RELOAD_CUTOFF, MIN_TIME_BEFORE_REAL_TIME, get_enable_caching
from trade.helpers.helper_types import DATE_HINT
from trade.helpers.helper import time_distance_helper  # noqa
from trade.helpers.helper import CustomCache, generate_option_tick_new
from trade.datamanager._enums import OptionSpotEndpointSource
from trade.helpers.helper import is_market_hours_today
from trade.helpers.helper_types import is_iterable  # noqa
from trade.helpers.Logging import setup_logger
from trade.optionlib.utils.format import assert_equal_length  # noqa
from dbase.DataAPI.ThetaData import list_dates
from dbase.DataAPI.ThetaExceptions import ThetaDataNotFound
from pathlib import Path
import os
from typing import Tuple, List, Optional, Union
from trade.datamanager.utils.logging import get_logging_level, UTILS_LOGGER_NAME
from trade import MARKET_CLOSE, MARKET_OPEN

logger = setup_logger(UTILS_LOGGER_NAME, stream_log_level=get_logging_level())

PATH = Path(os.environ["GEN_CACHE_PATH"]) / "dm_gen_cache"

## This cache will be used to save the min trading date for each option tick
## This is to avoid calling API all the time


LIST_DATE_CACHE = CustomCache(
    location=PATH.as_posix(),
    fname="list_date_cache",
    clear_on_exit=False,
    expire_days=365,
)


def _convert_expiration_to_datetime(expiration: Union[str, datetime]) -> datetime:
    """
    Converts an expiration date to a datetime object. If the input is already a datetime, it is returned as is.

    Args:
        expiration: The expiration date as a string or datetime.
    Returns:
        A datetime object representing the expiration date.
    """
    if isinstance(expiration, datetime):
        return expiration.replace(
            hour=MARKET_CLOSE.hour, minute=MARKET_CLOSE.minute
        )  # Set to end of day for accurate T calculation
    else:
        return to_datetime(expiration).replace(hour=MARKET_CLOSE.hour, minute=MARKET_CLOSE.minute)


def _convert_dates_to_datetime(dates: List[Union[str, datetime]]) -> List[datetime]:
    """
    Converts a list of dates to datetime objects. If an element is already a datetime, it is returned as is.

    Args:
        dates: A list of dates as strings or datetimes.
    Returns:
        A list of datetime objects representing the input dates.
    """
    converted_dates = []
    for d in dates:
        if isinstance(d, datetime):
            converted_dates.append(
                d.replace(hour=MARKET_OPEN.hour, minute=MARKET_OPEN.minute)
            )  # Set to market open time for accurate T calculation
        else:
            converted_dates.append(to_datetime(d).replace(hour=MARKET_OPEN.hour, minute=MARKET_OPEN.minute))
    return converted_dates


def _convert_date_to_datetime(date_input: Union[str, datetime]) -> datetime:
    """
    Converts a date to a datetime object. If the input is already a datetime, it is returned as is.

    Args:
        date_input: The date as a string or datetime.
    Returns:
        A datetime object representing the input date.
    """
    if isinstance(date_input, datetime):
        return date_input.replace(
            hour=MARKET_OPEN.hour, minute=MARKET_OPEN.minute
        )  # Set to market open time for accurate T calculation
    else:
        return to_datetime(date_input).replace(hour=MARKET_OPEN.hour, minute=MARKET_OPEN.minute)


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


def _max_allowable_eod_end_date() -> datetime:
    """Return the latest calendar date EOD market data may be requested through.

    Pre-market and regular session use the prior B-day (today's EOD not published).
    Post-market allows today. Overnight / closed uses prior B-day.

    Returns:
        Timezone-naive datetime for the runtime EOD upper bound.
    """
    today = ny_now()
    prev_busday = (today - BDay(1)).to_pydatetime()
    if is_pre_market_hours():
        return pd.Timestamp(to_datetime(prev_busday)).tz_localize(None).to_pydatetime()
    if is_post_market_hours() or is_market_hours_today():
        return pd.Timestamp(to_datetime(today.date())).tz_localize(None).to_pydatetime()
    ## Weekend / overnight: same as pre-market — no current-session EOD yet.
    return pd.Timestamp(to_datetime(prev_busday)).tz_localize(None).to_pydatetime()


def _sync_equity_date(
    start_date: DATE_HINT,
    end_date: DATE_HINT,
    *,
    symbol: Optional[str] = None,
    min_trade_date: Optional[DATE_HINT] = None,
) -> Tuple[datetime, datetime]:
    """Synchronize requested window for non-option EOD timeseries paths.

    Clamps ``end_date`` to the latest EOD observation available at runtime.
    Optionally floors ``start_date`` when ``min_trade_date`` is supplied (e.g. future
    per-symbol IPO listing lookup).

    Args:
        start_date: Requested window start.
        end_date: Requested window end.
        symbol: Reserved for future IPO listing-date resolution (unused today).
        min_trade_date: Optional explicit floor for ``start_date``.

    Returns:
        Adjusted ``(start_date, end_date)`` as datetimes.

    Examples:
        >>> start, end = _sync_equity_date("2020-01-01", "2026-06-18", symbol="AAPL")
    """
    start_dt = pd.Timestamp(to_datetime(start_date)).tz_localize(None).normalize()
    end_dt = pd.Timestamp(to_datetime(end_date)).tz_localize(None).normalize()
    max_allowable = pd.Timestamp(_max_allowable_eod_end_date()).tz_localize(None).normalize()

    if end_dt > max_allowable:
        logger.info(
            "Requested end_date %s is after latest available EOD %s. Adjusting.",
            end_dt.date(),
            max_allowable.date(),
        )
        end_dt = max_allowable

    if min_trade_date is not None:
        floor_dt = pd.Timestamp(to_datetime(min_trade_date)).tz_localize(None).normalize()
        if start_dt < floor_dt:
            logger.info(
                "Requested start_date %s is before min trade date %s. Adjusting.",
                start_dt.date(),
                floor_dt.date(),
            )
            start_dt = floor_dt

    if start_dt > end_dt:
        start_dt = end_dt

    return start_dt.to_pydatetime(), end_dt.to_pydatetime()


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
        - EOD: pre-market and market hours cap at previous B-day (EOD not published yet)
        - QUOTE: pre-market caps at previous B-day; during/post session allows today
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

    def _guard_rail_dates(
        start_date: datetime,
        end_date: datetime,
        min_trade_date: datetime,
        max_trade_date: datetime,
        dates: Optional[list] = None,
    ) -> Tuple[datetime, datetime]:
        """Ensures start_date and end_date are within min_trade_date and max_trade_date."""

        if start_date < min_trade_date:
            logger.warning(
                f"Requested start_date {start_date.date()} is before available data. Adjusting to {min_trade_date.date()}."
            )
            start_date = min_trade_date
        if end_date > max_trade_date:
            logger.warning(
                f"Requested end_date {end_date.date()} is after available data. Adjusting to {max_trade_date.date()}."
            )
            end_date = max_trade_date
        if start_date > end_date:
            start_date = end_date

        ## Check if date range is present in the list of dates, if provided
        if dates is not None:
            available_dates = set(to_datetime(dates))
            if start_date not in available_dates:
                logger.warning(
                    f"Adjusted start_date {start_date.date()} is not in available dates. Adjusting to nearest available date."
                )
                start_date = min(available_dates, key=lambda d: abs(d - start_date))
            end_is_not_today = end_date.date() != ny_now().date()
            if end_date not in available_dates and end_is_not_today:
                logger.warning(
                    f"Adjusted end_date {end_date.date()} is not in available dates. Adjusting to nearest available date."
                )
                end_date = min(available_dates, key=lambda d: abs(d - end_date))
        return start_date, end_date

    is_premarket = is_pre_market_hours()
    today = ny_now()
    timestamp_today = pd.Timestamp(today.date())
    timestamp_exp = pd.Timestamp(to_datetime(expiration).date())
    prev_busday = (today - BDay(1)).to_pydatetime()
    start_date = to_datetime(start_date)
    end_date = to_datetime(end_date)
    is_expired = timestamp_exp < timestamp_today

    def _compute_max_allowable() -> pd.Timestamp:
        """Returns the latest date we should allow for this request at runtime."""
        if not is_expired:
            if endpoint_source == OptionSpotEndpointSource.EOD:
                max_allowed_today = _max_allowable_eod_end_date()
            
            ## If QUOTE source, live prints exist during/post session; pre-market use prior B-day.
            elif endpoint_source == OptionSpotEndpointSource.QUOTE:
                if is_premarket:
                    max_allowed_today = prev_busday
                else:
                    max_allowed_today = today

            else:
                raise ValueError(f"Invalid endpoint source: {endpoint_source}. Must be EOD or QUOTE.")

            return min(timestamp_exp, pd.Timestamp(max_allowed_today.date()))

        # For expired contracts, expiration day is the latest safe upper bound
        # unless we fetched an explicit last-trade date from list_dates below.
        return timestamp_exp

    if opttick in LIST_DATE_CACHE.keys():
        logger.info(f"Using cached date range for {start_date} - {end_date} and option tick {opttick}")
        cached_dates = LIST_DATE_CACHE.get(key=opttick)
        min_date = to_datetime(cached_dates["min_date"])
        max_date_raw = cached_dates.get("max_date")
        max_date = to_datetime(max_date_raw) if max_date_raw is not None else _compute_max_allowable()

        start_date = max(min_date, start_date)
        end_date = min(max_date, end_date)
        return _guard_rail_dates(start_date, end_date, min_date, max_date)

    logger.info(f"Fetching date range from Thetadata for {opttick}")
    try:
        dates = list(
            list_dates(
                symbol=symbol,
                exp=expiration,
                right=right,
                strike=strike,
            )
        )
    except ThetaDataNotFound:
        logger.warning(f"ThetaData returned no data for {opttick}. Returning original requested date range.")
        return to_datetime(start_date), to_datetime(end_date)

    if not dates:
        logger.warning(f"No trading dates found for {opttick}. Returning original requested date range.")
        return to_datetime(start_date), to_datetime(end_date)

    dates = to_datetime(dates)
    max_trade_date = pd.Timestamp(max(dates))
    min_trade_date = pd.Timestamp(min(dates))

    ## Adjust start date to min
    start_date = max(min_trade_date, start_date)
    logger.info(f"Calculated date range for option spot timeseries: {start_date} to {end_date}")

    # This is how far into the future we can get data for.
    # For non-expired options, max changes with time, so do not persist it.
    if is_expired:
        max_allowable = max_trade_date
        if get_enable_caching():
            LIST_DATE_CACHE.set(
                key=opttick,
                value={
                    "min_date": pd.Timestamp(min_trade_date),
                    "max_date": pd.Timestamp(max_allowable),
                    "range": dates,
                },
                expire=None,
            )
    else:
        max_allowable = _compute_max_allowable()
        if get_enable_caching():
            LIST_DATE_CACHE.set(
                key=opttick,
                value={
                    "min_date": pd.Timestamp(min_trade_date),
                },
                expire=None,
            )

    return _guard_rail_dates(start_date, end_date, min_trade_date, max_allowable, dates=dates)


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
    _bool = max_date >= today and current_time >= TODAY_RELOAD_CUTOFF
    return _bool


def is_available_on_date(date: date) -> bool:
    """
    Returns True if the given date is a business day and not a US holiday, False otherwise.
    For when date == today, it checks current time to see if market is open.
    """
    date = to_datetime(date)
    is_today = date.date() == ny_now().date()
    is_trading_day = is_busday(date) and not is_USholiday(date)

    ## If both today and trading day, check time
    if is_today and is_trading_day:
        current_time = ny_now().time()

        ## If before min time, return False
        if current_time < MIN_TIME_BEFORE_REAL_TIME:
            return False

    ## Else just return trading day status
    return is_trading_day
