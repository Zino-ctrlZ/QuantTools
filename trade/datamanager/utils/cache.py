import pandas as pd
from datetime import date, datetime
from typing import Any, List, Optional, Union, Tuple
from trade.helpers.Logging import setup_logger
from trade.helpers.helper import CustomCache, change_to_last_busday, get_missing_dates, to_datetime
from trade import get_pricing_config
from dataclasses import dataclass
from .date import _should_save_today, DATE_HINT
from ..base import BaseDataManager
from .data_structure import _data_structure_sanitize
from trade.datamanager.utils.logging import get_logging_level, UTILS_LOGGER_NAME
logger = setup_logger(UTILS_LOGGER_NAME, stream_log_level=get_logging_level())
MARKET_OPEN = pd.Timestamp(get_pricing_config()["MARKET_OPEN_TIME"]).time()
MARKET_CLOSE = pd.Timestamp(get_pricing_config()["MARKET_CLOSE_TIME"]).time()
@dataclass
class _CachedData:
    """
    Represents cached timeseries data along with metadata about its date coverage and missing dates.
    The goal of this class is to encapsulate the cached data and provide utility methods to check if it fully covers a requested date range.
    This allows the cache checking logic to be more efficient by avoiding repeated calculations of missing dates and date range coverage.
    """
    key: str
    data: Union[pd.Series, pd.DataFrame]
    data_start_date: Optional[DATE_HINT] = None
    data_end_date: Optional[DATE_HINT] = None
    missing_dates_within_range: List[DATE_HINT] = None

    def __post_init__(self):
        if not isinstance(self.data, (pd.Series, pd.DataFrame)):
            raise TypeError(f"Expected pd.Series or pd.DataFrame for cached data, got {type(self.data)}")
        if not isinstance(self.data.index, pd.DatetimeIndex):
            raise TypeError("Expected DatetimeIndex for cached timeseries data.")
        
        self.data_start_date = self.data.index.min().date() if not self.data.empty else None
        self.data_end_date = self.data.index.max().date() if not self.data.empty else None
        if self.data_start_date and self.data_end_date:
            missing = get_missing_dates(self.data, _start=self.data_start_date, _end=self.data_end_date)
            self.missing_dates_within_range = [to_datetime(d).date() for d in missing]
        else:
            self.missing_dates_within_range = []

    def is_fully_covered(self, start_dt: DATE_HINT, end_dt: DATE_HINT) -> bool:
        """Checks if the cached data fully covers the requested date range."""
        if self.data.empty:
            logger.info(f"Cached data is empty for key: {self.key}.")
            return False
        if self.data_start_date > to_datetime(start_dt).date() or self.data_end_date < to_datetime(end_dt).date():
            logger.info(f"Cached data date range {self.data_start_date} to {self.data_end_date} does not cover requested range {to_datetime(start_dt).date()} to {to_datetime(end_dt).date()}.")
            return False
        if not self.missing_dates_within_range:
            logger.info(f"No missing dates within cached data range for key: {self.key}.")
            return True
        
        missing_in_range = [to_datetime(d).date() for d in self.missing_dates_within_range if to_datetime(start_dt).date() <= d <= to_datetime(end_dt).date()]
        _bool = len(missing_in_range) == 0
        if not _bool:
            logger.info(f"Missing dates within requested range for key {self.key}: {missing_in_range}. Fully covered: {_bool}")
        else:
            logger.info(f"No missing dates within requested range for key {self.key}. Fully covered: {_bool}")
        return _bool
    
    def _comprehensive_cache_check(self, start_dt: DATE_HINT, end_dt: DATE_HINT) -> Tuple[Optional[Union[pd.Series, pd.DataFrame]], bool, DATE_HINT, DATE_HINT]:
        """
        Performs a comprehensive check to determine if the cached data fully covers the requested date range, and identifies any missing dates.
        Return args order:
        - cached_data: The cached pd.Series or pd.DataFrame if fully present, else None
        - is_partial: True if some dates are missing, False if fully present
        - missing_start_date: The earliest missing date if partially present, else start_dt
        - missing_end_date: The latest missing date if partially present, else end_dt
        """
        if to_datetime(end_dt).date() == date.today() and datetime.now().time() < MARKET_OPEN:
            logger.info(f"Requested end date {end_dt} is today but market has not opened yet. Adjusting end date to last business day.")
            end_dt = to_datetime(end_dt) - pd.tseries.offsets.BDay(1)
        if self.is_fully_covered(start_dt, end_dt):
            logger.info(f"Cache hit for timeseries data structure key: {self.key}")
            sanitized_data = _data_structure_sanitize(
                self.data,
                start=start_dt,
                end=end_dt,
                source_name=f"cached timeseries for key {self.key}",
            )
            return sanitized_data, False, to_datetime(start_dt), to_datetime(end_dt)
        
        logger.info(
            f"Cache partially covers requested date range for timeseries data structure. "
            f"Key: {self.key}. Fetching missing dates within range: {[d for d in self.missing_dates_within_range if to_datetime(start_dt).date() <= d <= to_datetime(end_dt).date()]}"
        )
        
        missing_in_range = get_missing_dates(self.data, _start=start_dt, _end=end_dt)
        missing_start_date = to_datetime(min(missing_in_range) if missing_in_range else start_dt)
        missing_end_date = to_datetime(max(missing_in_range) if missing_in_range else end_dt)
        return self.data, True, missing_start_date, missing_end_date

def _simple_extract_from_cache(key: str, cache: CustomCache) -> Optional[Union[pd.Series, pd.DataFrame]]:
    """Simple helper to extract cached data, handling the _CachedData wrapper."""
    cached = cache.get(key, default=None)
    cached = _extract_data(cached)
    return cached

def _extract_data(data: Union[pd.Series, pd.DataFrame, _CachedData]) -> Union[pd.Series, pd.DataFrame]:
    """Extracts the actual data from a _CachedData object or returns it directly if it's already a Series/DataFrame."""
    if isinstance(data, _CachedData):
        return data.data
    return data

def _data_structure_cache_it(
    self: BaseDataManager, 
    key: str, 
    value: Union[pd.Series, pd.DataFrame], 
    *, 
    expire: Optional[int] = None,
):
    """Merges and caches rate timeseries, excluding today's partial data."""
    value = value.copy()
    if not isinstance(value, (pd.Series, pd.DataFrame)):
        raise TypeError(f"Expected pd.Series or pd.DataFrame for caching, got {type(value)}")

    if not isinstance(value.index, pd.DatetimeIndex):
        raise TypeError("Expected DatetimeIndex for caching timeseries data.")

    if not isinstance(self, BaseDataManager):
        raise TypeError(f"{self.__class__.__name__} must be a subclass of BaseDataManager.")
    
    existing: Optional[Union[pd.Series, pd.DataFrame]] = self.get(key, default=None)
    _cache_it_timeseries_data_structure(
        existing=existing,
        key=key,
        value=value,
        expire=expire,
        cache=self,
    )
    
def _cache_it_timeseries_data_structure(
    existing: Union[pd.Series, pd.DataFrame],
    key: str,
    value: Union[pd.Series, pd.DataFrame],
    expire: Optional[int] = None,
    cache: CustomCache = None,
    skip_today_check: bool = False,
):
    """Caches a timeseries data structure, merging with existing data and handling today's data."""
    if isinstance(existing, _CachedData):
        existing = existing.data
    assert isinstance(value, (pd.Series, pd.DataFrame)), f"Expected pd.Series or pd.DataFrame for caching, got {type(value)}"
    assert isinstance(existing, (pd.Series, pd.DataFrame, type(None))), f"Expected pd.Series, pd.DataFrame, or None for existing data, got {type(existing)}"

    ## Since it is a timeseries, we will append to existing if exists
    if existing is not None:
        # Merge existing and new values. We're expecting pd.Series
        merged = pd.concat([existing, value])
        value = merged[~merged.index.duplicated(keep="last")]

    if value.empty:
        logger.info(f"Not caching empty timeseries for key: {key}")
        return
    
    max_date = value.index.max().date()
    max_is_today = max_date == date.today()
    
    ## Really only makes sense to remove today's data if max date is today. if not just skip the check and save whatever we have since it won't be partial day data. 
    ## This also avoids the overhead of checking today's date and time for every cache entry that has a max date in the past.
    if max_is_today:
        if not _should_save_today(max_date=value.index.max().date()) and not skip_today_check:
            logger.info(f"Cutting off today's data for key: {key} to avoid saving partial day data.")
            value = value[value.index < pd.to_datetime(date.today())]
    else:
        logger.info(f"Max date {max_date} for key: {key} is not today. Skipping today's data check.")

    ## Do not cache rules:
    cache_data = True
    
    ## 1) If after removing today's data, there is no data left
    if value.empty:
        cache_data = False
        logger.info(f"No data left to cache for key: {key} after removing today's data.")
    
    ## 2) If all data points are NaN
    if value.isna().all().all():
        cache_data = False
        logger.info(f"All data points are NaN for key: {key}. Not caching.")


    if not cache_data:
        return
    

    value.sort_index(inplace=True)
    cache_data = _CachedData(key=key, data=value)
    logger.info(f"Caching timeseries data structure for key: {key} with date range {cache_data.data_start_date} to {cache_data.data_end_date}, missing dates within range: {cache_data.missing_dates_within_range}")

    cache.set(key, cache_data, expire=expire)


def _simple_list_cache_it(self: BaseDataManager, key: str, value: List[Any], *, expire: Optional[int] = None):
    """Cache a list of simple values. Will append and keep unique. Also sort"""

    if not isinstance(value, list):
        raise TypeError(f"Expected list. Recieved {type(value)}")

    existing: List = self.get(key, default=[])
    existing.extend(value)
    existing = sorted(list(set(existing)))
    self.set(key, existing, expire=expire)

def _check_cache_for_timeseries_data_structure(
    self: BaseDataManager,
    key: str,
    start_dt: DATE_HINT,
    end_dt: DATE_HINT,
) -> Tuple[Optional[Union[pd.Series, pd.DataFrame]], bool, DATE_HINT, DATE_HINT]:
    """
    Checks cache for existing timeseries data structure and identifies missing dates.

    Return args order:
    - cached_data: The cached pd.Series or pd.DataFrame if fully present, else None
    - is_partial: True if some dates are missing, False if fully present
    - missing_start_date: The earliest missing date if partially present, else start_dt
    - missing_end_date: The latest missing date if partially present, else end_dt
    """
    cached_data = self.get(key, default=None)
    if isinstance(cached_data, _CachedData):
        cached_data = cached_data.data

    if not isinstance(self, BaseDataManager):
        raise TypeError(f"{self.__class__.__name__} must be a subclass of BaseDataManager.")

    if not isinstance(cached_data, (pd.Series, pd.DataFrame, type(None))):
        logger.info(f"Cache entry for key: {key} is not a pd.Series, pd.DataFrame, or None. Found type: {type(cached_data)}. Ignoring cache entry.")
        return None, False, start_dt, end_dt

    if cached_data is None:
        logger.info(f"No cache entry found for key: {key}")
        return None, False, start_dt, end_dt
    
    return _data_structure_cache_check_missing(
        cached_data=cached_data,
        key=key,
        start_dt=start_dt,
        end_dt=end_dt,
    )
    
def _data_structure_cache_check_missing(
    cached_data: Union[pd.Series, pd.DataFrame, _CachedData],
    key: str,
    start_dt: DATE_HINT,
    end_dt: DATE_HINT,
) -> Tuple[Union[pd.Series, pd.DataFrame], bool, DATE_HINT, DATE_HINT]:
    """
    Checks cached timeseries data structure for missing dates within a specified range.
    Return args order:
    - cached_data: The cached pd.Series or pd.DataFrame, sanitized to the requested date range
    - is_partial: True if some dates are missing, False if fully present
    - missing_start_date: The earliest missing date if partially present, else start_dt
    - missing_end_date: The latest missing date if partially present, else end_dt
    """
    ## Firstly we want to ensure backward compatibility with old cache data structure which is just the raw pd.Series or pd.DataFrame. 
    ## We will convert it to the new cache data structure and save it back to cache for future use. This way we can also populate the missing dates info for old cache entries.

    if isinstance(cached_data, (pd.Series, pd.DataFrame)):
        logger.info(f"Converting old cache data structure to new for key: {key}")
        cached_data = _CachedData(key=key, data=cached_data)

    ## For start, we move forward if not busday
    start_dt = change_to_last_busday(start_dt, time_of_day_aware=False, offset=-1)

    ## For end, we move backward if not busday
    end_dt = change_to_last_busday(end_dt, time_of_day_aware=False, offset=1)
    return cached_data._comprehensive_cache_check(start_dt=start_dt, end_dt=end_dt)

    missing = get_missing_dates(cached_data, _start=start_dt, _end=end_dt)
    if not missing:
        logger.info(f"Cache hit for timeseries data structure key: {key}")
        cached_data = _data_structure_sanitize(
            cached_data,
            start=start_dt,
            end=end_dt,
            source_name=f"cached timeseries for key {key}",
        )
        return cached_data, False, start_dt, end_dt
    logger.info(
        f"Cache partially covers requested date range for timeseries data structure. "
        f"Key: {key}. Fetching missing dates: {missing}"
    )
    return cached_data, True, min(missing), max(missing)
