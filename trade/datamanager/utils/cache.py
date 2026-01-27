import pandas as pd
from datetime import date
from typing import Any, List, Optional, Union, Tuple
from trade.helpers.Logging import setup_logger
from trade.helpers.helper import get_missing_dates
from .date import _should_save_today, DATE_HINT
from ..base import BaseDataManager
from .data_structure import _data_structure_sanitize
logger = setup_logger("trade.datamanager.utils", stream_log_level="INFO")


def _data_structure_cache_it(
    self: BaseDataManager, key: str, value: Union[pd.Series, pd.DataFrame], *, expire: Optional[int] = None
):
    """Merges and caches rate timeseries, excluding today's partial data."""
    value = value.copy()
    if not isinstance(value, (pd.Series, pd.DataFrame)):
        raise TypeError(f"Expected pd.Series or pd.DataFrame for caching, got {type(value)}")

    if not isinstance(value.index, pd.DatetimeIndex):
        raise TypeError("Expected DatetimeIndex for caching timeseries data.")

    if not isinstance(self, BaseDataManager):
        raise TypeError(f"{self.__class__.__name__} must be a subclass of BaseDataManager.")

    ## Since it is a timeseries, we will append to existing if exists
    existing = self.get(key, default=None)
    if existing is not None:
        # Merge existing and new values. We're expecting pd.Series
        merged = pd.concat([existing, value])
        value = merged[~merged.index.duplicated(keep="last")]

    if value.empty:
        logger.info(f"Not caching empty timeseries for key: {key}")
        return

    if not _should_save_today(max_date=value.index.max().date()):
        logger.info(f"Cutting off today's data for key: {key} to avoid saving partial day data.")
        value = value[value.index < pd.to_datetime(date.today())]

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

    self.set(key, value, expire=expire)


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
    if not isinstance(self, BaseDataManager):
        raise TypeError(f"{self.__class__.__name__} must be a subclass of BaseDataManager.")

    if not isinstance(cached_data, (pd.Series, pd.DataFrame, type(None))):
        return None, False, start_dt, end_dt

    if cached_data is None:
        return None, False, start_dt, end_dt

    missing = get_missing_dates(cached_data, _start=start_dt, _end=end_dt)
    if not missing:
        logger.info(f"Cache hit for timeseries data structure key: {key}")
        cached_data = _data_structure_sanitize(
            cached_data,
            start=start_dt,
            end=end_dt,
        )
        return cached_data, False, start_dt, end_dt
    logger.info(
        f"Cache partially covers requested date range for timeseries data structure. "
        f"Key: {key}. Fetching missing dates: {missing}"
    )
    return cached_data, True, min(missing), max(missing)
