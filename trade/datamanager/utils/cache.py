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
from trade.datamanager.exceptions import EmptyDataException
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

    checked_missing_dates: Business dates that were explicitly requested from the source and confirmed
    to have no data. These count as covered so they are never re-fetched, even though the data rows
    for those dates are NaN.
    """

    key: str
    data: Union[pd.Series, pd.DataFrame]
    data_start_date: Optional[DATE_HINT] = None
    data_end_date: Optional[DATE_HINT] = None
    missing_dates_within_range: List[DATE_HINT] = None
    checked_missing_dates: List[DATE_HINT] = None

    def __post_init__(self):
        if not isinstance(self.data, (pd.Series, pd.DataFrame)):
            raise TypeError(f"Expected pd.Series or pd.DataFrame for cached data, got {type(self.data)}")
        if not isinstance(self.data.index, pd.DatetimeIndex):
            raise TypeError("Expected DatetimeIndex for cached timeseries data.")

        if self.checked_missing_dates is None:
            self.checked_missing_dates = []

        self.data_start_date = self.data.index.min().date() if not self.data.empty else None
        self.data_end_date = self.data.index.max().date() if not self.data.empty else None
        if self.data_start_date and self.data_end_date:
            missing = get_missing_dates(self.data, _start=self.data_start_date, _end=self.data_end_date)
            # A date that is confirmed checked-missing is not a gap to re-fetch.
            checked_missing_set = {to_datetime(d).date() for d in (self.checked_missing_dates or [])}
            self.missing_dates_within_range = [
                to_datetime(d).date() for d in missing if to_datetime(d).date() not in checked_missing_set
            ]
        else:
            self.missing_dates_within_range = []

    def is_fully_covered(self, start_dt: DATE_HINT, end_dt: DATE_HINT) -> bool:
        """Checks if the cached data fully covers the requested date range.

        A date is considered covered if it has a real row in the cached DataFrame
        OR if it was explicitly confirmed as having no data (checked_missing_dates).
        """
        start_d = to_datetime(start_dt).date()
        end_d = to_datetime(end_dt).date()

        checked_missing_set = {to_datetime(d).date() for d in (self.checked_missing_dates or [])}

        # Determine the outer boundary spanned by real data and checked-missing dates combined.
        all_covered_dates = set()
        if not self.data.empty:
            all_covered_dates.update(self.data.index.date)
        all_covered_dates.update(checked_missing_set)

        if not all_covered_dates:
            logger.info(f"Cached data is empty and no checked-missing dates for key: {self.key}.")
            return False

        effective_start = min(all_covered_dates)
        effective_end = max(all_covered_dates)

        if effective_start > start_d or effective_end < end_d:
            logger.info(
                f"Covered range {effective_start} to {effective_end} does not span "
                f"requested range {start_d} to {end_d} for key: {self.key}."
            )
            return False
        # Any index gap that is not in checked_missing_dates is a real gap.
        unfilled_gaps = [d for d in (self.missing_dates_within_range or []) if start_d <= d <= end_d]
        _bool = len(unfilled_gaps) == 0
        if not _bool:
            logger.info(
                f"Unfilled gaps within requested range for key {self.key}: {unfilled_gaps}. Fully covered: {_bool}"
            )
        else:
            logger.info(f"No unfilled gaps within requested range for key {self.key}. Fully covered: {_bool}")
        return _bool

    def _comprehensive_cache_check(
        self, start_dt: DATE_HINT, end_dt: DATE_HINT
    ) -> Tuple[Optional[Union[pd.Series, pd.DataFrame]], bool, DATE_HINT, DATE_HINT]:
        """
        Performs a comprehensive check to determine if the cached data fully covers the requested date range, and identifies any missing dates.
        Return args order:
        - cached_data: The cached pd.Series or pd.DataFrame if fully present, else None
        - is_partial: True if some dates are missing, False if fully present
        - missing_start_date: The earliest missing date if partially present, else start_dt
        - missing_end_date: The latest missing date if partially present, else end_dt
        """
        if to_datetime(end_dt).date() == date.today() and datetime.now().time() < MARKET_OPEN:
            logger.info(
                f"Requested end date {end_dt} is today but market has not opened yet. Adjusting end date to last business day."
            )
            end_dt = to_datetime(end_dt) - pd.tseries.offsets.BDay(1)
        if self.is_fully_covered(start_dt, end_dt):
            logger.info(f"Cache hit for timeseries data structure key: {self.key}")
            start_d = to_datetime(start_dt).date()
            end_d = to_datetime(end_dt).date()
            checked_missing_in_range = sorted(
                [
                    to_datetime(d)
                    for d in (self.checked_missing_dates or [])
                    if start_d <= to_datetime(d).date() <= end_d
                ]
            )

            try:
                sanitized_data = _data_structure_sanitize(
                    self.data,
                    start=start_dt,
                    end=end_dt,
                    source_name=f"cached timeseries for key {self.key}",
                )
            except EmptyDataException:
                # The whole window is checked-missing with no real rows at all.
                if not checked_missing_in_range:
                    raise
                if isinstance(self.data, pd.Series):
                    sanitized_data = pd.Series(
                        index=pd.DatetimeIndex(checked_missing_in_range), dtype=float, name=self.data.name
                    )
                else:
                    cols = self.data.columns.tolist() if len(self.data.columns) > 0 else ["close"]
                    sanitized_data = pd.DataFrame(
                        index=pd.DatetimeIndex(checked_missing_in_range), columns=cols, dtype=float
                    )
                sanitized_data = _data_structure_sanitize(
                    sanitized_data,
                    start=start_dt,
                    end=end_dt,
                    source_name=f"cached checked-missing placeholder for key {self.key}",
                )
            else:
                # Real rows exist but checked-missing dates in range may be absent.
                # Back-fill NaN rows for those dates so the caller sees the full window.
                if checked_missing_in_range:
                    existing_dates = set(sanitized_data.index.normalize())
                    absent = [d for d in checked_missing_in_range if pd.Timestamp(d).normalize() not in existing_dates]
                    if absent:
                        absent_idx = pd.DatetimeIndex(absent)
                        if isinstance(sanitized_data, pd.Series):
                            gap_rows = pd.Series(index=absent_idx, dtype=sanitized_data.dtype, name=sanitized_data.name)
                        else:
                            gap_rows = pd.DataFrame(index=absent_idx, columns=sanitized_data.columns, dtype=float)
                        sanitized_data = pd.concat([sanitized_data, gap_rows]).sort_index()
                        sanitized_data.index.name = "datetime"
            return sanitized_data, False, to_datetime(start_dt), to_datetime(end_dt)

        checked_missing_set = {to_datetime(d).date() for d in (self.checked_missing_dates or [])}

        # Unfilled gaps: index gaps not explained by checked_missing_dates.
        raw_missing = get_missing_dates(self.data, _start=start_dt, _end=end_dt)
        unfilled = [d for d in raw_missing if to_datetime(d).date() not in checked_missing_set]

        logger.info(
            f"Cache partially covers requested date range for timeseries data structure. "
            f"Key: {self.key}. Dates still needed: {unfilled}"
        )

        missing_start_date = to_datetime(min(unfilled) if unfilled else start_dt)
        missing_end_date = to_datetime(max(unfilled) if unfilled else end_dt)
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
    checked_missing_dates: Optional[List[DATE_HINT]] = None,
):
    """Merges and caches rate timeseries, excluding today's partial data.

    Args:
        checked_missing_dates: Business dates that were explicitly queried but
            returned no data. Stored in cache metadata so they are never
            re-fetched, even when all price values are NaN.
    """
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
        checked_missing_dates=checked_missing_dates,
    )


def _cache_it_timeseries_data_structure(
    existing: Union[pd.Series, pd.DataFrame],
    key: str,
    value: Union[pd.Series, pd.DataFrame],
    expire: Optional[int] = None,
    cache: CustomCache = None,
    skip_today_check: bool = False,
    checked_missing_dates: Optional[List[DATE_HINT]] = None,
):
    """Caches a timeseries data structure, merging with existing data and handling today's data.

    checked_missing_dates are merged with any already-stored checked-missing dates on
    the existing cache entry, so coverage knowledge accumulates across writes.
    """
    # Extract existing checked-missing dates before unwrapping _CachedData.
    existing_checked_missing: List[DATE_HINT] = []
    if isinstance(existing, _CachedData):
        existing_checked_missing = list(existing.checked_missing_dates or [])
        existing = existing.data

    assert isinstance(value, (pd.Series, pd.DataFrame)), (
        f"Expected pd.Series or pd.DataFrame for caching, got {type(value)}"
    )
    assert isinstance(existing, (pd.Series, pd.DataFrame, type(None))), (
        f"Expected pd.Series, pd.DataFrame, or None for existing data, got {type(existing)}"
    )

    # Merge all checked-missing dates (existing + new), keeping unique date values.
    merged_checked_missing: List[DATE_HINT] = list(
        {to_datetime(d).date() for d in (existing_checked_missing + (checked_missing_dates or []))}
    )

    ## Since it is a timeseries, we will append to existing if exists
    if existing is not None:
        merged = pd.concat([existing, value])
        value = merged[~merged.index.duplicated(keep="last")]

    max_date = value.index.max().date() if not value.empty else None
    max_is_today = max_date == date.today() if max_date is not None else False

    if max_is_today:
        if not _should_save_today(max_date=value.index.max().date()) and not skip_today_check:
            logger.info(f"Cutting off today's data for key: {key} to avoid saving partial day data.")
            value = value[value.index < pd.to_datetime(date.today())]
    elif max_date is not None:
        logger.info(f"Max date {max_date} for key: {key} is not today. Skipping today's data check.")

    ## Do not cache rules:
    skip_cache = False

    ## 1) If after removing today's data, there is no data left AND no checked-missing dates
    if value.empty and not merged_checked_missing:
        skip_cache = True
        logger.info(f"No data and no checked-missing dates to cache for key: {key}")

    ## 2) All-NaN is only rejected when there are no checked-missing dates to preserve.
    ##    If checked-missing dates are present the empty rows are intentional padding.
    if not value.empty and value.isna().all().all() and not merged_checked_missing:
        skip_cache = True
        logger.info(f"All data points are NaN and no checked-missing dates for key: {key}. Not caching.")

    if skip_cache:
        return

    value.sort_index(inplace=True)
    new_cache_data = _CachedData(
        key=key,
        data=value,
        checked_missing_dates=merged_checked_missing,
    )
    logger.info(
        f"Caching timeseries data structure for key: {key} with date range "
        f"{new_cache_data.data_start_date} to {new_cache_data.data_end_date}, "
        f"missing dates within range: {new_cache_data.missing_dates_within_range}, "
        f"checked-missing dates: {new_cache_data.checked_missing_dates}"
    )
    cache.set(key, new_cache_data, expire=expire)


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
    cached_entry = self.get(key, default=None)

    if not isinstance(self, BaseDataManager):
        raise TypeError(f"{self.__class__.__name__} must be a subclass of BaseDataManager.")

    # _CachedData is passed through intact so checked_missing_dates is preserved.
    if cached_entry is None:
        logger.info(f"No cache entry found for key: {key}")
        return None, False, start_dt, end_dt

    if not isinstance(cached_entry, (_CachedData, pd.Series, pd.DataFrame)):
        logger.info(
            f"Cache entry for key: {key} is not a recognised type. Found type: {type(cached_entry)}. Ignoring cache entry."
        )
        return None, False, start_dt, end_dt

    return _data_structure_cache_check_missing(
        cached_data=cached_entry,
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
