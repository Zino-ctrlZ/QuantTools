"""
Cache utility functions for DataManagers.

Provides helper functions for checking, retrieving, and saving data to caches.
"""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Literal, Optional, Tuple

import pandas as pd

from trade.helpers.helper import get_missing_dates
from trade.helpers.Logging import setup_logger
from .helpers import _BS_VOL_CACHE, _BINOMIAL_GREEKS_CACHE, _BS_GREEKS_CACHE, _OPTION_SPOT_CACHE, _BINOMIAL_VOL_CACHE

if TYPE_CHECKING:
    from trade.helpers.helper import CustomCache

logger = setup_logger('DataManagers.option_cache.utils')


def filter_today_from_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into cacheable (T-1 and earlier) and non-cacheable (T-0 today).
    
    For a date range T-20 to T-0, we save T-20 to T-1 and drop T-0.
    This avoids caching stale intraday data.
    
    Args:
        data: DataFrame with datetime index
    
    Returns:
        Tuple of (cacheable_data, today_data)
        - cacheable_data: Data from T-1 and earlier (can be cached)
        - today_data: Data from T-0 (today only, do not cache)
    """
    if data is None or data.empty:
        return data, pd.DataFrame()
    
    today = pd.Timestamp.now().normalize()
    
    # Check if index is DatetimeIndex
    if isinstance(data.index, pd.DatetimeIndex):
        # Split based on index
        cacheable_mask = data.index < today
        today_mask = data.index >= today
    elif 'datetime' in data.columns:
        # Split based on 'datetime' column (lowercase - from raw_spot_data)
        date_col = pd.to_datetime(data['datetime'])
        cacheable_mask = date_col < today
        today_mask = date_col >= today
    elif 'Datetime' in data.columns:
        # Split based on 'Datetime' column (capitalized - from final_data)
        date_col = pd.to_datetime(data['Datetime'])
        cacheable_mask = date_col < today
        today_mask = date_col >= today
    elif 'date' in data.columns:
        # Split based on 'date' column
        date_col = pd.to_datetime(data['date'])
        cacheable_mask = date_col < today
        today_mask = date_col >= today
    else:
        # No datetime information - cache everything (shouldn't happen, but safe fallback)
        logger.warning(f"No datetime index or columns found. Available columns: {data.columns.tolist()}. Caching all data.")
        return data, pd.DataFrame()
    
    cacheable_data = data[cacheable_mask].copy()
    today_data = data[today_mask].copy()
    
    if not today_data.empty:
        logger.debug(f"Filtered {len(today_data)} today rows (will not cache)")
    
    return cacheable_data, today_data


def get_cached_data(
    cache: CustomCache,
    interval: str,
    opttick: str
) -> Optional[pd.DataFrame]:
    """
    Retrieve cached data for a given opttick and interval.
    
    Args:
        cache: CustomCache instance to query
        interval: Time interval (e.g., '1d')
        opttick: Option ticker string
    
    Returns:
        pd.DataFrame if cached data exists (with lowercase columns), None otherwise
    """
    interval_dict = cache.get(interval, {})
    cached_df = interval_dict.get(opttick)
    
    if cached_df is not None:
        logger.debug(f"Cache HIT for {opttick} ({interval}): {len(cached_df)} rows")
        
        ## STANDARDIZE: Convert all column names to lowercase to match save_to_database format
        ## This handles legacy cached data that may have mixed-case column names
        cached_df = cached_df.copy()  # Don't modify the cached data in place
        cached_df.columns = cached_df.columns.str.lower()
        
        ## CRITICAL FIX: Standardize all variations of open interest to 'openinterest'
        _ = ['open_interest', 'open interest', 'openinterest']
        for col in cached_df.columns:
            if col.replace('_', ' ').replace(' ', '') == 'openinterest':
                if col != 'openinterest':
                    cached_df.rename(columns={col: 'openinterest'}, inplace=True)
        
        ## Deduplicate columns (keep last to prefer newer format)
        cached_df = cached_df.loc[:, ~cached_df.columns.duplicated(keep='last')]
        logger.debug(f"Standardized columns to lowercase: {list(cached_df.columns)[:5]}...")
    else:
        logger.debug(f"Cache MISS for {opttick} ({interval})")
    
    return cached_df


def check_cache_completeness(
    cached_data: Optional[pd.DataFrame],
    start_date: pd.Timestamp | datetime | str,
    end_date: pd.Timestamp | datetime | str
) -> Tuple[bool, list]:
    """
    Check if cached data covers the entire requested date range.
    
    Args:
        cached_data: Cached DataFrame (or None)
        start_date: Start of requested date range
        end_date: End of requested date range
    
    Returns:
        Tuple of (is_complete, missing_dates_list)
    """
    if cached_data is None or cached_data.empty:
        # No cached data - need entire range
        return False, [pd.Timestamp(start_date), pd.Timestamp(end_date)]
    
    # Get missing dates using existing helper
    missing_dates = get_missing_dates(cached_data, start_date, end_date)
    
    is_complete = len(missing_dates) == 0
    
    if is_complete:
        logger.debug("Cache COMPLETE: All dates present")
    else:
        logger.debug(f"Cache INCOMPLETE: {len(missing_dates)} missing dates")
    
    return is_complete, missing_dates


def save_to_cache(
    cache: CustomCache,
    interval: str,
    opttick: str,
    new_data: pd.DataFrame,
    merge_with_existing: bool = True,
    drop_today: bool = False
) -> None:
    """
    Save data to cache, optionally merging with existing cached data.
    
    Args:
        cache: CustomCache instance to save to
        interval: Time interval (e.g., '1d')
        opttick: Option ticker string
        new_data: New DataFrame to cache
        merge_with_existing: If True, merge with existing data (default)
        drop_today: If True, drop rows where index == today before saving (default False)
    """
    if new_data is None or new_data.empty:
        logger.warning(f"Attempted to cache empty data for {opttick}")
        return
    
    # Drop today's data if requested
    if drop_today:
        today = pd.Timestamp.now().normalize()
        original_len = len(new_data)
        new_data = new_data[new_data.index.normalize() != today]
        if len(new_data) < original_len:
            logger.info(f"Dropped {original_len - len(new_data)} rows for today ({today.date()}) before caching")
    
    # Get existing interval dict (or create new one)
    interval_dict = cache.get(interval, {})
    
    if merge_with_existing and opttick in interval_dict:
        # Merge with existing data
        existing_data = interval_dict[opttick]
        
        # First, ensure both have proper DatetimeIndex (handles timezone/precision issues)
        if not isinstance(existing_data.index, pd.DatetimeIndex):
            logger.info(f"Converting existing_data index to DatetimeIndex (was {type(existing_data.index)})")
            existing_data.index = pd.to_datetime(existing_data.index)
        if not isinstance(new_data.index, pd.DatetimeIndex):
            logger.info(f"Converting new_data index to DatetimeIndex (was {type(new_data.index)})")
            new_data.index = pd.to_datetime(new_data.index)
        
        # Debug: Check BEFORE dedup  
        logger.info(f"BEFORE dedup - existing: {len(existing_data)} rows, {existing_data.index.duplicated().sum()} dups")
        logger.info(f"BEFORE dedup - new: {len(new_data)} rows, {new_data.index.duplicated().sum()} dups")
        logger.info(f"existing.index full: {existing_data.index.tolist()}")
        logger.info(f"new.index full: {new_data.index.tolist()}")
        
        # Remove duplicate indices AFTER ensuring DatetimeIndex
        existing_data = existing_data[~existing_data.index.duplicated(keep='last')]
        new_data = new_data[~new_data.index.duplicated(keep='last')]
        
        # Remove duplicate COLUMNS (this is the key issue!)
        existing_data = existing_data.loc[:, ~existing_data.columns.duplicated(keep='last')]
        new_data = new_data.loc[:, ~new_data.columns.duplicated(keep='last')]
        
        # Debug: Check AFTER dedup
        logger.info(f"AFTER dedup - existing: {len(existing_data)} rows, is_unique={existing_data.index.is_unique}")
        logger.info(f"AFTER dedup - new: {len(new_data)} rows, is_unique={new_data.index.is_unique}")
        logger.info(f"existing columns: {existing_data.columns.tolist()}")
        logger.info(f"new columns: {new_data.columns.tolist()}")
        logger.info(f"existing cols duplicated: {existing_data.columns.duplicated().sum()}")
        logger.info(f"new cols duplicated: {new_data.columns.duplicated().sum()}")
        
        # Now concat the deduplicated data
        try:
            merged_data = pd.concat([existing_data, new_data]).sort_index()
        except Exception as e:
            logger.error(f"CONCAT FAILED: {e}")
            logger.error(f"existing_data index sample: {existing_data.index[:10].tolist()}")
            logger.error(f"new_data index sample: {new_data.index[:10].tolist()}")
            raise
        # Remove any overlapping dates, keeping last (most recent from new_data)
        merged_data = merged_data[~merged_data.index.duplicated(keep='last')]
        interval_dict[opttick] = merged_data
        logger.debug(f"Merged and saved {len(merged_data)} rows for {opttick} (added {len(new_data)} new)")
    else:
        # Save new data directly, but remove duplicates first
        new_data = new_data[~new_data.index.duplicated(keep='last')]
        interval_dict[opttick] = new_data
        logger.debug(f"Saved {len(new_data)} rows for {opttick} ({interval})")
    
    # CustomCache doesn't support in-place assignment, must reassign entire dict
    cache[interval] = interval_dict


def get_cache_for_type_and_model(
    type_: Literal['spot', 'vol', 'greeks'],
    model: Literal['bs', 'binomial'],
    caches: tuple
) -> CustomCache:
    """
    Get the appropriate cache instance based on data type and model.
    
    Args:
        type_: Data type ('spot', 'vol', or 'greeks')
        model: Pricing model ('bs' or 'binomial')
        caches: Tuple of 5 cache instances from load_option_data_cache()
    
    Returns:
        Appropriate CustomCache instance
    """
    try:
        spot_cache, bs_vol, bs_greeks, binom_vol, binom_greeks = caches
    except Exception:
        spot_cache, bs_vol, bs_greeks, binom_vol, binom_greeks = (
            _OPTION_SPOT_CACHE,
            _BS_VOL_CACHE,
            _BS_GREEKS_CACHE,
            _BINOMIAL_VOL_CACHE,
            _BINOMIAL_GREEKS_CACHE
        )
    
    if type_ == 'spot':
        # Spot is model-agnostic
        return spot_cache
    elif type_ == 'vol':
        return bs_vol if model == 'bs' else binom_vol
    elif type_ in ['greek', 'greeks']:
        return bs_greeks if model == 'bs' else binom_greeks
    else:
        raise ValueError(f"Unknown type: {type_}. Expected 'spot', 'vol', or 'greeks'")


def log_cache_stats(cache: CustomCache, cache_name: str) -> None:
    """
    Log statistics about cache contents (for debugging/monitoring).
    
    Args:
        cache: CustomCache instance
        cache_name: Name of the cache for logging
    """
    total_tickers = 0
    total_rows = 0
    
    for interval, interval_dict in cache.items():
        if isinstance(interval_dict, dict):
            ticker_count = len(interval_dict)
            row_count = sum(len(df) for df in interval_dict.values() if isinstance(df, pd.DataFrame))
            total_tickers += ticker_count
            total_rows += row_count
            logger.info(f"{cache_name}[{interval}]: {ticker_count} tickers, {row_count} total rows")
    
    logger.info(f"{cache_name} TOTAL: {total_tickers} tickers, {total_rows} rows")
