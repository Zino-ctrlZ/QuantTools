"""
CachedOptionDataManager - Option data manager with intelligent caching.

This module provides a cached version of OptionDataManager that implements
factor-level caching at the bottlenecks:
1. CachedSpotManager - caches ThetaData spot queries
2. CachedVolManager - caches IV calculations
3. CachedGreeksManager - caches greeks calculations

Features:
- Cascade caching: greeks → vol → spot
- Only caches interval='1d' data
- Column normalization: lowercase in cache, capitalized on retrieval
- Model-aware caching: separate caches for BS and binomial models

Author: DataManagers Cache Team
Date: 2025-11-29
"""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import pandas as pd
from trade.helpers.Logging import setup_logger

# Import cache infrastructure
from .option_cache import (
    load_option_data_cache,
    get_cached_data,
    check_cache_completeness,
    save_to_cache,
    get_cache_for_type_and_model,
)

# Import original DataManagers
from .DataManagers import (
    OptionDataManager,
    VolDataManager,
    GreeksDataManager,
    SpotDataManager,
)

# Import utils
from .utils import set_global_market_timeseries

# Import MarketTimeseries for underlier data caching
try:
    from EventDriven.riskmanager.market_data import get_timeseries_obj
    MARKET_TIMESERIES_AVAILABLE = True
except ImportError:
    import traceback
    MARKET_TIMESERIES_AVAILABLE = False
    logger = setup_logger('DataManagers.CachedOptionDataManager')
    logger.critical("MarketTimeseries not available - underlier caching disabled. Performance may be impacted.")
    logger.info(traceback.format_exc())

if TYPE_CHECKING:
    from trade.helpers.helper import CustomCache

logger = setup_logger('DataManagers.CachedOptionDataManager')


# ============================================================================
# Cached Manager Classes - Calculation Bottleneck Caching
# ============================================================================

class CachedVolManager:
    """
    Cached Vol Manager - checks cache before calculating IV.
    
    This sits at the vol calculation bottleneck. When calculate_iv() is called:
    1. Check if vol data exists in cache for the requested dates
    2. If complete cache hit: return cached vol
    3. If partial/miss: calculate only missing dates, merge with cache, save
    """
    def __init__(self, symbol: str, caches: tuple, opttick: str):
        self.symbol = symbol
        self.opttick = opttick
        self._caches = caches
        self._parent_manager = VolDataManager(symbol)
        logger.debug(f"CachedVolManager initialized for {symbol}")
    
    def calculate_iv(self, **kwargs):
        """
        Calculate IV with cache check.
        
        Checks cache first, only calculates missing data.
        """
        data_request = kwargs['data_request']
        model = data_request.model
        interval = data_request.interval
        raw_data = data_request.raw_spot_data
        
        # Only cache 1d data
        if interval != '1d':
            logger.debug(f"Skipping vol cache for interval={interval}")
            return self._parent_manager.calculate_iv(**kwargs)
        
        # If raw_data is empty, nothing to calculate
        if raw_data.empty:
            logger.debug("Raw data is empty, skipping vol calculation")
            return data_request
        
        # Extract date range from raw_data
        if isinstance(raw_data.index, pd.DatetimeIndex):
            dates = raw_data.index
        elif 'datetime' in raw_data.columns:
            dates = pd.to_datetime(raw_data['datetime'])
        else:
            logger.debug("Cannot determine dates from raw_data, calculating vol without cache")
            return self._parent_manager.calculate_iv(**kwargs)
        
        start = pd.Timestamp(dates.min())
        end = pd.Timestamp(dates.max())
        
        # Check vol cache
        vol_cache = get_cache_for_type_and_model('vol', model, self._caches)
        cached_vol = get_cached_data(vol_cache, interval, self.opttick)
        
        if cached_vol is not None and not cached_vol.empty:
            # Check if cache covers our date range
            is_complete, missing_dates = check_cache_completeness(cached_vol, start, end)
            
            if is_complete:
                logger.info(f"VOL cache HIT (complete) for {self.opttick} - extracting from cache")
                
                # Extract vol columns from cache and add to raw_spot_data
                # Cache has lowercase column names, need to match with vol_col_names
                vol_col_names = list(data_request.iv_cols.values())
                vol_col_names_lower = [x.lower() for x in vol_col_names]
                available_cols = [col for col in vol_col_names_lower if col in cached_vol.columns]
                
                if available_cols:
                    # Add vol columns from cache to raw_spot_data
                    raw_data.columns = [x.lower() for x in raw_data.columns]
                    for col in available_cols:
                        # Match on dates
                        for date in dates:
                            if date in cached_vol.index:
                                raw_data.loc[raw_data.index == date, col] = cached_vol.loc[date, col]
                    
                    logger.info(f"Added {len(available_cols)} vol columns from cache")
                    return data_request  # Vol already in raw_spot_data
                else:
                    logger.debug(f"Cache exists but vol columns {vol_col_names} not found")
            else:
                logger.info(f"VOL cache PARTIAL for {self.opttick} ({len(missing_dates)} missing dates)")
        else:
            logger.info(f"VOL cache MISS for {self.opttick}")
        
        # Cache miss or incomplete - calculate using parent
        logger.debug("Calculating vol using parent VolManager")
        result = self._parent_manager.calculate_iv(**kwargs)
        
        # After calculation, save to cache with lowercase column names
        # Note: Parent modifies raw_spot_data in place, adding vol columns
        # We need to extract the vol columns and save them
        if not raw_data.empty:
            vol_col_names = list(data_request.iv_cols.values())
            available_cols = [col for col in vol_col_names if col in raw_data.columns]
            
            if available_cols and isinstance(raw_data.index, pd.DatetimeIndex):
                # Extract vol columns as separate DataFrame with DatetimeIndex
                vol_data = raw_data[available_cols].copy()
                # Normalize to lowercase before saving
                vol_data.columns = [x.lower() for x in vol_data.columns]
                
                # Save to cache
                vol_cache = get_cache_for_type_and_model('vol', model, self._caches)
                save_to_cache(vol_cache, interval, self.opttick, vol_data)
                logger.info(f"Saved {len(vol_data)} vol rows to cache")
        
        return result


class CachedGreeksManager:
    """
    Cached Greeks Manager - checks cache before calculating greeks.
    
    This sits at the greeks calculation bottleneck. When calculate_greeks() is called:
    1. Check if greeks data exists in cache for the requested dates
    2. If complete cache hit: return cached greeks
    3. If partial/miss: calculate only missing dates, merge with cache, save
    """
    def __init__(self, symbol: str, caches: tuple, opttick: str):
        self.symbol = symbol
        self.opttick = opttick
        self._caches = caches
        self._parent_manager = GreeksDataManager(symbol)
        logger.debug(f"CachedGreeksManager initialized for {symbol}")
    
    def calculate_greeks(self, type_, **kwargs):
        """
        Calculate greeks with cache check.
        
        Checks cache first, only calculates missing data.
        """
        data_request = kwargs['data_request']
        model = data_request.model
        interval = data_request.interval
        raw_data = data_request.raw_spot_data
        
        # Only cache 1d data
        if interval != '1d':
            logger.debug(f"Skipping greeks cache for interval={interval}")
            return self._parent_manager.calculate_greeks(type_, **kwargs)
        
        # If raw_data is empty, nothing to calculate
        if raw_data.empty:
            logger.debug("Raw data is empty, skipping greeks calculation")
            return data_request
        
        # Extract date range from raw_data
        if isinstance(raw_data.index, pd.DatetimeIndex):
            dates = raw_data.index
        elif 'datetime' in raw_data.columns:
            dates = pd.to_datetime(raw_data['datetime'])
        else:
            logger.debug("Cannot determine dates from raw_data, calculating greeks without cache")
            return self._parent_manager.calculate_greeks(type_, **kwargs)
        
        start = pd.Timestamp(dates.min())
        end = pd.Timestamp(dates.max())
        
        # Check greeks cache
        greeks_cache = get_cache_for_type_and_model('greeks', model, self._caches)
        cached_greeks = get_cached_data(greeks_cache, interval, self.opttick)
        
        if cached_greeks is not None and not cached_greeks.empty:
            # DEBUG: Log cache info
            logger.debug(f"Cache has {len(cached_greeks)} rows from {cached_greeks.index.min().date()} to {cached_greeks.index.max().date()}")
            logger.debug(f"Requested {len(dates)} dates from {start.date()} to {end.date()}")
            
            # Check if cache covers our date range
            is_complete, missing_dates = check_cache_completeness(cached_greeks, start, end)
            
            if is_complete:
                # Extract greek columns from cache and add to raw_spot_data
                # Cache has lowercase columns like 'delta', 'gamma', 'vega', 'midpoint_delta', etc.
                # Look for any columns in cache that contain greek keywords
                greek_keywords = ['delta', 'gamma', 'vega', 'theta', 'rho', 'vanna', 'volga']
                available_cols = [col for col in cached_greeks.columns 
                                if any(g in col for g in greek_keywords)]
                
                if available_cols:
                    logger.info(f"GREEKS cache HIT (complete) for {self.opttick} - extracting from cache")
                    
                    # Add greek columns from cache to raw_spot_data
                    raw_data.columns = [x.lower() for x in raw_data.columns]
                    for col in available_cols:
                        # Match on dates
                        for date in dates:
                            if date in cached_greeks.index:
                                raw_data.loc[raw_data.index == date, col] = cached_greeks.loc[date, col]
                    
                    logger.info(f"Added {len(available_cols)} greek columns from cache")
                    return data_request  # Greeks already in raw_spot_data
                else:
                    logger.debug(f"GREEKS cache has data but no greek columns found in {list(cached_greeks.columns)}")
            else:
                logger.info(f"GREEKS cache PARTIAL for {self.opttick} ({len(missing_dates)} missing dates)")
        else:
            logger.info(f"GREEKS cache MISS for {self.opttick}")
        
        # Cache miss or incomplete - calculate using parent
        logger.debug("Calculating greeks using parent GreeksDataManager")
        self._parent_manager.calculate_greeks(type_, **kwargs)
        
        # After calculation, save to cache with lowercase column names
        # Note: Parent modifies data_request.raw_spot_data in place, adding greek columns
        # The parent function returns None, so we get the updated data from data_request
        updated_raw_data = data_request.raw_spot_data
        if not updated_raw_data.empty:
            # The actual greek columns are the ones with greek names in them
            # e.g., 'Delta', 'Gamma', 'Vega', 'Theta', 'Rho', 'Vanna', 'Volga', 
            # 'Midpoint_delta', 'Midpoint_gamma', etc.
            greek_keywords = ['delta', 'gamma', 'vega', 'theta', 'rho', 'vanna', 'volga']
            available_cols = [col for col in updated_raw_data.columns 
                            if any(g in col.lower() for g in greek_keywords)]
            
            if available_cols and isinstance(updated_raw_data.index, pd.DatetimeIndex):
                # Extract greek columns as separate DataFrame with DatetimeIndex
                greeks_data = updated_raw_data[available_cols].copy()
                # Normalize to lowercase before saving
                greeks_data.columns = [x.lower() for x in greeks_data.columns]
                
                # Save to cache (will merge with existing cached data)
                greeks_cache = get_cache_for_type_and_model('greeks', model, self._caches)
                save_to_cache(greeks_cache, interval, self.opttick, greeks_data)
                logger.info(f"Saved {len(greeks_data)} greeks rows to cache")
        
        # Parent function returns None, data_request was modified in place
        return data_request


class CachedSpotManager:
    """
    Cached Spot Manager - checks cache before querying ThetaData.
    
    This sits at the spot query bottleneck. When query_thetadata() is called:
    1. Check if spot data exists in cache for the requested dates
    2. If complete cache hit: return cached spot
    3. If partial/miss: query only missing dates, merge with cache, save
    """
    def __init__(self, symbol: str, caches: tuple, opttick: str):
        self.symbol = symbol
        self.opttick = opttick
        self._caches = caches
        self._parent_manager = SpotDataManager(symbol)
        logger.debug(f"CachedSpotManager initialized for {symbol}")
    
    def query_thetadata(self, start, end, **kwargs):
        """
        Query ThetaData with cache check.
        
        Checks cache first, only queries missing data.
        """
        data_request = kwargs.get('data_request')
        if data_request is None:
            # No data_request means can't determine interval
            return self._parent_manager.query_thetadata(start, end, **kwargs)
        
        interval = getattr(data_request, 'interval', None) or '1d'
        agg = getattr(data_request, 'agg', None) or 'eod'
        
        # Only cache 1d/eod data
        if interval != '1d' and agg != 'eod':
            logger.debug(f"Skipping spot cache for interval={interval}, agg={agg}")
            return self._parent_manager.query_thetadata(start, end, **kwargs)
        
        # Normalize dates
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        
        # Check spot cache
        spot_cache = get_cache_for_type_and_model('spot', 'bs', self._caches)  # Spot always uses 'bs' model
        cached_spot = get_cached_data(spot_cache, interval, self.opttick)
        
        if cached_spot is not None and not cached_spot.empty:
            # Check if cache covers our date range
            is_complete, missing_dates = check_cache_completeness(cached_spot, start_ts, end_ts)
            
            if is_complete:
                logger.info(f"SPOT cache HIT (complete) for {self.opttick}")
                # Filter to requested range and return
                result = cached_spot[(cached_spot.index >= start_ts) & (cached_spot.index <= end_ts)].copy()
                # Capitalize column names to match parent's output format
                result.columns = [x.capitalize() for x in result.columns]
                logger.info(f"Returning {len(result)} cached spot rows")
                return result
            else:
                logger.info(f"SPOT cache PARTIAL for {self.opttick} ({len(missing_dates)} missing dates)")
                # Query only missing dates
                if missing_dates:
                    query_start = min(missing_dates)
                    query_end = max(missing_dates)
                    logger.info(f"Querying missing spot dates: {query_start.date()} to {query_end.date()}")
                    
                    new_data = self._parent_manager.query_thetadata(query_start, query_end, **kwargs)
                    
                    if new_data is not None and not new_data.empty:
                        # Just use the new data - it already has the full range
                        # The query expanded to cover the missing dates
                        logger.info(f"Returning {len(new_data)} newly queried spot rows")
                        
                        # Save to cache for next time  
                        spot_cache = get_cache_for_type_and_model('spot', 'bs', self._caches)
                        save_to_cache(spot_cache, interval, self.opttick, new_data)
                        logger.info(f"Saved {len(new_data)} spot rows to cache")
                        
                        return new_data
                    else:
                        # Query failed, return cached data only
                        result = cached_spot[(cached_spot.index >= start_ts) & (cached_spot.index <= end_ts)]
                        return result
        else:
            logger.info(f"SPOT cache MISS for {self.opttick}")
        
        # Cache miss - query full range
        logger.debug("Querying spot data from parent SpotDataManager")
        result = self._parent_manager.query_thetadata(start, end, **kwargs)
        
        # After querying, save to cache with lowercase column names
        if result is not None and not result.empty and isinstance(result.index, pd.DatetimeIndex):
            spot_cache = get_cache_for_type_and_model('spot', 'bs', self._caches)
            # Normalize to lowercase before saving
            cache_data = result.copy()
            cache_data.columns = [x.lower() for x in cache_data.columns]
            save_to_cache(spot_cache, interval, self.opttick, cache_data)
            logger.info(f"Saved {len(result)} spot rows to cache")
        
        return result


# ============================================================================
# Main Cached Option Data Manager
# ============================================================================

class CachedOptionDataManager(OptionDataManager):
    """
    Cached version of OptionDataManager with intelligent caching.
    
    Features:
    - Caches option spot, vol, and greeks separately
    - Model-aware caching (bs vs binomial)
    - Cascade caching: greeks checks vol cache, vol checks spot cache
    - Only caches interval='1d' data
    - Filters out T-0 (today) to avoid stale intraday data
    
    Usage:
        dm = CachedOptionDataManager(
            symbol='SPY',
            exp='2025-12-20',
            right='C',
            strike=450.0
        )
        
        # First call - cache miss, queries data
        data = dm.get_timeseries('2024-01-01', '2024-12-31', type_='greeks', model='bs')
        
        # Second call - cache hit, instant retrieval
        data = dm.get_timeseries('2024-01-01', '2024-12-31', type_='greeks', model='bs')
    """
    
    # Class-level cache instances (shared across all instances)
    _CACHES: tuple[CustomCache, CustomCache, CustomCache, CustomCache, CustomCache] | None = None
    _CACHE_ENABLED = True  # Can be toggled off for debugging
    
    def __init__(self, *args, enable_cache: bool = True, use_market_timeseries: bool = True, **kwargs):
        """
        Initialize CachedOptionDataManager.
        
        Args:
            *args: Positional arguments passed to OptionDataManager
            enable_cache: Whether to enable caching (default: True)
            use_market_timeseries: Whether to use MarketTimeseries for underlier data (default: True)
            **kwargs: Keyword arguments passed to OptionDataManager
        """
        super().__init__(*args, **kwargs)
        
        # Load caches if not already loaded
        if CachedOptionDataManager._CACHES is None:
            CachedOptionDataManager._CACHES = load_option_data_cache()
            logger.info("Loaded option data caches")
        
        self.enable_cache = enable_cache and CachedOptionDataManager._CACHE_ENABLED
        
        # Setup MarketTimeseries for underlier data caching
        self.use_market_timeseries = use_market_timeseries and MARKET_TIMESERIES_AVAILABLE
        self._market_timeseries = None
        
        if self.use_market_timeseries:
            try:
                self._market_timeseries = get_timeseries_obj()
                logger.debug(f"MarketTimeseries enabled for {self.symbol} underlier data")
                
                # Set global MarketTimeseries so _ManagerLazyLoader can use it
                set_global_market_timeseries(self._market_timeseries)
                logger.debug("Set global MarketTimeseries for _ManagerLazyLoader")
            except Exception as e:
                logger.warning(f"Failed to initialize MarketTimeseries: {e}")
                self.use_market_timeseries = False
        
        # Replace parent managers with cached versions
        if self.enable_cache:
            logger.debug(f"Replacing managers with cached versions for {self.opttick}")
            self.vol_manager = CachedVolManager(self.symbol, self._CACHES, self.opttick)
            self.greek_manager = CachedGreeksManager(self.symbol, self._CACHES, self.opttick)
            self.spot_manager = CachedSpotManager(self.symbol, self._CACHES, self.opttick)
            logger.info(f"Cached managers initialized for {self.opttick}")
        
        if not self.enable_cache:
            logger.critical(f"Caching DISABLED for {self.opttick}. All queries will hit source.")
    
    @classmethod
    def disable_caching(cls):
        """Disable caching globally for all instances."""
        cls._CACHE_ENABLED = False
        logger.info("Caching globally DISABLED")
    
    @classmethod
    def enable_caching(cls):
        """Enable caching globally for all instances."""
        cls._CACHE_ENABLED = True
        logger.info("Caching globally ENABLED")
    
    def get_timeseries(self, 
                       start: str | datetime, 
                       end: str | datetime,
                       interval: str = '1d',
                       type_: str = 'spot',
                       model: str = 'bs',
                       extra_cols: list = None) -> pd.DataFrame:
        """
        Query timeseries data with intelligent caching.
        
        This method simply calls the parent OptionDataManager.get_timeseries().
        The caching interception happens at the FACTOR MANAGER level:
        - CachedSpotManager intercepts spot queries
        - CachedVolManager intercepts vol calculations
        - CachedGreeksManager intercepts greeks calculations
        
        Each cached manager checks its cache before calling the parent manager.
        This keeps the architecture clean and each manager responsible for its own caching.
        
        Args:
            start: Start date
            end: End date
            interval: Time interval
            type_: Data type ('spot', 'vol', 'greeks')
            model: Pricing model ('bs' or 'binomial')
            extra_cols: Extra columns to include
        
        Returns:
            Data request object (same as parent class)
        """
        # Cap end date at expiration date if necessary
        end_ts = pd.Timestamp(end)
        exp_ts = pd.Timestamp(self.exp)
        
        if end_ts > exp_ts:
            logger.warning(f"End date {end_ts.date()} is after expiration {exp_ts.date()} - capping at expiration")
            end = exp_ts
        
        logger.info(f"Query: {self.opttick} | type={type_} | model={model} | dates={pd.Timestamp(start).date()} to {pd.Timestamp(end).date()}")
        
        # Call parent - cached managers will intercept at factor level
        return super().get_timeseries(start, end, interval, type_, model, extra_cols)

