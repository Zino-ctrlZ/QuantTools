"""
Cache helpers for DataManagers with option data caching.

This module provides cache instances for option spot, volatility, and greeks data.
Follows the pattern from EventDriven.riskmanager.market_data.py for consistency.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from trade.helpers.helper import CustomCache
from trade.helpers.Logging import setup_logger

logger = setup_logger('DataManagers.cache_helpers')

# Global cache instances (initialized on first import)
_OPTION_SPOT_CACHE: Optional[CustomCache] = None
_BS_VOL_CACHE: Optional[CustomCache] = None
_BS_GREEKS_CACHE: Optional[CustomCache] = None
_BINOMIAL_VOL_CACHE: Optional[CustomCache] = None
_BINOMIAL_GREEKS_CACHE: Optional[CustomCache] = None


def _get_cache_base_path() -> Path:
    """
    Get the base path for DataManagers cache.
    
    Priority:
    1. GEN_CACHE_PATH environment variable
    2. WORK_DIR/.cache (fallback)
    
    Returns:
        Path: Base directory for cache storage
    """
    gen_cache = os.environ.get('GEN_CACHE_PATH')
    work_dir = os.environ.get('WORK_DIR')
    
    if gen_cache:
        base = Path(gen_cache) / 'data_manager_cache'
    elif work_dir:
        base = Path(work_dir) / '.cache' / 'data_manager_cache'
    else:
        raise ValueError(
            "Neither GEN_CACHE_PATH nor WORK_DIR environment variables are set. "
            "Please set one of them to use DataManagers caching."
        )
    
    base.mkdir(parents=True, exist_ok=True)
    logger.info(f"DataManagers cache base path: {base}")
    return base


def load_option_data_cache(
    expire_days: int = 200,
    clear_on_exit: bool = False
) -> tuple[CustomCache, CustomCache, CustomCache, CustomCache, CustomCache]:
    """
    Load all option data caches for DataManagers.
    
    Returns five cache instances:
    1. OPTION_SPOT_CACHE - Option spot prices (model-agnostic)
    2. BS_VOL_CACHE - Black-Scholes implied volatility
    3. BS_GREEKS_CACHE - Black-Scholes greeks
    4. BINOMIAL_VOL_CACHE - Binomial implied volatility
    5. BINOMIAL_GREEKS_CACHE - Binomial greeks
    
    Cache Structure:
        {
            'interval': {  # e.g., '1d'
                'opttick': pd.DataFrame  # e.g., 'SPY_C_20251220_450.0'
            }
        }
    
    Args:
        expire_days: Number of days before cache expires (default: 200)
        clear_on_exit: Whether to clear cache on exit (default: False)
    
    Returns:
        Tuple of 5 CustomCache instances
    """
    global _OPTION_SPOT_CACHE, _BS_VOL_CACHE, _BS_GREEKS_CACHE
    global _BINOMIAL_VOL_CACHE, _BINOMIAL_GREEKS_CACHE
    
    # Only initialize once (singleton pattern)
    if _OPTION_SPOT_CACHE is not None:
        logger.debug("Returning existing cache instances")
        return (
            _OPTION_SPOT_CACHE,
            _BS_VOL_CACHE,
            _BS_GREEKS_CACHE,
            _BINOMIAL_VOL_CACHE,
            _BINOMIAL_GREEKS_CACHE
        )
    
    base_path = _get_cache_base_path()
    
    logger.info(f"Initializing option data caches with expire_days={expire_days}")
    
    # Initialize all caches
    _OPTION_SPOT_CACHE = CustomCache(
        location=base_path,
        fname="dm_option_spot",
        expire_days=expire_days,
        clear_on_exit=clear_on_exit
    )
    
    _BS_VOL_CACHE = CustomCache(
        location=base_path,
        fname="dm_bs_vol",
        expire_days=expire_days,
        clear_on_exit=clear_on_exit
    )
    
    _BS_GREEKS_CACHE = CustomCache(
        location=base_path,
        fname="dm_bs_greeks",
        expire_days=expire_days,
        clear_on_exit=clear_on_exit
    )
    
    _BINOMIAL_VOL_CACHE = CustomCache(
        location=base_path,
        fname="dm_binomial_vol",
        expire_days=expire_days,
        clear_on_exit=clear_on_exit
    )
    
    _BINOMIAL_GREEKS_CACHE = CustomCache(
        location=base_path,
        fname="dm_binomial_greeks",
        expire_days=expire_days,
        clear_on_exit=clear_on_exit
    )
    
    logger.info("Successfully initialized all 5 cache instances")
    
    return (
        _OPTION_SPOT_CACHE,
        _BS_VOL_CACHE,
        _BS_GREEKS_CACHE,
        _BINOMIAL_VOL_CACHE,
        _BINOMIAL_GREEKS_CACHE
    )


def get_cache_instances() -> tuple[CustomCache, CustomCache, CustomCache, CustomCache, CustomCache]:
    """
    Get existing cache instances or load them if not initialized.
    
    Returns:
        Tuple of 5 CustomCache instances
    """
    return load_option_data_cache()


def clear_all_caches() -> None:
    """
    Clear all option data caches.
    Useful for testing or forced cache invalidation.
    """
    global _OPTION_SPOT_CACHE, _BS_VOL_CACHE, _BS_GREEKS_CACHE
    global _BINOMIAL_VOL_CACHE, _BINOMIAL_GREEKS_CACHE
    
    caches = [
        _OPTION_SPOT_CACHE,
        _BS_VOL_CACHE,
        _BS_GREEKS_CACHE,
        _BINOMIAL_VOL_CACHE,
        _BINOMIAL_GREEKS_CACHE
    ]
    
    for cache in caches:
        if cache is not None:
            cache.clear()
    
    logger.info("Cleared all option data caches")
