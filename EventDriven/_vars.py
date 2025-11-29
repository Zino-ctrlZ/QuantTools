from __future__ import annotations
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
from typing import Tuple
from pathlib import Path
import os
import yaml
from trade.helpers.helper import CustomCache
from trade.helpers.Logging import setup_logger
logger = setup_logger('EventDriven._vars')

CONTRACT_MULTIPLIER = 100
def ewm_smooth_data(series: pd.Series, window: int = 3) -> pd.Series:
    """
    Apply an exponential weighted moving average to a series.
    """
    window = CONFIG.get("smooth_ewn_span", window)
    return series.ewm(span=window).mean()


ADD_COLUMNS_FACTORY = {"ewm_smooth": ewm_smooth_data}


def add_columns(series: pd.Series, col_to_add: int, factory=ADD_COLUMNS_FACTORY):
    """
    Add new columns to a DataFrame using a factory function.
    """
    return factory[col_to_add](series)

BASE = Path(os.environ["WORK_DIR"]) / ".riskmanager_cache"  ## Main Cache for RiskManager
HOME_BASE = Path(os.environ["WORK_DIR"]) / ".cache"
BASE.mkdir(exist_ok=True)

OPTION_TIMESERIES_START_DATE: str|datetime = '2017-01-01'
Y1_LAGGED_START_DATE: str|datetime = (pd.to_datetime(OPTION_TIMESERIES_START_DATE) - relativedelta(years=1)).strftime('%Y-%m-%d')
Y2_LAGGED_START_DATE: str|datetime = (pd.to_datetime(OPTION_TIMESERIES_START_DATE) - relativedelta(years=2)).strftime('%Y-%m-%d')
Y3_LAGGED_START_DATE: str|datetime = (pd.to_datetime(OPTION_TIMESERIES_START_DATE) - relativedelta(years=3)).strftime('%Y-%m-%d')


# 1a) Create USE_TEMP_CACHE
USE_TEMP_CACHE = False


def set_use_temp_cache(use_temp_cache: bool) -> None:
    """
    Sets the USE_TEMP_CACHE variable to the given value.

    Args:
        use_temp_cache (bool): The value to set USE_TEMP_CACHE to.
    """
    global USE_TEMP_CACHE
    USE_TEMP_CACHE = use_temp_cache
    logger.critical(
        f"USE_TEMP_CACHE set to: {USE_TEMP_CACHE}. This will use a temporary cache that is cleared on exit. Utilize reset_persistent_cache() to reset the persistent cache."
    )


def get_use_temp_cache() -> bool:
    """
    Returns the current value of USE_TEMP_CACHE.

    Returns:
        bool: The current value of USE_TEMP_CACHE.
    """
    global USE_TEMP_CACHE
    return USE_TEMP_CACHE

with open(f'{os.environ["WORK_DIR"]}/EventDriven/riskmanager/config.yaml', "r") as f:
    CONFIG = yaml.safe_load(f)


def load_riskmanager_cache(target: str = None, 
                           create_on_missing: bool = False,
                           **kwargs)  -> CustomCache|Tuple[CustomCache, ...]:

    """ 
    Load the risk manager cache based on the USE_TEMP_CACHE setting.
    """
    if get_use_temp_cache():
        logger.info("Using Temporary Cache for RiskManager")        
        spot_timeseries = CustomCache(BASE/"temp", fname = "rm_spot_timeseries", expire_days=100)
        chain_spot_timeseries = CustomCache(BASE/"temp", fname = "rm_chain_spot_timeseries", expire_days=100) ## This is used for pricing, to account option strikes for splits
        processed_option_data = CustomCache(BASE/"temp", fname = "rm_processed_option_data", clear_on_exit=True)
        position_data = CustomCache(BASE/"temp", fname = "rm_position_data", clear_on_exit=True)
        dividend_timeseries = CustomCache(BASE/"temp", fname = "rm_dividend_timeseries", expire_days=100)
        adjusted_strike_cache = CustomCache(BASE/"temp", fname = "rm_adjusted_strike_cache", expire_days=100)
    else:
        spot_timeseries = CustomCache(BASE, fname = "rm_spot_timeseries", expire_days=100)
        chain_spot_timeseries = CustomCache(BASE, fname = "rm_chain_spot_timeseries", expire_days=100) ## This is used for pricing, to account option strikes for splits
        processed_option_data = CustomCache(BASE, fname = "rm_processed_option_data", expire_days=100)
        position_data = CustomCache(BASE, fname = "rm_position_data", clear_on_exit=True)
        dividend_timeseries = CustomCache(BASE, fname = "rm_dividend_timeseries", expire_days=100)
        adjusted_strike_cache = CustomCache(BASE, fname = "rm_adjusted_strike_cache", expire_days=100)
    
    ## Not dependent on USE_TEMP_CACHE, so always use the persistent cache.
    splits_raw =CustomCache(HOME_BASE, fname = "split_names_dates", expire_days = 1000)
    special_dividend = CustomCache(HOME_BASE, fname = 'special_dividend', expire_days=1000) ## Special dividend cache for handling special dividends
    special_dividend['COST'] = {
        '2020-12-01': 10,
        '2023-12-27': 15
    }
    
    if target is not None:
        logger.info(f"Loading risk manager cache for target: {target}")
        match target:
            case 'spot_timeseries':
                return spot_timeseries
            case 'chain_spot_timeseries':
                return chain_spot_timeseries
            case 'processed_option_data':
                return processed_option_data
            case 'position_data':
                return position_data
            case 'dividend_timeseries':
                return dividend_timeseries
            case 'splits_raw':
                return splits_raw
            case 'special_dividend':
                return special_dividend
            case 'adjusted_strike_cache':
                return adjusted_strike_cache
            case _:
                if create_on_missing:
                    logger.warning(f"Creating new cache for unknown target: {target}")
                    clear_on_exit = kwargs.get('clear_on_exit', True)
                    return CustomCache(BASE, fname = f"extra_{target}", clear_on_exit=clear_on_exit)
                raise ValueError(f"Unknown target: {target}")   
            
    return (spot_timeseries, 
            chain_spot_timeseries, 
            processed_option_data, 
            position_data, 
            dividend_timeseries, 
            splits_raw, 
            special_dividend)



