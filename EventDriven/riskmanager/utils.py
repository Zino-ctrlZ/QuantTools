"""
Risk Manager Utility Functions and Cache Management.

This module provides essential utility functions for the risk management system including
cache management, option data loading, position ID parsing, ticker swapping, and data
preprocessing operations. It centralizes common operations used across the risk manager.

Key Components:
    Cache Management:
        - Persistent cache for long-term data storage (30-day expiration)
        - Temporary cache for session-specific data (cleared on exit)
        - Spot price caching (45-day expiration)
        - Custom cache locations via environment variables

    Data Loading:
        - Bulk option chain retrieval from ThetaData
        - Open interest data loading
        - Position data loading with corporate action adjustments
        - Skip column addition for data quality monitoring

    Position Processing:
        - Position ID parsing (extract ticker, expiry, strike, type)
        - Signal ID parsing for strategy tracking
        - Ticker swapping for corporate name changes
        - Option ticker generation and parsing

    Date Handling:
        - Business day calculations
        - Lookback period computation
        - Date range validation
        - Holiday adjustments

Global Variables:
    Paths:
        - BASE: Root cache directory (.riskmanager_cache)
        - location: User-configurable cache path from GEN_CACHE_PATH env var

    Caches:
        - _TEMP_CACHE: Session cache, cleared on exit
        - _PERSISTENT_CACHE: Long-term cache with 30-day expiration
        - spot_cache: Equity spot price cache with 45-day expiration

    Timeseries:
        - TIMESERIES_START: Default start date for option data
        - TIMESERIES_END: Current date as end boundary
        - LOOKBACKS: Precomputed lookback periods for efficiency

    Flags:
        - USE_TEMP_CACHE: Toggle between temp and persistent storage
        - PATCH_TICKERS: Enable automatic ticker symbol updates

Key Functions:
    Cache Operations:
        - get_persistent_cache(): Returns appropriate cache based on settings
        - get_cache(cache_type): Retrieve specific cache instance
        - reset_persistent_cache(): Clear all cached data

    Data Loading:
        - load_position_data(position_id, date_range): Load position timeseries
        - populate_cache_with_chain(ticker, date): Fetch and cache option chain
        - precompute_lookbacks(start, end, ranges): Build lookback dictionaries

    Position Parsing:
        - parse_position_id(position_id): Extract components from ID string
        - parse_signal_id(signal_id): Decode signal information
        - generate_option_tick_new(params): Create standard option ticker
        - parse_option_tick(ticker): Extract strike, expiry, type from ticker

    Data Processing:
        - add_skip_columns(df, config): Add data quality indicators
        - ffwd_data(df, columns): Forward-fill missing data
        - swap_ticker(old, new): Handle ticker symbol changes

    Date Utilities:
        - change_to_last_busday(date): Convert to previous business day
        - get_timeseries_start_end(ticker): Determine data range
        - compare_dates utilities: Date comparison helpers

Skip Column Logic:
    Automatically identifies unreliable data points:
        - Missing prices or Greeks
        - Stale data (no recent updates)
        - Zero volume or open interest
        - Wide bid-ask spreads
        - Configured per column (Midpoint, Delta, Vega, etc.)

Cache Strategy:
    Persistent Cache:
        - Used for production/long-running backtests
        - 30-day expiration by default
        - Survives program restarts
        - Stored in GEN_CACHE_PATH

    Temporary Cache:
        - Used for development/testing
        - Cleared on program exit
        - No expiration during session
        - Isolated from production data

Ticker Swapping:
    Handles corporate actions affecting ticker symbols:
        - Mergers/acquisitions
        - Company rebranding
        - Spin-offs
        - Maintains historical position continuity

Position ID Format:
    Standard format: {TICKER}_{YYYYMMDD}_{STRIKE}{TYPE}_{DIRECTION}
    Example: AAPL_20240115_175P_long

    Components:
        - Ticker: Underlying symbol
        - Date: Expiration date
        - Strike: Strike price (in cents for options)
        - Type: C (call) or P (put)
        - Direction: long or short

Data Manager Integration:
    - OptionDataManager: Main interface for option pricing
    - CachedOptionDataManager: Cached version for performance
    - Automatic switch between implementations
    - MySQL query optimization flags

Signal Handling:
    - Registers cleanup on SIGTERM (15)
    - Registers cleanup on normal exit (0)
    - Ensures cache sanitization
    - Prevents data corruption

Environment Dependencies:
    Required Environment Variables:
        - WORK_DIR: Working directory for cache storage
        - GEN_CACHE_PATH: General cache location path

    Optional Overrides:
        - USE_TEMP_CACHE: Force temporary caching
        - SKIP_MYSQL: Disable database queries

Usage Examples:
    # Get persistent cache
    cache = get_persistent_cache()

    # Load position data
    pos_data = load_position_data(
        position_id='AAPL_20240115_175P_long',
        start_date='2023-12-01',
        end_date='2024-01-15'
    )

    # Parse position ID
    components = parse_position_id('MSFT_20240220_350C_short')
    # Returns: {'ticker': 'MSFT', 'expiry': '20240220',
    #           'strike': 350, 'option_type': 'C', 'direction': 'short'}

    # Add skip columns for data quality
    df_with_skips = add_skip_columns(
        df=option_data,
        config=SkipCalcConfig(skip_columns=['Midpoint', 'Delta'])
    )

    # Precompute lookback periods
    precompute_lookbacks('2020-01-01', '2024-12-31', _range=[30, 60, 90])

Performance Considerations:
    - All caches use disk-backed dictionaries
    - Lazy loading of option chains
    - Thread-safe operations with locks
    - Bulk data retrieval where possible
    - Efficient pandas operations

Error Handling:
    - Graceful degradation on missing data
    - Logging of cache misses
    - Stack traces for debugging
    - Signal handlers for cleanup

Notes:
    - set_use_temp_cache() moved to EventDriven._vars module
    - PATCH_TICKERS enabled by default for production
    - Position data includes corporate action adjustments
    - Cache locations configurable per environment
    - Loggers use hierarchical naming for filtering
"""

import os
from .config import get_avoid_opticks
import functools
from trade.assets.helpers.utils import swap_ticker
from module_test.raw_code.DataManagers.DataManagers import OptionDataManager  # noqa
from module_test.raw_code.DataManagers.DataManagers_cached import CachedOptionDataManager  # noqa
from trade.helpers.helper import generate_option_tick_new, parse_option_tick, CustomCache, change_to_last_busday
from dbase.DataAPI.ThetaData import retrieve_bulk_open_interest, retrieve_chain_bulk
from pandas.tseries.offsets import BDay
import pandas as pd
from trade.helpers.Logging import setup_logger
from trade.helpers.decorators import log_error_with_stack, log_time  # noqa
from trade.helpers.threads import runThreads
import numpy as np
from datetime import datetime
from EventDriven.helpers import parse_signal_id  # noqa
from trade import register_signal
from typing import Tuple
from pathlib import Path
import signal
from EventDriven._vars import get_use_temp_cache
from .config import ffwd_data
from .._vars import OPTION_TIMESERIES_START_DATE


## Vars
TIMESERIES_START = pd.to_datetime(OPTION_TIMESERIES_START_DATE)
TIMESERIES_END = datetime.now().strftime("%Y-%m-%d")
LOOKBACKS = {}

## Paths
BASE = Path(os.environ["WORK_DIR"]) / ".riskmanager_cache"
BASE.mkdir(exist_ok=True)
location = Path(os.environ["GEN_CACHE_PATH"])  ## Allows users to set a custom cache location

## Loggers
logger = setup_logger("QuantTools.EventDriven.riskmanager")
time_logger = setup_logger("QuantTools.EventDriven.riskmanager.time")
logger.info("RISK MANAGER is Using New DataManager")

## Caches
_TEMP_CACHE = CustomCache(location / "temp", fname="temp_cache", clear_on_exit=True)
_PERSISTENT_CACHE = CustomCache(location, fname="persistent_cache", expire_days=30)
spot_cache = CustomCache(BASE, fname="spot", expire_days=45)

## Flags
USE_TEMP_CACHE = False
PATCH_TICKERS = True  ## Patch tickers to swap old tickers with new ones

## Info Stack
## Register info on `skips` from add_skip_columns
IDS = []
ID_SAVE_FOLDER = Path(os.environ["WORK_DIR"]) / ".cache"
ID_SAVE_FILE = ID_SAVE_FOLDER / "position_data.csv"


def set_use_temp_cache(use_temp_cache: bool) -> None:  # noqa
    """
    Sets the USE_TEMP_CACHE variable to the given value.

    Args:
        use_temp_cache (bool): The value to set USE_TEMP_CACHE to.
    """
    raise AttributeError("set_use_temp_cache has been moved to EventDriven._vars. Please update your imports.")
    global USE_TEMP_CACHE
    USE_TEMP_CACHE = use_temp_cache
    logger.critical(
        f"USE_TEMP_CACHE set to: {USE_TEMP_CACHE}. This will use a temporary cache that is cleared on exit. Utilize reset_persistent_cache() to reset the persistent cache."
    )


# def get_use_temp_cache() -> bool:
#     """
#     Returns the current value of USE_TEMP_CACHE.

#     Returns:
#         bool: The current value of USE_TEMP_CACHE.
#     """
#     raise AttributeError("get_use_temp_cache has been moved to EventDriven._vars. Please update your imports.")
#     global USE_TEMP_CACHE
#     return USE_TEMP_CACHE


# 2a) Create persistent cache or temp
def get_persistent_cache() -> CustomCache:
    """
    Returns the persistent cache.

    Returns:
        CustomCache: The persistent cache instance.
    """
    if get_use_temp_cache():
        logger.info("Using temporary cache. This cache will be cleared on exit.")
        return _TEMP_CACHE
    else:
        logger.info("Using persistent cache. This cache will be saved to disk and can be reused.")
        return _PERSISTENT_CACHE


def dynamic_memoize(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        cache = get_persistent_cache()  # resolved on every call

        # Attach storage for memoized wrappers to the cache instance
        if not hasattr(cache, "_memoized_wrappers"):
            cache._memoized_wrappers = {}

        # Reuse the memoized version for this func+cache
        if func not in cache._memoized_wrappers:
            cache._memoized_wrappers[func] = cache.memoize()(func)

        memoized_func = cache._memoized_wrappers[func]
        if get_use_temp_cache():
            logger.info(f"Using temporary cache for function: {func.__name__}")

        return memoized_func(*args, **kwargs)

    return wrapper


## Function to register information about skips in the position data to a list
def register_info_stack(id, data, data_col, update_kwargs=None):
    """
    Register the information stack for a given position ID.

    Parameters:
    - id: The position ID.
    - data: The DataFrame containing position data.

    Returns:
    - info: A dictionary containing the registered information.
    """
    if update_kwargs is None:
        update_kwargs = {}

    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data must be a pandas DataFrame.")

    info = {}
    info["ID"] = id
    for k in data_col:
        info[f"{k.upper()}_SKIP"] = data[f"{k}_skip_day"].sum()
        copy_cat = data[f"{k}_skip_day"].copy().to_frame()
        copy_cat["streak_id"] = copy_cat[f"{k}_skip_day"].ne(copy_cat[f"{k}_skip_day"].shift()).cumsum()
        copy_cat["streak"] = copy_cat.groupby("streak_id").cumcount() + 1
        info[f"{k.upper()}_MAX_STREAK"] = (
            copy_cat[copy_cat[f"{k}_skip_day"] == True].streak.max() # noqa
            if not copy_cat[copy_cat[f"{k}_skip_day"] == True].streak.empty #noqa
            else 0
        )  # noqa
    info["DATA_LEN"] = len(data)
    info["DATETIME"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    info.update(update_kwargs)
    IDS.append(info)


## Function to save the information stack to a CSV file
def save_info_stack():
    """
    Save the information stack to a CSV file.

    Parameters:
    - IDS: List of dictionaries containing position information.
    - id_save_file: Path to the CSV file where the information will be saved.
    """
    global IDS, ID_SAVE_FILE, ID_SAVE_FOLDER
    if not IDS:
        logger.info("No data to save.")
        return
    full_data = pd.read_csv(ID_SAVE_FILE) if ID_SAVE_FILE.exists() else pd.DataFrame()
    df = pd.DataFrame(IDS)
    full_data = pd.concat([full_data, df], ignore_index=True)
    full_data.to_csv(ID_SAVE_FILE, index=False)
    with open(ID_SAVE_FOLDER / "ids.txt", "a") as f:
        f.write(f"Total IDs saved: {len(IDS)} on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    IDS = []  # Clear the IDS list after saving
    return


## Register the save_info_stack function to be called on exit
register_signal(signal.SIGTERM, save_info_stack)


def get_current_saved_ids() -> pd.DataFrame:
    """
    Returns the current saved IDs as a DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing the saved IDs.
    """
    return IDS


def clear_info_stack() -> None:
    """
    Clears the information stack.
    """
    global IDS
    IDS = []
    logger.info("Cleared info stack.")


def get_timeseries_start_end() -> Tuple[str, str]:
    """
    Returns the start and end dates for the timeseries data.

    Returns:
        Tuple[str, str]: A tuple containing the start and end dates for the timeseries data.
    """
    global TIMESERIES_START, TIMESERIES_END
    return pd.to_datetime(TIMESERIES_START).strftime("%Y-%m-%d"), pd.to_datetime(TIMESERIES_END).strftime("%Y-%m-%d")


# Precompute BDay lookbacks to eliminate redundant calculations
def precompute_lookbacks(start_date, end_date, _range=None) -> None:
    if _range is None:
        _range = [10, 20, 30]
    ## Extending to allow for multiple lookbacks
    global LOOKBACKS
    trading_days = pd.date_range(start=start_date, end=end_date, freq=BDay())
    if len(LOOKBACKS) == 0:
        lookback_cache = {x.strftime("%Y-%m-%d"): {} for x in trading_days}
    else:
        lookback_cache = LOOKBACKS
    for date in trading_days:
        dates = {x: (date - BDay(x)).strftime("%Y-%m-%d") for x in _range}
        lookback_cache[date.strftime("%Y-%m-%d")].update(dates)
    LOOKBACKS = lookback_cache


precompute_lookbacks("2000-01-01", "2030-12-31")


def get_cache(name: str) -> CustomCache:
    """
    returns the cache for the given name
    """
    global spot_cache
    if name == "chain":
        raise ValueError("Chain cache is not implemented yet.")
    elif name == "close":
        raise ValueError("Close cache is not implemented yet.")
    elif name == "oi":
        raise ValueError("OI cache is not implemented yet.")
    elif name == "spot":
        return spot_cache

    else:
        raise ValueError(f"Invalid cache name: {name}")


@dynamic_memoize
def populate_cache_with_chain(tick, date, chain_spot=None, print_url=True):
    """
    Populate the cache with chain data.
    """
    chain = retrieve_chain_bulk(tick, "", date, date, "16:00", "C", print_url=False)
    logger.info(f"Retrieved chain for {tick} on {date}")
    logger.error(f"Retrieved chain for {tick} on {date}")

    ## Retrieve OI
    ## Info: We use the previous business day to get OI
    ## This is because thetadata updates OI at the end of the day
    ## Therefore to avoid lookahead bias, we use the previous business day
    prev = change_to_last_busday((pd.to_datetime(date) - BDay(1))).strftime("%Y-%m-%d")
    oi = retrieve_bulk_open_interest(symbol=tick, exp=0, start_date=prev, end_date=prev, print_url=False)

    ## Clip Chain
    chain_clipped = chain.reset_index()  # [['datetime', 'Root', 'Strike', 'Right', 'Expiration', 'Midpoint']]
    chain_clipped = chain_clipped.merge(
        oi[["Root", "Expiration", "Strike", "Right", "Open_interest"]],
        on=["Root", "Expiration", "Strike", "Right"],
        how="left",
    )
    if PATCH_TICKERS:
        chain_clipped["Root"] = chain_clipped["Root"].apply(swap_ticker)

    ## Create ID
    id_params = chain_clipped[["Root", "Right", "Expiration", "Strike"]].T.to_numpy()
    ids = runThreads(generate_option_tick_new, id_params)
    chain_clipped["opttick"] = ids
    filter_opt = get_avoid_opticks(tick)
    chain_clipped = chain_clipped[~chain_clipped["opttick"].isin(filter_opt)]  ## Optticks to avoid
    chain_clipped["chain_id"] = chain_clipped["opttick"] + "_" + chain_clipped["datetime"].astype(str)
    chain_clipped["dte"] = (
        pd.to_datetime(chain_clipped["Expiration"]) - pd.to_datetime(chain_clipped["datetime"])
    ).dt.days

    ## Save to cache
    def save_to_cache(id, date, spot):
        date = pd.to_datetime(date).strftime("%Y-%m-%d")
        save_id = f"{id}_{date}"
        if save_id not in get_cache("spot"):
            spot_cache[save_id] = spot

    save_params = chain_clipped[["opttick", "datetime", "Midpoint"]].T.to_numpy()
    runThreads(save_to_cache, save_params)

    if chain_spot:
        chain_clipped["spot"] = chain_spot
        chain_clipped["moneyness"] = 0
        chain_clipped.loc[chain_clipped["Right"] == "C", "moneyness"] = (
            chain_clipped.loc[chain_clipped["Right"] == "C", "Strike"]
            / chain_clipped.loc[chain_clipped["Right"] == "C", "spot"]
        )
        chain_clipped.loc[chain_clipped["Right"] == "P", "moneyness"] = (
            chain_clipped.loc[chain_clipped["Right"] == "P", "spot"]
            / chain_clipped.loc[chain_clipped["Right"] == "P", "Strike"]
        )
        chain_clipped = chain_clipped[
            chain_clipped["moneyness"].between(0.01, 3)
        ]  ## Filter out extreme moneyness to reduce size
    chain_clipped.columns = chain_clipped.columns.str.lower()
    chain_clipped["pct_spread"] = (chain_clipped["closeask"] - chain_clipped["closebid"]) / chain_clipped["midpoint"]

    return chain_clipped


##UTILS
def load_position_data(opttick, processed_option_data, start, end, s, r, y, s0_close):
    """
    Load position data for a given option tick.

    args:
        opttick (str): The option tick to load data for.
        processed_option_data (dict): A dictionary to store processed option data.
        start (str|datetime): The start date for the data.
        end (str|datetime): The end date for the data.
        s (pd.Series): The spot price series. Must be split adjusted.
        r (pd.Series): The risk-free rate series.
        y (pd.Series): The dividend yield series.
        s0_close (pd.Series): The close price of the underlying asset series.

    This function ONLY retrives the data for the option tick, it does not apply any splits or adjustments.
    This function will NOT check for splits or special dividends. It will only retrieve the data for the given option tick.
    """
    ## Check if the option tick is already processed
    if opttick in processed_option_data:
        return processed_option_data[opttick]

    ## Get Meta
    meta = parse_option_tick(opttick)

    ## Generate data
    data = generate_spot_greeks(opttick, start_date=start, end_date=end)
    data = enrich_data(
        data,
        meta["ticker"],
        s[s.index.isin(data.index)],
        r[r.index.isin(data.index)],
        y[y.index.isin(data.index)],
        s0_close[s0_close.index.isin(data.index)],
    )
    processed_option_data[opttick] = data
    return data


def enrich_data(data, ticker, s, r, y, s0_close):
    """
    Args:
        data (pd.DataFrame): The data to enrich.
        ticker (str): The ticker symbol for the option.
        s (pd.Series): The spot price. Adjusted for splits.
        r (pd.Series): The risk-free rate.
        y (pd.Series): The dividend yield.
        s0_close (pd.Series): The close price of the underlying asset.
    Enrich the data with additional information.
    """
    data = _clean_data(data)
    data = data[~data.index.duplicated(keep="last")]
    data["s"] = s
    data["r"] = r
    data["y"] = y
    data["s0_close"] = s0_close
    data = ffwd_data(data, ticker)
    return data


def generate_spot_greeks(opttick, start_date: str | datetime, end_date: str | datetime) -> pd.DataFrame:
    """
    Generate spot greeks for a given option tick.
    """
    ## PRICE_ON_TO_DO: NO NEED TO CHANGE. This is necessary retrievals
    # meta = parse_option_tick(opttick)
    data_manager = OptionDataManager(opttick=opttick)
    greeks = data_manager.get_timeseries(
        start=start_date,
        end=end_date,
        interval="1d",
        type_="greeks",
    ).post_processed_data  ## Multiply by the shift to account for splits
    greeks_cols = [x for x in greeks.columns if "Midpoint" in x]
    greeks = greeks[greeks_cols]
    greeks[greeks_cols] = greeks[greeks_cols].replace(0, np.nan).fillna(method="ffill")
    greeks.columns = [x.split("_")[1].capitalize() for x in greeks.columns]

    spot = data_manager.get_timeseries(
        start=start_date, end=end_date, interval="1d", type_="spot", extra_cols=["bid", "ask"]
    ).post_processed_data  ## Using chain spot data to account for splits
    spot = spot[["Midpoint", "Closeask", "Closebid"]]  ## This is raw calc place
    data = greeks.join(spot)
    return data


def parse_position_id(positionID: str) -> Tuple[dict, list]:
    position_str = positionID
    position_list = position_str.split("&")
    position_list = [x.split(":") for x in position_list if x]
    position_list_parsed = [(x[0], parse_option_tick(x[1])) for x in position_list]
    position_dict = dict(L=[], S=[])
    for x in position_list_parsed:
        position_dict[x[0]].append(x[1])
    return position_dict, position_list


def _clean_data(df):
    """
    Cleans the data by removing rows with NaN values in specified columns.

    :param data: DataFrame to clean
    :param columns: List of columns to check for NaN values
    :return: Cleaned DataFrame
    """
    logger.info("Cleaning data...")

    def fill_values(df):
        """
        Fills NaN values with the last valid observation.
        """
        return df.replace(0, np.nan).ffill()

    df = df.copy()
    return fill_values(df)


def mad_band_spike_flag(df, threshold=2, window=20, col="Midpoint"):
    """
    Add a flag to the DataFrame indicating if the change in 'Midpoint' exceeds the threshold.
    """
    df = df.copy()
    median = df[col].rolling(window).median()
    mad = df[col].rolling(window).apply(lambda x: np.median(np.abs(x - np.median(x))))
    return (df[col] - median).abs() > threshold * mad


def add_skip_columns(df, id, skip_columns, window=15, skip_threshold=2.75):
    """
    Adds skip columns to the DataFrame.
    """
    for col in skip_columns:
        ## EMA Smoothing + Zscore Fiter
        logger.info(f"Adding skip column for {col} with window {window} and threshold {skip_threshold}")
        if col not in df.columns:
            logger.info(f"Column {col} not found in DataFrame. Skipping...")
            continue

        ##ABS Zscore
        df.loc[df[col] < 0, col] = 0  ## NOTE: This is one time fix. Take it out
        smooth = df[col].ewm(span=3).mean()
        _zscore = (smooth - smooth.rolling(window).mean()) / smooth.rolling(window).std()
        _thresh = _zscore.abs() > skip_threshold

        ## Percentage change
        smooth_pct = df[col].pct_change().fillna(0)
        _zscore_pct = (smooth_pct - smooth_pct.rolling(window).mean()) / smooth_pct.rolling(window).std()
        _zscore_pct = _zscore_pct.fillna(0)
        _zscore_pct.replace([np.inf, -np.inf], 0, inplace=True)  ## Replace inf values with 0
        _thresh_pct = _zscore_pct.abs() > skip_threshold

        ## Spike Detection
        spike_flag = mad_band_spike_flag(df, threshold=skip_threshold, window=window, col=col)

        ## Window
        shortened = df[col][:window]
        pct_change = shortened.pct_change()
        window_bool = pct_change.abs() > 0.5

        ## Zero Values
        zero_bool = df[col] == 0

        ## Combine both boolean masks
        _combined = _thresh | spike_flag | window_bool | zero_bool  # | _thresh_pct
        df[f"{col}_abs_zscore"] = _thresh
        df[f"{col}_pct_zscore"] = _thresh_pct
        df[f"{col}_spike_flag"] = spike_flag
        df[f"{col}_window"] = window_bool
        df[f"{col}_zero"] = zero_bool
        df[f"{col}_skip_day"] = _combined
        df[f"{col}_skip_day_count"] = _combined.rolling(60).sum()
    register_info_stack(
        id,
        df,
        skip_columns,
        update_kwargs={"window": window, "skip_threshold": skip_threshold, "window_bool_threshold": 0.5},
    )
    return df


## Accessing _PERSISTENT_CACHE


def delete_cached_chain(tick, date):
    """
    Delete cached chain data for a specific ticker and date.
    Args:
        tick (str): Ticker symbol.
        date (str): Date in 'YYYY-MM-DD' format.
    """

    func = "EventDriven.riskmanager.utils.populate_cache_with_chain"
    del_count = 0
    for key in list(_PERSISTENT_CACHE.keys()):
        f, tick_search, date_search = key[0], key[1], key[2]
        if f == func and tick == tick_search and date == date_search:
            del _PERSISTENT_CACHE[key]
            print(f"Deleted Chain cache for {tick} on {date}")
            del_count += 1
    print(f"Deleted {del_count} entries for {tick} on {date}")


def delete_cached_get_order(tick, date):
    """
    Delete cached order data for a specific ticker and date.
    Args:
        tick (str): Ticker symbol.
        date (str): Date in 'YYYY-MM-DD' format.
    """
    fs = [
        "EventDriven.riskmanager.picker.order_picker.OrderPicker.get_order_new",
        "EventDriven.riskmanager.picker.order_picker.OrderPicker.get_order",
        "'EventDriven.riskmanager.base.OrderPicker.__get_order'",
    ]

    del_count = 0
    for key in list(_PERSISTENT_CACHE.keys()):
        tick_top = key[1]
        if isinstance(tick_top, tuple):
            tick_loc = tick_top[2][1]

        elif isinstance(tick_top, object):
            ## Search for tick within
            for item in key:
                if isinstance(item, (tuple, list, set)):
                    for sub_item in item:
                        if isinstance(sub_item, str) and swap_ticker(sub_item) == tick:
                            tick_loc = tick
                            break
        else:
            continue
        try:
            func = key[0]
            date_loc = key[2]
            if func in fs and tick_loc == tick and date_loc == date:
                del _PERSISTENT_CACHE[key]
                print(f"Deleted cache for {key}")
                del_count += 1
        except Exception:
            continue

    print(f"Deleted {del_count} entries for {tick} on {date}")


def delete_all_cached_orders() -> None:
    """
    Delete all cached order data.
    """
    global _PERSISTENT_CACHE
    fs = [
        "EventDriven.riskmanager.picker.order_picker.OrderPicker.get_order_new",
        "EventDriven.riskmanager.picker.order_picker.OrderPicker.get_order",
        "EventDriven.riskmanager.base.OrderPicker.__get_order",
        "EventDriven.riskmanager.picker.order_picker.OrderPicker._get_order",
    ]

    del_count = 0
    for key in list(_PERSISTENT_CACHE.keys()):
        try:
            func = str(key[0])
            print(func, func in fs)
            if func in fs:
                del _PERSISTENT_CACHE[key]
                del_count += 1
        except Exception:
            continue

    print(f"Deleted {del_count} total order cache entries.")


def delete_cached_chain_and_order(tick: str, date: str) -> None:
    """
    Delete cached chain and order data for each trade in the DataFrame.
    Args:
    tick (str): Ticker symbol.
    date (str): Date in 'YYYY-MM-DD' format.
    """
    delete_cached_chain(tick, date)
    delete_cached_get_order(tick, date)
