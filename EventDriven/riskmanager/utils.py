import os, sys
from trade.assets.Stock import Stock
from trade.assets.Option import Option
from trade.assets.OptionStructure import OptionStructure
from trade.assets.Calculate import Calculate
# from trade.assets.helpers.DataManagers_new import OptionDataManager
from trade.assets.helpers.utils import (swap_ticker)
from module_test.raw_code.DataManagers.DataManagers import OptionDataManager, SaveManager, BulkOptionDataManager

from trade.helpers.Context import Context, clear_context
from trade.helpers.helper import (change_to_last_busday, 
                                  is_USholiday, 
                                  is_busday, 
                                  setup_logger, 
                                  generate_option_tick_new, 
                                  get_option_specifics_from_key,
                                  parse_option_tick,
                                  binomial_implied_vol,
                                  CustomCache,
                                  check_missing_dates,
                                  check_all_days_available,
                                  printmd)
from dbase.DataAPI.ThetaData import (list_contracts, 
                                     retrieve_openInterest, 
                                     retrieve_eod_ohlc, 
                                     retrieve_bulk_eod,
                                     retrieve_bulk_open_interest,
                                     retrieve_chain_bulk
                                     )
from pandas.tseries.offsets import BDay
from pandas.tseries.holiday import USFederalHolidayCalendar
from trade.helpers.decorators import log_error_with_stack, log_time
from itertools import product
import pandas as pd
from copy import deepcopy
from trade.helpers.Logging import setup_logger
from trade.helpers.decorators import log_error_with_stack, copy_doc
from pathos.multiprocessing import ProcessingPool as Pool
from trade.helpers.threads import runThreads
from trade.helpers.types import OptionModelAttributes
import numpy as np
import time
from datetime import datetime
import math
from trade.assets.rates import get_risk_free_rate_helper
from EventDriven.event import FillEvent
from EventDriven.helpers import parse_signal_id
from EventDriven.data import DataHandler
from EventDriven.eventScheduler import EventScheduler
from EventDriven.types import FillDirection, OpenPositionAction, ResultsEnum, SignalTypes
from threading import Thread, Lock
from trade import POOL_ENABLED, register_signal
import multiprocessing as mp
from module_test.raw_code.DataManagers.DataManagers import (
    BulkOptionQueryRequestParameter,
    BulkOptionDataManager,
    handle_extra_cols,
    build_name_format,
    extract_numeric_value,
    enforce_interval,
    enforce_inputs,
    determine_table_agg,
    determine_requested_columns,
    init_query,

)
from typing import List, Tuple
from functools import partial
from dbase.utils import default_timestamp, bus_range, add_eod_timestamp
from dbase.DataAPI.ThetaExceptions import is_thetadata_exception
from trade import HOLIDAY_SET
import shelve
from pathlib import Path
import atexit
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import signal



logger = setup_logger('QuantTools.EventDriven.riskmanager.utils')
time_logger = setup_logger('QuantTools.EventDriven.riskmanager.time')
logger.info("RISK MANAGER is Using Old DataManager")


## There's no point loading only within strt, end. Load all to avoid 
TIMESERIES_START = pd.to_datetime('2017-01-01')
TIMESERIES_END = datetime.today()

## Patch tickers to swap old tickers with new ones
PATCH_TICKERS = True

def get_patch_tickers():
    return PATCH_TICKERS

def set_patch_tickers(patch_tickers):
    global PATCH_TICKERS
    PATCH_TICKERS = patch_tickers

## To-Do:
## 1. Filter out contracts that have already been queried. Saves time
## 2. Move cache to class attribute.

##Test low memory cache
# 1) pick a folder for your caches
BASE = Path(os.environ["WORK_DIR"])/ ".riskmanager_cache"
BASE.mkdir(exist_ok=True)
location = Path(os.environ['GEN_CACHE_PATH'])

# 2) swap your dicts for Cache instances
chain_cache = CustomCache(BASE, fname="chain", expire_days=45)
close_cache = CustomCache(BASE, fname="close", expire_days=45)
oi_cache    = CustomCache(BASE, fname="oi", expire_days=45)
spot_cache  = CustomCache(BASE, fname="spot", expire_days=45)
formatted_flags = CustomCache(location = BASE, fname = 'formatted_flags', expire_days=45)
PERSISTENT_CACHE = CustomCache(location,
                               fname='persistent_cache',
                               expire_days=30)

# 3) Create clear_cache function
def clear_cache() -> None:
    """
    clears the cache
    """
    global chain_cache, close_cache, oi_cache, spot_cache
    chain_cache.clear()
    close_cache.clear()
    oi_cache.clear()
    spot_cache.clear()

## Register info on `skips` from add_skip_columns
IDS = []
ID_SAVE_FOLDER = Path(os.environ['WORK_DIR']) / '.cache'
ID_SAVE_FILE = ID_SAVE_FOLDER / 'position_data.csv'

## Function to register information about skips in the position data to a list
def register_info_stack(id, data, data_col, update_kwargs = {}):
    """
    Register the information stack for a given position ID.
    
    Parameters:
    - id: The position ID.
    - data: The DataFrame containing position data.
    
    Returns:
    - info: A dictionary containing the registered information.
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data must be a pandas DataFrame.")
    

    info = {}
    info['ID'] = id
    for k in data_col:
        info[f'{k.upper()}_SKIP'] = data[f"{k}_skip_day"].sum()
        copy_cat = data[f"{k}_skip_day"].copy().to_frame()
        copy_cat['streak_id'] = copy_cat[f"{k}_skip_day"].ne(copy_cat[f"{k}_skip_day"].shift()).cumsum()
        copy_cat['streak'] = copy_cat.groupby('streak_id').cumcount() + 1
        info[f'{k.upper()}_MAX_STREAK'] = copy_cat[copy_cat[f"{k}_skip_day"] ==True].streak.max() if not copy_cat[copy_cat[f"{k}_skip_day"] ==True].streak.empty else 0
    info['DATA_LEN'] = len(data)
    info['DATETIME'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
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
        print("No data to save.")
        return
    full_data = pd.read_csv(ID_SAVE_FILE) if ID_SAVE_FILE.exists() else pd.DataFrame()
    df = pd.DataFrame(IDS)
    full_data = pd.concat([full_data, df], ignore_index=True)
    full_data.to_csv(ID_SAVE_FILE, index=False)
    with open(ID_SAVE_FOLDER/'ids.txt', 'a') as f:
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



LOOKBACKS = {}

def _retrieve_openInterest(*args, **kwargs) -> pd.DataFrame|None:
    try:
        return retrieve_openInterest(*args, **kwargs)
    except Exception as e:
        if is_thetadata_exception(e):
            print(f"Error retrieving open interest: {e}")
            return None
        else:
            raise e

    
def _retrieve_eod_ohlc(*args, **kwargs) -> pd.DataFrame|None:
    try:
        return retrieve_eod_ohlc(*args, **kwargs)
    except Exception as e:
        if is_thetadata_exception(e):
            print(f"Error retrieving EOD OHLC data: {e}")
            return None
        else:
            raise e

## Keep track of deleted keys in the cache
DELETED_KEYS = []

def get_deleted_keys() -> List[str]:
    """
    Returns the list of deleted keys from the cache.
    
    Returns:
        List[str]: A list of deleted keys.
    """
    global DELETED_KEYS
    return DELETED_KEYS


def set_deleted_keys(keys: List[str]) -> None:
    """
    Sets the deleted keys in the cache.
    
    Args:
        keys (List[str]): A list of keys that were deleted from the cache.
    """
    global DELETED_KEYS
    DELETED_KEYS = keys
    logger.info(f"Deleted Keys: {DELETED_KEYS}")



def set_timeseries_start_end(start: str|datetime, end: str|datetime) -> None:
    """
    Sets the start and end dates for the timeseries data.
    
    Args:
        start (str|datetime): The start date for the timeseries data.
        end (str|datetime): The end date for the timeseries data.
    """
    global TIMESERIES_START, TIMESERIES_END
    TIMESERIES_START = pd.to_datetime(start)
    TIMESERIES_END = pd.to_datetime(end)
    logger.info(f"Timeseries Start: {TIMESERIES_START}, Timeseries End: {TIMESERIES_END}")

def get_timeseries_start_end() -> Tuple[str, str]:
    """
    Returns the start and end dates for the timeseries data.
    
    Returns:
        Tuple[str, str]: A tuple containing the start and end dates for the timeseries data.
    """
    global TIMESERIES_START, TIMESERIES_END
    return TIMESERIES_START.strftime('%Y-%m-%d'), TIMESERIES_END.strftime('%Y-%m-%d')

# Precompute BDay lookbacks to eliminate redundant calculations
def precompute_lookbacks(start_date, end_date, _range = [10, 20, 30]) -> None:

    ## Extending to allow for multiple lookbacks
    global LOOKBACKS
    trading_days = pd.date_range(start=start_date, end=end_date, freq=BDay())
    if len(LOOKBACKS) == 0:
        lookback_cache = {x.strftime('%Y-%m-%d'): {} for x in trading_days}
    else:
        lookback_cache = LOOKBACKS
    for date in trading_days:
        dates = {x: (date - BDay(x)).strftime('%Y-%m-%d') for x in _range}
        lookback_cache[date.strftime('%Y-%m-%d')].update(dates)
    LOOKBACKS = lookback_cache

precompute_lookbacks('2000-01-01', '2030-12-31')

# Function to check if a date is a holiday
def is_holiday(date) -> bool:
    return date in HOLIDAY_SET

def get_cache(name: str) -> CustomCache:
    """
    returns the cache for the given name
    """
    global chain_cache, close_cache, oi_cache, spot_cache
    if name == 'chain':
        return chain_cache
    elif name == 'close':
        return close_cache
    elif name == 'oi':
        return oi_cache
    elif name == 'spot':
        return spot_cache

    else:
        raise ValueError(f"Invalid cache name: {name}")

@PERSISTENT_CACHE.memoize()
def populate_cache_with_chain(tick, date, print_url = True):
    """
    Populate the cache with chain data.
    """
    chain = retrieve_chain_bulk(
        tick,
        '',
        date,
        date,
        '16:00',
        'C',
        print_url = print_url
    )
    logger.info(f"Retrieved chain for {tick} on {date}")


    ## Clip Chain
    chain_clipped = chain.reset_index()[['datetime', 'Root', 'Strike', 'Right', 'Expiration', 'Midpoint']]
    if PATCH_TICKERS:
        chain_clipped['Root'] = chain_clipped['Root'].apply(swap_ticker)

    ## Create ID
    id_params = chain_clipped[['Root', 'Right', 'Expiration', 'Strike']].T.to_numpy()
    ids = runThreads(
        generate_option_tick_new, 
        id_params)
    chain_clipped['opttick'] = ids
    chain_clipped['chain_id'] = chain_clipped['opttick'] + '_' + chain_clipped['datetime'].astype(str)
    chain_clipped['dte'] = (pd.to_datetime(chain_clipped['Expiration']) - pd.to_datetime(chain_clipped['datetime'])).dt.days

    ## Save to cache
    def save_to_cache(id, date, spot):
        date = pd.to_datetime(date).strftime('%Y-%m-%d')
        save_id = f"{id}_{date}"
        if save_id not in get_cache('spot'):
            spot_cache[save_id] = spot
    save_params = chain_clipped[['opttick', 'datetime', 'Midpoint']].T.to_numpy()
    runThreads(
        save_to_cache, 
        save_params)
    chain_clipped.columns = chain_clipped.columns.str.lower()

    return chain_clipped

def refresh_cache() -> None:
    """
    Refreshes the cache for the order picker
    """
    global order_cache, spot_cache, close_cache, oi_cache, chain_cache
    spot_cache = get_cache('spot')
    close_cache = get_cache('close')
    oi_cache = get_cache('oi')
    chain_cache = get_cache('chain')

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


def mad_zscore_spike_flag(df, threshold=10, window=10, col ='Midpoint'):
    """
    Add a flag to the DataFrame indicating if the change in 'Midpoint' exceeds the threshold.
    """
    df = df.copy()
    median = df[col].rolling(window).median()
    mad = lambda x: np.median(np.abs(x - np.median(x))) ## lambda function that calculates median absolute deviation. x is a series, therefore x - median(x)
    rolling_mad = df[col].rolling(window).apply(mad) ## Apply function
    zscore_like = (df[col] - median) / rolling_mad ## Z-score like calculation
    return zscore_like.abs() > threshold

def mad_band_spike_flag(df, threshold=2, window=20, col='Midpoint'):    
    """
    Add a flag to the DataFrame indicating if the change in 'Midpoint' exceeds the threshold.
    """
    df = df.copy()
    median = df[col].rolling(window).median()
    mad = df[col].rolling(window).apply(lambda x: np.median(np.abs(x - np.median(x))))
    return (df[col] - median).abs() > threshold * mad

def quantile_band_spike_flag(df, window=20, upper_quantile=0.90, lower_quantile=0.10, col='Midpoint'):
    """
    Add a flag to the DataFrame indicating if the change in 'Midpoint' exceeds the threshold.
    """
    df = df.copy()
    quantile = df[col].rolling(window).quantile(upper_quantile)
    quantile_down = df[col].rolling(window).quantile(lower_quantile)
    return (df[col] > quantile) | (df[col] < quantile_down)
    
    
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
        df.loc[df[col] < 0 , col] = 0 ## NOTE: This is one time fix. Take it out
        smooth = df[col].ewm(span=3).mean()
        _zscore = (smooth - smooth.rolling(window).mean()) / smooth.rolling(window).std()
        _thresh = _zscore.abs() > skip_threshold

        ## Percentage change
        smooth_pct = df[col].pct_change().fillna(0)
        _zscore_pct = (smooth_pct - smooth_pct.rolling(window).mean()) / smooth_pct.rolling(window).std()
        _zscore_pct = _zscore_pct.fillna(0)
        _zscore_pct.replace([np.inf, -np.inf], 0, inplace=True) ## Replace inf values with 0
        _thresh_pct = _zscore_pct.abs() > skip_threshold

        ## Spike Detection
        spike_flag = mad_band_spike_flag(df, threshold=skip_threshold, window = window, col=col)

        ## Window 
        shortened = df[col][:window]
        pct_change = shortened.pct_change()
        window_bool = pct_change.abs() > 0.5

        ## Zero Values
        zero_bool = df[col] == 0

        
        ## Combine both boolean masks
        _combined = _thresh  | spike_flag | window_bool| zero_bool# | _thresh_pct
        df[f'{col}_abs_zscore'] = _thresh
        df[f'{col}_pct_zscore'] = _thresh_pct
        df[f'{col}_spike_flag'] = spike_flag
        df[f'{col}_window'] = window_bool
        df[f'{col}_zero'] = zero_bool
        df[f'{col}_skip_day']= _combined
        df[f'{col}_skip_day_count'] = _combined.rolling(60).sum()
    register_info_stack(id, df, skip_columns, update_kwargs={'window': window, 'skip_threshold': skip_threshold, 'window_bool_threshold': 0.5})
    return df



def date_in_cache_index(date, opttick) -> bool:
    """
    Check if a date is in the index of a cache for an option tick.
    """
    if opttick not in get_cache('close').keys():
        return False
    
    if get_cache('close')[opttick] is None or get_cache('oi')[opttick] is None:
        return False
    
    return date in get_cache('close')[opttick].index and date in get_cache('oi')[opttick].index

def assemble_bulk_data_request(self, start: str | datetime, 
                       end: str | datetime,
                       interval: str = '1d',
                       type_: str = 'spot',
                       strikes_right: List[Tuple] = [],
                       model: str = 'bs',
                       extra_cols: list = []) -> BulkOptionQueryRequestParameter:
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    ivl_str, ivl_int = extract_numeric_value(interval)
    greek_names = self.greek_names
    _extra_cols = handle_extra_cols(extra_cols, type_, model)
    greek_cols = build_name_format('greek', model, extra_cols, self.default_fill)
    vol_cols = build_name_format('vol', model, extra_cols, self.default_fill)


    ## Enforce the interval
    enforce_interval(ivl_str)

    ## Assert inputs
    enforce_inputs(type_, model)

    ## Determine aggregation
    agg, database, table = determine_table_agg(ivl_str, type_, greek_names)
    input_params = getattr(self, agg)

    ## Determine the requested columns
    requested_col = determine_requested_columns(self.default_fill, type_, model, greek_names)

    data_request = BulkOptionQueryRequestParameter(table_name=table,
                                                        db_name=database, 
                                                        start_date=start, 
                                                        end_date=end, 
                                                        ticker=self.symbol, 
                                                        exp=self.exp, 
                                                        strikes=strikes_right)

    ## Set the parameters for the request to avoid having too many attributes
    data_request.symbol = self.symbol
    data_request.interval= interval
    data_request.type_ = type_
    data_request.input_params = input_params
    data_request.model = model
    data_request.ivl_str = ivl_str
    data_request.ivl_int = ivl_int
    data_request.default_fill = self.default_fill
    data_request.agg = agg
    data_request.requested_col = requested_col + _extra_cols + ['optiontick']
    data_request.iv_cols = vol_cols
    data_request.greek_cols = greek_cols
    data_request.col_kwargs = col_kwargs = {'underlier_price': 's0', 
            'expiration': 'exp_date', 
            'strike': 'k', 
            'right': 'right', 
            'rf_rate': 'r', 
            'dividend': 'y',
            'put/call': 'right',
            'datetime': 'datetime',}
    return data_request


def update_caches(x) -> None:
    refresh_cache()
    global oi_cache, close_cache, oi_cache, spot_cache
    key = f"{x.Optiontick.unique()[0]}"
    
    ## When updating cache, we either set the data if not in cache, or append the data if it is in cache
    if key in close_cache.keys(): ## Appending data
        data = close_cache[key]
        data.columns = data.columns.str.lower() ## This is to normalize for appending
        x.columns = x.columns.str.lower()
        x.set_index('datetime', inplace=True)
        x = pd.concat([data, x], axis=0)
        x.columns = x.columns.str.capitalize() ## Keeping the original format
        x = x[~x.index.duplicated(keep = 'first')]
    
    else: ## Setting data
        x = x.set_index('Datetime')
    close_cache[key] = x
    oi_cache[key] = x['Openinterest'].to_frame(name = 'Open_interest')
    return



def update_cache_with_missing_ticks(parsed_opts: pd.DataFrame, date: str|datetime ) -> None:
    """
    Updates the cache with missing ticks by retrieving EOD data and open interest data.

    Args:
        parsed_opts (pd.DataFrame): DataFrame containing parsed option ticks with start and end dates and key meta data.

    Returns:
        None
    """
    global oi_cache, close_cache, oi_cache, spot_cache
    
    tickOrderedList = parsed_opts[['ticker', 'put_call', 'exp_date', 'strike', ]].T.to_numpy()
    tick_results = (runThreads(generate_option_tick_new, tickOrderedList, 'map' , block=True))
    tick_results = list(tick_results)
    parsed_opts['opttick'] = tick_results
    ## We get ticks first, then filter out the ones that are already in the cache
    missing_ticks = [x for x in tick_results if x not in close_cache.keys() and \
                     x not in oi_cache.keys() \
                        and f"{x}_{date}" not in spot_cache.keys()]
    
    data_list = []
    for tick in tick_results:
        data = close_cache.get(tick, None)
        if data is not None and date not in data.index:
            ## If the data is not None, but the date is not in the index, we add it to the list
            data_list.append(tick)

    missing_ticks.extend(data_list)   
    if len(missing_ticks) == 0:
        ## If there are no missing ticks, we check for each of their datetime completeness (SQL sometimes is incomplete)


        return
    else:
        ## We filter out the ones that are already in the cache to reduce the number of requests
        parsed_opts = parsed_opts[parsed_opts.opttick.isin(missing_ticks)]

    OrderedList = parsed_opts[['ticker', 'end_date', 'exp_date', 'put_call', 'start_date', 'strike', ]].T.to_numpy()
    eod_results = (runThreads(_retrieve_eod_ohlc, OrderedList, 'map', block=True))
    oi_results = (runThreads(_retrieve_openInterest, OrderedList, 'map' , block=True))
    

    
    eod_results = list(eod_results)
    oi_results = list(oi_results)
    

    for oi, eod, tick in zip(oi_results, eod_results, tick_results):
        ## We won't filter for `None` since we are skipping ticks with None in cache
        cache_key = f"{tick}"
        if oi is None or eod is None:
            ## If either of them is None, we skip the tick
            close_cache[cache_key] = None
            oi_cache[cache_key] = None
            continue
        eod.index = default_timestamp(eod.index)
        eod['Optiontick'] = tick
        close_cache[cache_key] = eod

        ## OI Formatting for consistency
        oi.set_index('Datetime', inplace=True)
        oi.index = default_timestamp(oi.index)
        oi_cache[cache_key] = oi

    return



def organize_data_for_query(missing_list: list,
                            incomplete_dict: dict,
                            data_request: 'DataManagers.Request') -> pd.DataFrame:
    """
    Organizes the data for the query by parsing the option ticks and adding start and end dates.
    
    Args:
        missing_list (list): List of missing option ticks. These are ticks that are not in the database at all.
        incomplete_dict (dict): Dictionary of incomplete option ticks. These are ticks that are in the database but not complete.
        data_request (BulkOptionQueryRequestParameter): The data request object containing start and end dates.
        
    Returns:
    pd.DataFrame: A DataFrame containing the parsed option ticks with start and end dates.
    """
    parsed_opts = pd.DataFrame()
    ## First populate with the ticks completely missing.
    parsed_opts = pd.DataFrame([parse_option_tick(x) for x in missing_list])
    parsed_opts[['start_date', 'end_date']] = data_request.start_date, data_request.end_date

    ## Next populate with the ticks that are incomplete
    for opt, _list in incomplete_dict.items():
        if len(_list) == 0:
            continue
        opt_meta = parse_option_tick(opt)
        opt_meta['start_date'] = min(_list)
        opt_meta['end_date'] = data_request.end_date
        parsed_opts = pd.concat([parsed_opts, pd.DataFrame(opt_meta, index = [0])], axis=0)
    return parsed_opts


def format_cache() -> None:
    """
    Drops duplicates from the cache & capitalizes column names,
    but only for DataFrames we haven’t formatted yet.
    """
    global close_cache, oi_cache, formatted_flags

    def process_dataframe(df):
        # same as before…
        if df is None or df.empty:
            return df
        if not df.index.is_unique:
            df = df.loc[~df.index.duplicated(keep='first')]
        df.columns = [col.capitalize() for col in df.columns]
        return df

    def process_cache(cache):
        # 1) only keys not yet in formatted_flags
        to_process = [k for k in cache if not formatted_flags.get(k, False)]

        if not to_process:
            return

        with ThreadPoolExecutor() as executor:
            dfs = [cache[k] for k in to_process]
            results = executor.map(process_dataframe, dfs)

            # 2) write them back & 3) mark formatted
            for k, new_df in zip(to_process, results):
                cache[k] = new_df
                formatted_flags[k] = True

    process_cache(get_cache('close'))
    process_cache(get_cache('oi'))

def merge_incomplete_data_in_cache(
    incomplete_dict: dict,
    pre_processed_data: pd.DataFrame,
) -> None:
    global close_cache, oi_cache, spot_cache
    ## Now we have updated cache, since incomplete date updates cache with the missing dates, we have to add the data we already have
    for tick, _list in incomplete_dict.items():
        if len(_list) == 0:
            continue
        tick_data = pre_processed_data[pre_processed_data.Optiontick == tick]
        tick_data = tick_data.set_index('Datetime')
        close_cache[tick] = pd.concat([close_cache[tick], tick_data], axis=0).sort_index()
        oi_data = tick_data['Openinterest'].to_frame(name = 'Open_interest')
        oi_cache[tick] = pd.concat([oi_cache[tick], oi_data]).sort_index()

def update_spot_cache(opttick: list, target_date: str|datetime) -> None:
    """
    Updates the spot cache with the close price for the given option tick and target date.
    Args:
        opttick (list): List of option ticks.
        target_date (str|datetime): Target date to get the close price for.
    Returns:
        None
    """
    refresh_cache()
    global spot_cache, close_cache
    spot_results = runThreads(return_closePrice, [opttick, [target_date]*len(opttick)], 'map')
    for tick, spot in zip(opttick, spot_results):
        cache_key = f"{tick}_{target_date}"
        if spot is None:
            continue ## If spot is None, we don't update the cache
        spot_cache[cache_key] = spot


def should_update_cache(key, start, end) -> bool:

    """
    Check if the cache should be updated based on the key and date range.
    
    Args:
        key (str): The key to check in the cache.
        start (str): The start date for the date range.
        end (str): The end date for the date range.
        
    Returns:
        bool: True if the cache should be updated, False otherwise.
    """

    ## Update cache if not in close or oi cache
    if key not in get_cache('close').keys() :
        return True
    
    if key not in get_cache('oi').keys():
        return True
    
    ## Update cache if the date range is not in the cache
    close_df = get_cache('close')[key].copy()
    close_df['Datetime'] = close_df.index
    return not check_all_days_available(close_df, start, end)


@log_error_with_stack(logger)
def populate_cache_v1(start_date,
        end_date,
        order_candidates,
        target_date,) -> str|None:
    
    """
    populates the cache with the necessary data for the order candidates

    params:
    order_candidates: dict: dictionary containing the order candidates
        example: {'type': 'naked',
                    'specifics': [{'direction': 'long',
                    'rel_strike': .900,
                    'dte': 365,
                    'moneyness_width': 0.15},
                    {'direction': 'short',
                    'rel_strike': .80,
                    'dte': 365,
                    'moneyness_width': 0.15}],

                    'name': 'vertical_spread'}
    date: str: date to populate the cache for

    returns:
    str|None: returns 'holiday' if the date is a holiday, 'theta_data_error' if there is an error in the theta data, None otherwise
    """
    global close_cache, oi_cache, spot_cache

    tempholder1 = {}
    tempholder2 = {}

    if is_holiday(target_date):
        return 'holiday'
    
    else:

        ## Create necessary data structures
        ## Looping through the order candidates to get the necessary data, and organize into a list of lists that will be passed to runProcesses function
        for j, direction in enumerate(order_candidates):
            for i,data in enumerate(order_candidates[direction]):
                if isinstance(data, str) and data =='theta_data_error':
                    return 'theta_data_error'
                
                data[[ 'exp', 'strike', 'symbol', 'right']] = data[[ 'Expiration', 'Strike', 'ticker', 'Right']]
                if pd.to_datetime(target_date).weekday() >= 5:
                    return 'weekend'
                data[['end_date', 'start_date']] = end_date, start_date
                data['exp'] = data['exp'].dt.strftime('%Y-%m-%d')
                tempholder1[i+j] = (data[['symbol', 'end_date', 'exp', 'right', 'start_date', 'strike']].T.values.tolist())
                tempholder2[i+j] = data[['symbol', 'right', 'exp','strike']].T.values.tolist()
        symbol = data['symbol'].unique()[0]
        expiration = data['exp'].unique()[0]

        ## Extending lists, to ensure only one runProcesses call is made, instead of run per side
        for i, data in tempholder1.items():
            if i == 0:
                OrderedList = data
                tickOrderedList = tempholder2[i]
            else:
                for position, vars in enumerate(data):
                    OrderedList[position].extend(vars)
                for position, vars in enumerate(tempholder2[i]):
                    tickOrderedList[position].extend(vars)



        ## TO-DO:
        ## 1. Create hacked function to catch errors from ThetaData
        ## 2. Add OptionDataManager.one_off_save in hacked function
        eod_results = (runThreads(retrieve_eod_ohlc, OrderedList, 'map'))
        oi_results = (runThreads(_retrieve_openInterest, OrderedList, 'map'))
        tick_results = (runThreads(generate_option_tick_new, tickOrderedList, 'map'))

        ## Save to Dictionary Cache
        logger.info(f"Updating Cache for: {symbol}, {expiration}, {target_date}")
        for tick, eod, oi in zip(tick_results, eod_results, oi_results):
            cache_key = f"{tick}"
            close_cache[cache_key] = eod
            oi_cache[cache_key] = oi
        
        spot_results = runThreads(return_closePrice, [tick_results, [target_date]*len(tick_results)], 'map')
        for tick, spot in zip(tick_results, spot_results):
            cache_key = f"{tick}_{target_date}"
            spot_cache[cache_key] = spot


@log_error_with_stack(logger)
@log_time(time_logger)
def populate_cache_v2(
        start,
        end,
        candidates,
        target_date,
) -> str|None:
    """
    populates the cache with the necessary data for the order candidates
    This version will improve on the previous one by using the new BulkOptionDataManager
    The goal is to make use of our database to speed up queries where possible

    params:
    candidates: dict: dictionary containing the order candidates
        example: {'type': 'naked',
                    'specifics': [{'direction': 'long',
                    'rel_strike': .900,
                    'dte': 365,
                    'moneyness_width': 0.15},
                    {'direction': 'short',
                    'rel_strike': .80,
                    'dte': 365,
                    'moneyness_width': 0.15}],

                    'name': 'vertical_spread'}
    start: str: date to populate the cache for
    end: str: date to populate the cache for
    target_date: str: date to populate the cache for

    returns:
    str|None: returns 'holiday' if the date is a holiday, 'theta_data_error' if there is an error in the theta data, None otherwise
    """
    
    global oi_cache, close_cache, oi_cache, spot_cache
    start, end = pd.to_datetime(start), pd.to_datetime(end)
    full_data = pd.DataFrame()
    for direction in candidates:
        for data in candidates[direction]:
            if isinstance(data, str) and data =='theta_data_error':
                return 'theta_data_error'
            
            elif pd.to_datetime(target_date).weekday() >= 5:
                return 'weekend'
            
            elif isinstance(data, str) and data =='holiday':
                return 'holiday'

            elif isinstance(data, str):
                print(f"Data is a string: {data}, Error incoming...")
                
            full_data = pd.concat([full_data, data], axis=0)

    full_data.index.name = 'Date'
    full_data.columns.name = ''
    full_data['start_date'] = start
    full_data['end_date'] = end
    full_data.reset_index(inplace=True)
    tick = full_data.ticker.unique()[0]
    exp = full_data.Expiration.unique()[0]
    strikes_right = list(full_data[['Strike', 'Right']].itertuples(name=None, index=False))
    opttick = [generate_option_tick_new(tick, x[1], exp, x[0]) for x in strikes_right] 
    update_spot = [x for x in opttick if f"{x}_{target_date}" not in spot_cache.keys()]
    
    ## 1) Goal: Ensuring we don't requery the database for data we already have. Saves time
    ## Filtering out the ones that are already in the cache
    strikes_right = [x for x in strikes_right if opttick[strikes_right.index(x)] not in close_cache.keys()]
    
    ## 2) Goal: For all ticks in cache, we want to reupdate if start & end not in cache.
    ## Getting opttick already in cache

    cached_opttick = [x for x in opttick if x in close_cache.keys()]
    non_cached_opttick = [x for x in opttick if x not in close_cache.keys()]

    if len(strikes_right) != 0:
        logger.critical(f"Data needs to be queried for {len(strikes_right)} strikes_right. Load time ~1.5mins")
    
    # ## Updating strikes_right list with this keys once we have checked that they should be updated
    # if len(cached_opttick) != 0:
    #     update_list = [(parse_option_tick(key)["strike"], parse_option_tick(key)["put_call"]) \
    #                    for key in cached_opttick if should_update_cache(key, start, end)]
    #     strikes_right.extend(update_list)

    strikes_right = list(set(strikes_right)) ## Removing duplicates
    logger.info(f"Number of strikes_right: {len(strikes_right)}")
    logger.info(f"Strike Rights: {strikes_right}")

    logger.info(f"Info on {tick}, for date: {target_date}")

    ## Let's start with getting the requested data from database

    ## If everything is in the cache, we don't need to do anything. Go straight to updating spot
    if len(strikes_right) != 0:
        manager = BulkOptionDataManager(symbol=tick, exp=exp)
        logger.info(f"Generating Data for {manager.symbol} {manager.exp}")
        data_request = assemble_bulk_data_request(
            self = manager,
            start = start,
            end = end,
            type_ = 'spot',
            strikes_right = strikes_right,
            # strikes_right= [(225.0, 'C'), (280.0, 'P'), (250.0, 'C'), (290.0, 'P'), (270.0, 'C'), (270.0, 'P')],

        )
        # ## Second: we query our database to see what data we have
        query_time = time.time()
        init_query(data_request = data_request, query_category = 'bulk')
        logger.info(f"Time taken to query database: {time.time()-query_time}")
        num_optticks = len(data_request.opttick)
        size_database_data = len(data_request.database_data)
        expected_days = bus_range(data_request.start_date, data_request.end_date, freq = '1B')
        logger.info(f"Date Range: {data_request.start_date} to {data_request.end_date}")
        logger.info(f"Number of requested optticks: {num_optticks}")
        logger.info(f"Number of database data: {size_database_data}")
        logger.info(f"Number of expected days: {len(expected_days)}")
        logger.info(f"Expected Size of database data: {num_optticks * len(expected_days)}")
        logger.info(f"Amount discrepancy: {num_optticks * len(expected_days) - size_database_data}")
        logger.info(f"Time taken to query database: {time.time()-query_time}")

        # ## Third: we pre_process the data request to see if it is complete
        BulkOptionDataManager.pre_process_data(data_request = data_request) 


        is_complete = data_request.pre_process['is_complete']
        pre_processed_data = data_request.pre_processed_data.reset_index()
        is_complete_series = pre_processed_data.groupby('Optiontick').apply(check_all_days_available, _start = data_request.start_date, _end = data_request.end_date)
        logger.info(f"Is complete series: {is_complete_series.to_string()}")
        opttick = data_request.opttick
        logger.info(f"Data Is_complete bool: {is_complete}")

        ## If complete, Fantastic! We re done, now update cache and get out
        if is_complete:
            pre_processed_data.groupby('Optiontick').apply(update_caches)

        ## If NOT complete, do not fret. We'll simply run our process for incomplete/missing ticks
        else:
            ## We first check for the requested ticks. Which one is not in database at all?
            missing_opttick = [x for x in data_request.opttick if x not in pre_processed_data.Optiontick.unique()]
            logger.info(f"In missing opttick but not in opttick: ")
            logger.info([x for x in opttick if x not in missing_opttick])
            
            ## Next we check to see if the requested opttick data is COMPELETE. 
            ## If incomplete, we perform runthreads
            check_partial = partial(check_all_days_available, _start = data_request.start_date, _end = data_request.end_date)
            opttick_complete = pre_processed_data.groupby('Optiontick').apply(check_partial)
            incomplete_ticks = opttick_complete[opttick_complete==False].index.tolist()
            incomplete_dict = pre_processed_data.groupby('Optiontick').apply(check_missing_dates, _start = data_request.start_date, _end = data_request.end_date)
            if incomplete_dict.empty:
                incomplete_dict = {}
            else:
                incomplete_dict = incomplete_dict.to_dict()
            ## Before we perform run Threads, it is important we update cache with the Optticks that are COMPLETE
            available = opttick_complete[opttick_complete==True].index
            # pre_processed_data[pre_processed_data.Optiontick.isin(available)].groupby('Optiontick').apply(update_caches)
            
            ## We want to update the cache with whatever we have. merge_incomplete_data_in_cache will take care of the rest
            pre_processed_data.groupby('Optiontick').apply(update_caches)
            ## Produce the dataframe that stores names to update the cache
            to_update_cache_data = organize_data_for_query(
                missing_list=missing_opttick,
                incomplete_dict=incomplete_dict,
                data_request=data_request
            )



            # Now my dear friends, we update cache of unavailable ticks
            start_time = time.time()
            update_cache_with_missing_ticks(parsed_opts = to_update_cache_data, date = target_date)
            end_time = time.time()
            logger.info(f"Time taken to update cache: {end_time-start_time}")

            ## Merge the data we have in cache with the data we just retrieved for the incomplete ticks
            merge_incomplete_data_in_cache(incomplete_dict = incomplete_dict, pre_processed_data = pre_processed_data)
            format_cache()
            refresh_cache()

        
        ## Now we update the spot cache
        update_spot_cache(opttick = update_spot, target_date = target_date)
    else:
        format_cache()
        update_spot_cache(opttick = update_spot, target_date = target_date)

    BulkOptionDataManager.one_off_save(
        start=start,
        end=end,
        tick=tick,
        exp=exp
    ) ## We shouldn't keep going to thetadata, that takes time. Submit a process. Don't worry it runs on a new process.
    ## Wouldn't affect current procedures
    


    
@copy_doc(populate_cache_v2)
def populate_cache(start_date, end_date, order_candidates, target_date, version = 2) -> str|None:
    print(f"Populate Cache Dates: Start: {start_date}, End: {end_date}, Target: {target_date}")
    if version == 1:
        logger.info("Using V1")
        return populate_cache_v1(start_date, end_date, order_candidates, target_date)
    elif version == 2:
        logger.info("Using V2")
        return populate_cache_v2(start_date, end_date, order_candidates, target_date)


def return_closePrice(id: str, 
                      date: str) -> float:
    """
    returns the close price of the option contract
    id: str: id of the option contract, corresponding to cache keys.
        ps: Use spot_cache.keys() to get the keys
    date: str: date to get the close price for

    returns:
    float: close price of the option contract
    
    """
    refresh_cache()
    global close_cache, spot_cache, oi_cache
    cache_key = f"{id}"  ## Close Uses only the id, not the date
    close_data = close_cache[cache_key]
    if close_data is None:
        return None
    close_data = close_data[~close_data.index.duplicated(keep = 'first')]
    if date not in close_data.index:
        ## If the date is not in the close data, we remove that key from the cache
        ## There's no way to resolve this, so we remove the key from the cache
        try:
            logger.info(f"Removing {cache_key} from cache, since date {date} not in close data") 
            DELETED_KEYS.append(cache_key)
            # del close_cache[cache_key]
            # del oi_cache[cache_key]
        except KeyError:
            pass
        return None
    close = close_data['Midpoint'][date]
    return close


def load_chain(date: str, 
               ticker: str,  
               print_stderr: bool = False) -> None:
        """
        loads the option chain for the given date and ticker

        params:
        date: str: date to load the chain for
        ticker: str: ticker to load the chain for
        print_stderr: bool: whether to print to stderr or not

        returns:
        None
        
        """
        print(date, ticker) if print_stderr else None
        ## Get both calls and puts per moneyness. For 1 Moneyness, both will most be available. If not, if one is False, other True. 
        ## We will need to get two rows. 
        chain_key = f"{date}_{ticker}"
        with Context(end_date = date):
            if chain_key in chain_cache:
                Option_Chain = chain_cache[chain_key]
            else:
                start_time = time.time()
                Stock_obj = Stock(ticker, run_chain = False)
                end_time = time.time()
                print(f"Time taken to get stock object: {end_time-start_time}") if print_stderr else None
                Option_Chain = Stock_obj.option_chain()
                Spot = Stock_obj.spot(ts = False, spot_type = OptionModelAttributes.spot_type.name) ## need to use chain price to get the spot price, due to splits
                Spot = list(Spot.values())[0]
                Option_Chain['Spot'] = Spot
                Option_Chain['q'] = Stock_obj.div_yield()
                Option_Chain['r'] = Stock_obj.rf_rate
                chain_cache[chain_key] = Option_Chain





def chain_details(date: str, 
                  ticker: str, 
                  tgt_dte: int, 
                  tgt_moneyness: float, 
                  right: str ='P', 
                  moneyness_width: float =0.15, 
                  print_stderr: bool = False) -> pd.DataFrame:
    

    """
    Returns the option chain details for the given date, ticker, target days to expiration, target moneyness, right, and moneyness width

    params:
    date: str: date to get the chain for
    ticker: str: ticker to get the chain for
    tgt_dte: int: target days to expiration
    tgt_moneyness: float: target moneyness
    right: str: right of the option contract. Default is 'P'
    moneyness_width: float: moneyness width. Default is 0.15. This is the width of the moneyness spread
    print_stderr: bool: whether to print to stderr or not

    returns:
    pd.DataFrame: option chain details
    """
    return_dataframe = pd.DataFrame()
    errors = {}
    if is_holiday(date):  # Replaced is_USholiday() with the optimized function
        return 'holiday'
    try:
        print(date, ticker) if print_stderr else None
        chain_key = f"{date}_{ticker}"
        with Context(end_date=date):
            if chain_key in chain_cache:
                Option_Chain = chain_cache[chain_key]
            else:
                start_time = time.time()
                Stock_obj = Stock(ticker, run_chain=False)
                end_time = time.time()
                print(f"Time taken to get stock object: {end_time-start_time}") if print_stderr else None
                try:
                    Option_Chain = Stock_obj.option_chain()
                except:
                    return 'theta_data_error'
                Spot = Stock_obj.spot(ts=False, spot_type=OptionModelAttributes.spot_type.value) ## need to use chain price to get the spot price, due to splits
                Spot = list(Spot.values())[0]
                Option_Chain['Spot'] = Spot
                Option_Chain['q'] = Stock_obj.div_yield()
                Option_Chain['r'] = Stock_obj.rf_rate
                chain_cache[chain_key] = Option_Chain
    
            Option_Chain_Filtered = Option_Chain[Option_Chain[right.upper()]==True]
        
            if right == 'P':
                Option_Chain_Filtered['relative_moneyness'] = Option_Chain_Filtered.index.get_level_values('Strike') / Option_Chain_Filtered.Spot
            elif right == 'C':
                Option_Chain_Filtered['relative_moneyness'] = Option_Chain_Filtered.Spot / Option_Chain_Filtered.index.get_level_values('Strike')
            else:
                raise ValueError(f'Right dne. received {right}')
            
            Option_Chain_Filtered['moneyness_spread'] = (tgt_moneyness - Option_Chain_Filtered['relative_moneyness'])**2
            Option_Chain_Filtered['dte_spread'] = (Option_Chain_Filtered.index.get_level_values('DTE') - tgt_dte)**2
            Option_Chain_Filtered.sort_values(by=['dte_spread', 'moneyness_spread'], inplace=True)
            Option_Chain_Filtered = Option_Chain_Filtered.loc[Option_Chain_Filtered['dte_spread'] == Option_Chain_Filtered['dte_spread'].min()]
            
            ## Secondary fix for when dte_spread **2 leads to both lower bound and upper bound being the same. Which returns two Expirations
            if len(Option_Chain_Filtered.reset_index().DTE.unique()) > 1:
                Option_Chain_Filtered = Option_Chain_Filtered.loc[Option_Chain_Filtered.index.get_level_values('DTE') ==\
                                                                   Option_Chain_Filtered.index.get_level_values('DTE').max()]
            
            if float(moneyness_width) == 0.0:
                option_details = Option_Chain_Filtered.sort_values('moneyness_spread', ascending=True).head(1)
            else:
                option_details = Option_Chain_Filtered[(Option_Chain_Filtered['relative_moneyness'] >= tgt_moneyness - moneyness_width) & 
                                                    (Option_Chain_Filtered['relative_moneyness'] <= tgt_moneyness + moneyness_width)]
        
            if option_details.empty:
                return None
            
            option_details['build_date'] = date
            option_details['ticker'] = ticker
            option_details['moneyness'] = tgt_moneyness
            option_details['TGT_DTE'] = tgt_dte
            option_details.reset_index(inplace = True)
            option_details.set_index('build_date', inplace = True)
            option_details['Right'] = right
            option_details.drop(columns = ['C','P'], inplace = True)
            option_details['option_id'] = option_details.apply(lambda x: generate_option_tick_new(symbol = x['ticker'], 
                                                                        exp = x['Expiration'].strftime('%Y-%m-%d'), strike = float(x['Strike']), right = x['Right']), axis = 1)
            return_dataframe = pd.concat([return_dataframe, option_details])
        clear_context()
        return_dataframe.drop_duplicates(inplace = True)

    except Exception as e:
        raise e
        return 'error'
    
    
    return return_dataframe.sort_values('relative_moneyness', ascending=False)
    


def available_close_check(id: str, 
                          date: str, 
                          threshold: float = 0.7,
                          lookback: float = 30) -> bool:
    
    """
    checks if the close price is available for the given id and date

    params:
    id: str: id of the option contract
        ps: Use spot_cache.keys() to get the available ids
    date: str: date to check the close price for
    threshold: float: threshold to check if the close price is available. Default is 0.7

    returns:
    bool: True if the close price is available, False otherwise
    """
    cache_key = f"{id}"  ## Close Uses only the id, not the date
    if cache_key in DELETED_KEYS:
        ## If the cache key is in the deleted keys, we return False
        logger.info(f"Close Check: {id} is in DELETED_KEYS, returning False")
        return False
    sample_id = deepcopy(get_option_specifics_from_key(id))
    new_dict_keys = {'ticker': 'symbol', 'exp_date': 'exp', 'strike': 'strike', 'put_call': 'right'}
    transfer_dict = {}
    for k, v in sample_id.items():
        if k in new_dict_keys:
            if k == 'strike':
                transfer_dict[new_dict_keys[k]] = float(sample_id[k])
            else:
                transfer_dict[new_dict_keys[k]] = sample_id[k]
    
    if cache_key in close_cache:
        close_data_sample = close_cache[cache_key]
        close_data_sample = close_data_sample[(~close_data_sample.index.duplicated(keep = 'first')) & (close_data_sample.index <= date)] ## Filter out duplicates, and only dates before the target date
        close_data_sample = close_data_sample.iloc[-lookback:] ## Get the last lookback days
    else:
        start = LOOKBACKS[date][lookback]  # Used precomputed BDay(30)
        close_data_sample = retrieve_eod_ohlc(**transfer_dict, start_date=start, end_date=date)
        close_cache[cache_key] = close_data_sample
    close_mask_series = close_data_sample.Close != 0
    return close_mask_series.sum()/len(close_mask_series) > threshold

@log_time(time_logger)
def produce_order_candidates(settings: dict, 
                             tick: str, 
                             date: str, 
                             right: str = 'P',
                             thread: bool = False) -> dict:
    """
    returns the order candidates for the given settings, tick, date, and right

    params:
    settings: dict: settings for the order candidates
        example: {'type': 'naked',
                    'specifics': [{'direction': 'long',
                    'rel_strike': .900,
                    'dte': 365,
                    'moneyness_width': 0.15},
                    {'direction': 'short',
                    'rel_strike': .80,
                    'dte': 365,
                    'moneyness_width': 0.15}],

                    'name': 'vertical_spread'}

    tick: str: ticker to get the order candidates for
    date: str: date to get the order candidates for
    right: str: right of the option contract. Default is 'P'
    
    returns:
    dict: order_candidates
    """

    def hacked_chain_details(*args, **kwargs):
        direction = kwargs.pop('direction')
        chain = chain_details(*args, **kwargs)
        order_candidates[direction].append(chain)

    thread_lock = Lock()
    order_candidates = {'long': [], 'short': []}
    thread_list = []
    if thread:
        for spec in settings['specifics']:
            _thread = Thread(target=hacked_chain_details, args=(date, tick, spec['dte'], spec['rel_strike'], right, spec['moneyness_width']), kwargs={'direction': spec['direction']})
            _thread.start()
            thread_list.append(_thread)
            # chain = chain_details(date, tick, spec['dte'], spec['rel_strike'], right,  moneyness_width = spec['moneyness_width'], direction = spec['direction'])
            # order_candidates[spec['direction']].append(chain)
        for thread in thread_list:
            thread.join()
    else:
        for spec in settings['specifics']:
            direction = spec['direction']
            chain = chain_details(date, tick, spec['dte'], spec['rel_strike'], right,  moneyness_width = spec['moneyness_width'])
            order_candidates[spec['direction']].append(chain)
    return order_candidates


def liquidity_check(id: str, 
                    date: str, 
                    pass_threshold: int|float = 250, 
                    lookback: int = 10) -> bool:
    
    """
    returns True if the liquidity is greater than the pass_threshold, False otherwise

    params:
    id: str: id of the option contract
        ps: Use oi_cache.keys() to get the available ids
    date: str: date to check the liquidity for
    pass_threshold: int|float: threshold to check if the liquidity is greater than. Default is 250
    lookback: int: lookback to check the liquidity for. Default is 10

    returns:
    bool: True if the liquidity is greater than the pass_threshold, False otherwise
    """
    if id in DELETED_KEYS:
        ## If the cache key is in the deleted keys, we return False
        logger.info(f"Liquidity Check: {id} is in DELETED_KEYS, returning False")
        return False
    sample_id = deepcopy(get_option_specifics_from_key(id))
    new_dict_keys = {'ticker': 'symbol', 'exp_date': 'exp', 'strike': 'strike', 'put_call': 'right'}
    transfer_dict = {}
    
    for k, v in sample_id.items():

        if k in new_dict_keys:
            if k == 'strike':
                transfer_dict[new_dict_keys[k]] = float(sample_id[k])
            else:
                transfer_dict[new_dict_keys[k]] = sample_id[k]

    start = LOOKBACKS[date][lookback]  # Used precomputed BDay(30)
    oi_data = oi_cache[f"{id}"] ## OI Uses only the id, not the date
    if oi_data is None:
        return False
    oi_data = oi_data[~oi_data.index.duplicated(keep = 'first')]
    oi_data = oi_data[oi_data.index <= date]
    oi_data = oi_data.iloc[-lookback:] ## Get the last lookback days
    

    if isinstance(oi_data, pd.DataFrame):
        if oi_data.empty:
            return False
        
    elif oi_data is None:
        return False
    
    oi_data = oi_data[~oi_data.index.duplicated(keep = 'first')]
    oi_data = oi_data.iloc[:lookback]
    return oi_data.Open_interest.sum()/lookback > pass_threshold if isinstance(oi_data, pd.DataFrame) else False



