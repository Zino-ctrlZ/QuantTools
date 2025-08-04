## Notes
## 1. save_to_database has a check_if_already_saved function. This is to avoid taking up time processing already saved info
##      also, there is no checks on kwargs scheduled. So incase an already processed request is scheduled, it will avoid processing again
## 2. POOL_ENABLED is a global variable to control if we should use multiprocessing or not
## 3. Can set POOL_ENABLED using set_pool_enabled_dm(value: bool)
## 4. POOL_ENABLED is used in the parallel_apply function to control if we should use multiprocessing or not
## 5. Including `set_should_schedule` in save_to_database to avoid scheduling requests that go throgh ThetaData pipeline
## 6. There are timeframe requests where the data starts later than the requested start date, database has the complete data but that date is not available anywhere.
    ## So anytime we query data, it will still go thru ThetaData. Therefore, we will return an empty dataframe

import warnings
warnings.filterwarnings("ignore")
import time
from dotenv import load_dotenv
import os
import sys
import logging
from openpyxl import load_workbook
from datetime import datetime, date
from datetime import time as dtTime
import pandas as pd
import threading
import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool
from concurrent.futures import ProcessPoolExecutor, as_completed
import concurrent.futures
from trade import get_pool_enabled, PRICING_CONFIG
from trade.assets.Stock import Stock
from trade.helpers.helper import generate_option_tick_new
from trade.assets.rates import get_risk_free_rate_helper
from trade.helpers.helper import (IV_handler, 
                                  time_distance_helper, 
                                  binomial_implied_vol,
                                    wait_for_response, 
                                    HOLIDAY_SET,
                                    enforce_allowed_models,
                                    optionPV_helper,
                                    check_missing_dates,
                                    check_all_days_available)
from trade.helpers.helper import extract_numeric_value, change_to_last_busday, parse_option_tick
from trade.helpers.helper import optionPV_helper
from trade.helpers.exception import IncorrectExecutionError
from trade.helpers.Logging import setup_logger
from trade.assets.Calculate import Calculate
from trade.helpers.Context import Context
from dbase.DataAPI.ThetaData import (retrieve_ohlc, 
                                     retrieve_quote_rt, 
                                     retrieve_eod_ohlc, 
                                     resample, 
                                     retrieve_quote, 
                                     enforce_bus_hours,
                                     retrieve_bulk_eod,
                                     retrieve_openInterest,
                                     retrieve_chain_bulk,
                                     list_contracts,
                                     retrieve_bulk_open_interest,
                                     set_should_schedule
                                     )
from trade.helpers.pools import  parallel_apply
from trade.helpers.decorators import log_error, log_error_with_stack, log_time
from trade.helpers.types import OptionModelAttributes
from dateutil.relativedelta import relativedelta
from pandas.tseries.offsets import BDay
from dbase.database.SQLHelpers import DatabaseAdapter
from trade.models.VolSurface import fit_svi_model
from trade.models.utils import resolve_missing_vol
from threading import Thread, Lock
from dbase.utils import add_eod_timestamp, bus_range, enforce_bus_hours, default_timestamp
from trade.helpers.pools import runProcesses
from pathos.multiprocessing import ProcessingPool as Pool
from copy import deepcopy
import json
from queue import Queue, Full
from threading import Thread
from typing import TYPE_CHECKING, List, Tuple
from .SaveManager import (SaveManager, 
                          save_failed_request, 
                          flatten_all_dfs)
from .SaveManager_processes import ProcessSaveManager
from functools import partial

## DM NEEDED IMPORTS
from .utils import _ManagerLazyLoader, EMPTY_TIMESEIRES_TABLE
from .Requests import (
    create_request_bulk,
                       get_bulk_requests,
                       get_chain_requests,
                       get_single_requests,
                       ChainDataRequest,
                       OptionQueryRequestParameter,
                       BulkOptionQueryRequestParameter)
from .cache import get_cache
from .shared_obj import (get_request_list)
from dbase.DataAPI.ThetaExceptions import ThetaDataNotFound
import logging
from pathlib import Path
from functools import wraps

                       


logger = setup_logger('DataManager.py', stream_log_level = logging.CRITICAL)
time_logger = setup_logger('time_logger_test_dm')
vol_resolve_logger = setup_logger('DataManagers.Vol_Resolve')


CENTRAL_SAVE_THREAD = {}
DB_CACHE = get_cache()
SAVE_TO_DB = True ## This is a global variable to control if we save to DB or not
SHOULD_SCHEDULE = True ## This is a global variable to control if we should schedule the save or enqueue
POOL_ENABLED = get_pool_enabled() ## This is a global variable to control if we should use multiprocessing or not
                                  ## Switching to function based to avoid stickyness when variable is changed
                                  ## Thinking of it, it wouldn't make any difference because the func is called once.

def refresh_cache():
    """
    Refresh the cache. Ensures that the cache is not stale.
    """
    global DB_CACHE
    DB_CACHE = get_cache()

def set_pool_enabled_dm(value: bool):
    """
    Set the pool enabled flag.
    """
    global POOL_ENABLED
    POOL_ENABLED = value

def get_pool_enabled_dm():
    """
    Get the pool enabled flag.
    """
    return POOL_ENABLED


def cached_get_timeseries(func):
    """
    Decorator to cache the entire return value (Request object) of get_timeseries(...)
    in a CustomCache. The cache key is built from the OptionDataManager instance's identifying
    attributes plus all get_timeseries arguments.
    """
    @wraps(func)
    def wrapper(self, start, end, interval="1d", type_="spot", model="bs", extra_cols=None):
        # 1) Normalize / stringify the inputs so the key is reproducible:
        if extra_cols is None:
            extra_cols = []
        # Ensure start/end are strings (or ISO format), so the key is stable:
        start_str = pd.to_datetime(start).strftime("%Y-%m-%d_%H:%M:%S")
        end_str   = pd.to_datetime(end).strftime("%Y-%m-%d_%H:%M:%S")
        # extra_cols as a comma‐separated string:
        extra_str = ",".join(sorted(extra_cols))

        # Build a unique cache key. We assume `self.opttick` uniquely identifies
        # symbol/exp/right/strike internally; if you want to be more explicit you
        # could do f"{self.symbol}|{self.exp}|{self.right}|{self.strike}|…"
        key = "|".join([
            str(self.opttick),
            start_str,
            end_str,
            interval,
            type_,
            model,
            extra_str
        ])

        # 2) Try to fetch from disk:
        if key in DB_CACHE:
            return DB_CACHE[key]

        # 3) Not in cache? Call the real method and store its return value:
        result = func(self, start, end, interval, type_, model, extra_cols)
        DB_CACHE[key] = result
        return result

    return wrapper

## Skip MySql Query. MySql query is not optimized so just go straight to API
SKIP_MYSQL_QUERY = False
def set_skip_mysql_query(value: bool):
    """
    Set the skip MySQL query flag.
    """
    if value:
        logger.critical("Skipping MySQL query. This is not optimized and may lead to performance issues.")
    global SKIP_MYSQL_QUERY
    SKIP_MYSQL_QUERY = value

def get_skip_mysql_query():
    """
    Get the skip MySQL query flag.
    """
    return SKIP_MYSQL_QUERY


## Set Save Manager 
def get_save_manager():
    from .SaveManager import SaveManager
    from .SaveManager_processes import ProcessSaveManager
    from trade import get_pool_enabled
    return ProcessSaveManager if get_pool_enabled_dm() else SaveManager

_SaveManager = get_save_manager()

def reset_save_manager():
    """
    Resets the save manager to the default.
    """
    global _SaveManager
    _SaveManager = get_save_manager()


def set_save_bool(b:bool):
    """
    Sets the save to DB boolean value.
    """
    global SAVE_TO_DB
    SAVE_TO_DB = b

def get_save_bool() -> bool:
    """
    Returns the save to DB boolean value.
    """
    return SAVE_TO_DB

def set_schedule_bool(b:bool):
    """
    Sets the schedule boolean value.
    """
    global SHOULD_SCHEDULE
    SHOULD_SCHEDULE = b

def get_schedule_bool() -> bool:  
    """
    Returns the schedule boolean value.
    """
    return SHOULD_SCHEDULE


TABLES = {
    'eod':{
        'attribution': 'securities_master.attribution_eod',
        'spot': 'securities_master.temp_options_eod_new',
        'vol': 'securities_master.temp_options_eod_new',
        'greeks': 'securities_master.temp_options_eod_new',
        'chain': 'vol_surface.option_chain'
    },
    'intra':{
        'attribution': 'securities_master.attribution_intra',
        'spot': 'securities_master.temp_options_intra_new',
        'vol': 'securities_master.temp_options_intra_new',
        'greeks': 'securities_master.temp_options_intra_new',
    }
}


#### Empty classes. Not yet implemented.

class AttributionDataManager:
    def __init__(self):
        raise NotImplementedError("AttributionDataManager is not implemented yet.")

class ScenarioDataManager:
    def __init__(self):
        raise NotImplementedError("ScenarioDataManager is not implemented yet.")

    

class SpotDataManager:
    def __init__(self, symbol:str):
        self.symbol = symbol

    @log_error_with_stack(logger)
    def query_thetadata(self,
                        start: str | datetime,
                        end: str | datetime,
                        strike: float = None,
                        exp: str | datetime = None,
                        right: str = None,
                        bulk: bool = False,
                        **kwargs) -> pd.DataFrame:
        """
        Query the spot data & Open Interest from ThetaData API.
        """
        data_request = kwargs.get('data_request')
        print_url = kwargs.get('print_url', False)
        agg = data_request.agg
        if agg == 'eod':
            if not bulk:
                data = retrieve_eod_ohlc(symbol=self.symbol, end_date=end, exp=exp, right=right, start_date=start, strike=strike, print_url=print_url)
                data = data[~data.index.duplicated(keep='first')]
                open_interest = retrieve_openInterest(symbol=self.symbol, end_date=end, exp=exp, right=right, start_date=start, strike=strike, print_url=print_url).set_index('Datetime')
                open_interest.drop_duplicates(inplace = True)
                data['Open_interest'] = open_interest['Open_interest']
                data.index = default_timestamp(data.index)
                return data

            else:
                bulk = retrieve_bulk_eod(
                    symbol = self.symbol,
                    exp = exp,
                    start_date = start,
                    end_date = end,
                )

                ## Add Option Tick
                bulk_eod = bulk.reset_index()
                tick_col = ['Root', 'Right', 'Expiration', 'Strike']
                bulk_eod['OptionTick'] = parallel_apply(bulk_eod[tick_col], generate_option_tick_new, pool = False)
                if data_request.opttick is not None:
                    bulk_eod = bulk_eod[bulk_eod['OptionTick'].isin(data_request.opttick)]


                ## Query Bulk Open Interest
                bulk_oi = retrieve_bulk_open_interest(
                    symbol = self.symbol,
                    exp = exp,
                    start_date = start,
                    end_date = end,
                )
                ## Add Option Tick
                bulk_oi['OptionTick'] = parallel_apply(bulk_oi[tick_col], generate_option_tick_new, pool = False)
                if data_request.opttick is not None:
                    bulk_oi = bulk_oi[bulk_oi['OptionTick'].isin(data_request.opttick)]
                ## Add EOD Timestamp
                bulk_oi['Datetime'] = add_eod_timestamp(pd.DatetimeIndex(bulk_oi['Datetime']))
                data = bulk_eod.merge(bulk_oi[['Datetime','OptionTick', 'Open_interest']], on = ['Datetime', 'OptionTick'], how = 'left')
                data = data.rename(columns = {'Root': 'ticker', 'Strike':'k', 'Expiration': 'exp_date'})
                data.set_index('Datetime', inplace = True)
                data.index = default_timestamp(pd.DatetimeIndex(data.index))
                return data
            
        elif agg == 'intra':
            if not bulk:
                data = retrieve_ohlc(symbol=self.symbol, end_date=end, exp=exp, right=right, start_date=pd.to_datetime(start) - BDay(1), strike=strike, print_url=print_url)
                ## For open Interest we will query from Start - 1BDay to End
                open_interest = retrieve_openInterest(symbol=self.symbol, end_date=end, exp=exp, right=right, 
                                                      start_date=pd.to_datetime(start) - BDay(1), strike=strike, 
                                                      print_url=print_url).set_index('Datetime')['Open_interest']
                
                ## PS: Quering for Open Interest uses Start - 1BDAY to END
                ## This is because open interest returns EOD. Resampling to intraday moves previous day data to current day
                ## Therefore, first date will be NaN because it will be the previous day data, which is not included in the query results
                open_interest = resample(open_interest, PRICING_CONFIG['INTRADAY_AGG'] )
                data['Open_interest']=open_interest
                
                return data
            else:
                raise NotImplementedError("Bulk data not implemented for intra data")
            

class VolDataManager:
    def __init__(self, symbol:str):
        self.symbol = symbol

    @log_error_with_stack(logger)
    def calculate_iv(self, **kwargs):
        """
        Calculate the implied volatility using the model.
        """
        data_request = kwargs['data_request']
        model = data_request.model
        raw_data = data_request.raw_spot_data
        raw_data.columns = [x.lower() for x in raw_data.columns]
        raw_data['datetime'] = raw_data.index
        return_cols = []
        for col, name in data_request.iv_cols.items():
            calc_vol_for_data_parallel(raw_data, col, name, model, col_kwargs = data_request.col_kwargs, pool = False)

        raw_data.drop(columns=['datetime'], inplace=True)


class GreeksDataManager:
    def __init__(self, symbol:str):
        self.symbol = symbol

    @log_error_with_stack(logger)
    def calculate_greeks(self, type_, **kwargs):
       
        data_request = kwargs['data_request']
        model = data_request.model
        raw_data = data_request.raw_spot_data
        raw_data.columns = [x.lower() for x in raw_data.columns]
        raw_data['datetime'] = raw_data.index
        if type_ in ['greek', 'greeks']:
            ## Greeks
            for col, format_name in data_request.greek_cols.items():
                calc_greeks_for_data_parallel(raw_data, model, col, format_name, col_kwargs = data_request.col_kwargs, pool = False)
        else:
            ## Individual Greeks
            for col, format_name in data_request.greek_cols.items():
                calc_greeks_for_data_parallel(raw_data, model, col, format_name, col_kwargs = data_request.col_kwargs, greek_name=type_, pool = False)
        
        raw_data.drop(columns=['datetime'], inplace=True)

## Writing this as a separate class to handle the chain data
## I don't want it to depend on OptionDataManager
## Because it isn't tethered to a specific option.
class ChainDataManager(_ManagerLazyLoader):
    """
    Class to manage the chain data for a given symbol.
    It inherits from the _ManagerLazyLoader class to load data on demand.
    It uses the ChainDataRequest class to handle the data requests.
    It uses the DatabaseAdapter class to handle the database operations.
    """

    CLASS_THREADS = {}
    def __init__(self, symbol):
        """
        Initialize the ChainDataManager with the symbol.
        """
        super().__init__(symbol)
        self.symbol = symbol
        self.requests = {}
        self.current_request = ''
        self.db = DatabaseAdapter()

    @log_error_with_stack(logger)
    def get_at_time(self, date:str, organize:bool = False) -> pd.DataFrame:
        database, table = TABLES['eod']['chain'].split('.')
        self.exp = date
        self.current_request = datetime.now().strftime("%Y%m%d %H:%M:%S")
        data_request = ChainDataRequest(
            symbol=self.symbol,
            date = date,
            table_name = table,
            db_name = database )
        self.requests[self.current_request] = data_request
        init_query(data_request=data_request, db=self.db, query_category='chain')
        self.__pre_process(data_request=data_request)
        is_empty = data_request.is_empty
        if is_empty:
            self.__post_process(data_request=data_request)
            kwargs = dict(
                exp=date,
                start=date,
                end=date,
                tick=self.symbol,
                save_func=partial(save_chain_data),
                set_attributes={'post_processed_data': data_request.post_processed_data,},
                type_='chain',
            )
            _SaveManager.schedule(kwargs) if SHOULD_SCHEDULE else _SaveManager.enqueue(kwargs)

        else:
            data_request.post_processed_data = data_request.database_data

        if organize:
                data = data_request.post_processed_data.copy()
                data.columns = data.columns.str.capitalize() 
                data.rename(columns = {'Dte': 'DTE', 'Price': 'Midpoint'}, inplace=True)
                chain = data.pivot_table(
                                    index = ['Expiration', 'DTE', 'Strike'],
                                    columns = ['Right'],
                                    values = ['Midpoint']
                                )
                data_request.organized_data = chain
        else:
            data_request.organized_data = data_request.post_processed_data
        
        return data_request
                
            
    @log_error_with_stack(logger)
    def __pre_process(self, **kwargs):
        """
        Preprocess the data for the request
        """
        data_request = kwargs.get('data_request')
        database_data = data_request.database_data
        database_data.columns = [x.lower() for x in database_data.columns]
        if database_data.empty:
            data_request.is_empty = True
        else:
            data_request.is_empty = False
        
        return database_data

    def __post_process(self, **kwargs):
        """
        Postprocess the data for the request
        """
        logger.warning(f"ChainDataManger will not be returning Volatility data due to performance.")
        data_request = kwargs.get('data_request')
        date = data_request.date
        chain = retrieve_chain_bulk(self.symbol, 0, date, date, PRICING_CONFIG['MARKET_CLOSE_TIME'])
        chain.index.name = 'build_date'
        chain_v2 = chain.rename(columns = {'Root':'ticker', 'Midpoint': 'price'}).drop(columns = ['Date']).reset_index()
        chain_v2.columns = [x.lower() for x in chain_v2.columns]
        chain_v2['dte'] = (chain_v2['expiration'] - chain_v2['build_date']).dt.days
        chain_v2['spot'] = self.eod['s0_chain']['close'][date]
        chain_v2['r'] = self.eod['r'][date]
        chain_v2['q'] = self.eod['y'][date]
        chain_v2['option_tick'] = chain_v2.apply(lambda x: generate_option_tick_new(x['ticker'], x['right'], x['expiration'].strftime('%Y-%m-%d'), x['strike']), axis=1)
        chain_v2['moneyness'] = chain_v2.apply(lambda x: x['spot'] / x['strike'], axis=1)
        data_request.post_processed_data = chain_v2


    @staticmethod
    def one_off_save(date:str,
                     tick:str,
                     print_info:bool = True):
        """
        This function is used to save the data to the database without initializing the data manager
        """
        kwargs = dict(
            exp=date,
            start=date,
            end=date,
            tick=tick,
            save_func=partial(save_chain_data),
            type_='chain',
        )
        _SaveManager.schedule(kwargs)
        




class BulkOptionDataManager(_ManagerLazyLoader):
    """
    Class to manage the bulk option data for a given symbol.
    It inherits from the _ManagerLazyLoader class to load data on demand.
    It uses the BulkOptionQueryRequestParameter class to handle the data requests.
    It uses the DatabaseAdapter class to handle the database operations.
    """
    
    

    @log_time(time_logger)
    def __init__(self,
                symbol: str = None,
                exp: str | datetime = None,
                default_fill: str = 'midpoint',
                **kwargs) -> None:
        """
        Returns an object for querying data

        Params:
        symbol: Underlier symbol
        exp: expiration
        right: Put(P) or Call (C)
        strike: Option Strike
        default_fill: How to fill zero values for close. 'midpoint' or 'weighted_midpoint'
        opttick: Option ticker, if provided, will ignore symbol, exp, right, strike and be initialized with the string
        """
        

        super().__init__(symbol)
        if default_fill not in ['midpoint', 'weighted_midpoint', None]:
            raise ValueError("Expected default_fill to be one of: 'midpoint', 'weighted_midpoint', None ")
        
        assert all([symbol, exp,]), "symbol, exp, are required"
        self.exp = exp
        self.symbol = symbol

        self.default_fill = default_fill
        self.db = DatabaseAdapter()
        self.data_request = {}
        self.save_thread = {}
        self.current_request =None
        self.spot_manager = SpotDataManager(self.symbol)
        self.vol_manager = VolDataManager(self.symbol)
        self.greek_manager = GreeksDataManager(self.symbol)
        self.chain_manager = ChainDataManager(self.symbol)
        self.greek_names = PRICING_CONFIG["AVAILABLE_GREEKS"] + ['greek', 'greeks']
        self.print_info = kwargs.get('print_info', False)

        ## Prefer to use dicts to avoid having too many attributes
        self._eod = {}
    
    def get_timeseries(self, 
                       start: str | datetime, 
                       end: str | datetime,
                       interval: str = '1d',
                       type_: str = 'spot',
                       strikes_right: List[Tuple] = [],
                       model: str = 'bs',
                       extra_cols: list = []) -> pd.DataFrame:
        """
        Query the timeseries data from ThetaData API or SQL Database.
        Params:
        start: Start date for the query
        end: End date for the query
        interval: Interval for the query. Options are: h, d, w, M, q, y
        type_: Type of data to query. Options are: spot, vol, greeks, greek, attribution, scenario
        model: Model to use for the query. Options are: bs, binomial
        extra_cols: Extra columns to include in the query. Options are: ask, bid, open
        strikes_right: List of tuples containing the strike and right for the options. Eg: [(250, 'C'), (225, 'P')]
        """
        
        if not strikes_right:
            raise ValueError("Strikes cannot be empty")
        
        assert isinstance(strikes_right, list), f"Strikes has to be type list, recieved {type(strikes_right)}"
        assert all([isinstance(x, tuple) for x in strikes_right]), f"Strikes has to be type list of tuples, recieved {type(strikes_right)}"
        
        ## Organize inputs
        self.current_request = datetime.now().strftime("%Y%m%d %H:%M:%S")
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
        self.data_request[self.current_request] = data_request ## save the request for future reference

        ## Start by getting query
        init_query(data_request=data_request, db=self.db, query_category='bulk')
        ## Next, pre process data available in database
        self.pre_process_data(data_request=data_request)

        ## Before handling missing/incomplete data, we begin save to database
        is_complete = data_request.pre_process['is_complete']
        is_empty = data_request.pre_process['is_empty']
        if is_empty or not is_complete:
            if SAVE_TO_DB:
                kwargs = dict(exp = self.exp,
                start = start,type_ = 'bulk',
                end = end,
                tick = self.symbol,
                save_func = partial(save_to_database, print_info=True),)
                _SaveManager.schedule(kwargs) if SHOULD_SCHEDULE else _SaveManager.enqueue(kwargs)


        ## Handle missing or incomplete data if any
        self.__handle_incomplete_data(data_request=data_request)

        ## Post process the data
        post_process(data_request=data_request, bulk = True)
        
        ## Format the data
        format_final_data(data_request=data_request, bulk = True)
        return data_request 
 
    ## Make a function
    def __handle_incomplete_data(self, **kwargs):
        data_request = kwargs['data_request']
        is_complete = data_request.pre_process['is_complete']
        is_empty = data_request.pre_process['is_empty']
        start, end, type_ = data_request.start_date, data_request.end_date, data_request.type_

        if is_empty:
            raw_spot_data = self.spot_manager.query_thetadata(start=start, end=end, 
                                                              strike=None, exp=self.exp, 
                                                              right=None, bulk=True, 
                                                              data_request=data_request)
            data_request.raw_spot_data = raw_spot_data
            if type_ != 'spot':
                ## Add inputs to raw data, this is necessary for vol calculation
                add_inputs_to_raw(self, data_request=data_request, bulk = True) ## Not formatting yet, this is to utilize joins on datetime
                vol_data = self.vol_manager.calculate_iv(data_request=data_request)
                if type_ in self.greek_names:
                    greek_data = self.greek_manager.calculate_greeks(type_, data_request = data_request)
            format_raw_spot_data(data_request=data_request)
            


        elif not is_complete:
            start_missing, end_missing = min(data_request.missing_dates), max(data_request.missing_dates)
            raw_spot_data = self.spot_manager.query_thetadata(start=start_missing, end=end_missing, 
                                                              strike=None, exp=self.exp, 
                                                              right=None, bulk=True, 
                                                              data_request=data_request)

            data_request.raw_spot_data = raw_spot_data
            if type_ != 'spot':
                add_inputs_to_raw(self, data_request=data_request, bulk = True)
                vol_data = self.vol_manager.calculate_iv(data_request=data_request)
                data_request.raw_spot_data = pd.concat([data_request.raw_spot_data, vol_data], axis=1)
                if type_ in self.greek_names:
                    greek_data = self.greek_manager.calculate_greeks(type_, data_request = data_request)
                    data_request.raw_spot_data = pd.concat([data_request.raw_spot_data, greek_data], axis=1)
            format_raw_spot_data(data_request=data_request)
        
        else:
            data_request.raw_spot_data = pd.DataFrame()

    @classmethod
    def pre_process_data(cls, **kwargs):
        data_request = kwargs.get('data_request')
        data = data_request.database_data
        data_request.pre_process = {}

        ## Check timeseries is complete
        ## Considering we're taking a resample approach, where base intraday data is 5 minutes, and EOD is 1 day
        ## We will only check 5 minutes and 1 day is complete

        start, end = data_request.start_date, data_request.end_date
        date_range = bus_range(start, end, '5Min') if data_request.agg == 'intra' else bus_range(start, end, '1B')

        ## Transform the data to Opttick as columns, close as values, datetime as index
        transformed = data_request.database_data.pivot_table(
            index = ['datetime'],
            columns = ['optiontick'],
            values = ['close']
        )
        transformed.columns = transformed.columns.droplevel(0)


        ## First Completeness check: Do we have all OptTicks?
        first_check = all(x in transformed.columns for x in data_request.opttick)
        ## This will fill missing option ticks with NaN
        transformed[[x for x in data_request.opttick if x not in transformed.columns.get_level_values(0)]] = np.nan


        ## Second Completeness check: Do we have all dates?
        missing_dates_second_check = [x for x in date_range if x not in (transformed.index)]
        second_check = all(x in pd.DatetimeIndex(transformed.index) for x in date_range)
        transformed = transformed.reindex(date_range, fill_value=np.nan) ## This will fill the missing dates with NaN

        ## Third Completeness check: If we have all Ticks, do all ticks have all dates? 
        ## This will exclude dates from second check
        if not first_check:
            third_check = False
        else:
            ## Check if all dates are present for all ticks
            third_check = not transformed.isna().any().any()

        complete_check = first_check and second_check and third_check

        ## Is empty Check
        is_empty = data_request.database_data.empty

        # Missing Dates: This will be tiered
        if not first_check: ## This means all dates are missing for one name, we have to query all dates
            missing_dates = date_range
        else: ## This means some dates are missing for some names.
            missing_dates = transformed[transformed.isna().any(axis = 1)].index.to_list()

        ## Save preprocessed data
        data_request.pre_process['is_complete'] = complete_check
        data_request.pre_process['is_empty'] = is_empty
        data_request.missing_dates = missing_dates

        data = data[data_request.requested_col] ## Select only the requested columns
        data.columns = [x.capitalize() for x in data.columns] ## Capitalize the columns
        data.set_index([ 'Optiontick', 'Datetime',], inplace = True) ## Set the index to datetime
        data = data[~data.index.duplicated(keep='first')] ## Remove duplicates
        data.sort_index(inplace = True)
        data_request.pre_processed_data = data

    @staticmethod
    def one_off_save(start:str, 
                     end:str, 
                     tick:str, 
                     exp:str, 
                     print_info:bool = False):
        """
        This function is used to save the data to the database without initializing the data manager
        """
        bulk_one_off_save(start, end, tick, exp, print_info)


class OptionDataManager(_ManagerLazyLoader):
    """
    Class to manage the option data for a given symbol.
    It inherits from the _ManagerLazyLoader class to load data on demand.
    It uses the OptionQueryRequestParameter class to handle the data requests.
    It uses the DatabaseAdapter class to handle the database operations.
    """


    @log_time(time_logger)
    def __init__(self,
                symbol: str = None,
                exp: str | datetime = None,
                right: str = None,
                strike: float = None,
                default_fill: str = 'midpoint',
                opttick: str = None,
                **kwargs) -> None:
        """
        Returns an object for querying data

        Params:
        symbol: Underlier symbol
        exp: expiration
        right: Put(P) or Call (C)
        strike: Option Strike
        default_fill: How to fill zero values for close. 'midpoint' or 'weighted_midpoint'
        opttick: Option ticker, if provided, will ignore symbol, exp, right, strike and be initialized with the string
        """

        
        if opttick is not None:
            assert isinstance(opttick, str), f"opttick has to be type str, recieved {type(opttick)}"
            option_meta = parse_option_tick(opttick)
            self.symbol = option_meta['ticker']
            self.exp = option_meta['exp_date']
            self.right = option_meta['put_call']
            self.strike = option_meta['strike']
            self.opttick = opttick

        else:
            assert isinstance(strike, float), f"Strike has to be type float, recieved {type(strike)}"
            if default_fill not in ['midpoint', 'weighted_midpoint', None]:
                raise ValueError("Expected default_fill to be one of: 'midpoint', 'weighted_midpoint', None ")
            
            assert all([symbol, exp, right, strike]), "symbol, exp, right, strike are required"
            self.exp = exp
            self.symbol = symbol
            self.right = right.upper()
            self.strike = strike
            self.opttick = generate_option_tick_new(symbol, right, exp, strike)
        
        ## Switching location of the symbol to be in the constructor,
        ## Incase initiated with ticker
        super().__init__(self.symbol)
        self.default_fill = default_fill
        self.db = DatabaseAdapter()
        self.data_request = {}
        self.save_thread = {}
        self.current_request =None
        self.spot_manager = SpotDataManager(self.symbol)
        self.vol_manager = VolDataManager(self.symbol)
        self.greek_manager = GreeksDataManager(self.symbol)
        self.chain_manager = ChainDataManager(self.symbol)
        self.greek_names = PRICING_CONFIG["AVAILABLE_GREEKS"] + ['greek', 'greeks']
        self.print_info = kwargs.get('print_info', False)

        ## Prefer to use dicts to avoid having too many attributes


    def get_timeseries(self, 
                       start: str | datetime, 
                       end: str | datetime,
                       interval: str = '1d',
                       type_: str = 'spot',
                       model: str = 'bs',
                       extra_cols: list = []) -> pd.DataFrame:
        """
        Query the timeseries data from ThetaData API or SQL Database.

        Params:
        start: Start date for the query
        end: End date for the query
        interval: Interval for the query. Options are: h, d, w, M, q, y
        type_: Type of data to query. Options are: spot, vol, greeks, greek, attribution, scenario
        model: Model to use for the query. Options are: bs, binomial
        extra_cols: Extra columns to include in the query. Options are: ask, bid, open
        """
        
        
        ## Organize inputs
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        ivl_str, ivl_int = extract_numeric_value(interval)
        greek_names = self.greek_names
        self.current_request = datetime.now().strftime("%Y%m%d %H:%M:%S")
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
        

        ## TO-DO: Move this to a method
        data_request = OptionQueryRequestParameter(table_name=table, 
                                                   db_name=database, 
                                                   start_date=start, 
                                                   end_date=end, 
                                                   ticker=self.symbol, 
                                                   exp=self.exp, 
                                                   strike=self.strike,
                                                   right=self.right)
        
        ## Set the parameters for the request to avoid having too many attributes
        data_request.opttick = self.opttick
        data_request.symbol = self.symbol
        data_request.interval= interval
        data_request.type_ = type_
        data_request.input_params = input_params
        data_request.model = model
        data_request.ivl_str = ivl_str
        data_request.ivl_int = ivl_int
        data_request.default_fill = self.default_fill
        data_request.agg = agg
        data_request.requested_col = requested_col + _extra_cols
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
        self.data_request[self.current_request] = data_request ## save the request for future reference
        
        ## Start by getting query
        init_query(data_request=data_request, db=self.db, query_category='single')

        ## Next, pre process data available in database
        self.__pre_process_data(data_request=data_request)

        ## Before handling missing/incomplete data, we begin save to database

        is_complete = data_request.pre_process['is_complete']
        is_empty = data_request.pre_process['is_empty']
        if is_empty or not is_complete:
            if SAVE_TO_DB:
                kwargs = dict(
                    exp=self.exp,
                    right=self.right,
                    strike=self.strike,
                    start=start,
                    end=end,
                    tick=self.symbol,
                    type_ = 'single',
                    save_func = partial(save_to_database, print_info=self.print_info))
                _SaveManager.schedule(kwargs) if SHOULD_SCHEDULE else _SaveManager.enqueue(kwargs)
            

        
        ## Handle missing or incomplete data if any
        self.__handle_incomplete_data(data_request=data_request)
        ## Post process the data
        post_process(data_request=data_request)
        format_final_data(data_request=data_request)
        

            
        return data_request
    
    def get_at_time(self, 
                   date: str | datetime, 
                   type_: str = 'spot',
                   model: str = 'bs',
                   **kwargs) -> pd.DataFrame:
        """
        Get data at a specific time
        params:

        """
        
        if type_ == 'chain':
            return_price = kwargs.get('return_price', False)
            if return_price:
                self.current_request = datetime.now().strftime("%Y%m%d %H:%M:%S")
                request = self.chain_manager.get_at_time(date)
                data = request.post_processed_data.copy()
                self.data_request[self.current_request] = request
                data.columns = data.columns.str.capitalize() 
                data.rename(columns = {'Dte': 'DTE', 'Price': 'Midpoint'}, inplace=True)
                chain = data.pivot_table(
                                    index = ['Expiration', 'DTE', 'Strike'],
                                    columns = ['Right'],
                                    values = ['Midpoint']
                                )
            else: 
                chain = self.Stock.option_chain(date = date)
            return chain
        
        elif type_ in ['spot', 'vol'] + self.greek_names:
            extra_cols = kwargs.get('extra_cols', [])
            return self.get_timeseries(date, date, 
                                       interval = '1d',
                                       type_ = type_,
                                       model = model,
                                       extra_cols=extra_cols).post_processed_data


    def __pre_process_data(self, **kwargs):
        
        data_request = kwargs.get('data_request')
        data = data_request.database_data
        data_request.pre_process = {}

        ## Check if data is empty
        if data.empty:
            ## If data is empty, we will not be able to process it
            data_request.pre_process['is_empty'] = True
        else:
            data_request.pre_process['is_empty'] = False

        ## Check timeseries is complete
        ## Considering we're taking a resample approach, where base intraday data is 5 minutes, and EOD is 1 day
        ## We will only check 5 minutes and 1 day is complete

        start, end = data_request.start_date, data_request.end_date
        date_range = bus_range(start, end, '5Min') if data_request.agg == 'intra' else bus_range(start, end, '1B')

        ## Now we will check if the data is complete
        is_complete = all([(x in pd.DatetimeIndex(data.datetime)) for x in date_range])
        missing_dates = [x for x in date_range if x not in pd.DatetimeIndex(data.datetime)]
        data_request.pre_process['is_complete'] = is_complete
        data_request.missing_dates = missing_dates

        ## Save preprocessed data
        data = data[data_request.requested_col] ## Select only the requested columns
        data.columns = [x.capitalize() for x in data.columns] ## Capitalize the columns
        data.set_index('Datetime', inplace = True) ## Set the index to datetime
        data = data[~data.index.duplicated(keep='first')] ## Remove duplicates
        data_request.pre_processed_data = data

    def __handle_incomplete_data(self, **kwargs):
        data_request = kwargs['data_request']
        is_complete = data_request.pre_process['is_complete']
        is_empty = data_request.pre_process['is_empty']
        start, end, type_ = data_request.start_date, data_request.end_date, data_request.type_

        if is_empty:
            try:
                raw_spot_data = self.spot_manager.query_thetadata(start=start, end=end, 
                                                                strike=self.strike, exp=self.exp, 
                                                                right=self.right, bulk=False, 
                                                                data_request=data_request)
                data_request.raw_spot_data = raw_spot_data
            
            except ThetaDataNotFound as e:
                ## This means we have no data for the given date range. Happens for edge cases when the range is cut from original date range
                ## This happens when option didn't start/end trading on the exact ranges
                logger.info(f"Error querying data: {e}")
                data_request.raw_spot_data = pd.DataFrame(columns = data_request.database_data.columns)
                return

            
            if type_ != 'spot':
                ## Add inputs to raw data, this is necessary for vol calculation
                add_inputs_to_raw(self, data_request=data_request) ## Not formatting yet, this is to utilize joins on datetime
                vol_data = self.vol_manager.calculate_iv(data_request=data_request)
                # data_request.raw_spot_data = pd.concat([data_request.raw_spot_data, vol_data], axis=1)
                if type_ in self.greek_names:
                    greek_data = self.greek_manager.calculate_greeks(type_, data_request = data_request)
            format_raw_spot_data(data_request=data_request)
            


        elif not is_complete:
            start_missing, end_missing = min(data_request.missing_dates), max(data_request.missing_dates)
            try:
                raw_spot_data = self.spot_manager.query_thetadata(start=start_missing, end=end_missing, 
                                                                strike=self.strike, exp=self.exp, 
                                                                right=self.right, bulk=False, 
                                                                data_request=data_request)
            except ThetaDataNotFound as e:
                ## This means we have no data for the given date range. Happens for edge cases when the range is cut from original date range
                ## This happens when option didn't start/end trading on the exact ranges
                logger.info(f"Error querying data: {e}")
                data_request.raw_spot_data = pd.DataFrame(columns = data_request.database_data.columns)
                return

            data_request.raw_spot_data = raw_spot_data
            if type_ != 'spot':
                add_inputs_to_raw(self, data_request=data_request)
                vol_data = self.vol_manager.calculate_iv(data_request=data_request)
                data_request.raw_spot_data = pd.concat([data_request.raw_spot_data, vol_data], axis=1)
                if type_ in self.greek_names:
                    greek_data = self.greek_manager.calculate_greeks(type_, data_request = data_request)
                    data_request.raw_spot_data = pd.concat([data_request.raw_spot_data, greek_data], axis=1)
            format_raw_spot_data(data_request=data_request)
        
        else:
            data_request.raw_spot_data = pd.DataFrame()

    @staticmethod
    def one_off_save(start:str, 
                     end:str, 
                     tick:str, 
                     exp:str, 
                     strike:float,
                     right:str,
                     print_info:bool = True):
        """
        This function is used to save the data to the database without initializing the data manager
        """
        kwargs = dict(
            exp=exp,
            right=right,
            strike=strike,
            start=start,
            end=end,
            tick=tick,
            type_ = 'single',
            save_func = partial(save_to_database, print_info=print_info))
        _SaveManager.schedule(kwargs)
        



#### Save to Database Functions
@log_error(logger) 
def save_to_database(data_request: 'RequestParameter', 
                     print_info: bool = False, 
                     pool: bool = None,
                     **kwargs) -> None:
    """
    Saves the data to the database
    """
    set_should_schedule(False) ## Avoid scheduling. This will occur when SpotDataManager.query_thetadata is called
    func_start = time.time()
    req_class = data_request.__class__.__name__

    ## This function is using parallel apply to reduce overhead on the current process.
    if pool is None:
        pool = get_pool_enabled()
    print(f"Pool: {pool}", flush=True) if print_info else None
    logger.info(f"Saving data to {data_request.db_name}.{data_request.table_name}")
    print(f"Saving data to {data_request.db_name}.{data_request.table_name}", flush=True) if print_info else None
    
    ## Determine if the data is bulk or not
    if isinstance(data_request, OptionQueryRequestParameter):
        bulk = False
    elif isinstance(data_request, BulkOptionQueryRequestParameter):
        data_request.strike = None
        data_request.right = None
        bulk = True
    else:
        raise ValueError(f"Expected data_request to be of type OptionQueryRequestParameter or BulkOptionQueryRequestParameter, recieved {type(data_request)}")
    
    db = DatabaseAdapter()

    ## If there is no missing data, we will not save the data to the database
    if len(data_request.missing_dates) == 0:
        print("No missing data, skipping save to database", flush=True) if print_info else None
        logger.warning("No missing data, skipping save to database")
        return
    
    ## Because of bulk size, we want to query data for 1 month before and after the missing dates
    if bulk:
        start_date, end_date  = pd.to_datetime(min(data_request.missing_dates)) - relativedelta(months=1), pd.to_datetime(max(data_request.missing_dates)) + relativedelta(months=1)
    else:
        start_date, end_date = pd.to_datetime(min(data_request.missing_dates)) - relativedelta(months=3), pd.to_datetime(max(data_request.missing_dates)) + relativedelta(months=3)
    logger.info(f"Querying data from {start_date} to {end_date}")
    print(f"Querying data from {start_date} to {end_date} for {req_class}", flush=True) if print_info else None
    
    ## Start by populating initial data from spot_manager & Querying Data
    spot_manager = SpotDataManager(data_request.symbol)
    start = time.time()
    print("Starting to query data", flush=True) if print_info else None
    spot_sm = spot_manager.query_thetadata(start_date, end_date,
                                                   strike=data_request.strike, exp=data_request.exp, 
                                                   right=data_request.right, bulk=bulk, 
                                                   data_request=data_request)
    print("Time taken to query data: ", time.time() - start, flush=True) if print_info else None
    spot_sm = spot_sm.fillna(0)
    spot_sm.drop_duplicates(inplace=True)

    ## Check in with DB to abort if data is already in DB
    spot_sm = check_if_already_saved(data_request, spot_sm)
    if spot_sm.empty:
        print("All data in Database, returning") if print_info else None
        return 
    print("Starting to save data to database", flush=True) if print_info else None
    if not bulk:
        spot_sm['Strike'] = data_request.strike
        spot_sm['Expiration'] = data_request.exp
        spot_sm['Put/Call'] = data_request.right
        spot_sm['OptionTick'] = data_request.opttick
        spot_sm['Underlier'] = data_request.symbol
        
    else:
        spot_sm.rename(columns = {'k':'strike','exp_date':'expiration', 'Right':'put/call', 'ticker':'Underlier'}, inplace = True)

    spot_sm['Underlier_price'] = data_request.input_params['s0_chain']['close']
    spot_sm['RF_rate'] = data_request.input_params['r']
    spot_sm['dividend'] = data_request.input_params['y']
    spot_sm['RF_rate_name'] = data_request.input_params['r_name']
    spot_sm['Datetime'] = spot_sm.index
    spot_sm.columns = [x.lower() for x in spot_sm.columns]
    spot_sm.rename(columns = {'open_interest':'openinterest'}, inplace = True)

    ## Filter the data to avoid large computation times
    print("Size of spot data: ", spot_sm.shape, flush=True) if print_info else None
    logger.info("Filtering data based on moneyness")
    logger.info(f"Spot data size: {spot_sm.shape}")
    if bulk:
        spot_sm['Moneyness'] = spot_sm.apply(lambda x: x['underlier_price'] / x['strike'], axis=1)
        spot_sm = spot_sm[(spot_sm.Moneyness <= PRICING_CONFIG["UPPER_BOUND_MONEYNESS"]) \
                    & (spot_sm.Moneyness >= PRICING_CONFIG["LOWER_BOUND_MONEYNESS"])]
        
        print("Size of spot data after filtering: ", spot_sm.shape, flush=True) if print_info else None
        logger.info(f"Spot data size after filtering: {spot_sm.shape}")
        spot_sm.drop(columns = 'Moneyness', inplace = True)

    ## Fix for missing data on intraday spot. 
    if data_request.agg == 'intra':
        spot_sm = spot_sm[~spot_sm.underlier_price.isna()]
        if spot_sm.empty:
            logger.warning("Spot data is empty, skipping save to database")
            print("Spot data is empty, skipping save to database")
            return
    
    
    ## Add the vol columns
    print("Calculating Vols", flush=True) if print_info else None
    logger.info("Calculating Vols")
    bs_vol_start = time.time()
    print("Calculating BS Vols", flush=True) if print_info else None
    calc_vol_for_data_parallel(spot_sm, 'close', 'BS_IV', 'bs', pool=pool)
    calc_vol_for_data_parallel(spot_sm, 'midpoint', 'Midpoint_BS_IV', 'bs', pool=pool)
    calc_vol_for_data_parallel(spot_sm, 'weighted_midpoint', 'Weighted_Midpoint_BS_IV', 'bs', pool=pool)
    calc_vol_for_data_parallel(spot_sm, 'closebid', 'bid_bs_iv', 'bs', pool=pool)
    calc_vol_for_data_parallel(spot_sm, 'closeask', 'ask_bs_iv', 'bs',  pool=pool)
    print(f"Time taken to calculate BS vols for {req_class}: ", time.time() - bs_vol_start, flush=True) if print_info else None
    print("\n")
    
    print("Calculating Binomial Vols", flush=True) if print_info else None
    logger.info("Calculating Binomial Vols")
    bt_vol_start = time.time()
    calc_vol_for_data_parallel(spot_sm, 'close', 'Binomial_IV','binomial', pool=pool)
    calc_vol_for_data_parallel(spot_sm, 'midpoint', 'Midpoint_binomial_IV', 'binomial', pool=pool)
    calc_vol_for_data_parallel(spot_sm, 'weighted_midpoint', 'Weighted_Midpoint_binomial_IV', 'binomial', pool=pool)
    calc_vol_for_data_parallel(spot_sm, 'closebid', 'bid_binomial_iv', 'binomial', pool=pool)
    calc_vol_for_data_parallel(spot_sm, 'closeask', 'ask_binomial_iv', 'binomial', pool=pool)
    print(f"Time taken to calculate Binomial vols for {req_class}: ", time.time() - bt_vol_start, flush=True) if print_info else None
    print("\n")
    spot_sm.columns = spot_sm.columns.str.lower()
    data_request.spot_data = spot_sm.copy()
    
    

    ## Vol Resolve before Calculating Greeks, this vol is necessary for Greeks

    if data_request.agg != 'intra':
        ## Will not be resolving vols for intra data
        print("Resolving Vols\n") if print_info else None
        resolve_start = time.time()
        logger.info("Resolving Vols")
        try:
            resolve_missing_vols_in_data(spot_sm, 
                                        [ 'midpoint_bs_iv',  'midpoint_binomial_iv'], 
                                        ['bs', 'binomial'],
                                        ['midpoint', 'midpoint'],
                                        agg = data_request.agg,)
        except Exception as e:
            vol_resolve_logger.error(f"Error resolving vols: {e}")
            data_request.spot_data = spot_sm.copy()
            data_request.error = e
            save_failed_request(data_request, 'requests_json/failed_vol_resolve.jsonl')
            print(f"Error resolving vols: {e}") if print_info else None
        print(f"Time taken to resolve vols for {req_class}: ", time.time() - resolve_start, flush=True) if print_info else None
        print("\n")

        

    ## Add the greek columns
    print("Calculating Greeks", flush=True) if print_info else None
    logger.info("Calculating Greeks")
    bs_greek_start = time.time()
    print("Calculating BS Greeks", flush=True) if print_info else None
    calc_greeks_for_data_parallel(spot_sm, 'bs', 'bs_iv', '{x}', pool=pool)
    calc_greeks_for_data_parallel(spot_sm, 'bs', 'midpoint_bs_iv', 'midpoint_{x}', pool=pool)
    calc_greeks_for_data_parallel(spot_sm, 'bs', 'weighted_midpoint_bs_iv', 'weighted_midpoint_{x}', pool=pool)
    print(f"Time taken to calculate BS Greeks for {req_class}: ", time.time() - bs_greek_start, flush=True) if print_info else None

    print("\n")
    print("Calculating Binomial Greeks", flush=True) if print_info else None
    logger.info("Calculating Binomial Greeks")
    bt_greek_start = time.time()
    calc_greeks_for_data_parallel(spot_sm, 'binomial', 'bid_binomial_iv', 'bid_binomial_{x}', pool=pool)
    calc_greeks_for_data_parallel(spot_sm, 'binomial', 'ask_binomial_iv', 'ask_binomial_{x}', pool=pool)
    calc_greeks_for_data_parallel(spot_sm, 'binomial', 'binomial_iv', 'binomial_{x}', pool=pool)
    calc_greeks_for_data_parallel(spot_sm, 'binomial', 'midpoint_binomial_iv', 'midpoint_binomial_{x}', pool=pool)
    spot_sm.columns = spot_sm.columns.str.lower()
    print(f"Time taken to calculate Binomial Greeks for {req_class}: ", time.time() - bt_greek_start, flush=True) if print_info else None
    print("\n")
    
    ## Add the dollar delta columns
    calc_dollar_delta_from_data(spot_sm, 'delta', 'dollar_delta')
    calc_dollar_delta_from_data(spot_sm, 'midpoint_delta', 'midpoint_dollar_delta')
    calc_dollar_delta_from_data(spot_sm, 'weighted_midpoint_delta', 'weighted_midpoint_dollar_delta')

    ## Add the last updated column
    spot_sm['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    ## Add the vol resolve columns
    spot_sm['midpoint_bs_vol_resolve'] = 0
    spot_sm['midpoint_binomial_vol_resolve'] = 0

    ## Finally, save the data to the database
    print("Saving data to database", flush=True) if print_info else None
    logger.info("Saving data to database")
    data_request.pre_save_to_db_data = spot_sm.copy()
    db.save_to_database(spot_sm, data_request.db_name, data_request.table_name)
    data_request.saved_to_db_data = spot_sm
    print(f"{data_request.__class__.__name__} took {time.time() - func_start} seconds to save to database", flush=True) if print_info else None
    return spot_sm

@log_error(logger)
def check_if_already_saved(request, spot_sm):
    """
    Check if the data has already been saved to the database.
    """
    # Check if the data has already been saved
    db = DatabaseAdapter()
    if isinstance(request,OptionQueryRequestParameter):
        filter_data = init_query(data_request=request, db=db, query_category='single')
        filter_data['Datetime'] = pd.to_datetime(filter_data['datetime'])
        missing_dates = check_missing_dates(filter_data, request.start_date, request.end_date)
        return spot_sm[spot_sm.index.isin(missing_dates)]
    elif isinstance(request,BulkOptionQueryRequestParameter):
        ## Get all strikes to query
        strikes = list(set(spot_sm[['k', 'Right']].itertuples(name = None, index = False)))
        request.strikes = strikes
        filter_data = init_query(data_request=request, db=db, query_category='bulk')
        available = filter_data.set_index(['optiontick', 'datetime']).index
        unavailable = spot_sm.reset_index().set_index(['OptionTick', 'Datetime']).index
        mask = ~unavailable.isin(available)

        ## Filter the data to get the missing data
        missing_data = spot_sm[mask]
        return missing_data
    else:
        raise ValueError("Invalid request type. Must be OptionQueryRequestParameter or BulkOptionQueryRequestParameter.")



@log_error(logger)
def save_chain_data(data_request, **kwargs):
    """
    Save the chain data to the database
    """
    col_kwargs = {
        'underlier_price': 'spot',
        'strike': 'strike',
        'expiration': 'expiration',
        'datetime': 'build_date',
        'rf_rate': 'r',
        'dividend': 'q',
        'put/call': 'right'
    }
    db = DatabaseAdapter()
    pool = kwargs.get('pool', get_pool_enabled())
    chain_data = data_request.post_processed_data
    calc_vol_for_data(chain_data, 'price', 'bs_vol', 'bs', col_kwargs=col_kwargs)
    binomial_col = ['price', 'spot', 'strike', 'r', 'expiration', 'right', 'build_date', 'q']
    chain_data['binomial_vol'] = parallel_apply(chain_data[binomial_col], binomial_implied_vol, timeout=10, pool=pool)

    db.save_to_database(chain_data, data_request.db_name, data_request.table_name,)



@log_error(logger)
def bulk_one_off_save(
        start: str | datetime,
        end: str | datetime,
        tick: str,
        exp: str,
        print_info: bool = False,
):
    """
    This function is used to save the data to the database without initializing the data manager
    """
    
    kwargs = {
        'start': start,
        'end': end,
        'tick': tick,
        'exp': exp,
        'print_info': print_info,
        'save_func': partial(save_to_database, print_info=print_info),
        'type_': 'bulk',
    }

    _SaveManager.schedule(kwargs) if SHOULD_SCHEDULE else _SaveManager.enqueue(kwargs)


## DataManager Procedure Functions

def enforce_interval(ivl_str: str):
        ## Enforce the interval
    try:
        ## We want to throw an error if the interval is not in the available intervals + if we get query for minute data 'm'
        PRICING_CONFIG['AVAILABLE_INTERVALS'].remove('m') ## Remove minute data from available intervals 
    except:
        pass
    

    if ivl_str.lower() not in PRICING_CONFIG["AVAILABLE_INTERVALS"] and ivl_str != 'M': ## Want to avoid minute data
        raise ValueError(f"Expected interval to be one of: {PRICING_CONFIG['AVAILABLE_INTERVALS']}")
    
    if ivl_str == 'm': ## Minute data not available
        raise AttributeError("Minute data currently unavailable, please go higher")
    
    return

def enforce_inputs(type_:str, model:str) -> None:
        ## Assert inputs
    if type_ not in ['spot', 'vol', 'vega', 'vanna', 'volga', 'delta', 'gamma', 'theta', 'rho', 'greeks', 'greek', 'attribution', 'scenario']:
        raise ValueError("Expected type_ to be one of: ['spot', 'vol', 'vega', 'vanna', 'volga', 'delta', 'gamma', 'theta', 'rho', 'greeks', 'greek', 'attribution', 'scenario']")
    if model not in ['bs', 'binomial', 'mc']: ## Only Black Scholes, binomial tree, monte carlo
        raise ValueError(f"Expected model to be one of: {PRICING_CONFIG['AVAILABLE_PRICING_MODELS']}")
    return

def determine_table_agg(ivl_str: str, type_: str, greek_names: list) -> tuple:
    ## Determine aggregation
    if ivl_str in ['h', 'm']:
        agg = 'intra'
    else:
        agg = 'eod'
    
    ## Table to query, picking based on interval & type
    if type_ in greek_names:
        database, table = TABLES[agg]['greeks'].split('.')
    else:
        database, table = TABLES[agg][type_].split('.')
    
    return agg, database, table


def determine_requested_columns(default_fill:str, type_:str, model:str, greek_names:list) -> list:
    if type_ == 'spot':
        requested_col = ['datetime', 'open', 'high', 'low', 'close', default_fill.lower(),'volume', 'openinterest']

    elif type_ == 'vol':
        requested_col = ['datetime', f"{model}_iv".lower(), f"{default_fill.lower()}_{model}_iv".lower()]

    elif type_ in greek_names:
        ## If Statement logic to format a the list of greek names to be used
        if type_ not in ['greek', 'greeks']:
            if model == 'bs':
                requested_col = ['datetime'] + [f"{default_fill}_{type_}".lower()] + [f"{type_}".lower()]
            else:
                requested_col = ['datetime'] + [f"{model}_{type_}".lower()] + [f"{default_fill.lower()}_{model}_{type_}".lower()]
        else:
            if model == 'bs':
                requested_col = ['datetime'] + [f"{x}".lower() for x in greek_names if x not in ['greek', 'greeks']] + [f"{default_fill.lower()}_{x}".lower() for x in greek_names if x not in ['greek', 'greeks']]
            else:
                requested_col = ['datetime'] + [f"{model}_{x}".lower() for x in greek_names if x not in ['greek', 'greeks']] + [f"{default_fill.lower()}_{model}_{x}".lower() for x in greek_names if x not in ['greek', 'greeks']]

    elif type_ == 'attribution':
        raise NotImplementedError("Attribution data not implemented yet")
    
    elif type_ == 'scenario':
        raise NotImplementedError("Scenario data not implemented yet")
    
    elif type_ == 'chain':
        raise IncorrectExecutionError("Chain Data does not return a timeseries, returns at_time")
    else:
        raise KeyError(f"Type {type_} not in requested columns")
    return requested_col

        
def format_raw_spot_data( **kwargs):
    """
    Adds necessary formatting. To avoid overpopulating the __handle_incomplete_data method
    """
    data_request = kwargs.get('data_request')
    raw_spot_data = data_request.raw_spot_data
    if raw_spot_data.empty:
        print("Format raw found this empty")
        data_request.raw_spot_data = pd.DataFrame()

    else:
        raw_spot_data.reset_index(inplace = True)
        raw_spot_data.columns = [x.lower() for x in raw_spot_data.columns]
        raw_spot_data = raw_spot_data[raw_spot_data.datetime.isin(data_request.missing_dates)]
        if 'index' in raw_spot_data.columns:
            raw_spot_data.drop(columns=['index'], inplace=True)
        data_request.raw_spot_data = raw_spot_data


def post_process( **kwargs):
    """
    Post process the data after all the calculations
    """

    data_request = kwargs.get('data_request')
    bulk = kwargs.get('bulk', False)
    is_complete = data_request.pre_process['is_complete']
    is_empty = data_request.pre_process['is_empty']
    raw_spot_data = data_request.raw_spot_data.copy()

    if not is_empty and is_complete:
        ## If not empty and data complete, no need formatting. Just return from db
        final_data = data_request.pre_processed_data.copy()
        data_request.post_processed_data = final_data
        return
    
    ## Start by renaming the columns to match the database.
    rename_columns = {'open_interest': 'openinterest'}
    try:
        raw_spot_data.rename(columns=rename_columns, inplace=True)
    except KeyError as e:
        pass
    
    ## Filter the columns to only the requested columns
    raw_spot_data = raw_spot_data[data_request.requested_col]

    ## Capitalize the columns & set the index to datetime
    raw_spot_data.columns = [x.capitalize() for x in raw_spot_data.columns]
    if bulk:
        raw_spot_data.set_index(['Optiontick','Datetime'], inplace=True)
    else:
        raw_spot_data.set_index('Datetime', inplace=True)
    raw_spot_data = raw_spot_data[~raw_spot_data.index.duplicated(keep='first')]
    
    if is_empty:
        ## If the data is empty, the final data is the raw data
        final_data = raw_spot_data
    elif not is_complete:
        ## Else we will have to merge the data
        final_data = pd.concat([data_request.pre_processed_data, raw_spot_data], axis=0)
    
    
    final_data = final_data[~final_data.index.duplicated(keep='first')]
    final_data.columns = [x.capitalize() for x in final_data.columns]
    final_data.sort_index(inplace=True)

    if bulk:
        ## For final data, we will filter for the Option Tick we need
        final_data = final_data[final_data.index.get_level_values('Optiontick').isin(data_request.opttick)]
    

    data_request.post_processed_data = final_data
    return data_request.post_processed_data


def format_final_data(**kwargs):
    """
    Format the final data to match the database
    """
    data_request = kwargs.get('data_request')
    bulk = kwargs.get('bulk', False)
    ## Resample the data to the requested interval

    if len(data_request.post_processed_data) > 1:
        resampled = resample( data_request.post_processed_data, 
                                                        data_request.interval, 
                                                        {col: 'ffill' for col in data_request.post_processed_data.columns})
    else:
        resampled = data_request.post_processed_data.copy()
    if bulk:
        resampled.index = resampled.index.swaplevel()
    data_request.post_processed_data = resampled


@log_error(logger)
def init_query_old(**kwargs):

    """
    Initialize the query for the data request and save the data to the request
    """
    data_request = kwargs.get('data_request')
    db = kwargs.get('db', DatabaseAdapter())
    try:
        query_category = kwargs['query_category']
    except KeyError:
        raise KeyError("Query category not specified, expected one of: ['single', 'bulk', 'chain']")
    
    if query_category == 'single':
        query = f"""SELECT *
        FROM {data_request.db_name}.{data_request.table_name}
        WHERE OPTIONTICK = '{data_request.opttick}' AND
        DATETIME >= '{data_request.start_date} 00:00:00' AND 
        DATETIME <= '{data_request.end_date} 00:00:00'
        """
        database_data = db.query_database(data_request.db_name, data_request.table_name, query)
        database_data.columns = [x.lower() for x in database_data.columns]
        data_request.query = query
        data_request.database_data = database_data
        return database_data
    
    ############# Bulk Query
    elif query_category == 'bulk':
        strikes = data_request.strikes
        opttick_list = [f"{generate_option_tick_new(data_request.symbol, right, data_request.exp, strike)}" for strike, right in strikes]
        data_request.opttick = opttick_list ## save the opttick list for future reference
        str_list = [f"'{x}'" for x in opttick_list]
        filter_str = f"({', '.join(str_list)})"
        query = f"""SELECT *
        FROM {data_request.db_name}.{data_request.table_name}
        WHERE OPTIONTICK in {filter_str} AND
        DATETIME >= '{data_request.start_date}' AND 
        DATETIME <= '{data_request.end_date}'
        """
        database_data = db.query_database(data_request.db_name, data_request.table_name, query)
        database_data.columns = [x.lower() for x in database_data.columns]
        data_request.query = query
        data_request.database_data = database_data
        return database_data
    
    ############# Chain Query
    elif query_category == 'chain':
        query = f"""SELECT *
        FROM {data_request.db_name}.{data_request.table_name}
        WHERE TICKER = '{data_request.symbol}' AND
        DATE(BUILD_DATE) = '{data_request.date}' 
        """
        
        database_data = db.query_database(data_request.db_name, data_request.table_name, query)
        database_data.columns = [x.lower() for x in database_data.columns]
        data_request.query = query
        data_request.database_data = database_data
        return database_data
    
@log_error(logger)
def init_query(**kwargs):

    """
    Initialize the query for the data request and save the data to the request
    """
    global DB_CACHE
    data_request = kwargs.get('data_request')
    db = kwargs.get('db', DatabaseAdapter())
    try:
        query_category = kwargs['query_category']
    except KeyError:
        raise KeyError("Query category not specified, expected one of: ['single', 'bulk', 'chain']")
    
    ## Return Empty DataFrame and skip to API if single & SKIP_MYSQL is True
    if query_category == 'single' and get_skip_mysql_query():
        EMPTY_TIMESEIRES_TABLE.columns = [col.lower() for col in EMPTY_TIMESEIRES_TABLE.columns]
        data_request.database_data = EMPTY_TIMESEIRES_TABLE
        return data_request
    

    ## 1) Check if it already cached
    skip_db_query = query_cache(data_request, query_category)
    

    ## 2) If all available, skip, else query needed
    if skip_db_query:
        database_data = data_request.cache_data
        database_data.columns = [x.lower() for x in database_data.columns]
        data_request.database_data = database_data
        return database_data
    
    ## 2.1) If not all available, query database for missing
    else:
        if query_category == 'single':
            opttick = data_request.query_ticks[0]
            query = f"""SELECT *
            FROM {data_request.db_name}.{data_request.table_name}
            WHERE OPTIONTICK = '{opttick}' AND
            DATETIME >= '{data_request.start_date} 00:00:00' AND 
            DATETIME <= '{data_request.end_date} 00:00:00'
            """
        
        ############# Bulk Query
        elif query_category == 'bulk':
            opttick = data_request.query_ticks
            strikes = data_request.strikes
            opttick_list = [f"{generate_option_tick_new(data_request.symbol, right, data_request.exp, strike)}" for strike, right in strikes]
            data_request.opttick = opttick_list ## save the opttick list for future reference
            str_list = [f"'{x}'" for x in opttick]
            filter_str = f"({', '.join(str_list)})"
            query = f"""SELECT *
            FROM {data_request.db_name}.{data_request.table_name}
            WHERE OPTIONTICK in {filter_str} AND
            DATETIME >= '{data_request.start_date}' AND 
            DATETIME <= '{data_request.end_date}'
            """
        
        ############# Chain Query
        elif query_category == 'chain':
            query = f"""SELECT *
            FROM {data_request.db_name}.{data_request.table_name}
            WHERE TICKER = '{data_request.symbol}' AND
            DATE(BUILD_DATE) = '{data_request.date}' 
            """ 
        
        ## 2.2) Join Cache Data & Missing queried data
        database_data = db.query_database(data_request.db_name, data_request.table_name, query)
        database_data.columns = [x.lower() for x in database_data.columns]
        data_request.query = query
        database_data = pd.concat([database_data, data_request.cache_data])
        data_request.database_data = database_data

        ## 2.3) Update Cache
        
        key = f"{data_request.db_name}.{data_request.table_name}"
        db_data = DB_CACHE.get(key, pd.DataFrame())
        db_data = pd.concat([db_data, database_data]).drop_duplicates(inplace=False)
        DB_CACHE[key] = db_data
        DB_CACHE[key] = DB_CACHE[key].drop_duplicates(inplace = False)
        return database_data

    

def add_inputs_to_raw(self, **kwargs):
    """
    Adds Inputs to raw_spot_data for Vol & other necessary uses
    """
    data_request = kwargs.get('data_request')
    bulk = kwargs.get('bulk', False)
    if not bulk:
        if not data_request.raw_spot_data.empty:
            data_request.raw_spot_data['s0'] = data_request.input_params['s0_chain']['close']
            data_request.raw_spot_data['y'] = data_request.input_params['y']
            data_request.raw_spot_data['r'] = data_request.input_params['r']
            data_request.raw_spot_data['K'] = self.strike
            data_request.raw_spot_data['exp_date'] = self.exp
            data_request.raw_spot_data['right'] = self.right
    else:
        if not data_request.raw_spot_data.empty:
            data_request.raw_spot_data['s0'] = data_request.input_params['s0_chain']['close']
            data_request.raw_spot_data['y'] = data_request.input_params['y']
            data_request.raw_spot_data['r'] = data_request.input_params['r']




##### Calculation Functions   
@log_error(logger)
def calc_vol_for_data(
        df,
        price_col,
        col_name,
        model,
        col_kwargs = None
) -> pd.DataFrame:
    """
    Adds a vol column to passed dataframe.

    Parameters:
    df: DataFrame containing following columns: `underlier_price`, `strike`, `expiration`, `datetime`, `rf_rate`, `dividend`, `put/call`
    price_col: Column to back out Implied Vol from
    model: bs or binomial
    col_name: name of added column
    col_kwargs: dictionary with keys as column names in df and values as the corresponding column names in the model
        expected keys: `underlier_price`, `strike`, `expiration`, `datetime`, `rf_rate`, `dividend`, `put/call`

    ps: This function both returns a DataFrame and modifies the passed DataFrame in place. You can use the returned DataFrame or the modified one.

    returns pd.DataFrame
    """
    enforce_allowed_models(model)
    
    if not col_kwargs:
        col_kwargs = {
            'underlier_price': 'underlier_price',
            'strike': 'strike',
            'expiration': 'expiration',
            'datetime': 'datetime',
            'rf_rate': 'rf_rate',
            'dividend': 'dividend',
            'put/call': 'put/call',
        }

    if model == 'bs':
        df[col_name] = df.apply(
        lambda x:IV_handler(S = x[col_kwargs['underlier_price']], 
                            K = x[col_kwargs['strike']], 
                            price = x[price_col], 
                            t = time_distance_helper(x[col_kwargs['expiration']], x[col_kwargs['datetime']]), 
                            r = x[col_kwargs['rf_rate']], 
                            q = x[col_kwargs['dividend']], 
                            flag = x[col_kwargs['put/call']].lower()), axis = 1
    )
        
    elif model == 'binomial':
        df[col_name] = df.apply(
            lambda x: binomial_implied_vol(price = x[price_col], 
                                           S = x[col_kwargs['underlier_price']], 
                                           K = x[col_kwargs['strike']],  
                                           r = x[col_kwargs['rf_rate']], 
                                           exp_date=x[col_kwargs['expiration']], 
                                           option_type=x[col_kwargs['put/call']].lower(), 
                                           pricing_date=x[col_kwargs['datetime']], 
                                           dividend_yield=x[col_kwargs['dividend']]), axis=1
        )
    return df

@log_error(logger)
def calc_vol_for_data_parallel(
        df,
        price_col,
        col_name,
        model,
        col_kwargs = None,
        pool = None
) -> pd.DataFrame:
    """
    Adds a vol column to passed dataframe. But parallelizes the calculation using multiprocessing or threading.

    Parameters:
    df: DataFrame containing following columns: `underlier_price`, `strike`, `expiration`, `datetime`, `rf_rate`, `dividend`, `put/call`
    price_col: Column to back out Implied Vol from
    model: bs or binomial
    col_name: name of added column
    col_kwargs: dictionary with keys as column names in df and values as the corresponding column names in the model
        expected keys: `underlier_price`, `strike`, `expiration`, `datetime`, `rf_rate`, `dividend`, `put/call`
    pool: bool, if True, use multiprocessing pool for parallel processing. Default is POOL_ENABLED. False will use single-threaded processing.

    ps: This function both returns a DataFrame and modifies the passed DataFrame in place. You can use the returned DataFrame or the modified one.

    returns pd.DataFrame
    """
    enforce_allowed_models(model)
    if pool is None:
        pool = get_pool_enabled()


    if not col_kwargs:
        col_kwargs = {
            'underlier_price': 'underlier_price',
            'strike': 'strike',
            'expiration': 'expiration',
            'datetime': 'datetime',
            'rf_rate': 'rf_rate',
            'dividend': 'dividend',
            'put/call': 'put/call',
        }
    temp_df = df.copy()
    temp_df.rename(columns=col_kwargs, inplace=True)
    temp_df['t'] = temp_df.apply(lambda x: time_distance_helper(x[col_kwargs['expiration']], x[col_kwargs['datetime']]), axis=1)
    binomial_column = [price_col, col_kwargs['underlier_price'], 
                       col_kwargs['strike'], col_kwargs['rf_rate'], col_kwargs['expiration'],
                        col_kwargs['put/call'], col_kwargs['datetime'], col_kwargs['dividend'],]
    
    bs_column = [price_col, col_kwargs['underlier_price'], col_kwargs['strike'], 't', col_kwargs['rf_rate'], col_kwargs['dividend'], col_kwargs['put/call']]
    if model == 'bs':
        df[col_name] = parallel_apply(temp_df[bs_column], IV_handler, pool = pool)
        
    elif model == 'binomial':
        df[col_name] = parallel_apply(temp_df[binomial_column], binomial_implied_vol, pool = pool)
    return df

@log_error(logger)
def calc_greeks_for_data(
        df,
        model,
        vol_col,
        greek_name_format,
        col_kwargs = None,
        greek_name = None
) -> pd.DataFrame:
    """
    Adds a greek column to passed dataframe.

    Parameters:
    df: DataFrame containing following columns: `underlier_price`, `strike`, `expiration`, `datetime`, `rf_rate`, `dividend`, `put/call`
    model: bs or binomial
    greek_name: Format of greek name. Eg "Midpoint_BS_{x}" or "Midpoint_binomial_{x}"

    ps: This function both returns a DataFrame and modifies the passed DataFrame in place. You can use the returned DataFrame or the modified one.
    returns pd.DataFrame
    """
    enforce_allowed_models(model) 
    if not col_kwargs:
        col_kwargs = {
            'underlier_price': 'underlier_price',
            'strike': 'strike',
            'expiration': 'expiration',
            'datetime': 'datetime',
            'rf_rate': 'rf_rate',
            'dividend': 'dividend',
            'put/call': 'put/call',
        } 
    
    if not greek_name:
        if model == 'bs':
            greek = df.apply(
                        lambda x:Calculate.greeks(S = x[col_kwargs['underlier_price']], 
                                                K = x[col_kwargs['strike']], 
                                                r = x[col_kwargs['rf_rate']], 
                                                sigma = x[vol_col], 
                                                start = x[col_kwargs['datetime']].strftime('%Y-%m-%d'), 
                                                flag =x[col_kwargs['put/call']].lower(),
                                                exp =   x[col_kwargs['expiration']], 
                                                y = x[col_kwargs['dividend']]), axis = 1, result_type = 'expand'
                    )
        elif model == 'binomial':
            greek = df.apply(
                        lambda x:Calculate.greeks(S = x[col_kwargs['underlier_price']], 
                                                K = x[col_kwargs['strike']], 
                                                r = x[col_kwargs['rf_rate']], 
                                                sigma = x[vol_col], 
                                                start = x[col_kwargs['datetime']].strftime('%Y-%m-%d'), 
                                                flag =x[col_kwargs['put/call']].lower(), 
                                                exp =   x[col_kwargs['expiration']], 
                                                y = x[col_kwargs['dividend']],
                                                model = model), axis = 1, result_type = 'expand'
                    )
        greek.columns = [greek_name_format.format(x=x) for x in greek.columns]
        df[greek.columns] = greek
        return df
    else:
        calc_func = getattr(Calculate, greek_name.lower())
        greek = df.apply(
                    lambda x:calc_func(S = x[col_kwargs['underlier_price']], 
                                       K = x[col_kwargs['strike']], 
                                       r = x[col_kwargs['rf_rate']], 
                                       sigma = x[vol_col], 
                                       start = x[col_kwargs['datetime']].strftime('%Y-%m-%d'), 
                                       flag =x[col_kwargs['put/call']].lower(), 
                                       exp =   x[col_kwargs['expiration']], 
                                       y = x[col_kwargs['dividend']]), axis = 1)
        
        df[greek_name_format.format(x=greek_name)] = greek
        return df

@log_error(logger)
def calc_greeks_for_data_parallel(
        df,
        model,
        vol_col,
        greek_name_format,
        col_kwargs = None,
        greek_name = None,
        pool = None
) -> pd.DataFrame:
    """
    Adds a greek column to passed dataframe. This function parallelizes the calculation using multiprocessing or threading.

    Parameters:
    df: DataFrame containing following columns: `underlier_price`, `strike`, `expiration`, `datetime`, `rf_rate`, `dividend`, `put/call`
    model: bs or binomial
    greek_name: Format of greek name. Eg "Midpoint_BS_{x}" or "Midpoint_binomial_{x}"
    pool: bool, if True, use multiprocessing pool for parallel processing. Default is POOL_ENABLED. False will use single-threaded processing.
    ps: This function both returns a DataFrame and modifies the passed DataFrame in place. You can use the returned DataFrame or the modified one.

    returns pd.DataFrame
    """
    enforce_allowed_models(model) 
    if pool is None:
        pool = get_pool_enabled()
    
    if not col_kwargs:
        col_kwargs = {
            'underlier_price': 'underlier_price',
            'strike': 'strike',
            'expiration': 'expiration',
            'datetime': 'datetime',
            'rf_rate': 'rf_rate',
            'dividend': 'dividend',
            'put/call': 'put/call',
        } 

    temp_df = df.copy()
    temp_df.rename(columns=col_kwargs, inplace=True)
    temp_df['t'] = temp_df.apply(lambda x: time_distance_helper(x[col_kwargs['expiration']], x[col_kwargs['datetime']]), axis=1)
    temp_df['asset'] = None
    temp_df['model'] = model
    greeks_colums_use = ['asset',col_kwargs['underlier_price'], col_kwargs['strike'], 
                         col_kwargs['rf_rate'], vol_col,
                         col_kwargs[ 'datetime'], col_kwargs['put/call'], col_kwargs['expiration'], col_kwargs['dividend'], 'model']
                         
    if not greek_name:
        greek = parallel_apply(temp_df[greeks_colums_use], Calculate.greeks, pool = pool)
        greek = pd.DataFrame(greek)
        greek.columns = [greek_name_format.format(x=x) for x in greek.columns]
        greek.index = temp_df.index
        df[greek.columns] = greek
        return df
    else:
        calc_func = getattr(Calculate, greek_name.lower())
        greek = parallel_apply(temp_df[greeks_colums_use], calc_func, pool = pool)
        df[greek_name_format.format(x=greek_name)] = greek
        return df

@log_error(logger)
def calc_dollar_delta_from_data(
        df,
        delta_col,
        col_name   
) -> pd.DataFrame:
    """
    Adds a Dollar Delta Column to passed dataframe

    parameters:
    df: DataFrame containing following columns: `underlier_price`, `strike`, `expiration`, `datetime`, `rf_rate`, `dividend`, `put/call`
    delta_col: Delta Column to use in multiplication
    col_name: Format for added columns. Eg "midpoint_dollar_delta"
    """

    df[col_name] = df['underlier_price'] * df[delta_col]
    return df

@log_error(logger)
def resolve_missing_vols_in_data(
        df,
        vol_resolve_list,
        model_map_list,
        price_map_list,
        agg,
):
    """
    Resolves missing vols in passed dataframe

    parameters:
    df: DataFrame containing following columns: `underlier_price`, `strike`, `expiration`, `datetime`, `rf_rate`, `dividend`, `put/call`
    vol_resolve_list: List of columns to resolve missing vols in
    model_map_list: List of models to use for resolving missing vols. Maps according to vol_resolve_list index
    price_map_list: List of columns to use for pricing. Maps according to vol_resolve_list index
    """
    for col, model, price_col in zip(vol_resolve_list, model_map_list, price_map_list):
        zero_vol = df[col] == 0
        resolved_vols = df[zero_vol].apply(
                lambda x:resolve_missing_vol(
                    underlier = x['underlier'],
                    expiration = x['expiration'],
                    strike = x['strike'],
                    put_call = x['put/call'],
                    datetime = x['datetime'],
                    S = x['underlier_price'],
                    r = x['rf_rate'],
                    q = x['dividend'],
                    pricing_model = model,
                    agg = agg,
            ), axis = 1)
        df.loc[zero_vol, col] = resolved_vols
        new_pv = df[zero_vol].apply(
            lambda x: optionPV_helper(
                spot_price = x['underlier_price'],
                strike_price = x['strike'],
                exp_date = x['expiration'],
                risk_free_rate = x['rf_rate'],
                dividend_yield = x['dividend'],
                volatility = x[col],
                putcall = x['put/call'],
                settlement_date_str= x['datetime'],
                model = model,
            ), axis = 1
        )
        df.loc[zero_vol, price_col] = new_pv

        
    return df


### DataManager Helper Functions

def handle_extra_cols(extra_cols, type_, model):
        return_cols = []
        if extra_cols:
            assert all([x in ['ask', 'bid', 'open'] for x in extra_cols]), f"Expected extra_cols to be one of: ['ask', 'bid', 'open'] received {extra_cols}"
        
        if type_ == 'spot':
              for col in extra_cols:
                if col == 'ask':
                     return_cols.append('closeask')
                elif col == 'bid':
                    return_cols.append('closebid')
                elif col == 'open':
                     return_cols.append('open')

        elif type_ == 'vol':
            for col in extra_cols:
                if col == 'ask':
                     return_cols.append(f'ask_{model}_iv')
                elif col == 'bid':
                    return_cols.append(f'bid_{model}_iv')
                elif col == 'open':
                    logger.critical("Open not implemented for vol data")
                    return []
                
        elif type_ in ['greeks', 'greek']:
            if model == 'bs':
                  logger.critical("Extra Cols not implemented for BS Greeks")
                  return []
            
            elif model == 'binomial':
                 for col in extra_cols:
                    if col == 'ask':
                        return_cols.extend([f'ask_{model}_{x}' for x in PRICING_CONFIG['AVAILABLE_GREEKS']])
                    elif col == 'bid':
                        return_cols.extend([f'bid_{model}_{x}' for x in PRICING_CONFIG['AVAILABLE_GREEKS']])
                    elif col == 'open':
                        logger.critical("Open not implemented for greek data")
                        return []
                        
        elif type_ in PRICING_CONFIG['AVAILABLE_GREEKS']:
            if model == 'bs':
                logger.critical("Extra Cols not implemented for BS Greeks")
                return []
            for col in extra_cols: ## Only binomial will have extra cols
                if col == 'ask':
                    return_cols.append(f'ask_{model}_{type_}')
                elif col == 'bid':
                    return_cols.append(f'bid_{model}_{type_}')
                elif col == 'open':
                    logger.critical("Open not implemented for greek data")
                    return []
             
        return return_cols

def build_name_format(type_, model, extra_cols, default_fill):
    """
    Build the name format for the columns
    """
    name_format = {}

    if type_ == 'vol':
        if model == 'bs':
            name_format['close'] = 'bs_iv'
            name_format[f"{default_fill}"] = f"{default_fill}_bs_iv"

            ## Handle extra columns
            for col in extra_cols:
                if col.lower() in ['open']:
                    continue
                name_format[f"close{col}"] = handle_extra_cols([col], type_, model)[0]


        elif model == 'binomial':
            name_format['close'] = 'binomial_iv'
            name_format[f"{default_fill}"] = f"{default_fill}_binomial_iv"
            for col in extra_cols:
                if col.lower() in ['open']:
                    continue
                name_format[f"close{col}"] = handle_extra_cols([col], type_, model)[0]
    
    elif type_ in ['greek', 'greeks']:
        if model == 'bs':
            name_format['bs_iv'] = '{x}'
            name_format[f"{default_fill}_bs_iv"] = f'{default_fill}_'+'{x}'
            if extra_cols:
                pass ## Figure out how to handle extra cols
            
        elif model == 'binomial':
            name_format['binomial_iv'] = 'binomial_{x}'
            name_format[f"{default_fill}_binomial_iv"] = f'{default_fill}_binomial_'+'{x}'
            for col in extra_cols:
                name_format[f"{col}_binomial_iv"] = f"{col}_{model}_" +"{x}"
    
    elif type_ in PRICING_CONFIG['AVAILABLE_GREEKS']:
        if model == 'bs':
            name_format['bs_iv'] = '{x}'
            name_format[f"{default_fill}_bs_iv"] = f'{default_fill}_'+'{x}'
        elif model == 'binomial':
            name_format['binomial_iv'] = 'binomial_{x}'
            name_format[f"{default_fill}_binomial_iv"] = f'{default_fill}_binomial_'+'{x}'
            for col in extra_cols:
                name_format[f"{col}_binomial_iv"] = f"{col}_{model}_" +"{x}"


    return name_format


## CACHE FUNCTIONS
def query_cache(data_request, query_category):
    """
    Check if the data is available in the cache.

    Args:
        data_request (object): The data request object containing the parameters.
        query_category (str): The category of the query ('single', 'bulk', 'chain').
    Returns:
        bool: True if the data is available in the cache, False otherwise.

    Refer to:
        data_request.cache_data: The data retrieved from the cache.
        data_request.query_ticks: The ticks that are missing from the cache.
    """
    # 1) Get data from cache.
    cache_data = _check_cache(data_request, query_category)

    # 2) Check if it is complete or empty
    if query_category in ['single', 'bulk']:
        data_request.query_ticks, skip_db_query = timeseries_cache_validation(data_request, query_category)

    else:
        skip_db_query = not cache_data.empty
    
    data_request.cache_data = cache_data
    return skip_db_query

def _check_cache(data_request, query_category):
    """
    Check if the data is available in the cache.

    Args:
        data_request (object): The data request object containing the parameters.
        query_category (str): The category of the query ('single', 'bulk', 'chain').
    Returns:
        bool: True if the data is available in the cache, False otherwise.

    Refer to:
        data_request.cache_data: The data retrieved from the cache.
        data_request.query_ticks: The ticks that are missing from the cache.
    """
    global DB_CACHE
    tick = data_request.symbol

    if query_category in ['single', 'bulk']:
        exp = data_request.exp
        start, end = pd.to_datetime(data_request.start_date), pd.to_datetime(data_request.end_date)
    else:
        date = pd.to_datetime(data_request.date).date()

    # 1) What Table are we using?
    key = f"{data_request.db_name}.{data_request.table_name}"
    
    # 2) Handle Filtering based on query_category
    if query_category == 'single':
        opttick = [data_request.opttick]
        data_request.cache_opttick = opttick

    elif query_category == 'bulk':
        strikes_right = data_request.strikes
        opttick = [generate_option_tick_new(tick, x[1], exp, x[0]) for x in strikes_right]
        data_request.cache_opttick = opttick
        data_request.opttick = opttick

    elif query_category == 'chain':
        pass
    
    else: raise ValueError("Unknown query_category. Recieved {query_category}".format(query_category))

    # 3) Get Data from db cache
    cache_data = DB_CACHE.get(key)
    if cache_data is None or cache_data.empty:
        return DB_CACHE.setdefault(key, pd.DataFrame())
    # 3.1) Filter based on type
    else:
        if query_category in ['single', 'bulk']:
            return cache_data[(cache_data.optiontick.isin(opttick)) & \
                              (cache_data.datetime >= start ) & \
                              (cache_data.datetime >= start )]
        else:
            return cache_data[(cache_data.ticker == tick) & \
                              (pd.DatetimeIndex(cache_data.build_date).date == date)]


def timeseries_cache_validation(data_request, query_category):
    """
    Validate the cache data for timeseries requests.
    Args:
        data_request (object): The data request object containing the parameters.
        query_category (str): The category of the query ('single', 'bulk', 'chain').
    Returns:
        tuple: A tuple containing the missing option ticks and a boolean indicating whether to skip the database query.
    """
    

    bulk_cache_data = _check_cache(data_request, query_category).copy()
    if bulk_cache_data.empty:
        return data_request.cache_opttick, False
    bulk_cache_data['Datetime'] = bulk_cache_data.datetime
    has_all_ticks = all(x in bulk_cache_data.optiontick.unique() for x in data_request.cache_opttick)
    missing_opttick = [x for x in data_request.cache_opttick if x not in bulk_cache_data.optiontick.unique()]
    bool_series = bulk_cache_data.groupby('optiontick').apply(check_all_days_available, _start = data_request.start_date, _end = data_request.end_date)
    is_available_option_tick_complete = bool_series.all()
    missing_opttick = missing_opttick + bool_series[bool_series==False].index.to_list()
    is_empty = bulk_cache_data.empty
    skip_db_query = has_all_ticks and is_available_option_tick_complete and not is_empty

    return missing_opttick, skip_db_query
