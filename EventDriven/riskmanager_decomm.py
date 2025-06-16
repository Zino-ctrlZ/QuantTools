import os, sys
from trade.assets.Stock import Stock
from trade.assets.Option import Option
from trade.assets.OptionStructure import OptionStructure
from trade.assets.Calculate import Calculate
from trade.assets.rates import get_risk_free_rate_helper
from trade.assets.helpers.DataManagers import OptionDataManager
from trade.helpers.Context import Context, clear_context
from trade.helpers.helper import (change_to_last_busday, 
                                  is_USholiday, 
                                  is_busday, 
                                  setup_logger, 
                                  generate_option_tick_new, 
                                  get_option_specifics_from_key,
                                  parse_option_tick,
                                  binomial_implied_vol)
from dbase.DataAPI.ThetaData import (list_contracts, 
                                     retrieve_openInterest, 
                                     retrieve_eod_ohlc, 
                                     retrieve_bulk_eod,
                                     retrieve_bulk_open_interest
                                     )
from pandas.tseries.offsets import BDay
from pandas.tseries.holiday import USFederalHolidayCalendar
from trade.helpers.decorators import log_error_with_stack, log_time
from itertools import product
import pandas as pd
from copy import deepcopy
from trade.helpers.Logging import setup_logger
from trade.helpers.decorators import log_error_with_stack
from pathos.multiprocessing import ProcessingPool as Pool
from trade.helpers.threads import runThreads
from trade.helpers.types import OptionModelAttributes
from EventDriven.types import ResultsEnum
import numpy as np
import time
from datetime import datetime
import math
from EventDriven.event import FillEvent
from EventDriven.helpers import parse_signal_id
from EventDriven.data import DataHandler
from EventDriven.eventScheduler import EventScheduler
from EventDriven.types import OpenPositionAction, ResultsEnum
from threading import Thread, Lock
from trade import POOL_ENABLED
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
from trade import HOLIDAY_SET

## To-Do:
## 1. Filter out contracts that have already been queried. Saves time
## 2. Move cache to class attribute.


logger = setup_logger('QuantTools.EventDriven.riskmanager')
time_logger = setup_logger('QuantTools.EventDriven.riskmanager.time')
LOOKBACKS = {}

def _retrieve_openInterest(*args, **kwargs):
    try:
        return retrieve_openInterest(*args, **kwargs)
    except Exception as e:
        return None



# Precompute BDay lookbacks to eliminate redundant calculations
def precompute_lookbacks(start_date, end_date, _range = [10, 20, 30]):

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
def is_holiday(date):
    return date in HOLIDAY_SET

chain_cache = {}
close_cache = {}
oi_cache = {}
spot_cache = {}
order_cache = {}


def clear_cache():
    """
    clears the cache
    """
    global chain_cache, close_cache, oi_cache, spot_cache, order_cache
    chain_cache = {}
    close_cache = {}
    oi_cache = {}
    spot_cache = {}
    order_cache = {}



def check_all_days_available(x, _start, _end):
    # print(x)
    date_range = bus_range(_start, _start, freq = '1B')
    dates_available = x.Datetime
    missing_dates_second_check = [x for x in date_range if x not in pd.DatetimeIndex(dates_available)]
    return all(x in pd.DatetimeIndex(dates_available) for x in date_range)

def update_caches(x):
    global oi_cache, close_cache, oi_cache, spot_cache
    key = f"{x.Optiontick.unique()[0]}"
    x = x.set_index('Datetime')
    close_cache[key] = x
    oi_cache[key] = x['Openinterest'].to_frame(name = 'Open_interest')
    pass


def update_cache_with_missing_ticks(query_ticks, data_request):
    global oi_cache, close_cache, oi_cache, spot_cache
    parsed_opts = pd.DataFrame([parse_option_tick(x) for x in query_ticks])
    parsed_opts[['start_date', 'end_date']] = data_request.start_date, data_request.end_date
    OrderedList = parsed_opts[['ticker', 'end_date', 'exp_date', 'put_call', 'start_date', 'strike', ]].T.to_numpy()
    tickOrderedList = parsed_opts[['ticker', 'put_call', 'exp_date', 'strike', ]].T.to_numpy()
    eod_results = (runThreads(retrieve_eod_ohlc, OrderedList, 'map', block=False))
    oi_results = (runThreads(_retrieve_openInterest, OrderedList, 'map' , block=False))
    tick_results = (runThreads(generate_option_tick_new, tickOrderedList, 'map' , block=False))

    eod_results = list(eod_results)
    oi_results = list(oi_results)
    tick_results = list(tick_results)

    for oi, eod, tick in zip(oi_results, eod_results, tick_results):
        cache_key = f"{tick}"
        eod.index = default_timestamp(eod.index)
        close_cache[cache_key] = eod
        oi_cache[cache_key] = oi

    return





def assemble_bulk_data_request(self, start: str | datetime, 
                       end: str | datetime,
                       interval: str = '1d',
                       type_: str = 'spot',
                       strikes_right: List[Tuple] = [],
                       model: str = 'bs',
                       extra_cols: list = []) :
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



        
        eod_results = (runThreads(retrieve_eod_ohlc, OrderedList, 'map'))
        oi_results = (runThreads(_retrieve_openInterest, OrderedList, 'map'))
        tick_results = (runThreads(generate_option_tick_new, tickOrderedList, 'map'))

        ## Save to Dictionary Cache
        print("Updating Cache")
        for tick, eod, oi in zip(tick_results, eod_results, oi_results):
            cache_key = f"{tick}"
            close_cache[cache_key] = eod
            oi_cache[cache_key] = oi
        
        spot_results = runThreads(return_closePrice, [tick_results, [target_date]*len(tick_results)], 'map')
        for tick, spot in zip(tick_results, spot_results):
            cache_key = f"{tick}_{target_date}"
            spot_cache[cache_key] = spot


@log_error_with_stack(logger)
def populate_cache_v2(
        start,
        end,
        candidates,
        target_date,
):
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
    
    print(f"Looks like our young fellow is targetting: {target_date}")
    global oi_cache, close_cache, oi_cache, spot_cache
    start, end = pd.to_datetime(start), pd.to_datetime(end)
    full_data = pd.DataFrame()
    for direction in candidates:
        for data in candidates[direction]:
            if isinstance(data, str) and data =='theta_data_error':
                return 'theta_data_error'
            if pd.to_datetime(target_date).weekday() >= 5:
                return 'weekend'
            full_data = pd.concat([full_data, data], axis=0)

    full_data.index.name = 'Date'
    full_data.columns.name = ''
    full_data['start_date'] = start
    full_data['end_date'] = end
    full_data.reset_index(inplace=True)
    tick = full_data.ticker.unique()[0]
    exp = full_data.Expiration.unique()[0]
    strikes_right = list(full_data[['Strike', 'Right']].itertuples(name=None, index=False))

    ## Let's start with getting the requested data from database
    manager = BulkOptionDataManager(symbol=tick, exp=exp)
    print(f"Generting Data for {manager.symbol} {manager.exp}")
    data_request = assemble_bulk_data_request(
        self = manager,
         start = start,
        end = end,
        type_ = 'spot',
        strikes_right= strikes_right

    )
    # ## Second: we query our database to see what data we have
    init_query(data_request = data_request, query_category = 'bulk')

    # ## Third: we pre_process the data request to see if it is complete
    BulkOptionDataManager.pre_process_data(data_request = data_request) 
    # BulkOptionDataManager.one_off_save(
    #     start=start,
    #     end=end,
    #     tick=tick,
    #     exp=exp
    # ) ## We shouldn't keep going to thetadata, that takes time. Submit a process. Don't worry it runs on a new process.
    ## Wouldn't affect current procedures


    is_complete = data_request.pre_process['is_complete']
    pre_processed_data = data_request.pre_processed_data.reset_index()
    opttick = data_request.opttick
    print(f"Data Is_complete bool: {is_complete}")

    ## If complete, Fantastic! We re done, now update cache and get out
    if is_complete:
        pre_processed_data.groupby('Optiontick').apply(update_caches)

    ## If NOT complete, do not fret. We'll simply run our process for incomplete/missing ticks
    else:
        ## We first check for the requested ticks. Which one is not in database at all?
        missing_opttick = [x for x in data_request.opttick if x not in pre_processed_data.Optiontick.unique()]
        
        ## Next we check to see if the requested opttick data is COMPELETE. 
        ## If incomplete, we perform runthreads
        check_partial = partial(check_all_days_available, _start = data_request.start_date, _end = data_request.end_date)
        opttick_complete = pre_processed_data.groupby('Optiontick').apply(check_partial)
        query_ticks = opttick_complete[opttick_complete==False].index.tolist() + missing_opttick

        ## Before we perform run Threads, it is important we update cache with the Optticks that are COMPLETE
        available = opttick_complete[opttick_complete==True].index
        pre_processed_data[pre_processed_data.Optiontick.isin(available)].groupby('Optiontick').apply(update_caches)

        ## Now my dear friends, we update cache of unavailable ticks
        update_cache_with_missing_ticks(query_ticks=query_ticks, data_request=data_request)
        print("I'm proud of you, we are finally done")

    print("Actually! We are not done yet. We need to get the spot prices for the requested date")
    
    spot_results = runThreads(return_closePrice, [opttick, [target_date]*len(opttick)], 'map')
    for tick, spot in zip(opttick, spot_results):
        cache_key = f"{tick}_{target_date}"
        spot_cache[cache_key] = spot

    print("Now, my dear friend, we are done")
    return
    

def populate_cache(start_date, end_date, order_candidates, target_date, version = 2):

    if version == 1:
        print("Using V1")
        return populate_cache_v1(start_date, end_date, order_candidates, target_date)
    elif version == 2:
        print("Using V2")
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
    global close_cache, spot_cache
    cache_key = f"{id}"  ## Close Uses only the id, not the date
    close_data = close_cache[cache_key]
    if close_data is None:
        return None
    close_data = close_data[~close_data.index.duplicated(keep = 'first')]
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
            
            if float(moneyness_width) == 0.0:
                option_details = Option_Chain_Filtered.sort_values('moneyness_spread', ascending=False).head(1)
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
    oi_data = oi_data[oi_data.Datetime <= date]
    oi_data = oi_data.iloc[-lookback:] ## Get the last lookback days
    

    if isinstance(oi_data, pd.DataFrame):
        if oi_data.empty:
            return False
        
    elif oi_data is None:
        return False
    
    oi_data = oi_data[~oi_data.index.duplicated(keep = 'first')]
    oi_data = oi_data.iloc[:lookback]
    # print(f'Open Interest > {pass_threshold} for {id}:', oi_data.Open_interest.mean() )
    # return oi_data.Open_interest.mean() > pass_threshold if isinstance(oi_data, pd.DataFrame) else False
    return oi_data.Open_interest.sum()/lookback > pass_threshold if isinstance(oi_data, pd.DataFrame) else False




class OrderPicker:
    def __init__(self, 
                 start_date: str|datetime,
                 end_date: str|datetime,
                 liquidity_threshold: int = 250, 
                 data_availability_threshold: float = 0.7, 
                 lookback: int = 30):
        """
        initializes the OrderPicker class
        
        params:
        liquidity_threshold: int: liquidity threshold. Default is 250
        data_availability_threshold: float: data availability threshold. Default is 0.7
        lookback: int: lookback. Default is 30
        """
        self.liquidity_threshold = liquidity_threshold
        self.data_availability_threshold = data_availability_threshold
        self.__lookback = lookback
        self.start_date = start_date
        self.end_date = end_date
        
    @property
    def lookback(self):
        return self.__lookback
    
    @lookback.setter
    def lookback(self, value):
        global LOOKBACKS
        initial_lookback_key = list(LOOKBACKS.keys())[0]
        if value not in LOOKBACKS[initial_lookback_key].keys():
            precompute_lookbacks('2000-01-01', '2030-12-31', _range = [value])
        self.__lookback = value

        
    @log_error_with_stack(logger)
    def get_order(self, 
                  tick: str, 
                  date: str,
                  right: str, 
                  max_close: str,
                  order_settings: dict) -> dict:
        
        """
        returns the order for the given tick, date, right, max_close, and order_settings

        params:
        tick: str: ticker to get the order for
        date: str: date to get the order for
        right: str: right of the option contract (P or C)
        max_close: str: maximum close price
        order_settings: dict: settings for the order
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

        returns:
        dict: order
        """
        global order_cache
        order_cache.setdefault(date, {})
        order_cache[date].setdefault(tick, {})

        ## Create necessary data structures
        direction_index = {}
        str_direction_index = {}
        for indx, v in enumerate(order_settings['specifics']):
            if v['direction'] == 'long':
                str_direction_index[indx] = 'long'
                direction_index[indx] = 1
            elif v['direction'] == 'short':
                str_direction_index[indx] = 'short'
                direction_index[indx] = -1


        order_candidates = produce_order_candidates(order_settings, tick, date, right)
        if any([x2 is None for x in order_candidates.values() for x2 in x]):
            return_item = {
                'result': "MONEYNESS_TOO_TIGHT",
                'data': None
            } 
            order_cache[date][tick] = return_item
            return return_item


        returned = populate_cache(order_candidates, target_date=date, start_date=self.start_date, end_date=self.end_date) 

        if returned == 'holiday':
            return_item = {
                'result': "IS_HOLIDAY",
                'data': None
            }
            order_cache[date][tick] = return_item
            return return_item
        
        elif returned == 'theta_data_error':

            return_item = {
                'result': "UNAVAILABLE_CONTRACT",
                'data': None
            }
            order_cache[date][tick] = return_item
            return return_item
        
        elif returned == 'weekend':
            return_item = {
                'result': "IS_WEEKEND",
                'data': None
            }
            order_cache[date][tick] = return_item
            return return_item
    

        for direction in order_candidates: ## Fix this to use .items()
            for i,data in enumerate(order_candidates[direction]):
                data['liquidity_check'] = data.option_id.apply(lambda x: liquidity_check(x, date, pass_threshold=self.liquidity_threshold, lookback=self.lookback))
                data = data[data.liquidity_check == True]
                if data.empty:
                    return_item = {
                        'result': "TOO_ILLIQUID",
                        'data': None
                    }
                    order_cache[date][tick] = return_item
                    return return_item
                
                data['available_close_check'] = data.option_id.apply(lambda x: available_close_check(x, date, threshold=self.data_availability_threshold))
                data = data[data.available_close_check == True] ## Filter out contracts that do not have close data.
                if data.empty:
                    return_item = {
                        'result': "NO_TRADED_CLOSE",
                        'data': None
                    }
                    order_cache[date][tick] = return_item
                    return return_item
                
                # print("After Available Close Check")
                # print(data)
                order_candidates[direction][i] = data



        ## Filter Unique Combinations per leg.
        unique_ids = {'long': [], 'short': []}
        for direction in order_candidates:
            for i,data in enumerate(order_candidates[direction]):
                unique_ids[direction].append(data[(data.liquidity_check == True) & (data.available_close_check == True)].option_id.unique().tolist())

        ## Produce Tradeable Combinations
        tradeable_ids = list(product(*unique_ids['long'], *unique_ids['short']))
        tradeable_ids, unique_ids 

        ## Keep only unique combinations. Not repeating a contract.
        filtered = [t for t in tradeable_ids if len(set(t)) == len(t)]


        ## Get the price of the structure
        ## Using List Comprehension to sum the prices of the structure per index
        results = [
            (*items, sum([direction_index[i] * spot_cache[f'{item}_{date}'] for i, item in enumerate(items)])) for items in filtered
        ]

        ## Convert to DataFrame, and sort by the price of the structure.
        return_dataframe = pd.DataFrame(results)
        if return_dataframe.empty:
            return_item = {
                'result': ResultsEnum.MONEYNESS_TOO_TIGHT.value,
                'data': None
            }
            order_cache[date][tick] = return_item

            return return_item
        cols = return_dataframe.columns.tolist()
        cols[-1] = 'close'
        return_dataframe.columns= cols
        # print(return_dataframe)
        return_dataframe = return_dataframe[(return_dataframe.close<= max_close) & (return_dataframe.close> 0)].sort_values('close', ascending = False).head(1) ## Implement for shorts. Filtering automatically removes shorts.


        if return_dataframe.empty:
            return_item = {
                'result': ResultsEnum.MAX_PRICE_TOO_LOW.value,
                'data': None
            }
            order_cache[date][tick] = return_item
            return return_item
            
        ## Rename the columns to the direction names
        return_dataframe.columns = list(str_direction_index.values()) + ['close']
        return_order = return_dataframe[list(str_direction_index.values())].to_dict(orient = 'list')
        return_order

        ## Create the trade_id with the direction and the id of the contract.
        id = ''
        for k, v in return_order.items():
            if len(v) > 0:
                id += f"&{k[0].upper()}:{v[0]}"
        return_order['trade_id'] = id
        return_order['close'] = return_dataframe.close.values[0]
        
        return_dict = {
            'result': ResultsEnum.SUCCESSFUL.value,
            'data': return_order
        }
        order_cache[date][tick] = return_dict

        return return_dict

class RiskManager:
    def __init__(self,
                 bars: DataHandler,
                 events: EventScheduler,
                 initial_capital: int|float,
                 start_date: str|datetime,
                 end_date: str|datetime,
                 portfolio_manager: 'Portfolio' = None,
                 price_on = 'close',
                 option_price = 'Midpoint',
                 sizing_type = 'delta',
                 leverage = 5.0,
                 max_moneyness = 1.2,
                 ):
        
        """
        initializes the RiskManager class

        params:
        bars: Bars: bars
        events: Events: events
        initial_capital: float: initial capital
        start_date: str: start date, recommended to match with the start date of the bars
        end_date: str: end date, recommended to match with the end date of the bars
        portfolio_manager: PortfolioManager: portfolio manager. Default is None
        price_on: str: price on. Default is 'mkt_close'
        option_price: str: option price. The Option Price used for pricing. Default is 'Midpoint'. Available Options are 'Midpoint', 'Bid', 'Ask', 'Close', 'Weighted Midpoint'
        sizing_type: str: sizing type. This is what you want your quantity to be calculated on. Default is 'delta'. Available Options are 'delta', 'vega', 'gamma', 'price'
        leverage: float: Multiplier for Equity Equivalent Size. Default is 5.0. Eg (Cash Available/Spot Price) * Leverage = Equity Equivalent Size
        max_moneyness: float: Maximum Moneyness before rolling. Default is 1.2

        Other Attributes:


        """
        
        assert sizing_type in ['delta', 'vega', 'gamma', 'price'], f"Sizing Type {sizing_type} not recognized, expected 'delta', 'vega', 'gamma', or 'price'"
        self.bars = bars
        self.events = events
        self.initial_capital = initial_capital
        self.__pm = portfolio_manager
        self.start_date = start_date
        self.end_date = end_date
        self.symbol_list = self.bars.symbol_list
        self.OrderPicker = OrderPicker(start_date, end_date)
        self.spot_timeseries = {}
        self.chain_spot_timeseries = {} ## This is used for pricing, to account option strikes for splits
        self.processed_option_data = {}
        self.position_data = {}
        self.dividend_timeseries = {}
        self.sizing_type = sizing_type
        self.sizing_lev = leverage
        self.limits = {
            'delta': True,
            'gamma': False,
            'vega': False,
            'theta': False,
            'dte': False,
            'moneyness': False
        }
        self.greek_limits = {
            'delta': {},
            'gamma': {},
            'vega': {},
            'theta': {}
        }
        self.data_managers = {}

        ## Might want to make this changeable in future
        self.rf_timeseries = get_risk_free_rate_helper()['annualized']
        self.price_on = price_on
        self.max_moneyness = max_moneyness
        self.option_price = option_price


    @property 
    def option_data(self):
        global close_cache
        return close_cache  
    
    @property
    def pm(self):
        return self.__pm
    
    @pm.setter
    def pm(self, value):
        self.__pm = value


    def print_settings(self):
        msg = f"""
Risk Manager Settings:
Start Date: {self.start_date}
End Date: {self.end_date}
Current Limits State (Position Adjusted when these thresholds are reached):
    Delta: {self.limits['delta']}
    Gamma: {self.limits['gamma']}
    Vega: {self.limits['vega']}
    Theta: {self.limits['theta']}
    Roll On DTE: {self.limits['dte']}
        Min DTE Threshold: {self.pm.min_acceptable_dte_threshold}
    Roll On Moneyness: {self.limits['moneyness']}
        Max Moneyness: {self.max_moneyness}
Quanitity Sizing Type: {self.sizing_type}
            """
        print(msg)
        

    def get_order(self, *args, **kwargs):
        signalID = kwargs.pop('signal_id')
        print(f"Signal ID: {signalID}")

        ## I cannot calculate greeks here. I need option_data to be available first.
        order = self.OrderPicker.get_order(*args, **kwargs)     
        print(f"Order Produced: {order}")

        if order['result'] == ResultsEnum.SUCCESSFUL.value:
            position_id = order['data']['trade_id']
        else:
            return order
        
        print(f"Position ID: {position_id}")
        print("Calculating Position Greeks")
        self.calculate_position_greeks(position_id, kwargs['date'])
        print('Updating Signal Limits')
        self.update_greek_limits(signalID, position_id)
        print("Calculating Quantity")
        quantity = self.calculate_quantity(position_id, signalID, kwargs['date'])
        print(f"Quantity for Position ({position_id}): {quantity}")
        order['data']['quantity'] = quantity
        print(order)
        return order


        

    @log_time(time_logger)
    def calculate_position_greeks(self, positionID, date):
        """
        Calculate the greeks of a position

        date: Evaluation Date for the greeks (PS: This is not the pricing date)
        positionID: str: position string. (PS: This function assumes ticker for position is the same)
        """

        ## Initialize the Long and Short Lists
        long = []
        short = []
        threads = []
        thread_input_list = [
            [], [], [], [], [], []
        ]

        date = pd.to_datetime(date) ## Ensure date is in datetime format
        
        ## First get position info
        position_dict, positon_meta = self.parse_position_id(positionID)

        ## Now ensure that the spot and dividend data is available
        for p in position_dict.values():
            for s in p:
                self.generate_data(s['ticker'])
        ticker = s['ticker']

        ## Get the spot, risk free rate, and dividend yield for the date
        s = self.chain_spot_timeseries[ticker]
        s0_close = self.spot_timeseries[ticker]
        r = self.rf_timeseries
        y = self.dividend_timeseries[ticker]

        @log_time(time_logger)
        def get_timeseries(data_manager, s, r, y, s0_close, direction):
            greeks = data_manager.get_timeseries(start = self.start_date,
                                                    end = self.end_date,
                                                    interval = '1d',
                                                    type_ = 'greeks',)
            greeks_cols = [x for x in greeks.columns if 'Midpoint' in x]
            greeks = greeks[greeks_cols]
            greeks.columns = [x.split('_')[1].capitalize() for x in greeks.columns]

            spot = data_manager.get_timeseries(start = self.start_date,
                                                end = self.end_date,
                                                interval = '1d',
                                                type_ = 'spot',)
            spot = spot[[self.option_price.capitalize()]]
            # data[['symbol', 'put_call', 'exp_date', 'strike']] = data_manager.symbol, data_manager.put_call, data_manager.exp_date, data_manager.strike
            data = greeks.join(spot)
            data['s'] = s
            data['r'] = r
            data['y'] = y
            data['s0_close'] = s0_close
            self.processed_option_data[data_manager.opttick] = data
            if direction == 'L':
                long.append(data)
            elif direction == 'S':
                short.append(data)
            else:
                raise ValueError(f"Position Type {_set[0]} not recognized")
            
            return data
        
        ## Calculating IVs & Greeks for the options
        for _set in positon_meta:
            # To-do: Thread thisto speed up the process
            print(_set)
            id = _set[1]
            data_manager = OptionDataManager(opttick = id)
            for input, list_ in zip([data_manager, s, r, y, s0_close, _set[0]], thread_input_list):
                list_.append(input)
        
        runThreads(get_timeseries, thread_input_list)
            
        position_data = sum(long) - sum(short)
        position_data = position_data[~position_data.index.duplicated(keep = 'first')]
        self.position_data[positionID] = position_data
        self.position_data[positionID].columns = [x.capitalize() for x in self.position_data[positionID].columns]
        ## Retain the spot, risk free rate, and dividend yield for the position, after the greeks have been calculated & spread values subtracted
        self.position_data[positionID]['s0_close'] = s0_close
        self.position_data[positionID]['s'] = s
        self.position_data[positionID]['r'] = r
        self.position_data[positionID]['y'] = y

    @log_time(time_logger)
    def update_greek_limits(self, signal_id, position_id):
        """
        Updates the limits associated with a signal
        ps: This should only be updated on first purchase of the signal
            Limits are saved in absolute values to account for both long and short positions
        
        """

        ## We want to update delta limits for now.
        ## This should be based on the SignalID.
        ## I will use The date from Signal ID To create the limit
        ## Goal is to enfore the limit on the signal, not the position

        id_details = parse_signal_id(signal_id)
        cash_available = self.pm.allocated_cash_map[id_details['ticker']]
        delta_at_purchase = self.position_data[position_id]['Delta'][id_details['date']] 
        s0_at_purchase = self.position_data[position_id]['s0_close'][id_details['date']]
        equivalent_delta_size = (math.floor(cash_available/s0_at_purchase)/100) * self.sizing_lev
        self.greek_limits['delta'][signal_id] = abs(equivalent_delta_size)
        print(f"Spot Price at Purchase: {s0_at_purchase} at time {id_details['date']}")
        print(f"Delta at Purchase: {delta_at_purchase}")
        print(f"Equivalent Delta Size: {equivalent_delta_size}, with Cash Available: {cash_available}, and Leverage: {self.sizing_lev}")
        print(f"Equivalent Delta Size: {equivalent_delta_size}")

    def calculate_quantity(self, positionID, signalID, date) -> int:
        """
        Returns the quantity of the position that can be bought based on the sizing type
        """

        if positionID not in self.position_data: ## If the position data isn't available, calculate the greeks
            self.calculate_position_greeks(positionID, date)
        
        ## First get position info and ticker
        position_dict, _ = self.parse_position_id(positionID)
        key = list(position_dict.keys())[0]
        ticker = position_dict[key][0]['ticker']

        ## Now calculate the max size cash can buy
        cash_available = self.pm.allocated_cash_map[ticker]
        purchase_date = pd.to_datetime(date)
        s0_at_purchase = self.position_data[positionID]['s0_close'][purchase_date]
        print(f"Spot Price at Purchase: {s0_at_purchase} at time {purchase_date}")
        opt_price = self.position_data[positionID]['Midpoint'][purchase_date]
        max_size_cash_can_buy = abs(math.floor(cash_available/(opt_price*100))) ## Assuming Allocated Cash map is already in 100s

        if self.sizing_type == 'price':
            return max_size_cash_can_buy
          
        elif self.sizing_type.capitalize() == 'Delta':
            delta = self.position_data[positionID]['Delta'][purchase_date]
            if signalID not in self.greek_limits['delta']:
                self.update_greek_limits(signalID,positionID )
            target_delta = self.greek_limits['delta'][signalID]
            print(f"Target Delta: {target_delta}")
            delta_size = (math.floor(target_delta/abs(delta)))
            print(f"Delta from Full Cash Spend: {max_size_cash_can_buy * delta}, Size: {max_size_cash_can_buy}")
            print(f"Delta with Size Limit: {delta_size * delta}, Size: {delta_size}")
            return delta_size if abs(delta_size) <= abs(max_size_cash_can_buy) else max_size_cash_can_buy
        
        elif self.sizing_type.capitalize() in ['Gamma', 'Vega']:
            raise NotImplementedError(f"Sizing Type {self.sizing_type} not yet implemented, please use 'delta' or 'price'")
        
        else:
            raise ValueError(f"Sizing Type {self.sizing_type} not recognized")
        
    def analyze_position(self):
        """
        Analyze the current positions and determine if any need to be rolled, closed, or adjusted
        """
        
        action_dict = {}
        date = pd.to_datetime(self.pm.events.current_date)
        print(f"Analyzing Positions on {date}")
        is_holiday = is_USholiday(date)
        if is_holiday:
            self.pm.logger.warning(f"Market is closed on {date}, skipping")
            print(f"Market is closed on {date}, skipping")
            return "IS_HOLIDAY"

        ## First check if the position needs to be rolled
        if self.limits['dte']:
            roll_dict = self.dte_check()
        else:
            print("Roll Check Not Enabled")
            roll_dict = {}
            for sym in self.pm.symbol_list:
                current_position = self.pm.current_positions[sym]
                if 'position' not in current_position:
                    continue
                roll_dict[current_position['position']['trade_id']] = OpenPositionAction.HOLD.value

        print("Roll Dict", roll_dict)

        ## Check if the position needs to be adjusted based on moneyness
        if self.limits['moneyness']:
            moneyness_dict = self.moneyness_check()
        else:
            print("Moneyness Check Not Enabled")
            moneyness_dict = {}
            for sym in self.pm.symbol_list:
                current_position = self.pm.current_positions[sym]
                if 'position' not in current_position:
                    continue
                moneyness_dict[current_position['position']['trade_id']] = OpenPositionAction.HOLD.value
        print("Moneyness Dict", moneyness_dict)

        ## Check if the position needs to be adjusted based on greeks
        greek_dict = self.limits_check()
        print("Greek Dict", greek_dict)

        check_dicts = [roll_dict, moneyness_dict, greek_dict]
        all_empty = all([len(x)==0 for x in check_dicts])

        if all_empty: ## Return if all are empty
            self.pm.logger.info(f"No positions need to be adjusted on {date}")
            print(f"No positions need to be adjusted on {date}")
            return "NO_POSITIONS_TO_ADJUST"
        
        ## Aggregate the results
        for sym in self.pm.symbol_list:
            current_position = self.pm.current_positions[sym]
            if 'position' not in current_position:
                continue
            k = current_position['position']['trade_id']

            actions = [roll_dict[k], moneyness_dict[k], greek_dict[k]]
            sub_action_dict = {'action': '', 'quantity_diff': 0}

            ## If the position needs to be rolled or exercised, do that first, no need to check other actions or adjust quantity
            if OpenPositionAction.ROLL.value in actions:
                sub_action_dict['action'] = OpenPositionAction.ROLL.value
                continue
            elif OpenPositionAction.EXERCISE.value in actions:
                sub_action_dict['action'] = OpenPositionAction.EXERCISE.value
                continue

            ## If the position is a hold, check if it needs to be adjusted based on greeks
            elif OpenPositionAction.HOLD.value in actions:
                sub_action_dict['action'] = OpenPositionAction.HOLD.value

            quantity_change_list = []
            for key, value in greek_dict.items():
                for greek, res in value.items():
                    quantity_change_list.append(res['quantity_diff'])
            sub_action_dict['quantity_diff'] = min(quantity_change_list)
            if sub_action_dict['quantity_diff'] < 0:
                sub_action_dict['action'] = OpenPositionAction.ADJUST.value
            action_dict[k] = sub_action_dict
        return action_dict

                

        
    def limits_check(self):
        """
        Checks if the order is within the limits of the portfolio
        """
        limits = self.limits
        delta_limit = limits['delta']
        position_limit = {}

        date = pd.to_datetime(self.pm.events.current_date)
        print(f"Checking Limits on {date}")
        if is_USholiday(date):
            self.pm.logger.warning(f"Market is closed on {date}, skipping")
            return 
        

        current_positions = self.pm.current_positions
        for symbol, position in current_positions.items():
            if 'position' not in position:
                continue

            ## Initialize the greeks limits to False and other essentials variables
            status = {'status': False, 'quantity_diff': 0} ## Status is False by default
            greek_limit_bool = dict(vega = status, gamma = status, delta = status, theta = status) ## Initialize the greek limits to False
            max_delta = self.greek_limits['delta'][position['signal_id']]
            quantity, q = position['quantity'], position['quantity']
            trade_id = position['position']['trade_id']
            date = pd.to_datetime(self.pm.events.current_date)
            current_delta = abs(self.position_data[trade_id]['Delta'][date] * quantity)

            if delta_limit:
                quantity_diff = 0 ## Quantity difference to be used in case of limit breach, I want to return negative values
                if current_delta < max_delta:
                    print(f"Delta for Position {trade_id} is within limits")
                else:
                    print(f"Delta for Position {trade_id} is above limits")
                    while current_delta > max_delta:
                        ## Reduce the quantity of the position until it is within limits
                        quantity_diff -= 1
                        q = q -1
                        current_delta = abs(self.position_data[trade_id]['Delta'][date]) * q
                        print(f"Current Delta: {current_delta}, Max Delta: {max_delta}, Quantity: {q}")
                    greek_limit_bool['delta'] = {'status': True, 'quantity_diff': quantity_diff}
                position_limit[trade_id] = greek_limit_bool
        return position_limit
    
    def dte_check(self):
        """
        Analyze the current positions and determine if any need to be rolled
        """
        date = pd.to_datetime(self.pm.events.current_date)
        print(f"Checking DTE on {date}")
        if is_USholiday(date):
            self.pm.logger.warning(f"Market is closed on {date}, skipping")
            return
        
        roll_dict = {}
        for symbol in self.pm.symbol_list:
            current_position = self.pm.current_positions[symbol] 
            
            if 'position' not in current_position:
                continue

            id = current_position['position']['trade_id']
            expiry_date = ''
            
            if 'long' in current_position['position']:
                for option_id in current_position['position']['long']:
                    option_meta = parse_option_tick(option_id)
                    expiry_date = option_meta['exp_date']
                    break
            elif 'short' in current_position['position']:
                for option_id in current_position['position']['short']:
                    option_meta = parse_option_tick(option_id)
                    expiry_date = option_meta['exp_date']
                    break


            dte = (pd.to_datetime(expiry_date) - pd.to_datetime(date)).days

            
            if symbol in self.pm.roll_map and dte <= self.pm.roll_map[symbol]:
                roll_dict[id] = OpenPositionAction.ROLL.value
            elif symbol not in self.pm.roll_map and dte == 0:  # exercise contract if symbol not in roll map
                roll_dict[id] = OpenPositionAction.EXERCISE.value
            else:
                roll_dict[id] = OpenPositionAction.HOLD.value
        return roll_dict
    
    def moneyness_check(self):
        """
        Analyze the current positions and determine if any need to be rolled based on moneyness
        """
        date = pd.to_datetime(self.pm.events.current_date)
        print(f"Checking Moneyness on {date}")
        if is_USholiday(date):
            self.pm.logger.warning(f"Market is closed on {date}, skipping")
            return
        
        strike_list = []
        roll_dict = {}
        for symbol in self.pm.symbol_list:
            current_position = self.pm.current_positions[symbol] 
            if 'position' not in current_position:
                continue

            id = current_position['position']['trade_id']
            spot = self.chain_spot_timeseries[symbol][date] ## Use the spot price on the date (from chain cause of splits)
            
            if 'long' in current_position['position']:
                for option_id in current_position['position']['long']:
                    option_meta = parse_option_tick(option_id)
                    strike_list.append(option_meta['strike']/spot if option_meta['put_call'] == 'P' else spot/option_meta['strike'])

            if 'short' in current_position['position']:
                for option_id in current_position['position']['short']:
                    option_meta = parse_option_tick(option_id)
                    strike_list.append(option_meta['strike']/spot if option_meta['put_call'] == 'P' else spot/option_meta['strike'])
            
            roll_dict[id] = OpenPositionAction.ROLL.value if any([x > self.max_moneyness for x in strike_list]) else OpenPositionAction.HOLD.value
        return roll_dict
        

        
            




    ## Lazy Loading Spot Data
    def generate_data(self, symbol):
        stk = self.pm.get_underlier_data(symbol)  ## Performance isn't affected because of singletons in stock class
        if symbol not in self.spot_timeseries:
            self.spot_timeseries[symbol] = stk.spot(
                ts = True,
                ts_start = pd.to_datetime(self.start_date) - BDay(30),
                ts_end = pd.to_datetime(self.end_date),
            )[self.price_on]

        if symbol not in self.chain_spot_timeseries:
            self.chain_spot_timeseries[symbol] = stk.spot(
                ts = True,
                spot_type = OptionModelAttributes.spot_type.value,
                ts_start = pd.to_datetime(self.start_date) - BDay(30),
                ts_end = pd.to_datetime(self.end_date),
            )[self.price_on]
        
        if symbol not in self.dividend_timeseries:
            divs = stk.div_yield_history(start = pd.to_datetime(self.start_date) - BDay(30))
            if not isinstance(divs, (pd.DataFrame, pd.Series)): ## When a ticker has no dividends, it returns None/0
                divs = pd.Series(divs, index = self.spot_timeseries[symbol].index)
            self.dividend_timeseries[symbol] = divs

    def parse_position_id(self, positionID):
        position_str = positionID
        position_list = position_str.split('&')
        position_list = [x.split(':') for x in position_list if x]
        position_list_parsed = [(x[0], parse_option_tick(x[1])) for x in position_list]
        position_dict = dict(L = [], S = [])
        for x in position_list_parsed:
            position_dict[x[0]].append(x[1])
        return position_dict, position_list

    def get_position_dict(self, positionID):
        return self.parse_position_id(positionID)[0]

    def get_position_list(self, positionID):
        return self.parse_position_id(positionID)[1]

    def get_option_price(self, optID, date):
        portfolio = self.pm
        return portfolio.options_data[optID][self.option_price][date]
    
    