import os, sys
from trade.assets.Stock import Stock
from trade.assets.Option import Option
from trade.assets.OptionStructure import OptionStructure
from trade.helpers.Context import Context, clear_context
from trade.helpers.helper import (change_to_last_busday, 
                                  is_USholiday, 
                                  is_busday, 
                                  setup_logger, 
                                  generate_option_tick_new, 
                                  get_option_specifics_from_key,
                                  identify_length,
                                  extract_numeric_value)
from scipy.stats import percentileofscore
from dbase.DataAPI.ThetaData import (list_contracts, retrieve_openInterest, retrieve_eod_ohlc, retrieve_quote)
from pandas.tseries.offsets import BDay
from pandas.tseries.holiday import USFederalHolidayCalendar
from trade.helpers.decorators import log_error_with_stack
from itertools import product
import pandas as pd
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from trade.helpers.Logging import setup_logger
from trade.helpers.decorators import log_error_with_stack
from pathos.multiprocessing import ProcessingPool as Pool
from trade.helpers.threads import runThreads
from trade.helpers.types import ResultsEnum, OptionModelAttributes
import numpy as np
import time

logger = setup_logger('QuantTools.EventDriven.riskmanager')
LOOKBACKS = {}

def _retrieve_openInterest(*args, **kwargs):
    try:
        return retrieve_openInterest(*args, **kwargs)
    except Exception as e:
        return None


# Caching holidays to avoid redundant function calls
HOLIDAY_SET = set(USFederalHolidayCalendar().holidays(start='2000-01-01', end='2030-12-31').strftime('%Y-%m-%d'))

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


@log_error_with_stack(logger)
def populate_cache(order_candidates: dict, 
                   date: str = '2024-03-12',) -> str|None:
    
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

    if is_holiday(date):
        return 'holiday'
    
    else:

        ## Create necessary data structures
        ## Looping through the order candidates to get the necessary data, and organize into a list of lists that will be passed to runProcesses function
        for j, direction in enumerate(order_candidates):
            for i,data in enumerate(order_candidates[direction]):
                if isinstance(data, str) and data =='theta_data_error':
                    return 'theta_data_error'
                
                data[[ 'exp', 'strike', 'symbol']] = data[[ 'expiration', 'strike', 'ticker']]
                start = LOOKBACKS[date][20]  # Used precomputed BDay(20) instead of recalculating
                data[['end_date', 'start_date']] = date, start
                data['exp'] = data['exp'].dt.strftime('%Y-%m-%d')
                tempholder1[i+j] = (data[['symbol', 'end_date', 'exp', 'right', 'start_date', 'strike']].T.values.tolist())
                tempholder2[i+j] = data[['symbol', 'right', 'exp','strike']].T.values.tolist()

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
        for tick, eod, oi in zip(tick_results, eod_results, oi_results):

            cache_key = f"{tick}_{date}"

            close_cache[cache_key] = eod
            oi_cache[cache_key] = oi


        ## Test1: Run spot_cache process after close_cache has been populate.
        
        spot_results = runThreads(return_closePrice, [tick_results, [date]*len(tick_results)], 'map')
        for tick, spot in zip(tick_results, spot_results):
            cache_key = f"{tick}_{date}"
            spot_cache[cache_key] = spot


        ## Test2: We will edit the populate spot_cache populate function to make an api call instead of using the cache.



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
    cache_key = f"{id}_{date}"
    close_data = close_cache[cache_key]
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
    
            Option_Chain_Filtered = Option_Chain[Option_Chain[right.upper()] == True]
        
            if right == 'P':
                Option_Chain_Filtered['relative_moneyness'] = Option_Chain_Filtered.index.get_level_values('strike') / Option_Chain_Filtered.Spot
            elif right == 'C':
                Option_Chain_Filtered['relative_moneyness'] = Option_Chain_Filtered.Spot / Option_Chain_Filtered.index.get_level_values('strike')
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
            option_details['right'] = right
            option_details.drop(columns = ['C','P'], inplace = True)
            option_details['option_id'] = option_details.apply(lambda x: generate_option_tick_new(symbol = x['ticker'], 
                                                                        exp = x['expiration'].strftime('%Y-%m-%d'), strike = float(x['strike']), right = x['right']), axis = 1)
            return_dataframe = pd.concat([return_dataframe, option_details])
        clear_context()
        return_dataframe.drop_duplicates(inplace = True)

    except Exception as e:
        raise e
        return 'error'
    
    
    return return_dataframe.sort_values('relative_moneyness', ascending=False)
    


def available_close_check(id: str, 
                          date: str, 
                          threshold: float = 0.7) -> bool:
    
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
    cache_key = f"{id}_{date}"
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
    else:
        start = LOOKBACKS[date][30]  # Used precomputed BDay(30)
        close_data_sample = retrieve_eod_ohlc(**transfer_dict, start_date=start, end_date=date)
        close_cache[cache_key] = close_data_sample
    close_mask_series = close_data_sample.Close != 0
    return close_mask_series.sum()/len(close_mask_series) > threshold


def produce_order_candidates(settings: dict, 
                             tick: str, 
                             date: str, 
                             right: str = 'P') -> dict:
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

    order_candidates = {'long': [], 'short': []}
    for spec in settings['specifics']:
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
    # oi_data = retrieve_openInterest(**transfer_dict, end_date=date, start_date=start)
    oi_data = oi_cache[f"{id}_{date}"]

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
    def __init__(self, liquidity_threshold: int = 250, data_availability_threshold: float = 0.7, lookback: int = 30):
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


        returned = populate_cache(order_candidates, date=date)

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
                 bars,
                 events,
                 initial_capital,
                 ):
        
        """
        initializes the RiskManager class

        params:
        bars: Bars: bars
        events: Events: events
        initial_capital: float: initial capital
        """
        
        self.bars = bars
        self.events = events
        self.initial_capital = initial_capital
        # self.symbol_list = self.bars.symbol_list
        self.OrderPicker = OrderPicker()



    def get_order(self, symbol, date, order_settings):
        pass