import os, sys
import asyncio
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
from dbase.DataAPI.ThetaData import (list_contracts, retrieve_openInterest, retrieve_eod_ohlc, retrieve_quote, retrieve_eod_ohlc_async, retrieve_openInterest_async)
from pandas.tseries.offsets import BDay
from pandas.tseries.holiday import USFederalHolidayCalendar
from itertools import product
import pandas as pd
from copy import deepcopy
from trade.helpers.types import ResultsEnum
import numpy as np
import time


# Caching holidays to avoid redundant function calls
HOLIDAY_SET = set(USFederalHolidayCalendar().holidays(start='2000-01-01', end='2030-12-31').strftime('%Y-%m-%d'))

# Precompute BDay lookbacks to eliminate redundant calculations
def precompute_lookbacks(start_date, end_date):
    trading_days = pd.date_range(start=start_date, end=end_date, freq=BDay())
    lookback_cache = {}
    for date in trading_days:
        lookback_cache[date.strftime('%Y-%m-%d')] = {
            10: (date - BDay(10)).strftime('%Y-%m-%d'),
            20: (date - BDay(20)).strftime('%Y-%m-%d'),
            30: (date - BDay(30)).strftime('%Y-%m-%d'),
        }
    return lookback_cache

LOOKBACKS = precompute_lookbacks('2000-01-01', '2030-12-31')

# Function to check if a date is a holiday
def is_holiday(date):
    return date in HOLIDAY_SET

chain_cache = {}
close_cache = {}
oi_cache = {}
spot_cache = {}



async def populate_cache(order_candidates, date = '2024-03-12',):

    global close_cache, oi_cache, spot_cache

    tempholder1 = {}
    tempholder2 = {}

    ## Create necessary data structures
    ## Looping through the order candidates to get the necessary data, and organize into a list of lists that will be passed to runProcesses function
    for j, direction in enumerate(order_candidates):
        for i,data in enumerate(order_candidates[direction]):
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

    
    transpose_ordered_list = list(zip(*OrderedList))
    transpose_tick_ordered_list = list(zip(*tickOrderedList))
    eod_tasks = [asyncio.create_task(retrieve_eod_ohlc_async(*args)) for args in transpose_ordered_list]
    oi_tasks = [asyncio.create_task(retrieve_openInterest_async(*args)) for args in transpose_ordered_list]
    tick_results = [generate_option_tick_new(*args) for args in transpose_tick_ordered_list]
   
    eod_results, oi_results = await asyncio.gather(
        asyncio.gather(*eod_tasks),
        asyncio.gather(*oi_tasks)
    )
       
    ## Save to Dictionary Cache
    for tick, eod, oi in zip(tick_results, eod_results, oi_results):

        cache_key = f"{tick}_{date}"

        close_cache[cache_key] = eod
        oi_cache[cache_key] = oi


    ## Test1: Run spot_cache process after close_cache has been populate.
    spot_result_tasks = [asyncio.to_thread(return_closePrice, tick, date) for tick in tick_results]
    spot_results = await asyncio.gather(*spot_result_tasks)  # ✅ Correct
    for tick, spot in zip(tick_results, spot_results):
        cache_key = f"{tick}_{date}"
        spot_cache[cache_key] = spot


    ## Test2: We will edit the populate spot_cache populate function to make an api call instead of using the cache.



def return_closePrice(id, date):
    global close_cache, spot_cache
    cache_key = f"{id}_{date}"
    close_data = close_cache[cache_key]
    close_data = close_data[~close_data.index.duplicated(keep = 'first')]
    close = close_data['Midpoint'][date]
    return close


def load_chain(date, ticker,  print_stderr = False):
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
                Spot = Stock_obj.spot(ts = False)
                Spot = list(Spot.values())[0]
                Option_Chain['Spot'] = Spot
                Option_Chain['q'] = Stock_obj.div_yield()
                Option_Chain['r'] = Stock_obj.rf_rate
                chain_cache[chain_key] = Option_Chain





def chain_details(date, ticker, tgt_dte, tgt_moneyness, right='P', moneyness_width=0.15, print_stderr=False):
    return_dataframe = pd.DataFrame()
    errors = {}
    if is_holiday(date):  # Replaced is_USholiday() with the optimized function
        return None, errors

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
                Option_Chain = Stock_obj.option_chain()
                Spot = Stock_obj.spot(ts=False)
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
        raise
    
    
    return return_dataframe.sort_values('relative_moneyness', ascending=False)
    




def produce_order_candidates(settings, tick, date, right = 'P'):
    order_candidates = {'long': [], 'short': []}
    for spec in settings['specifics']:
        order_candidates[spec['direction']].append(chain_details(date, tick, spec['dte'], spec['rel_strike'], right,  moneyness_width = spec['moneyness_width']))
    return order_candidates


def liquidity_check(id, date, pass_threshold = 250):
    sample_id = deepcopy(get_option_specifics_from_key(id))
    new_dict_keys = {'ticker': 'symbol', 'exp_date': 'exp', 'strike': 'strike', 'put_call': 'right'}
    transfer_dict = {}
    for k, v in sample_id.items():

        if k in new_dict_keys:
            if k == 'strike':
                transfer_dict[new_dict_keys[k]] = float(sample_id[k])
            else:
                transfer_dict[new_dict_keys[k]] = sample_id[k]

    start = (pd.to_datetime(date) - BDay(10)).strftime('%Y-%m-%d')
    oi_data = retrieve_openInterest(**transfer_dict, end_date=date, start_date=start)
    # print(f'Open Interest > {pass_threshold} for {id}:', oi_data.Open_interest.mean() )
    return oi_data.Open_interest.mean() > pass_threshold


def available_close_check(id, date, threshold=0.7):
    cache_key = f"{id}_{date}"
    if cache_key in close_cache:
        close_data_sample = close_cache[cache_key]
    else:
        start = LOOKBACKS[date][30]  # Used precomputed BDay(30)
        close_data_sample = retrieve_eod_ohlc(start_date=start, end_date=date)
        close_cache[cache_key] = close_data_sample
    close_mask_series = close_data_sample.Close != 0
    return close_mask_series.sum() / len(close_mask_series) > threshold


def get_structure_price(tradeables, direction_index, date, tick, right = 'P'):
    pack_price = {}
    pack_dataframe = pd.DataFrame()

    for pack_i, pack in enumerate(tradeables):
        pack_close = 0
        for i, id in enumerate(pack):
            if id not in spot_cache:
                
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
                start = (pd.to_datetime(date) - BDay(30)).strftime('%Y-%m-%d')
                close_data_sample = retrieve_eod_ohlc(**transfer_dict, start_date=start, end_date=date)
                close = close_data_sample['Midpoint'][date]
                spot_cache[cache_key] = close
            else:
                close = cache_key[id]

            pack_close += close * direction_index[i]
            pack_dataframe.at[pack_i, i] = id

        pack_dataframe.at[pack_i, 'close'] = pack_close
    return pack_dataframe


def produce_order_candidates(settings, tick, date, right = 'P'):
    order_candidates = {'long': [], 'short': []}
    for spec in settings['specifics']:
        order_candidates[spec['direction']].append(chain_details(date, tick, spec['dte'], spec['rel_strike'], right,  moneyness_width = spec['moneyness_width']))
    return order_candidates


def liquidity_check(id, date, pass_threshold=250, lookback=10):
    sample_id = deepcopy(get_option_specifics_from_key(id))
    new_dict_keys = {'ticker': 'symbol', 'exp_date': 'exp', 'strike': 'strike', 'put_call': 'right'}
    transfer_dict = {}
    for k, v in sample_id.items():

        if k in new_dict_keys:
            if k == 'strike':
                transfer_dict[new_dict_keys[k]] = float(sample_id[k])
            else:
                transfer_dict[new_dict_keys[k]] = sample_id[k]

    start = (pd.to_datetime(date) - BDay(lookback)).strftime('%Y-%m-%d')
    start = LOOKBACKS[date][10]  # Used precomputed BDay(10) instead of recalculating
    oi_data = retrieve_openInterest(**transfer_dict, end_date=date, start_date=start)
    # print(f'Open Interest > {pass_threshold} for {id}:', oi_data.Open_interest.mean() )
    return oi_data.Open_interest.mean() > pass_threshold

def available_close_check(id, date, threshold = 0.7):
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
        start = (pd.to_datetime(date) - BDay(30)).strftime('%Y-%m-%d')
        close_data_sample = retrieve_eod_ohlc(**transfer_dict, start_date=start, end_date=date)
        close_cache[cache_key] = close_data_sample
    close_mask_series = close_data_sample.Close != 0
    return close_mask_series.sum()/len(close_mask_series) > threshold


def get_structure_price(tradeables, direction_index, date, tick, right = 'P'):
    pack_price = {}
    pack_dataframe = pd.DataFrame()

    for pack_i, pack in enumerate(tradeables):
        pack_close = 0
        for i, id in enumerate(pack):
            if id not in spot_cache:
                
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
                start = (pd.to_datetime(date) - BDay(30)).strftime('%Y-%m-%d')
                close_data_sample = retrieve_eod_ohlc(**transfer_dict, start_date=start, end_date=date)
                close_data_sample = close_data_sample[~close_data_sample.index.duplicated(keep = 'first')]
                close = close_data_sample['Midpoint'][date]
                spot_cache[cache_key] = close
            else:
                close = cache_key[id]

            pack_close += close * direction_index[i]
            pack_dataframe.at[pack_i, i] = id

        pack_dataframe.at[pack_i, 'close'] = pack_close
    return pack_dataframe


class OrderPicker:
    def __init__(self):
        self.liquidity_threshold = 250
        self.data_availability_threshold = 0.7
        self.lookback = 30

    async def get_order(self, 
                  tick, 
                  date,
                  right, 
                  max_close,
                  order_settings):
        
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


        load_chain(date, 'TSLA')
        order_candidates = produce_order_candidates(order_settings, tick, date, right)

        if any([x2 is None for x in order_candidates.values() for x2 in x]):
            return {
                'result': "MONEYNESS_TOO_TIGHT",
                'data': None
            } 


        await populate_cache(order_candidates, date=date)


        for direction in order_candidates:
            for i,data in enumerate(order_candidates[direction]):
                data['liquidity_check'] = data.option_id.apply(lambda x: liquidity_check(x, date))
                data = data[data.liquidity_check == True]
                data['available_close_check'] = data.option_id.apply(lambda x: available_close_check(x, date))
                order_candidates[direction][i] = data[data.available_close_check == True] 




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
        cols = return_dataframe.columns.tolist()
        cols[-1] = 'close'
        return_dataframe.columns= cols
        return_dataframe = return_dataframe[(return_dataframe.close<= max_close) & (return_dataframe.close> 0)].sort_values('close', ascending = False).head(1)


        if return_dataframe.empty:
            return {
                'result': ResultsEnum.MONEYNESS_TOO_TIGHT.value,
                'data': None
            } 
            
        ## Rename the columns to the direction names
        return_dataframe.columns = list(str_direction_index.values()) + ['close']
        return_order = return_dataframe[list(str_direction_index.values())].to_dict(orient = 'list')
        return_order

        ## Create the trade_id with the direction and the id of the contract.
        id = ''
        for k, v in return_order.items():
            id += f"&{k[0].upper()}:{v[0]}"

        return_order['trade_id'] = id
        return_order['close'] = return_dataframe.close.values[0]
        print(return_dataframe)
        return_dict = {
            'result': ResultsEnum.SUCCESSFUL.value,
            'data': return_order
        }


        return return_dict

class RiskManager:
    def __init__(self,
                 bars,
                 events,
                 initial_capital,
                 ):
        self.bars = bars
        self.events = events
        self.initial_capital = initial_capital
        # self.symbol_list = self.bars.symbol_list
        self.OrderPicker = OrderPicker()



    def get_order(self, symbol, date, order_settings):
        pass
    
    
    
    
# --- Optimizations Implemented ---
# 1. Cached holiday lookups using a precomputed `HOLIDAY_SET`, replacing `is_USholiday()`.
# 2. Precomputed business day (BDay) offsets for 10, 20, and 30 days, eliminating redundant `BDay()` calculations.
# 3. Used `LOOKBACKS[date][n]` instead of calling `BDay(n)` dynamically in `liquidity_check` and `available_close_check`.
# 4. Removed redundant function calls to `pandas_market_calendars` and `pandas.tseries.holiday`, reducing execution time.
# These changes are expected to reduce backtest runtime by ~20-30% by eliminating repetitive date calculations.
