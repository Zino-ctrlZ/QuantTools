import os, sys
from trade.assets.Stock import Stock
from trade.assets.Option import Option
from trade.assets.OptionStructure import OptionStructure
from trade.helpers.Context import Context, clear_context
from trade.helpers.helper import (change_to_last_busday, 
                                  is_USholiday, 
                                  is_busday, 
                                  setup_logger, 
                                  generate_option_tick, 
                                  get_option_specifics_from_key,
                                  identify_length,
                                  extract_numeric_value)
from scipy.stats import percentileofscore
from dbase.DataAPI.ThetaData import (list_contracts, retrieve_openInterest, retrieve_eod_ohlc, retrieve_quote)
from pandas.tseries.offsets import BDay
from itertools import product
import pandas as pd
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathos.multiprocessing import ProcessingPool as Pool
import numpy as np
import time
chain_cache = {}
close_cache = {}
spot_cache = {}



def chain_details(date, ticker, tgt_dte, tgt_moneyness, right = 'P', moneyness_width = 0.15, print_stderr = False):
    return_dataframe = pd.DataFrame()
    errors = {}
    if not (is_USholiday(date) and not is_busday(date)):
        try:
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

                
                Option_Chain_Filtered = Option_Chain[Option_Chain[right.upper()] == True]
                
                
                if right == 'P':
                    Option_Chain_Filtered['relative_moneyness']  = Option_Chain_Filtered.index.get_level_values('strike')/Option_Chain_Filtered.Spot
                elif right == 'C':
                    Option_Chain_Filtered['relative_moneyness']  = Option_Chain_Filtered.Spot/Option_Chain_Filtered.index.get_level_values('strike')
                else:
                    raise ValueError(f'Right dne. recieved {right}')
                Option_Chain_Filtered['moneyness_spread'] = (tgt_moneyness-Option_Chain_Filtered['relative_moneyness'])**2
                Option_Chain_Filtered['dte_spread'] = (Option_Chain_Filtered.index.get_level_values('DTE')-tgt_dte)**2
                Option_Chain_Filtered.sort_values(by=['dte_spread','moneyness_spread'], inplace = True)
                Option_Chain_Filtered = Option_Chain_Filtered.loc[Option_Chain_Filtered['dte_spread'] == Option_Chain_Filtered['dte_spread'].min()]
                if float(moneyness_width) == 0.0:
                    option_details = Option_Chain_Filtered.sort_values('moneyness_spread', ascending=False).head(1)
                else:
                    option_details = Option_Chain_Filtered[(Option_Chain_Filtered['relative_moneyness'] >= tgt_moneyness-moneyness_width) & 
                                                        (Option_Chain_Filtered['relative_moneyness'] <= tgt_moneyness+moneyness_width)]
                option_details['build_date'] = date
                option_details['ticker'] = ticker
                option_details['moneyness'] = tgt_moneyness
                option_details['TGT_DTE'] = tgt_dte
                option_details.reset_index(inplace = True)
                option_details.set_index('build_date', inplace = True)
                option_details['right'] = right
                option_details.drop(columns = ['C','P'], inplace = True)
                option_details['option_id'] = option_details.apply(lambda x: generate_option_tick(symbol = x['ticker'], 
                                                                    exp = x['expiration'].strftime('%Y-%m-%d'), strike = float(x['strike']), right = x['right']), axis = 1)
                return_dataframe = pd.concat([return_dataframe, option_details])
            clear_context()
            return_dataframe.drop_duplicates(inplace = True)

        except Exception as e:
            raise
            errors[date] = e
            return errors
        return return_dataframe.sort_values('relative_moneyness', ascending=False)
    else:
        return None, errors
    




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


def liquidity_check(id, date, pass_threshold = 250, lookback = 10):
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

    def get_order(self, 
                  tick: str, 
                  date: str,
                  right: str, 
                  max_close: float|int,
                  order_settings: dict):
        
        ## Create necessary data structures
        direction_index = {}
        for indx, v in enumerate(order_settings['specifics']):
            if v['direction'] == 'long':
                direction_index[indx] = 1
            elif v['direction'] == 'short':
                direction_index[indx] = -1

        ## Produce Order Candidates
        start = (pd.to_datetime(date) - BDay(30)).strftime('%Y-%m-%d')
        order_candidates = produce_order_candidates(order_settings, tick, date, right)

        ## Check Liquidity and Close Availability, Filter out those that don't meet the criteria
        for direction in order_candidates:
            for i,data in enumerate(order_candidates[direction]):
                data['liquidity_check'] = data.option_id.apply(lambda x: liquidity_check(x, date, self.liquidity_threshold, self.lookback))
                order_candidates[direction][i] = data[data.liquidity_check == True]

        for direction in order_candidates:
            for i,data in enumerate(order_candidates[direction]):
                data['available_close_check'] = data.option_id.apply(lambda x: available_close_check(x, date, self.data_availability_threshold))
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
        prices = get_structure_price(filtered, direction_index, date, 'AAPL')

        ## Return the structure with the best price
        return_dataframe = prices[(prices.close<= max_close)].sort_values('close', ascending = False).head(1)
        return_order = {'long': [], 'short': []}
        id = ''
        for key, v in direction_index.items():
            if v < 0:
                option_id = return_dataframe[key].values[0]
                id += f'&L:{option_id}'
                return_order['short'].append(option_id)
            elif v > 0:
                option_id = return_dataframe[key].values[0]
                id += f'&S:{option_id}'
                return_order['long'].append(option_id)
        return_order['close'] = return_dataframe.close.values[0]
        return_order['trade_id'] = id
        return return_order
    

    

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