import time
start_time = time.time() 
import os, sys
sys.path.append(os.environ['WORK_DIR'])
sys.path.append(os.environ['DBASE_DIR'])
from trade.models.VolSurface import SurfaceLab
from dbase.database.SQLHelpers import DatabaseAdapter
from trade.helpers.helper import (change_to_last_busday, 
                                  is_busday, 
                                  is_USholiday,
                                  IV_handler, 
                                  implied_volatility, 
                                  time_distance_helper, 
                                  generate_option_tick, 
                                  setup_logger,
                                  binomial_implied_vol)

from trade.helpers.Context import Context
from trade.assets.rates import get_risk_free_rate_helper
from dbase.DataAPI.ThetaData import (retrieve_quote_rt, 
                                     retrieve_eod_ohlc,
                                     retrieve_quote, 
                                     list_contracts)

from dbase.database.SQLHelpers import store_SQL_data_Insert_Ignore
from pathos.multiprocessing import ProcessingPool as Pool
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, root_mean_squared_error
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
from threading import Thread
import pandas as pd
logger = setup_logger('OptionChain')





shutdown_event = False


def shutdown(pool):
    global shutdown_event
    shutdown_event = True
    print('shutdown_event set')
    pool.terminate()


def get_set(
        ticker,
        date,
        exp, 
        right,
        strike,
        spot,
        r,
        q
) -> dict:
    try:
        price = retrieve_quote(ticker, date, exp, right, date, strike, start_time = '9:00')['Midpoint'][-1] #To-do: Handle None values
        vol = IV_handler(S = spot, K = strike, t = time_distance_helper(exp = exp, strt = date), r = r, flag = right.lower(), price = price, q = q)
    except Exception as e:
        logger.error(f'Error in get_set: {e}', exc_info=True)
        raise e
    return {'price': price, 'vol': vol}

def get_df_set(split_df, id):
    if len(split_df) == 0:
        return pd.DataFrame()
    split_df[['expiration', 'build_date']] = split_df[['expiration', 'build_date']].astype(str)
    # print(f'Size in ID {id} is {split_df.shape}')
    split_df[['price', 'vol']] = split_df.apply(lambda x: get_set(x['ticker'], x['build_date'], x['expiration'], x['right'], x['strike'], x['Spot'], x['r'], x['q']), axis = 1, result_type = 'expand')
    return split_df

def produce_chain_values(chain, date, ticker, stock):
    results = []
    global shutdown_event # To-do: Why after shut down event set in some situations, it is not being reset?
    shudown_event = False
    # with Context(end_date = date):
    # stk = Stock(ticker)
    stk = stock
    spot = list(stk.spot().values())[0]
    q = stk.div_yield()
    rf_rate = stk.rf_rate
    chain['Spot'] = spot
    chain['r'] = rf_rate
    chain['q'] = q
    chain['build_date'] = date
    chain['ticker'] = ticker
    
    workers = 25
    split_data = np.array_split(chain.reset_index(), workers)


    #Multiprocessing to speed up chain retrieval
    pool = Pool(nodes = workers)
    pool.restart()
    try:
        for result in pool.imap(get_df_set, split_data, [i for i in range(workers)]):
            results.append(result)
            if shutdown_event:
                break
            
        
    except KeyboardInterrupt:
        print('Interrupted by Keyboard')
        shutdown(pool)
        

    except Exception as e:
        print('Exception:', e)
        shutdown(pool)
        

    finally:
        pool.close()
        pool.join()
    
    if len(results) == 0:
        return None

    return pd.concat(results)





class OptionChain:
    """
    Class responsible for creating option chains for a given stock. 
    Expected behavior is to return the chain, corresponding vol, price and in select instances, Volatility surface
    """

    def __init__(self, ticker, build_date, stock, dumas_width = 0.75, **kwargs) -> None:
        self.ticker = ticker.upper()
        self.build_date = build_date
        self.chain = None
        self.lab = None
        self.dbAdapter = DatabaseAdapter()
        self.dumas_width = dumas_width
        self.simple_chain = None
        self.stock = stock
        self.kwargs = kwargs
        self.__initiate_chain()


    def __repr__(self) -> str:
        return f"OptionChain({self.ticker}, {self.build_date})"
    
    def __str__(self) -> str:
        return f"OptionChain({self.ticker}, {self.build_date})"

    def __initiate_chain(self):
        query = f"""
        SELECT * FROM vol_surface.option_chain 
        WHERE ticker = '{self.ticker}' 
        AND build_date = '{self.build_date}'"""

        chain = self.dbAdapter.query_database('vol_surface', 'option_chain',query)
        if not chain.empty:
            self.chain = chain
            self.__option_chain_bool()
            return 
        

        chain =  self.__option_chain_bool()
        chain_new = produce_chain_values(chain, self.build_date, self.ticker, self.stock)
        self.chain = chain_new
        self.save_thread = None
        self.save_thread = Thread(target = self._save_chain)
        self.save_thread.start()
        return chain_new

        
    def get_chain(self, return_values = False):
        if return_values:
            return self.chain
        
        else:
            return self.simple_chain


    def __option_chain_bool(self):

        # Set build date
        date = self.build_date

        contracts = list_contracts(self.ticker, date)
        contracts.expiration = pd.to_datetime(contracts.expiration, format='%Y%m%d')

        ## Producing the DTE for the contracts
        contracts['DTE'] = (contracts['expiration'] - pd.to_datetime(date)).dt.days
        contracts_v2 = contracts.pivot_table(index=['expiration', 'DTE','strike','right'],values = 'root', aggfunc='count')
        
        ## Formatting the pivot table
        contracts_v2.fillna(0, inplace = True)
        contracts_v2.where(contracts_v2 == 1, False, inplace = True)
        contracts_v2.where(contracts_v2 == 0, True, inplace = True)
        self.simple_chain = contracts_v2
        return contracts_v2
    

    def initiate_lab(self):
        force_build = self.kwargs.get('force_build', False)
        if self.chain is None:
            self.chain = self.get_chain(True)
        self.lab = SurfaceLab(self.ticker, self.build_date, self.chain.copy(),self.dumas_width, force_build = force_build)

    

    def _save_chain(self):
        if self.chain is None:
            self.get_chain(True)
        chain = self.chain.copy()
        chain.drop(columns = ['root'], inplace = True)
        chain['moneyness'] = chain['strike'] / chain['Spot']
        chain['option_tick'] = chain.apply(lambda x: generate_option_tick(x['ticker'],x['right'], x['expiration'],  x['strike']), axis = 1)
        chain['build_date'] = self.build_date
        self.dbAdapter.save_to_database(chain, 'vol_surface', 'option_chain')



