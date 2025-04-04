from dotenv import load_dotenv
import sys
import os
load_dotenv()
# sys.path.append(
#     os.environ.get('DBASE_DIR', ''))
from dbase.database.SQLHelpers import *
from dbase.DataAPI.ThetaData import *
import datetime
import yfinance as yf
import pandas as pd
import warnings
from trade.helpers.helper import change_to_last_busday, setup_logger, retrieve_timeseries
from threading import Thread
import logging
warnings.filterwarnings("ignore")
logger = setup_logger('rates')
rates_thread_logger = setup_logger('rates_thread', stream_log_level = logging.INFO)

## To-do: Check why rates data doesn't start from far back

## FYI: A thread saves _rates_cache. This is necessary to avoid blocking the main thread and makes sure the data is ready when needed. 

## Rates cache variable
_rates_cache = None




## Does the actual fetching of the rates
def fetch_rates_save_cache(interval = '1d', use = 'yf', return_data = False):
    # download 3-month us treasury bills rates
    """
    Return timeseries of 3-month US treasury bills rates

    Parameters
    ----------
    interval : str
        Interval to resample the data to. Default is '1d'
    
    use: str
        Source of the data. Default is 'yf', other option is 'db'
    
    """
    logger.info('Saving rates timeseries to cache with fetch_rates_save_cache function')
    global _rates_cache
    annualized = retrieve_timeseries('^IRX',  '1960-01-01', end = datetime.datetime.today().strftime('%Y-%m-%d'), interval = interval)['close']
    rates = yf.Ticker("^IRX")
    annualized.index = pd.to_datetime(annualized.index)
    annualized.index = [x.replace(minute = 30) for x in annualized.index]
    annualized.index.name = 'Datetime'

   
    if use == 'yf':

        def get_data(interval = '1d'):
            daily = annualized.apply(deannualize)
            data = pd.DataFrame({"annualized": annualized, "daily": daily})
            data['name'] = "^IRX"
            data['description']=rates.info['longName']

            if data.index.name.lower() != 'datetime':
                data.index.name = 'Datetime'
            return data

        data = get_data(interval)
        data['annualized'] = data.annualized/100


    elif use == 'db':
        ## Added Datetime min Filter to reduce time taken to query

        data = query_database('securities_master','rates_timeseries' ,"SELECT * FROM rates_timeseries WHERE yf_tick = '^IRX' AND DATETIME >= '2010-01-01'")
        data['datetime'] = pd.to_datetime(data['datetime'])
        data.set_index('datetime',inplace = True)
        data.rename(columns = {'daily_rate':'daily', 'annualized_rate':'annualized', 'yf_tick': 'name'}, inplace = True)
        data.index.name = 'Datetime'

    # create dataframe
    try:
        _rates_cache =  resample(data, interval, {'daily':'last', 'annualized': 'last', 'name': 'last', 'description': 'last'})
        if return_data:
            return _rates_cache

    except Exception as e:
        
        if isinstance(e, TypeError):
            logger.critical(f'YFinance did not return data. Rerun the function')
            return e
        else:
            logger.critical(f'Error in fetching rates data: {e}')
            return e



## Start a thread to update the rates data in the db
# rates_cache_thread = Thread(target = fetch_rates_save_cache, name = 'rates_cache_thread', args = ('1h', 'db'))
# rates_cache_thread.start()




def is_rates_thread_still_running():
    return rates_cache_thread.is_alive()

## Save the rates data to the db
def run_routine_rates_save():
    h1_rates = fetch_rates_save_cache('1h', 'yf', return_data=True)
    
    if isinstance(h1_rates, Exception) or isinstance(h1_rates, TypeError):
        return 

    
    h1_rates.reset_index(inplace = True)
    h1_rates.rename(columns = {'Datetime': 'datetime', 'daily': 'daily_rate',
                            'name':'yf_tick', 'annualized':'annualized_rate'}, inplace = True)
    
    store_SQL_data_Insert_Ignore('securities_master', 'rates_timeseries', h1_rates)
    logger.info('Rates data updated in db')



## Handles all the workflow. It doesn't fetch the rates data, it just checks if the rates cache is ready and updates it if necessary
## Checks if the rates cache is ready, if not, it waits for the rates thread to finish
## Updates the cache if the last date is not today
def _fetch_rates():

    # ## Check if the rates cache is ready
    # import time
    # timer = 0
    # while is_rates_thread_still_running():
    #     time.sleep(5)
    #     rates_thread_logger.info(f'Waiting for rates data to be ready. {timer} seconds elasped')
    #     timer += 5
    #     if timer > 60:
    #         rates_thread_logger.error('Rates data not ready after 60 seconds. Exiting')
    #         return None
        
    #     if not is_rates_thread_still_running():
    #         break


    global _rates_cache
    if _rates_cache is None:
        logger.info('Saving to cache')
        ## In saving to cache, I want to use db data
        print('Saving to cache from db')
        fetch_rates_save_cache(use = 'db')

    #Update the cache if the last date is not today
    if _rates_cache is not None:
        
        ## If db data is not up to date, update it db data AND update the cache using yf to save time. DB will be updated with a Thread

        if not isinstance(_rates_cache.index, pd.DatetimeIndex):
            _rates_cache.index = pd.to_datetime(_rates_cache.index)
        

        if change_to_last_busday(datetime.datetime.now()).date() > _rates_cache.index[-1].date():
            logger.info('Updating cache')
            logger.info('Updating db rates data')
            print('Updating db rates data')
            db_update_thread = Thread(target = run_routine_rates_save)
            db_update_thread.start()
            fetch_rates_save_cache()
        
        else:
            logger.info('Cache is up to date')
    
    return _rates_cache


def deannualize(annual_rate, periods=365):
    return (1 + annual_rate) ** (1/periods) - 1


## User facing function
def get_risk_free_rate_helper(interval = '1d', use = 'db'):
    # download 3-month us treasury bills rates
    """
    Return timeseries of 3-month US treasury bills rates

    Parameters
    ----------
    interval : str
        Interval to resample the data to. Default is '1d'
    
    use: str
        Source of the data. Default is 'yf', other option is 'db'
    
    """

    global _rates_cache
    data =  _fetch_rates().copy()
    return resample(data, interval, {'daily':'last', 'annualized': 'last', 'name': 'last', 'description': 'last'})


def save_previous_rates_date():
    import yfinance as yf
    rtes = yf.download('^IRX', progress=False)
    rtes['annualized'] = rtes['Adj Close']/100
    rtes['daily'] = (1 + rtes['annualized']) ** (1/365) - 1
    rtes['yf_tick'] = '^IRX'
    rtes['description'] = '13 week treasury bill'
    rtes = rtes[rtes.index< get_risk_free_rate_helper().index.min()][['yf_tick', 'description', 'annualized', 'daily']].reset_index()
    rtes.rename(columns = {'annualized': 'annualized_rate', 'daily': 'daily_rate', 'Date': 'datetime'}, inplace = True)
    store_SQL_data_Insert_Ignore('securities_master', 'rates_timeseries', rtes)