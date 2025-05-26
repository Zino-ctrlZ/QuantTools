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
from pandas.tseries.offsets import BDay
warnings.filterwarnings("ignore")
logger = setup_logger('rates')
rates_thread_logger = setup_logger('rates_thread', stream_log_level = logging.INFO)

## To-do: Check why rates data doesn't start from far back

## FYI: A thread saves _rates_cache. This is necessary to avoid blocking the main thread and makes sure the data is ready when needed. 

## Rates cache variable
_rates_cache = None

def reset_rates_cache():
    """
    Reset the rates cache
    """
    global _rates_cache
    _rates_cache = None
    logger.info('Rates cache reset')

def deannualize(annual_rate, periods=365):
    return (1 + annual_rate) ** (1/periods) - 1


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

    data = _fetch_rates(interval = interval).copy()
    # , {'daily':'last', 'annualized': 'last', 'name': 'last', 'description': 'last'}
    data = resample(data, interval)
    data = data[~data.index.duplicated(keep = 'first')]
    return data ## Not adding the resample schema for now


def _fetch_rates(interval):
    """
    Handles _rates_cache logic picking
    """
    global _rates_cache
    ## First check data base.
    if _rates_cache is None:
        data = query_database('securities_master','rates_timeseries' ,"SELECT * FROM rates_timeseries WHERE yf_tick = '^IRX' AND DATETIME >= '2010-01-01'")
        data['datetime'] = pd.to_datetime(data['datetime'])
        data.set_index('datetime',inplace = True)
        data.rename(columns = {'daily_rate':'daily', 'annualized_rate':'annualized', 'yf_tick': 'name'}, inplace = True)
        data.index.name = 'Datetime'
    else:
        data = _rates_cache.copy()

    ## Now, if data is not up to date, update it
    if data.index.max().date() < change_to_last_busday(datetime.datetime.now()).date():
        # logger.info('Updating rates data')
        ## Query from max to today from retrieve_timeseries
        ## Prefer using yf because openbb timezone is UTC for IRX
        ## For YF, we have to do end_date + 1 day to get the last day
        data_min = yf.download('^IRX', data.index.max().date(), end = (datetime.datetime.today()+ BDay(1)).strftime('%Y-%m-%d') , interval = '1h', progress=False, multi_level_index = False)
        data_min.tz_convert('America/New_York')
        data_min.index = data_min.index.tz_convert('America/New_York')
        data_min.index = data_min.index.tz_localize(None)
        data_min.index = [x.replace(minute = 30) for x in data_min.index]
        data_min.columns = data_min.columns.str.lower()
        data_min['daily'] = data_min['close'].apply(deannualize)
        data_min['annualized'] = data_min['close']/100
        data_min['name'] = '^IRX'
        data_min['description'] = '13 WEEK TREASURY BILL'
        data_min.index.name = 'Datetime'
        data_min = data_min[['name', 'description', 'daily', 'annualized']]
        data = pd.concat([data, data_min])
        data = data[~data.index.duplicated(keep = 'first')]
    
    _rates_cache =  resample(data, '30m', {'daily':'ffill', 'annualized': 'ffill', 'name': 'ffill', 'description': 'ffill'})
    return data


    

def save_previous_rates_date():
    import yfinance as yf
    print("Saving previous rates date")
    max_date = get_risk_free_rate_helper().index.max()
    rtes = yf.download('^IRX', progress=False, multi_level_index=False,start = max_date, end = (datetime.datetime.today()+ BDay(1)).strftime('%Y-%m-%d'), interval = '1h')
    print("DOWNLOAD COMPLETE")
    rtes.tz_convert('America/New_York')
    rtes.index = rtes.index.tz_convert('America/New_York')
    rtes.index = rtes.index.tz_localize(None)
    rtes.index = [x.replace(minute = 30) for x in rtes.index]
    rtes['annualized'] = rtes['Close']/100
    rtes['daily'] = (1 + rtes['annualized']) ** (1/365) - 1
    rtes['yf_tick'] = '^IRX'
    rtes['description'] = '13 WEEK TREASURY BILL'
    rtes = rtes[rtes.index> get_risk_free_rate_helper().index.min()][['yf_tick', 'description', 'annualized', 'daily']]
    rtes.rename(columns = {'annualized': 'annualized_rate', 'daily': 'daily_rate', 'Date': 'datetime'}, inplace = True)
    rtes.index.name = 'datetime'
    rtes.reset_index(inplace = True)
    print("RATES TO SAVE")
    print(rtes.to_string())
    if rtes.empty:
        print("No new data to save")
        return
    store_SQL_data_Insert_Ignore('securities_master', 'rates_timeseries', rtes)
    return rtes


if __name__ == '__main__':
    ## Save to db
    save_previous_rates_date()














# ## Does the actual fetching of the rates
# def fetch_rates_save_cache(interval = '1d', use = 'yf', return_data = False):
#     # download 3-month us treasury bills rates
#     """
#     Return timeseries of 3-month US treasury bills rates

#     Parameters
#     ----------
#     interval : str
#         Interval to resample the data to. Default is '1d'
    
#     use: str
#         Source of the data. Default is 'yf', other option is 'db'
    
#     """
#     logger.info('Saving rates timeseries to cache with fetch_rates_save_cache function')
#     global _rates_cache
#     annualized = retrieve_timeseries('^IRX',  '1960-01-01', end = datetime.datetime.today().strftime('%Y-%m-%d'), interval = interval)['close']
#     rates = yf.Ticker("^IRX")
#     annualized.index = pd.to_datetime(annualized.index)
#     annualized.index = [x.replace(minute = 30) for x in annualized.index]
#     annualized.index.name = 'Datetime'

   
#     if use == 'yf':

#         def get_data(interval = '1d'):
#             daily = annualized.apply(deannualize)
#             data = pd.DataFrame({"annualized": annualized, "daily": daily})
#             data['name'] = "^IRX"
#             data['description']=rates.info['longName']

#             if data.index.name.lower() != 'datetime':
#                 data.index.name = 'Datetime'
#             return data

#         data = get_data(interval)
#         data['annualized'] = data.annualized/100


#     elif use == 'db':
#         ## Added Datetime min Filter to reduce time taken to query

#         data = query_database('securities_master','rates_timeseries' ,"SELECT * FROM rates_timeseries WHERE yf_tick = '^IRX' AND DATETIME >= '2010-01-01'")
#         data['datetime'] = pd.to_datetime(data['datetime'])
#         data.set_index('datetime',inplace = True)
#         data.rename(columns = {'daily_rate':'daily', 'annualized_rate':'annualized', 'yf_tick': 'name'}, inplace = True)
#         data.index.name = 'Datetime'

#     # create dataframe
#     try:
#         _rates_cache =  resample(data, interval, {'daily':'last', 'annualized': 'last', 'name': 'last', 'description': 'last'})
#         if return_data:
#             return _rates_cache

#     except Exception as e:
        
#         if isinstance(e, TypeError):
#             logger.critical(f'YFinance did not return data. Rerun the function')
#             return e
#         else:
#             logger.critical(f'Error in fetching rates data: {e}')
#             return e



# def is_rates_thread_still_running():
#     return rates_cache_thread.is_alive()

# ## Save the rates data to the db
# def run_routine_rates_save():
#     h1_rates = fetch_rates_save_cache('1h', 'yf', return_data=True)
    
#     if isinstance(h1_rates, Exception) or isinstance(h1_rates, TypeError):
#         return 

    
#     h1_rates.reset_index(inplace = True)
#     h1_rates.rename(columns = {'Datetime': 'datetime', 'daily': 'daily_rate',
#                             'name':'yf_tick', 'annualized':'annualized_rate'}, inplace = True)
    
#     store_SQL_data_Insert_Ignore('securities_master', 'rates_timeseries', h1_rates)
#     logger.info('Rates data updated in db')



# ## Handles all the workflow. It doesn't fetch the rates data, it just checks if the rates cache is ready and updates it if necessary
# ## Checks if the rates cache is ready, if not, it waits for the rates thread to finish
# ## Updates the cache if the last date is not today
# def _fetch_rates():
#     """handles logic picking"""

#     global _rates_cache
#     if _rates_cache is None:
#         logger.info('Saving to cache')
#         ## In saving to cache, I want to use db data
#         print('Saving to cache from db')
#         fetch_rates_save_cache(use = 'db')

#     #Update the cache if the last date is not today
#     if _rates_cache is not None:
        
#         ## If db data is not up to date, update it db data AND update the cache using yf to save time. DB will be updated with a Thread

#         if not isinstance(_rates_cache.index, pd.DatetimeIndex):
#             _rates_cache.index = pd.to_datetime(_rates_cache.index)
        
#         print(change_to_last_busday(datetime.datetime.now()).date(), _rates_cache.index[-1].date())
#         if change_to_last_busday(datetime.datetime.now()).date() > _rates_cache.index[-1].date():
#             logger.info('Updating cache')
#             logger.info('Updating db rates data')
#             print('Updating db rates data')
#             db_update_thread = Thread(target = run_routine_rates_save)
#             db_update_thread.start()
#             fetch_rates_save_cache()
        
#         else:
#             logger.info('Cache is up to date')
    
#     return _rates_cache


# def deannualize(annual_rate, periods=365):
#     return (1 + annual_rate) ** (1/periods) - 1


# ## User facing function



# def save_previous_rates_date():
#     import yfinance as yf
#     print("Saving previous rates date")
#     rtes = yf.download('^IRX', progress=False)
#     print("DOWNLOAD COMPLETE")
#     rtes['annualized'] = rtes['Adj Close']/100
#     rtes['daily'] = (1 + rtes['annualized']) ** (1/365) - 1
#     rtes['yf_tick'] = '^IRX'
#     rtes['description'] = '13 week treasury bill'
#     rtes = rtes[rtes.index< get_risk_free_rate_helper().index.min()][['yf_tick', 'description', 'annualized', 'daily']].reset_index()
#     rtes.rename(columns = {'annualized': 'annualized_rate', 'daily': 'daily_rate', 'Date': 'datetime'}, inplace = True)
#     store_SQL_data_Insert_Ignore('securities_master', 'rates_timeseries', rtes)