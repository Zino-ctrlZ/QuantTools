from dotenv import load_dotenv
import sys
import os
load_dotenv()
sys.path.append(
    os.environ.get('DBASE_DIR'))

from dbase.database.SQLHelpers import query_database
from dbase.DataAPI.ThetaData import resample
import yfinance as yf
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# de-annualize yearly interest rates




def deannualize(annual_rate, periods=365):
    return (1 + annual_rate) ** (1/periods) - 1


def get_risk_free_rate_helper(interval = '1d', use = 'yf'):
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
    annualized = yf.download("^IRX", progress=False, interval=interval)["Adj Close"]
    rates = yf.Ticker("^IRX")
    
    if use == 'yf':
        def get_data(interval = '1d'):
            daily = annualized.apply(deannualize)
            data = pd.DataFrame({"annualized": annualized, "daily": daily})
            data['name'] = "^IRX"
            data['description']=rates.info['longName']

            try:
                data.index = data.index.tz_convert('America/New_York')
                data.index = data.index.tz_localize(None)
                data = data.reset_index()
                data.drop_duplicates(inplace = True)
                data.set_index('Datetime',inplace = True)
            except:
                pass

            if data.index.name.lower() != 'datetime':
                data.index.name = 'Datetime'
            return data

        data = get_data(interval)
        data['annualized'] = data.annualized/100
    
    elif use == 'db':
        data = query_database('securities_master','rates_timeseries' ,"SELECT * FROM rates_timeseries WHERE yf_tick = '^IRX'")
        data['datetime'] = pd.to_datetime(data['datetime'])
        data.set_index('datetime',inplace = True)
        data.rename(columns = {'daily_rate':'daily', 'annualized_rate':'annualized', 'yf_tick': 'name'}, inplace = True)
        data.index.name = 'Datetime'


    # create dataframe
    return resample(data, interval, {'daily':'last', 'annualized': 'last', 'name': 'last', 'description': 'last'})
# Make risk free rate a stand alone file, rates related maybe?
