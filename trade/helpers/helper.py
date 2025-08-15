## To-Do: Switch Binomial Pricing to Leisen-Reimer Formulas
import inspect
import time
import os
import backoff
from dotenv import load_dotenv
load_dotenv()
import sys
import pstats
import warnings
from typing import Union
from trade.helpers.Configuration import ConfigProxy
Configuration = ConfigProxy()
import re
from datetime import datetime
import QuantLib as ql
from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
from datetime import datetime
from trade.helpers.parse import parse_date, parse_time
import yfinance as yf
from py_vollib.black_scholes import black_scholes as bs
from py_vollib.black_scholes.greeks.numerical import delta, vega, theta, rho
from py_vollib.black_scholes_merton.implied_volatility import implied_volatility
from py_vollib.black_scholes_merton import black_scholes_merton
from py_lets_be_rational.exceptions import BelowIntrinsicException
from scipy.stats import norm
import yfinance as yf
from openbb import obb
import pandas as pd
import inspect
from datetime import datetime
from copy import deepcopy
from trade.helpers.Logging import setup_logger
from trade.helpers.types import OptionTickMetaData
from pathlib import Path
import os
from trade.helpers.exception import YFinanceEmptyData, OpenBBEmptyData
import traceback
from pandas.tseries.holiday import USFederalHolidayCalendar
import pandas_market_calendars as mcal
from trade import HOLIDAY_SET, PRICING_CONFIG
import json
from dbase.utils import add_eod_timestamp, bus_range, enforce_bus_hours
from pathlib import Path
from diskcache import Cache
from pprint import pprint, pformat
import atexit
import signal
import shortuuid

logger = setup_logger('trade.helpers.helper')

# To-Dos: 
# If still using binomial, change the r to prompt for it rather than it calling a function

option_keys = {}


def _ipython_shutdown(_callable):
    """
    Register a shutdown function to be called when the IPython kernel is shutting down.
    """
    if not callable(_callable):
        raise TypeError("The shutdown function must be callable.")
    from IPython import get_ipython
    try:
        ipython = get_ipython()
        if ipython is not None:
            ipython.events.register('shutdown', _callable)
    except ImportError as e:
        pass

    except Exception as e:
        logger.error(f"Error during IPython shutdown registration: {e}")


class Scalar:
    def __init__(self, value):
        self.value = value

    def __add__(self, other):
        return Scalar(self.value + self._get_val(other))

    def __sub__(self, other):
        return Scalar(self.value - self._get_val(other))

    def __mul__(self, other):
        return Scalar(self.value * self._get_val(other))

    def __truediv__(self, other):
        return Scalar(self.value / self._get_val(other))

    def __pow__(self, other):
        return Scalar(self.value ** self._get_val(other))

    def __repr__(self):
        return f"Scalar({self.value})"

    def __float__(self):
        return float(self.value)

    def __array__(self):
        return np.array(self.value)

    def _get_val(self, other):
        return other.value if isinstance(other, Scalar) else other



class CustomCache(Cache):
    """
    CustomCache is a dictionary-like object that stores data on disk. It is a subclass of diskcache.Cache and provides additional functionality
    """
    def __init__(self, 
                 location: str | Path = None, 
                 fname: str = None,
                 log_path: str | Path = None, 
                 clear_on_exit: bool = False, 
                 expire_days: int = 7,
                 data: dict = None,
                 **kwargs):
        """
        Important Behavior:
        1. The cache is pegged to a specific on disk data. Represented by location/fname
        2. The cache is cleared on exit if clear_on_exit is set to True. Else, it will remain populated and open. But the location of the directory 
            will be recorded in a file for later clean-up.
        
        :params location: str | Path: Folder to store the cache. If None, it will use the WORK_DIR environment variable.
        :params fname: str: Name of the cache file. Defaults to 'cache'.
        :params log_path: str | Path: Path to the log file. If None, it will use the WORK_DIR environment variable.
        :params clear_on_exit: bool: Whether to clear the cache on exit. Defaults to False.
        :params kwargs: Additional arguments to pass to the Cache constructor.

        Example usage:
        cache = CustomCache(location='/path/to/cache', fname='my_cache', log_path='/path/to/log.txt', clear_on_exit=True)
        """
        
        #1. Check dir & create cache
        fname = str(fname) if fname else shortuuid.random(length=8)
        dir = Path(location) / fname if location else Path(os.environ.get('WORK_DIR'))/'.cache'/fname
        self.dir = dir
        self.fname = fname
        self.expiry_date = (datetime.today() + relativedelta(days=expire_days)).date().strftime('%Y-%m-%d')
        self._register_location = f'{os.environ["WORK_DIR"]}/trade/helpers/clear_dirs.json'
        
        ## Avoid non path like objects
        if isinstance(log_path, (str, os.PathLike)):
            log_path = Path(log_path)
        elif log_path is None:
            log_path = Path(os.environ.get('WORK_DIR'))/'trade'/'helpers'/'cache_clear_log.txt'
        else:
            logger.error(f"log_path must be str, Path or None, not {type(log_path)}, recieved {log_path}")
            log_path = str(Path(os.environ.get('WORK_DIR'))/'trade'/'helpers'/'cache_clear_log.txt')

        self.__log_path = log_path
        os.makedirs(dir, exist_ok=True)
        
        #2. Create cache
        super().__init__(dir, **kwargs)

        #3. Check if the cache is empty
        self.clear_on_exit = clear_on_exit
        
        #4. If data is passed, load it into the cache
        if data is not None:
            if not isinstance(data, dict):
                raise ValueError("Data must be a dictionary.")
            for key, value in data.items():
                self[key] = value

    def __getstate__(self):
        """
        Custom serialization to avoid pickling the cache directory.
        """
        return dict(
            location=str(self.dir),
            fname=self.fname,
            log_path=str(self.log_path),
            clear_on_exit=self.clear_on_exit,
            expire_days=(pd.to_datetime(self.expiry_date).date() - datetime.today().date()).days,
            data=dict(self.items())
        )
    
    def __setstate__(self, state):
        """
        Custom deserialization to restore the cache state.
        """
        self.__init__(
            location=state['location'],
            fname=state['fname'],
            log_path=state['log_path'],
            clear_on_exit=state['clear_on_exit'],
            expire_days=state['expire_days'],
            data=state['data']
        )



    @property
    def clear_on_exit(self):
        return self._clear_on_exit
    
    @clear_on_exit.setter
    def clear_on_exit(self, value):
        if not isinstance(value, bool):
            raise ValueError("clear_on_exit must be a boolean value.")
        self._clear_on_exit = value
        self._install_handlers()
    
    @property
    def log_path(self):
        return self.__log_path
    
    @log_path.setter
    def log_path(self, value):
        if not isinstance(value, (str, Path)):
            raise TypeError(f"log_path must be a str or Path, not {type(value)}")
        self.__log_path = Path(value) if isinstance(value, str) else value
        if not self.__log_path.exists():
            self.__log_path.parent.mkdir(parents=True, exist_ok=True)
            self.__log_path.touch()

    @property
    def register_location(self):
        return self._register_location
    
    def _install_handlers(self):
        """
        Central place to register whatever needs doing
        depending on self._clear_on_exit.
        """
        if self._clear_on_exit:
            atexit.register(self._on_exit)
            signal.signal(signal.SIGTERM, self._on_signal)
        else:
            # just record the dir for later weekly cron clean-up
            with open(self.register_location, 'r') as f:
                json_file = json.load(f)
            with open(self.register_location, 'w') as f:
                loc = str(self.dir)
                json_file.update({loc: self.expiry_date})
                json.dump(json_file, f, default=str)

    def keys(self):
        return list(self)

    def values(self):
        return [self[k] for k in self]

    def items(self):
        return [(k, self[k]) for k in self]
    
    def update(self, other):
        if isinstance(other, dict):
            for key, value in other.items():
                self[key] = value
        elif isinstance(other, CustomCache):
            for key, value in other.items():
                self[key] = value
        else:
            raise ValueError("Other must be a dictionary or CustomCache instance.")
        
    def filter_keys(self, x):
        """
        Filter the cache keys based on a condition.
        Args:
            x (function): A function that takes a key and returns True or False.
        Returns:
            list: A list of keys that satisfy the condition.
        """
        return [key for key in self.keys() if x(key)]
    
    
    def __repr__(self):
        sample = dict(list(self.items())[:10])
        return f"<CustomCache {len(self)} entries; sample={pformat(sample)}>"
    def __str__(self):
        sample = dict(list(self.items())[:10])
        return f"<CustomCache {len(self)} entries; sample={pformat(sample)}>"
    
    def setdefault(self, key, default):
        if key not in self:
            self[key] = default
        return self[key]
    
    def _on_exit(self):
        self.clear()
        with open(f'{self.log_path}', 'a') as f:
            f.write(f"Cache cleared by AtExit at {datetime.now()}\n")

    def _on_signal(self, signum, frame):
        self.clear()
        self._on_exit()
        os.kill(os.getpid(), signum)


def check_all_days_available(x, _start, _end):
    """
    Check if all business days in the range are available in the DataFrame x.
    Args:
        x (pd.DataFrame): DataFrame with a 'Datetime' column.
        _start (str or datetime): Start date of the range.
        _end (str or datetime): End date of the range.
        
    Returns:
        bool: True if all business days in the range are available, False otherwise.
    """
    date_range = bus_range(_start, _end, freq = '1B')
    dates_available = x.Datetime
    missing_dates_second_check = [x for x in date_range if x not in pd.DatetimeIndex(dates_available)]
    return all(x in pd.DatetimeIndex(dates_available) for x in date_range)

def check_missing_dates(x, _start, _end):
    """
    Check for missing business days in the DataFrame x within the specified date range.
    Args:
        x (pd.DataFrame): DataFrame with a 'Datetime' column.
        _start (str or datetime): Start date of the range.
        _end (str or datetime): End date of the range.
    Returns:
        list: List of missing business days in the range.
    """
    if 'Datetime' not in x.columns:
        logger.warning(f"DataFrame does not contain 'Datetime' column. Will default to index")
        x['Datetime'] = x.index
    date_range = bus_range(_start, _end, freq = '1B')
    dates_available = x.Datetime
    missing_dates_second_check = [x for x in date_range if x not in pd.DatetimeIndex(dates_available)]
    x.drop(columns=['Datetime'], inplace=True, errors='ignore')
    return missing_dates_second_check

def vol_backout_errors(sigma, K, S0, T, r, q, market_price, flag):

    """Check for errors in the input parameters for the vol backout function"""
    assert isinstance(sigma, (int, float)), f"Recieved '{type(sigma)}' for sigma. Expected 'int' or 'float'"
    assert isinstance(K, (int, float)), f"Recieved '{type(K)}' for K. Expected 'int' or 'float'"
    assert isinstance(S0, (int, float)), f"Recieved '{type(S0)}' for S0. Expected 'int' or 'float'"
    assert isinstance(r, (int, float)), f"Recieved '{type(r)}' for r. Expected 'int' or 'float'"
    assert isinstance(q, (int, float)), f"Recieved '{type(q)}' for q. Expected 'int' or 'float'"
    assert isinstance(market_price, (int, float)), f"Recieved '{type(market_price)}' for market_price. Expected 'int' or 'float'"
    assert isinstance(flag, str), f"Recieved '{type(flag)}' for flag. Expected 'str'"

    if sigma <= 0:
        raise ValueError("Volatility must be positive.")
    if K <= 0:
        raise ValueError("Strike price must be positive.")
    if S0 <= 0:
        raise ValueError("Spot price must be positive.")
    if T < 0:
        raise ValueError("Time to expiration must be positive.")
    if r < 0:
        raise ValueError("Risk-free rate must be non-negative.")
    if q < 0:
        raise ValueError("Dividend yield must be non-negative.")
    if market_price <= 0:
        raise ValueError("Market price must be positive.")
    if flag not in ['c', 'p']:
        raise ValueError("Flag must be 'c' for call or 'p' for put.")
    
    if pd.isna(sigma) or pd.isna(K) or pd.isna(S0) or pd.isna(r) or pd.isna(q) or pd.isna(market_price):
        raise ValueError("Input values cannot be NaN.")

def save_vol_resolve(opt_tick, datetime, vol_resolve, agg = 'eod'):
    """Utility function to save vol_resolve to json file"""
    import os, json
    with open(f'{os.environ["WORK_DIR"]}/trade/helpers/vol_resolve_{agg}.json', 'r') as f:
        data = json.load(f)
    datetime = pd.to_datetime(datetime).strftime('%Y-%m-%d')
    data.setdefault(datetime, {})
    data[datetime][opt_tick]= {}
    data[datetime][opt_tick]['VolResolve'] = vol_resolve
    with open(f'{os.environ["WORK_DIR"]}/trade/helpers/vol_resolve_{agg}.json', 'w') as f:
        json.dump(data, f)


def import_option_keys():
    global option_keys
    import json
    with open(f'{os.environ["WORK_DIR"]}/trade/assets/option_key.json', 'rb') as f:
        option_keys = json.load(f)


def save_option_keys(key, info):
    import json
    global option_keys
    import_option_keys()
    if key not in option_keys.keys():
        option_keys[key] = info    
        with open(f'{os.environ["WORK_DIR"]}/trade/assets/option_key.json', 'w') as f:
            json.dump(option_keys, f)


def save_block_option_keys(block_option_keys):
    option_keys.update(block_option_keys)
    save_option_keys()


def get_option_specifics_from_key(key):
    try:
        return parse_option_tick(key)
    except:
        return None


def filter_inf(data):
    data = data.replace([np.inf, -np.inf], np.nan)
    return data.ffill()

def filter_zeros(data):
    data = data.replace(0, np.nan)
    return data.ffill()

@backoff.on_exception(backoff.expo, OpenBBEmptyData, max_tries=5, logger=logger)
def retrieve_timeseries(tick, 
                        start, 
                        end, 
                        interval = '1d', 
                        provider = 'yfinance', 
                        spot_type='close',
                        **kwargs):
    """
    Returns an OHLCV for provided ticker.

    Utilizes OpenBB historical api. Default provider is yfinance.
    """
    if spot_type == 'chain_price':
        df = retrieve_timeseries(tick, end =change_to_last_busday(datetime.today()).strftime('%Y-%m-%d'), 
                                    start = '1960-01-01', interval= interval, provider = provider)
        df.index = pd.to_datetime(df.index)
        df = df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
        df['close'] = df['chain_price']
        df['cum_split_from_start'] = df['split_ratio'].cumprod()
        return df
    else:
        try:
            res = obb.equity.price.historical(symbol=tick, start_date = start, end_date = end, provider=provider, interval =interval)
        except:
            raise OpenBBEmptyData(f"OpenBB raised an unexpected error for {tick} with {provider} provider. Check the logs for more details")
        
        
        ## OpenBB has an issue where if a column is all None (incases of no splits witin the date range), it doesn't return the column
        data = res.to_df()
        if 'split_ratio' not in data.columns:
            res_vs = [r.__dict__ for r in res.results]
            data = pd.DataFrame(res_vs, index = [r['date'] for r in res_vs])
            data['split_ratio'] = 0


        data.split_ratio.replace(0, 1, inplace = True)
        data['cum_split'] = data.split_ratio.cumprod()
        data['max_cum_split'] = data.cum_split.max()
        data['unadjusted_close'] = data.close * data.max_cum_split
        data['split_factor'] = data.max_cum_split / data.cum_split
        data['chain_price'] = data.close * data.split_factor
        data = data[['open', 'high', 'low', 'close', 'volume','chain_price','unadjusted_close',  'split_ratio', 'cum_split']]
        data['is_split_date'] = data['split_ratio'] != 1
        data.index = pd.to_datetime(data.index)
        ## To-Do: Add a data cleaning function to remove zeros and inf and check for other anomalies. 
        ## In the function, add a logger to log the anomalies

        if data.empty and provider == 'yfinance':
            logger.warning(f"yfinance returned empty data for {tick} is empty")
            raise YFinanceEmptyData(f"yfinance returned empty data for {tick} is empty")

        ## Fix intraday data missing 16:00:00 timestamp
        if 'h' in interval or 'm' in interval:
            if 'm' in interval:
                ## Pandas doesn't like the 'm' in the interval, so we need to replace it with 'min'. 'm' is month in pandas
                interval = interval.replace('m', 'min')
            data = enforce_bus_hours(data)
            reindex = bus_range(data.index[0], data.index[-1], interval)
            data = data.reindex(reindex, method='ffill').dropna()

            
        return data

def identify_interval(timewidth, timeframe, provider = 'default'):
    if provider == 'yfinance':
        TIMEFRAMES = {'day': 'd', 'hour': 'h', 'minute': 'm', 'week': 'W', 'month': 'M', 'quarter': 'Q'}
        assert timeframe.lower() in TIMEFRAMES.keys(), f"For '{provider}' provider timeframes, these are your options, {TIMEFRAMES.keys()}"
        return f"{str(timewidth)}{TIMEFRAMES[timeframe.lower()]}"
    
    elif provider == 'default':
        TIMEFRAMES = {'day': 'd', 'hour': 'h', 'minute': 'm', 'week': 'w', 'month': 'M', 'quarter': 'q', 'year': 'y'}
        assert timeframe.lower() in TIMEFRAMES.keys(), f"For '{provider}' provider timeframes, these are your options, {TIMEFRAMES.keys()}"
        return f"{str(timewidth)}{TIMEFRAMES[timeframe.lower()]}"
    
    
    elif provider == 'fmp':
        TIMEFRAMES = {'day': 'd', 'hour': 'h', 'minute': 'm'}
        assert timeframe.lower() in TIMEFRAMES.keys(), f"For '{provider}' provider timeframes, these are your options, {TIMEFRAMES.keys()}"
        return f"{str(timewidth)}{TIMEFRAMES[timeframe.lower()]}"
    


def identify_length(string, integer):
    TIMEFRAMES_VALUES = {'d': 1, 'w': 5, 'm': 30, 'y': 252, 'q': 91}
    assert string in TIMEFRAMES_VALUES.keys(), f'Available timeframes are {TIMEFRAMES_VALUES.keys()}, recieved "{string}"'
    return integer * TIMEFRAMES_VALUES[string]

def extract_numeric_value(timeframe_str):
    match = re.findall(r'(\d+)([a-zA-Z]+)', timeframe_str)
    integers = [int(num) for num, _ in match][0]
    strings = [str(letter) for _, letter in match][0]
    return strings, integers

def enforce_allowed_models(model: list) -> list:
    """
    Ensures that the model is in the allowed models list.
    """
    assert model in PRICING_CONFIG['AVAILABLE_PRICING_MODELS'], f"Model {model} is not in the allowed models list. Expected {PRICING_CONFIG['AVAILABLE_PRICING_MODELS']}"



def date_inbetween(date, start, end, inclusive=True):
    """
    Check if a date is within a given range.
    Args:
        date (str or datetime): The date to check.
        start (str or datetime): The start of the range.
        end (str or datetime): The end of the range.
    Returns:
        bool: True if date is within the range, False otherwise.
    """
    start, end, date = pd.to_datetime(start), pd.to_datetime(end), pd.to_datetime(date)
    if inclusive:
        return start <= date <= end
    else:
        return start < date < end

class compare_dates:
    """
    A class to compare dates with various methods.
    """
    @staticmethod
    def is_before(date1, date2):
        """ Check if date1 is before date2."""
        return pd.to_datetime(date1) < pd.to_datetime(date2)

    @staticmethod
    def is_after(date1, date2):
        """ Check if date1 is after date2."""
        return pd.to_datetime(date1) > pd.to_datetime(date2)
    
    @staticmethod
    def is_on_or_before(date1, date2):
        """ Check if date1 is on or before date2."""
        return pd.to_datetime(date1) <= pd.to_datetime(date2)
    
    @staticmethod
    def is_on_or_after(date1, date2):
        """ Check if date1 is on or after date2."""
        return pd.to_datetime(date1) >= pd.to_datetime(date2)

    @staticmethod
    def is_equal(date1, date2):
        return pd.to_datetime(date1) == pd.to_datetime(date2)
    
    @staticmethod
    def inbetween(date, start, end, inclusive=True):
        """
        Check if a date is within a given range.
        Args:
            date (str or datetime): The date to check.
            start (str or datetime): The start of the range.
            end (str or datetime): The end of the range.
        Returns:
            bool: True if date is within the range, False otherwise.
        """
        return date_inbetween(date, start, end, inclusive)



def print_cprofile_internal_time_share(_stats, top_n=20, sort_by='tottime', full_name=False):
    """
    Print top n functions by internal (self) time, with their share of total self time.
    """
    _stats = deepcopy(_stats)
    _stats.sort_stats(sort_by)
    
    all_stats = _stats.stats.items()
    total_self_time = sum(stat[2] for _, stat in all_stats) 

    top_list = sorted(all_stats, key=lambda x: x[1][2], reverse=True)[:top_n]

    print(f"{'Function':<70} {'SelfTime':>10} {'ShareOfTotal':>12}")
    print('-' * 95)

    for func, stat in top_list:
        filename, line, funcname = func
        label = f"{filename}:{line} {funcname}" if full_name else funcname
        self_time = stat[2]
        ratio = self_time / total_self_time if total_self_time else 0
        print(f"{label:<70} {self_time:>10.4f} {ratio:>12.2%}")

def print_top_cprofile_stats(_stats, top_n=20, sort_by='cumulative', full_name=False):
    """
    Display the top n functions from a cProfile stats file,
    showing cumulative time and ratio to the top function.

    :param stats: pstats.Stats object
    :param top_n: Number of functions to display
    :param sort_by: 'cumulative', 'time', etc.
    :param full_name: If True, show full path:line:function_name
    """
    _stats = deepcopy(_stats)
    _stats.sort_stats(sort_by)
    top_stats = _stats.stats.items()
    top_list = sorted(top_stats, key=lambda x: x[1][3], reverse=True)[:top_n]

    top_cum_time = top_list[0][1][3]

    # Header
    print(f"{'Function':<80} {'CumTime':>10} {'RatioToTop':>12}")
    print('-' * 105)

    for func, stat in top_list:
        filename, line, funcname = func
        cum_time = stat[3]
        ratio = cum_time / top_cum_time if top_cum_time else 0

        if full_name:
            label = f"{filename}:{line} {funcname}"
        else:
            label = funcname

        print(f"{label:<80} {cum_time:>10.4f} {ratio:>12.2f}")


def find_split_dates_within_range(tick: str, 
                                 start: str, 
                                 end: str):
    """
    Find split dates within a range
    params:
    tick: str, stock ticker
    start: str, start date
    end: str, end date

    return:
    list of split dates within the range
    
    
    """
    data = retrieve_timeseries(tick, '1900-01-01', end, '1d')
    data = data[data.index.date >= pd.to_datetime(start).date()]
    return list(data[data['is_split_date'] == True]['split_ratio'].to_frame().itertuples(name = None))



def printmd(string):
    from IPython.display import Markdown, display
    display(Markdown(string))

def copy_doc_from(func):
    def wrapper(method):
        method.__doc__ = func.__doc__
        return method
    return wrapper



def contains_time_format(date_str: str) -> bool:
    try:
        datetime.strptime(date_str, '%H:%M:%S')
        return True
    except ValueError:
        return False

def time_distance_helper(exp: str, strt: str = None) -> float:
    """
    Calculate the time distance between two dates in years.
    """
    if strt is None:
        strt = datetime.today()
    
    exp = pd.to_datetime(exp)
    exp = exp.replace(hour = 16, minute = 0, second = 0, microsecond = 0,)
    parsed_dte, start_date = pd.to_datetime(exp), pd.to_datetime(strt)
    if start_date.hour == 0 and start_date.minute == 0 and start_date.second == 0:
        start_date = start_date.replace(hour=16, minute=0, second=0, microsecond=0)
    days = (parsed_dte - start_date).total_seconds()

    T = days/(365.25*24*3600)
    return T


def binomial(K: Union[int, float], exp_date: str, sigma: float, r: float = None, N: int = 100, S0: Union[int, float, None] = None, y: float = None, tick: str = None,  opttype='P', start: str = None) -> float:
    '''
    Returns the price of an american option

        Parameters:
            K: Strike price
            exp_date: Expiration date
            S0: Spot at current time (Optional)
            r: Risk free rate (Optional)
            N: Number of steps to use in the calculation (Optional)
            y: Dividend yield (Optional)
            Sigma: Implied Volatility of the option
            opttype: Option type ie put or call (Defaults to "P")
            start: Start date of the pricing model. If nothing is passed, defaults to today. If initiated within a context and nothing is passed, defaults to context start date (Optional)
    '''
    if start is None:
        if Configuration.start_date is not None:
            start = Configuration.start_date
        else:
            today = datetime.today()
            start = today.strftime("%Y-%m-%d")
    if tick is not None:
        stock = Stock(tick)
        if y is None:
            y = stock.div_yield()
        if S0 is None:
            S0 = stock.prev_close()
            S0 = S0.close
    else:
        if y is None:
            y = 0
    if r is None:
        rates = 0.005
        r = rates.iloc[len(rates)-1, 0]/100

    # Create a formula to get implied vol
    T = time_distance_helper(exp_date, start)
    dt = T/N
    nu = r - 0.5*sigma**2
    u = np.exp(nu*dt + sigma*np.sqrt(dt))
    d = np.exp(nu*dt - sigma*np.sqrt(dt))
    q = (np.exp((r-y)*dt) - d) / (u-d)
    disc = np.exp(-(r-y)*dt)
    opttype = opttype.upper()

    # initialise stock prices at maturity (calculating final stock values at the last nodes)
    S = np.zeros(N+1)
    for j in range(0, N+1):
        S[j] = S0 * u**j * d**(N-j)

    # option payoff, (calculating the payoffs at each final node.)
    C = np.zeros(N+1)
    for j in range(0, N+1):
        if opttype == 'P':
            C[j] = max(0, K - S[j])
        else:
            C[j] = max(0, S[j] - K)

    # backward recursion through the tree
    for i in np.arange(N-1, -1, -1):
        for j in range(0, i+1):
            S = S0 * u**j * d**(i-j)
            C[j] = disc * (q*C[j+1] + (1-q)*C[j])
            if opttype == 'P':
                C[j] = max(C[j], K - S)
            else:
                C[j] = max(C[j], S - K)

    return C[0]


def implied_vol_bs_helper(S0, K, T, r, market_price, flag='c', tol=1e-3, exp_date='2024-03-08'):
    """Compute the implied volatility of a European Option
        S0: initial stock price
        K:  strike price
        T:  maturity
        r:  risk-free rate
        market_price: market observed price
        tol: user choosen tolerance
    """
    max_iter = 200  # max number of iterations
    vol_old = 0.5  # initial guess
    count = 0
    for k in range(max_iter):
        bs_price = bs(flag, S0, K, T, r, vol_old)
        Cprime = vega(flag, S0, K, T, r, vol_old)*100
        C = bs_price - market_price
        vol_new = vol_old - C/Cprime
        bs_new = bs(flag, S0, K, T, r, vol_new)
        if (abs((vol_old - vol_new)/vol_old)) < tol:
            break
        vol_old = vol_new
    implied_vol = vol_old

    return implied_vol


def implied_vol_bt(S0, K, r, market_price,exp_date: str, flag='c', tol=0.000000000001,  y=None, start = None, break_time = 60):
    """Compute the implied volatility of an American Option
        S0: initial stock price
        K:  strike price
        r:  risk-free rate
        y:  Dividend yield
        market_price: market observed price
        tol: user choosen tolerance
    """
    if pd.to_datetime(exp_date) == pd.to_datetime(start):
        logger.warning(f"Expiration date {exp_date} is the same as start date {start}. Include HH:MM:SS in the start date, to prevent pricing EOD")

    T = time_distance_helper(exp_date, start)
    max_iter = 200  # max number of iterations
    vol_old = 0.2  # initial guess
    count = 0
    vol_backout_errors(vol_old, K, S0, T, r, y, market_price, flag)
    start_time = time.time()
    for k in range(max_iter):
        current_time = time.time()
        if current_time - start_time > break_time:
            logger.error(f"Binomial Implied vol took too long to calculate for {S0}, {K}, {r}, {market_price}, {exp_date}, {flag}, total time: {current_time - start_time}")
            return 0.0
        bs_price = binomial(
            K=K, exp_date=exp_date, S0=S0,  r=r, sigma=vol_old, opttype=flag, y=y, start = start)

        Cprime = vega(flag, S0, K, T, r, vol_old)*100
        C = bs_price - market_price
        vol_new = vol_old - C/Cprime
        vol_new = np.clip(vol_new, 0.0001, 5)
        if (abs((vol_old - vol_new)/vol_old)) < tol:
            break
        vol_old = vol_new
        count += 1
    implied_vol = vol_old
    if pd.isna(implied_vol) or implied_vol == 0.0:
        logger.warning(f"Binomial Implied vol is NaN for {S0}, {K}, {r}, {market_price}, Exp: {exp_date},  Flag: {flag}, Start: {start}")
        return 0.0
    return implied_vol


def d1_helper(S, K, r, T, sigma, q):
    return (np.log(S / K) + ((r-q) + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def d2_helper(S, K, r, T, sigma, q):
    return d1_helper(S, K, r, T, sigma, q) - sigma * np.sqrt(T)


def volga(S, K, r, T, sigma, flag, q):
    d1 = d1_helper(S, K, r, T, sigma, q)
    d2 = d2_helper(S, K, r, T, sigma, q)
    flag = flag.upper()
    if sigma <= 0:
        raise ValueError("Volatility must be positive.")
    if flag == 'C' or flag == 'P':
        if flag.upper() == 'C':
            volga = (d1*d2*S*np.exp(-q*T)*norm.cdf(d1)*np.sqrt(T))/sigma
        else:
            volga = (d1*d2*S*np.exp(-q*T)*norm.cdf(-d1)*np.sqrt(T))/sigma
    else:
        raise ValueError("Invalid Option Type. Only 'C' for Call and 'P' for Put are available.")
    return volga


def vanna(S, K, r, T, sigma, flag, q):
    d1 = d1_helper(S, K, r, T, sigma, q)
    d2 = d2_helper(S, K, r, T, sigma, q)
    flag = flag.upper()
    if sigma <= 0:
        raise ValueError("Volatility must be positive.")
    flag = flag.upper()
    if flag == 'C' or flag == 'P':
        if flag.upper() == 'C':
            vanna = -(d2 * np.exp(-q*T)*norm.cdf(d1))/sigma
        else:
            vanna = -(d2 * np.exp(-q*T)*norm.cdf(-d1))/sigma
    else:
        raise ValueError("Invalid Option Type. Only 'C' for Call and 'P' for Put are available.")
    return vanna



import QuantLib as ql
from datetime import datetime




def optionPV_helper(
    spot_price: float,
    strike_price: float | int,
    exp_date: str | datetime, 
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
    putcall: str,
    settlement_date_str: str,
    model: str = 'bs'
):

    """
    Price an American option using QuantLib Engine

    params:
    _________

    spot_price: Underlying Spot
    strike_price: Options Strike price
    exp_date: Options expiration date
    risk_free_rate: Prevailing discount rate, annualized and expressed as 0.01 for 1%
    volatility: Underlying Volatility
    settlement_date_str: pricing date
    model: Preferred pricing method. 
        Available options:
        'bsm': Black Scholes Model
        'bt': Binomial Tree Model
        'mcs': Monte Carlo Simulation

    Returns: 
    ____________

    PV (float): Option present value

    """
    enforce_allowed_models(model)
    try:
        # Option Parameters


        if model == 'binomial':
            binomial_price = binomial(
                K = strike_price,
                exp_date = exp_date,
                sigma = volatility,
                r = risk_free_rate,
                S0=spot_price,
                y = dividend_yield,
                opttype=putcall,
                start = settlement_date_str
            )
            return binomial_price

        spot_price = spot_price # Current stock price
        strike_price = strike_price  # Option strike price
        maturity_date_str = exp_date  # Option maturity date as a string
        risk_free_rate = risk_free_rate  # Risk-free interest rate
        volatility = volatility  # Volatility of the underlying asset
        dividend_yield = dividend_yield # Continuous dividend yield

        # Convert string date to QuantLib Date
        maturity_date = ql.Date(pd.to_datetime(maturity_date_str).day,
                                pd.to_datetime(maturity_date_str).month,
                                pd.to_datetime(maturity_date_str).year)

        # QuantLib Settings
        calendar = ql.UnitedStates(ql.UnitedStates.NYSE)  # U.S. market calendar (NYSE)
        day_count = ql.Actual365Fixed()
        settlement_date = ql.Date(pd.to_datetime(settlement_date_str).day,
                                pd.to_datetime(settlement_date_str).month,
                                pd.to_datetime(settlement_date_str).year)

        ql.Settings.instance().evaluationDate = settlement_date

        # Construct the payoff and the exercise objects
        if putcall.upper() == 'P':
            right = ql.Option.Put
        elif putcall.upper() == 'C':
            right = ql.Option.Call
        else:
            raise ValueError(f"Recieved '{putcall}' for putcall. Expected 'P' or 'C'")
        payoff = ql.PlainVanillaPayoff(right, strike_price)
        exercise = ql.AmericanExercise(settlement_date, maturity_date)

        # Market data
        spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price))
        dividend_ts = ql.YieldTermStructureHandle(ql.FlatForward(settlement_date, dividend_yield, day_count))
        risk_free_ts = ql.YieldTermStructureHandle(ql.FlatForward(settlement_date, risk_free_rate, day_count))
        volatility_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(settlement_date, calendar, volatility, day_count))

        # Black-Scholes-Merton Process (with dividend yield)
        bsm_process = ql.BlackScholesMertonProcess(spot_handle, dividend_ts, risk_free_ts, volatility_ts)


        
        if model == 'mcs':
            # Monte Carlo Pricing (Longstaff-Schwartz)
            monte_carlo_engine = ql.MCAmericanEngine(bsm_process, "PseudoRandom", timeSteps=250, requiredSamples=10000)
            american_option = ql.VanillaOption(payoff, exercise)
            american_option.setPricingEngine(monte_carlo_engine)
            monte_carlo_price = american_option.NPV()
            return monte_carlo_price
        
        elif model == 'bs':
            # Black-Scholes Pricing (Treated as European for comparison)
            european_exercise = ql.EuropeanExercise(maturity_date)
            european_option = ql.VanillaOption(payoff, european_exercise)
            black_scholes_engine = ql.AnalyticEuropeanEngine(bsm_process)
            european_option.setPricingEngine(black_scholes_engine)
            black_scholes_price = european_option.NPV()
            return black_scholes_price
    except Exception as e:
        logger.info('')
        logger.info('"optionPV_helper" raised the below error')
        logger.info(e)
        logger.info(f'Kwargs: {locals()}')
        return 0.0



def pad_string(input_value):
    # Convert float to string and remove the decimal point if needed
    if isinstance(input_value, float):
        input_value = str(input_value).replace('.', '')

    # Convert to string and pad with leading zeros to ensure length of 8
    padded_string = str(input_value).zfill(6)
    
    return padded_string


def IV_handler(*args, **kwargs):
    """
    Calculate the Black-Scholes-Merton implied volatility.

    :param S: underlying asset price
    :type S: float
    :param K: strike price
    :type K: float
    :param sigma: annualized standard deviation, or volatility
    :type sigma: float
    :param t: time to expiration in years
    :type t: float
    :param r: risk-free interest rate
    :type r: float
    :param q: annualized continuous dividend rate
    :type q: float 
    :param flag: 'c' or 'p' for call or put.
    :type flag: str

    >>> S = 100
    >>> K = 100
    >>> sigma = .2
    >>> r = .01
    >>> flag = 'c'
    >>> t = .5
    >>> q = 0

    >>> price = black_scholes_merton(flag, S, K, t, r, sigma, q)
    >>> iv = implied_volatility(price, S, K, t, r, q, flag)

    >>> expected_price = 5.87602423383
    >>> expected_iv = 0.2

    >>> abs(expected_price - price) < 0.00001
    True
    >>> abs(expected_iv - iv) < 0.00001
    
    """

    keys = ['price', 'S', 'K', 't', 'r', 'q', 'flag']
    if args:
        extra_kwargs = {k: v for k, v in zip(keys, args)}
        kwargs.update(extra_kwargs)
    try:
        kwargs['flag'] = kwargs['flag'].lower()
        iv = implied_volatility(**kwargs)

        if np.isinf(iv):
            iv = 0
        return iv
    except (BelowIntrinsicException, ZeroDivisionError) as e:
        ## Add AboveMaximumException
        logger.warning('')
        logger.warning('"implied_volatility" raised the below error')
        logger.warning(e)
        logger.warning(f'Kwargs: {kwargs}')
        return 0.0
    
    except Exception as j:
        logger.warning('')
        logger.warning('"implied_volatility" unrelated error')
        logger.warning(j)
        logger.warning(f'Kwargs: {kwargs}')
        return 0.0
        



def binomial_implied_vol(price, S, K, r, exp_date, option_type, pricing_date, dividend_yield):
    """
    Calculate the implied volatility of an option using the binomial tree model.

    :param price: option price
    :type price: float
    :param S: underlying asset price
    :type S: float
    :param K: strike price
    :type K: float
    :param r: risk-free interest rate
    :type r: float
    :param exp_date: Expiration date
    :type exp_date: str
    :param option_type: 'c' or 'p' for call or put
    :type option_type: str
    :param pricing_date: Pricing date
    :type pricing_date: str
    :param dividend_yield: annualized continuous dividend rate
    :type dividend_yield: float

    >>> price = 5.87602423383
    >>> S = 100
    >>> K = 100
    >>> r = .01
    >>> option_type = 'c'
    >>> exp_date = '2024-03-08'
    >>> pricing_date = '2024-03-08'
    >>> dividend_yield = 0

    >>> iv = binomial_implied_vol(price, S, K, r, exp_date, option_type, pricing_date, dividend_yield)

    
    """
    kwargs = {
        'price': price,
        'S': S,
        'K': K,
        'r': r,
        'T': exp_date,
        'option_type': option_type,
        'pricing_date': pricing_date,
        'dividend_yield': dividend_yield
    }
    try:
        if price <= 0:
            logger.warning('Market price is less than or equal to 0')
            return 0.0
        
        return implied_vol_bt(
        S0 = S,
        K = K,
        exp_date = exp_date,
        r = r,
        y = dividend_yield,
        market_price=price,
        flag = option_type.lower(),
        start = pricing_date
        )


    except Exception as e:
        logger.warning('')
        logger.warning('"binomial_implied_vol" raised the below error')
        logger.warning(e)
        logger.warning(f"Traceback: {traceback.format_exc()}")
        logger.warning(f'Kwargs: {kwargs}')
        raise e
        return 0.0

def generate_option_tick(symbol, right, exp, strike):
    assert right.upper() in ['P', 'C'], f"Recieved '{right}' for right. Expected 'P' or 'C'"
    assert isinstance(exp, str), f"Recieved '{type(exp)}' for exp. Expected 'str'" 
    assert isinstance(strike, ( float)), f"Recieved '{type(strike)}' for strike. Expected 'float'"
    
    tick_date = pd.to_datetime(exp).strftime('%Y%m%d')
    if str(strike)[-1] == '0':
        strike = int(strike)
    else:
        strike = float(strike)
    
    key = symbol.upper() + tick_date + pad_string(strike) +right.upper()
    return key


def parse_option_tick(tick : str):
    """
    Parse the option tick into its components.
    returns a dictionary with the following keys
    ticker: str
    put_call: str C|P
    exp_date: str
    strike: float
    """
    # Regex pattern to extract components
    pattern = r"([A-Za-z]+)(\d{8})([CP])(\d+(\.\d+)?)"
    match = re.match(pattern, tick)
    
    if not match:
        raise ValueError(f"Invalid option string format, got: {tick}")
    
    # Extract components from the regex groups
    ticker = match.group(1)
    exp_date_raw = match.group(2)
    put_call = match.group(3)
    strike = float(match.group(4))
    
    # Convert the expiration date to the desired format
    exp_date = datetime.strptime(exp_date_raw, "%Y%m%d").strftime("%Y-%m-%d")
    
    # Construct and return the dictionary
    return {
        "ticker": ticker,
        "put_call": put_call,
        "exp_date": exp_date,
        "strike": strike
    }


def generate_option_tick_new(symbol, right, exp, strike) -> str:
    from datetime import datetime
    assert right.upper() in ['P', 'C'], f"Recieved '{right}' for right. Expected 'P' or 'C'"
    assert isinstance(exp, (str, datetime)), f"Recieved '{type(exp)}' for exp. Expected 'str'" 
    assert isinstance(strike, ( float)), f"Recieved '{type(strike)}' for strike. Expected 'float'"
    
    tick_date = pd.to_datetime(exp).strftime('%Y%m%d')
    if str(strike)[-1] == '0':
        strike = int(strike)
    else:
        strike = float(strike)
    
    key = symbol.upper() + tick_date + right.upper() + f'{strike}' 
    return key

def wait_for_response(wait_time, condition_func, interval):

    ## Can use time.time to ensure it is not counting (meaning not taking func time into consideration)
    ## This is better to ensure if it at least reaches 15 secs it ends, rather than 15 secs + loop of time to run 
    ## the func call
    elapsed_time = 0
    while elapsed_time < wait_time:
        if condition_func():
            return
        time.sleep(interval)
        elapsed_time += 1

def is_busday(date):
    return bool(len(pd.bdate_range(date, date)))


def is_USholiday(date):
    """
    Returns True if the date is a US holiday, False otherwise
    """

    # import holidays
    import pandas_market_calendars as mcal
    date = pd.to_datetime(date)
    return date.date().strftime('%Y-%m-%d') in HOLIDAY_SET


def change_to_last_busday(end, offset = 1):
    from pandas.tseries.offsets import BDay
    
    #Enfore time is passed


    if not isinstance(end, str):
        end = end.strftime('%Y-%m-%d %H:%M:%S')

    if pd.to_datetime(end).time() == pd.Timestamp('00:00:00').time():
        end = end + ' 16:00:00' 
    
    ## Make End Comparison Busday
    isBiz = is_busday(end)
    while not isBiz:

        end_dt = pd.to_datetime(end)
        end = (end_dt - BDay( offset)).strftime('%Y-%m-%d %H:%M:%S')
        isBiz = bool(len(pd.bdate_range(end, end)))

    ## Make End Comparison prev day if before 9:30
    if pd.Timestamp(end).time() <pd.Timestamp('9:30').time():
        end = pd.to_datetime(end)-BDay(offset)

    # Make End Comparison prev day if holiday
    while is_USholiday(end):
        end_dt = pd.to_datetime(end)
        end = (end_dt - BDay(offset)).strftime('%Y-%m-%d %H:%M:%S')

    return pd.to_datetime(end)

def is_class_method(cls, obj):
    """
    Check if an object is a method of a class.

    Params:
    cls: The class to check.
    obj: The object to check.

    Returns:
    bool: True if the object is a method of the class, False otherwise.
    """
    if inspect.isroutine(obj):
        for name, member in inspect.getmembers(cls):
            if member is obj:
                return True
    return False
