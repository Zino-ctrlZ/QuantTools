from datetime import datetime
from openbb import obb
import pandas as pd
from ..config.defaults import(
    OPTION_TIMESERIES_START_DATE,
)
from trade.helpers.helper import (
    CustomCache
)
from trade.helpers.Logging import setup_logger
logger = setup_logger('trade.optionlib.utils.market_data')

DIVIDEND_CACHE = CustomCache(
    location = '/Users/chiemelienwanisobi/cloned_repos/QuantTools/.cache',
    fname='dividend_cache',
    clear_on_exit=True,)


def get_div_schedule(ticker, filter_specials=True):
    """
    Fetch the dividend schedule for a given ticker.
    If the ticker is not in the cache, it fetches the data from yfinance and caches it.
    If the ticker is in the cache, it retrieves the data from the cache.
    If filter_specials is True, it filters out dividends >= 7.5.
    Returns a DataFrame with the dividend schedule.
    """
    if ticker not in DIVIDEND_CACHE:
        try:
            div_history = obb.equity.fundamental.dividends(symbol=ticker, provider='yfinance').to_df()
            div_history.set_index('ex_dividend_date', inplace = True)
            DIVIDEND_CACHE.set(ticker, div_history)

            div_history['amount'] = div_history['amount'].astype(float)
            div_history.index = pd.to_datetime(div_history.index)
        except Exception as e:
            logger.error(f"Error fetching dividend schedule for {ticker}: {e}")
            div_history = pd.DataFrame({'amount':[0]}, index = pd.bdate_range(start=OPTION_TIMESERIES_START_DATE, end=datetime.now(), freq='1Q'))
        DIVIDEND_CACHE[ticker] = div_history
    
    else:
        div_history = DIVIDEND_CACHE[ticker]
    
    # Filter out dividends >= 7.5
    if filter_specials:
        div_history = div_history[div_history['amount'] < 7.5]
        
    return div_history.sort_index()