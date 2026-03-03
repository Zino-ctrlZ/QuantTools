from datetime import datetime, timedelta
from openbb import obb
import pandas as pd
from trade.optionlib.config.defaults import (
    OPTION_TIMESERIES_START_DATE, # noqa
)
from trade.helpers.Logging import setup_logger
from trade.helpers.helper import CustomCache
from dataclasses import dataclass
from trade.datamanager.vars import DM_GEN_PATH
from trade.optionlib.assets.dividend import infer_frequency, FREQ_MAP
from trade.datamanager.utils.logging import get_logging_level, register_to_factor_list
from trade.datamanager.config import OptionDataConfig
logger = setup_logger("trade.datamanager.market_data_helpers.dividends", stream_log_level=get_logging_level())
register_to_factor_list("trade.datamanager.market_data_helpers.dividends")


@dataclass
class SavedDividendsResult:
    symbol: str
    historicals: pd.Series
    resampled_timeseries: pd.Series
    last_updated: datetime


## Cache has to be in memory. Incase dividends update on another date
DIVIDEND_CACHE = CustomCache(
    location=DM_GEN_PATH, fname="discrete_dividends_timeseries", clear_on_exit=False, expire_days=365
)
def resample_dividends_to_daily(div_series: pd.Series, buffer: int = 30) -> pd.Series:
    """Resample dividend series to daily frequency with forward fill."""

    freq = infer_frequency(div_series)
    if freq is None:
        raise ValueError("Could not infer frequency.")
    freq_days = FREQ_MAP[freq] * 30  # Approximate to days
    freq_days += buffer

    ## First, resample to 1b (daily business days)
    resampled = div_series.resample("1b").ffill()

    ## Next, the resampled is clearly missing last dividends to today or end_date,
    ## SO we will forward fill the last known dividend to today. But with some rules.
    ## There are cases where dividends were discontinued, so we will only forward fill if the last known dividend date - today is less than freq_days
    ## If not we fill with zeros
    last_div_date = div_series.dropna().index[-1]
    today = datetime.now()
    days_since_last_div = (today - last_div_date).days

    ## Add additional days to ffill into
    resampled = resampled.reindex(pd.date_range(start=resampled.index[0], end=today, freq="1b"))
    if days_since_last_div <= freq_days:
        resampled = resampled.ffill()
    else:
        logger.info("Filling with zeros as dividends seem to be discontinued.")
        resampled.loc[last_div_date + timedelta(days=1) : today] = 0.0
    resampled.index = pd.to_datetime(resampled.index, format="%Y-%m-%d")
    resampled.name = "dividend_amount"
    resampled.index.name = "datetime"
    resampled.sort_index(inplace=True)  
    return resampled

def get_div_schedule(ticker: str):
    """
    Fetch the dividend schedule for a given ticker.
    If the ticker is not in the cache, it fetches the data from yfinance and caches it.
    If the ticker is in the cache, it retrieves the data from the cache.
    If filter_specials is True, it filters out dividends >= 7.5.
    Returns a DataFrame with the dividend schedule.
    """

    ## We're going to use a multi-level dividend retrieval. CustomCache is on disk cache
    ## 1. We first check if the symbol is in the on disk DIVIDEND_CACHE
    ## 2. If not, we fetch from yfinance via openbb and store in DIVIDEND_CACHE and save with last_updated
    ## 3. If in cache, we retrieve from cache, but still check last_updated.
    ##    We will use a weekly update policy to refresh dividends
    ## 4. Return the dividend schedule DataFrame

    # Check if ticker is in cache
    filter_specials = OptionDataConfig().filter_out_special_dividends
    key = (ticker, filter_specials)
    if key not in DIVIDEND_CACHE:
        try:
            div_history = obb.equity.fundamental.dividends(symbol=ticker, provider="yfinance").to_df()
            div_history.set_index("ex_dividend_date", inplace=True)
            div_history["amount"] = div_history["amount"].astype(float)
            div_history.index = pd.to_datetime(div_history.index)
            dividends_data = SavedDividendsResult(
                symbol=ticker,
                historicals=div_history["amount"],
                resampled_timeseries=None,
                last_updated=datetime.now(),
            )
        except Exception as e:  # noqa
            div_history = pd.DataFrame(
                {"amount": [0]}, index=pd.bdate_range(start="2001-01-01", end=datetime.now(), freq="1Q")
            )
            dividends_data = SavedDividendsResult(
                symbol=ticker,
                historicals=div_history["amount"],
                resampled_timeseries=None,
                last_updated=datetime.now(),
            )
        DIVIDEND_CACHE[key] = dividends_data

    else:
        logger.info(f"Ticker {ticker} found in dividend cache.")
        dividends_data: SavedDividendsResult = DIVIDEND_CACHE[key]
        # Check if we need to refresh (if last_updated > 7 days)
        if (datetime.now() - dividends_data.last_updated).days > 7:
            del DIVIDEND_CACHE[key]
            return get_div_schedule(ticker)

    # Filter out dividends >= 7.5
    if filter_specials:
        dividends_data.historicals = dividends_data.historicals[dividends_data.historicals < 7.5]

    return dividends_data.historicals.sort_index()

def get_daily_dividends_timeseries(ticker, start, end):
    """
    Get daily resampled dividend timeseries for a given ticker between start and end dates.
    This function retrieves the dividend schedule, resamples it to daily frequency, and filters it to the specified date range.
    Returns a pd.Series with daily dividend amounts.
    """

    # Else we fallthrough to refetching the schedule
    div_series = get_div_schedule(ticker)
    daily_div_series = resample_dividends_to_daily(div_series)
    daily_div_series = daily_div_series[start:end]
    return daily_div_series
