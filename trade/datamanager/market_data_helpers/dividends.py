from datetime import datetime, timedelta
from openbb import obb # noqa
import pandas as pd
from trade.optionlib.config.defaults import (
    OPTION_TIMESERIES_START_DATE,  # noqa
)
from trade.helpers.Logging import setup_logger
from trade.helpers.helper import CustomCache
from dataclasses import dataclass
from trade.datamanager.vars import get_dm_gen_path, get_enable_caching
from trade.optionlib.assets.dividend import infer_frequency, FREQ_MAP
from trade.datamanager.utils.logging import get_logging_level, register_to_factor_list
from trade.datamanager.config import OptionDataConfig

from trade.datamanager.utils.date import business_day_grid
from trade.helpers.helper import to_datetime

logger = setup_logger("trade.datamanager.market_data_helpers.dividends", stream_log_level=get_logging_level())
register_to_factor_list("trade.datamanager.market_data_helpers.dividends")

## Earliest date for the daily dividend B-day grid (pre-IPO / pre-dividend rows are zero).
DIVIDEND_DAILY_GRID_START = "1999-01-01"


@dataclass
class SavedDividendsResult:
    symbol: str
    historicals: pd.Series
    resampled_timeseries: pd.Series
    last_updated: datetime


## Cache has to be in memory. Incase dividends update on another date
DIVIDEND_CACHE = CustomCache(
    location=get_dm_gen_path().as_posix(), fname="discrete_dividends_timeseries", clear_on_exit=False, expire_days=365
)


def resample_dividends_to_daily(div_series: pd.Series, buffer: int = 30) -> pd.Series:
    """Resample dividend series to daily business-day frequency with forward fill.

    Sparse ex-dividend prints are resampled and forward-filled through today (or
    zero-filled when dividends appear discontinued). The series is then aligned
    to ``business_day_grid`` from ``DIVIDEND_DAILY_GRID_START``; dates before the
    first observed dividend are set to zero.

    Args:
        div_series: Historical dividend amounts indexed by ex-dividend date.
        buffer: Extra day buffer added to inferred payment cadence for discontinuation check.

    Returns:
        Daily dividend amounts on the shared B-day grid.
    """
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
    first_div_date = pd.Timestamp(div_series.dropna().index[0]).normalize()
    today = datetime.now()
    days_since_last_div = (today - last_div_date).days

    ## Add additional days to ffill into
    resampled = resampled.reindex(pd.date_range(start=resampled.index[0], end=today, freq="1b"))
    dividends_discontinued = days_since_last_div > freq_days
    if dividends_discontinued:
        logger.info("Filling with zeros as dividends seem to be discontinued.")
        resampled.loc[last_div_date + timedelta(days=1) : today] = 0.0
    else:
        resampled = resampled.ffill()

    ## Align to shared B-day grid from 1999; pre-first-dividend dates are zero.
    grid = pd.DatetimeIndex(
        business_day_grid(DIVIDEND_DAILY_GRID_START, to_datetime(today).date())
    )
    resampled = resampled.reindex(grid)
    prior_mask = resampled.index.normalize() < first_div_date
    resampled.loc[prior_mask] = 0.0

    if dividends_discontinued:
        after_last_mask = resampled.index.normalize() > pd.Timestamp(last_div_date).normalize()
        resampled.loc[after_last_mask] = resampled.loc[after_last_mask].fillna(0.0)
    else:
        resampled = resampled.ffill()

    resampled.loc[prior_mask] = 0.0

    resampled.index = pd.to_datetime(resampled.index, format="%Y-%m-%d")
    resampled.name = "dividend_amount"
    resampled.index.name = "datetime"
    resampled.sort_index(inplace=True)
    return resampled


def get_div_schedule(
    ticker: str,
    filter_specials: bool = None,
) -> pd.Series:
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
    if filter_specials is None:
        filter_specials = OptionDataConfig().filter_out_special_dividends

    def _fetch_from_source() -> SavedDividendsResult:
        try:
            div_history = obb.equity.fundamental.dividends(symbol=ticker, provider="yfinance").to_df()
            div_history.set_index("ex_dividend_date", inplace=True)
            div_history["amount"] = div_history["amount"].astype(float)
            div_history.index = pd.to_datetime(div_history.index)
            return SavedDividendsResult(
                symbol=ticker,
                historicals=div_history["amount"],
                resampled_timeseries=None,
                last_updated=datetime.now(),
            )
        except Exception as e:  # noqa
            div_history = pd.DataFrame(
                {"amount": [0]}, index=pd.bdate_range(start="2001-01-01", end=datetime.now(), freq="1Q")
            )
            return SavedDividendsResult(
                symbol=ticker,
                historicals=div_history["amount"],
                resampled_timeseries=None,
                last_updated=datetime.now(),
            )

    key = (ticker, filter_specials)
    if not get_enable_caching():
        dividends_data = _fetch_from_source()
    elif key not in DIVIDEND_CACHE:
        dividends_data = _fetch_from_source()
        DIVIDEND_CACHE[key] = dividends_data

    else:
        logger.info(f"Ticker {ticker} found in dividend cache.")
        dividends_data: SavedDividendsResult = DIVIDEND_CACHE[key]
        # Check if we need to refresh (if last_updated > 7 days)
        if (datetime.now() - dividends_data.last_updated).days > 7:
            del DIVIDEND_CACHE[key]
            return get_div_schedule(ticker, filter_specials=filter_specials)

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
