from dotenv import load_dotenv
import sys
import os

load_dotenv()

from dbase.database.SQLHelpers import *
from dbase.DataAPI.ThetaData import *
import datetime
import yfinance as yf
import pandas as pd
import warnings
from trade.helpers.helper import change_to_last_busday, setup_logger, ny_now
from threading import Thread
import logging
from pandas.tseries.offsets import BDay
import time

warnings.filterwarnings("ignore")
logger = setup_logger("rates")
rates_thread_logger = setup_logger("rates_thread", stream_log_level=logging.INFO)

## To-do: Check why rates data doesn't start from far back

## FYI: A thread saves _rates_cache. This is necessary to avoid blocking the main thread and makes sure the data is ready when needed.

## Rates cache variable
_rates_cache = None
DAILY_RATES_CACHE = None


def reset_rates_cache():
    """
    Reset the rates cache
    """
    global _rates_cache
    global DAILY_RATES_CACHE
    _rates_cache = None
    DAILY_RATES_CACHE = None
    logger.info("Rates cache reset")


def deannualize(annual_rate, periods=365):
    return (1 + annual_rate) ** (1 / periods) - 1


def get_risk_free_rate_helper(interval="1d", use="db") -> pd.DataFrame:
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
    data = _fetch_rates(interval=interval).copy()

    if interval != "1d":
        data = resample(data, interval)
    data = data[~data.index.duplicated(keep="first")]
    return data  ## Not adding the resample schema for now


def _fetch_rates(interval):
    """
    Handles _rates_cache logic picking
    """
    global _rates_cache
    global DAILY_RATES_CACHE

    choice_cache = _rates_cache if interval != "1d" else DAILY_RATES_CACHE
    
    if ny_now().hour < 25:
        print("Fetching rates data from yfinance directly during market hours")
        ## Just use yfinance directly during market hours to avoid stale data
        data = yf.download(
            "^IRX",
            start="2010-01-01",
            end=(datetime.datetime.today() + BDay(1)).strftime("%Y-%m-%d"),
            interval="1d",
            progress=False,
            multi_level_index=False,
        )

        data["daily"] = data["Close"].apply(deannualize)
        data["annualized"] = data["Close"] / 100
        return data[["daily", "annualized"]]

    resample_bool = interval != "1d" or DAILY_RATES_CACHE is None
    ## First check data base.
    if choice_cache is None:
        data = query_database(
            "securities_master",
            "rates_timeseries",
            "SELECT * FROM rates_timeseries WHERE yf_tick = '^IRX' AND DATETIME >= '2010-01-01'",
        )
        data["datetime"] = pd.to_datetime(data["datetime"])
        data.set_index("datetime", inplace=True)
        data.rename(columns={"daily_rate": "daily", "annualized_rate": "annualized", "yf_tick": "name"}, inplace=True)
        data.index.name = "Datetime"
    else:
        data = choice_cache.copy()

    ## Drop today's date to ensure forced update
    if ny_now().hour >= 16:
        data = data[data.index.date <= change_to_last_busday(datetime.datetime.now()).date()]
    else:
        data = data[data.index.date < change_to_last_busday(datetime.datetime.now()).date()]

    ## Now, if data is not up to date, update it
    if data.index.max().date() < change_to_last_busday(datetime.datetime.now()).date():
        ## Query from max to today from retrieve_timeseries
        ## Prefer using yf because openbb timezone is UTC for IRX
        ## For YF, we have to do end_date + 1 day to get the last day
        data_min = yf.download(
            "^IRX",
            data.index.max().date(),
            end=(datetime.datetime.today() + BDay(1)).strftime("%Y-%m-%d"),
            interval="1h",
            progress=False,
            multi_level_index=False,
        )
        data_min.tz_convert("America/New_York")
        data_min.index = data_min.index.tz_convert("America/New_York")
        data_min.index = data_min.index.tz_localize(None)
        data_min.index = [x.replace(minute=30) for x in data_min.index]
        data_min.columns = data_min.columns.str.lower()
        data_min["daily"] = data_min["close"].apply(deannualize)
        data_min["annualized"] = data_min["close"] / 100
        data_min["name"] = "^IRX"
        data_min["description"] = "13 WEEK TREASURY BILL"
        data_min.index.name = "Datetime"
        data_min = data_min[["name", "description", "daily", "annualized"]]
        data = pd.concat([data, data_min])
        data = data[~data.index.duplicated(keep="first")]

    ## Have to resample all intervals
    resample_int = interval if interval == "1d" else "30m"
    if resample_bool:
        data = resample(
            data, resample_int, {"daily": "ffill", "annualized": "ffill", "name": "ffill", "description": "ffill"}
        )
    if interval != "1d":
        _rates_cache = data.copy()
    else:
        DAILY_RATES_CACHE = data.copy()
    return data


def save_previous_rates_date():
    import yfinance as yf

    print("Saving previous rates date")
    max_date = get_risk_free_rate_helper().index.max()
    rtes = yf.download(
        "^IRX",
        progress=False,
        multi_level_index=False,
        start=max_date,
        end=(datetime.datetime.today() + BDay(1)).strftime("%Y-%m-%d"),
        interval="1h",
    )
    print("DOWNLOAD COMPLETE")
    rtes.tz_convert("America/New_York")
    rtes.index = rtes.index.tz_convert("America/New_York")
    rtes.index = rtes.index.tz_localize(None)
    rtes.index = [x.replace(minute=30) for x in rtes.index]
    rtes["annualized"] = rtes["Close"] / 100
    rtes["daily"] = (1 + rtes["annualized"]) ** (1 / 365) - 1
    rtes["yf_tick"] = "^IRX"
    rtes["description"] = "13 WEEK TREASURY BILL"
    rtes = rtes[rtes.index > get_risk_free_rate_helper().index.min()][["yf_tick", "description", "annualized", "daily"]]
    rtes.rename(columns={"annualized": "annualized_rate", "daily": "daily_rate", "Date": "datetime"}, inplace=True)
    rtes.index.name = "datetime"
    rtes.reset_index(inplace=True)
    print("RATES TO SAVE")
    print(rtes.to_string())
    if rtes.empty:
        print("No new data to save")
        return
    store_SQL_data_Insert_Ignore("securities_master", "rates_timeseries", rtes)
    return rtes


if __name__ == "__main__":
    ## Save to db
    save_previous_rates_date()
