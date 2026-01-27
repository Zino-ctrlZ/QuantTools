from pathlib import Path
import os
from datetime import time
from datetime import datetime
from trade.helpers.Logging import setup_logger
from EventDriven.riskmanager.market_data import MarketTimeseries
from trade.optionlib.config.defaults import OPTION_TIMESERIES_START_DATE
logger = setup_logger("trade.datamanager.vars", stream_log_level="INFO")

DM_GEN_PATH = Path(os.getenv("GEN_CACHE_PATH")) / "dm_gen_cache"
TS = MarketTimeseries(_end=datetime.now(), _start=OPTION_TIMESERIES_START_DATE)
LOADED_NAMES = set()
MARKET_OPEN_TIME = time(9, 30)
MARKET_CLOSE_TIME = time(16, 0)

def load_name(symbol: str):
    global LOADED_NAMES
    if symbol not in LOADED_NAMES:
        logger.info(f"Loading timeseries for {symbol}...")
        TS.load_timeseries(symbol, start_date=OPTION_TIMESERIES_START_DATE, end_date=datetime.now())
        LOADED_NAMES.add(symbol)
    else:
        logger.info(f"Timeseries for {symbol} already loaded.")

def is_name_loaded(symbol: str) -> bool:
    global LOADED_NAMES
    return symbol in LOADED_NAMES

def clear_loaded_names():
    global LOADED_NAMES
    LOADED_NAMES.clear()
    logger.info("Cleared loaded names cache.")


def get_loaded_names() -> set:
    global LOADED_NAMES
    return LOADED_NAMES

## This is the time after which today data can be cached. Any time before this time,
## today data should be reloaded to ensure completeness.
TODAY_RELOAD_CUTOFF = time(18, 30)  # 6:30 PM