from pathlib import Path
import os
import pandas as pd
from datetime import time
from datetime import datetime
from typing import List, Dict, Any
from trade.helpers.Logging import setup_logger
from typing import TYPE_CHECKING
from trade.optionlib.config.defaults import OPTION_TIMESERIES_START_DATE
from trade.datamanager.utils.logging import get_logging_level
from trade import register_signal
import signal
if TYPE_CHECKING:
    from trade.datamanager.market_data import MarketTimeseries
logger = setup_logger("trade.datamanager.vars", stream_log_level=get_logging_level())

DM_GEN_PATH = Path(os.getenv("GEN_CACHE_PATH")) / "dm_gen_cache"
TS: "MarketTimeseries" = None  # type: MarketTimeseries
_LOG_TO_DISK_BUCKET : List[Dict[str, Any]] = []
LOADED_NAMES = set()
MARKET_OPEN_TIME = time(9, 30)
MARKET_CLOSE_TIME = time(16, 0)
DEFAULT_SCENARIOS = [0.9, 0.95, 1.0, 1.05, 1.1]
DEFAULT_VOL_SCENARIOS = [-0.02, -0.01, 0.0, 0.01, 0.02]

def _parse_bool_env(var_name: str, default: bool = True) -> bool:
    raw = os.getenv(var_name)
    if raw is None:
        return default

    value = raw.strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False

    logger.warning(
        f"Invalid boolean value '{raw}' for env var {var_name}. Falling back to default={default}."
    )
    return default


ENABLE_CACHING: bool = _parse_bool_env("DATAMANAGER_ENABLE_CACHE", default=True)


def get_enable_caching() -> bool:
    global ENABLE_CACHING
    return ENABLE_CACHING

def disable_caching() -> None:
    global ENABLE_CACHING
    ENABLE_CACHING = False

def enable_caching() -> None:
    global ENABLE_CACHING
    ENABLE_CACHING = True

def is_caching_enabled() -> bool:
    global ENABLE_CACHING
    return ENABLE_CACHING

def get_dm_gen_path(is_live: bool = None) -> Path:
    from .config import OptionDataConfig
    if is_live is None:
        is_live = OptionDataConfig().is_live
        
    global DM_GEN_PATH
    if not DM_GEN_PATH.exists():
        DM_GEN_PATH.mkdir(parents=True)
    return DM_GEN_PATH if not is_live else DM_GEN_PATH / "live"

def set_times_series()-> "MarketTimeseries":
    from trade.datamanager.market_data import MarketTimeseries
    global TS
    TS = MarketTimeseries(_end=datetime.now(), _start=OPTION_TIMESERIES_START_DATE)
    return TS

def get_times_series() -> "MarketTimeseries":
    global TS
    if TS is None:
        set_times_series()
    return TS

def send_log_to_disk() -> None:
    global _LOG_TO_DISK_BUCKET
    if not _LOG_TO_DISK_BUCKET:
        logger.info("No logs to write to disk.")
        return

    log_path = DM_GEN_PATH / "dm_runtime_logs.csv"
    DM_GEN_PATH.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(_LOG_TO_DISK_BUCKET)

    if log_path.exists() and log_path.stat().st_size > 0:
        try:
            df_existing = pd.read_csv(log_path)
            if not df_existing.empty:
                df = pd.concat([df_existing, df], ignore_index=True)
        except pd.errors.EmptyDataError:
            logger.warning("Existing runtime log file %s is empty; overwriting.", log_path)

    df.to_csv(log_path, index=False)
    logger.info(f"Wrote {_LOG_TO_DISK_BUCKET.__len__()} log entries to disk at {log_path}.")
    _LOG_TO_DISK_BUCKET.clear()

def add_to_log_bucket(entry: Dict[str, Any]) -> None:
    global _LOG_TO_DISK_BUCKET
    _LOG_TO_DISK_BUCKET.append(entry)

register_signal("exit", send_log_to_disk)
register_signal(signal.SIGINT, send_log_to_disk)
register_signal(signal.SIGTERM, send_log_to_disk)

def load_name(symbol: str):
    key = (symbol, datetime.now().date())
    global LOADED_NAMES
    if key not in LOADED_NAMES:
        logger.info(f"Loading timeseries for {symbol}...")
        get_times_series().load_timeseries(symbol, start_date=OPTION_TIMESERIES_START_DATE, end_date=datetime.now())
        LOADED_NAMES.add(key)
    else:
        logger.info(f"Timeseries for {symbol} already loaded.")

def is_name_loaded(symbol: str) -> bool:
    global LOADED_NAMES
    key = (symbol, datetime.now().date())
    return key in LOADED_NAMES

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

## This is the minimum time before real-time data is requested
## ie if current time is before this time, real-time will not query for today and instead 
## rely on RealTimeFallback option
MIN_TIME_BEFORE_REAL_TIME = time(9, 45)  # 9:45 AM

