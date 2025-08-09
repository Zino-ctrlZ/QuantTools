## Initialisation of the trade package
import warnings
warnings.filterwarnings("ignore")
import os, sys
import json
import pandas_market_calendars as mcal
import pandas as pd
from datetime import datetime
from .helpers.Logging import setup_logger
sys.path.append(os.environ.get('DBASE_DIR', ''))
sys.path.append(os.environ.get('WORK_DIR', ''))
import signal
POOL_ENABLED = None
SIGNALS_TO_RUN = {}
logger = setup_logger('trade.__init__')


def register_signal(signum, signal_func):
    """
    Register a signal to be run when the process is interrupted.

    Parameters:
    ----------
    signum : int
        The signal number (e.g., signal.SIGINT, signal.SIGTERM).
    signal_func : callable
        The function to execute when the signal is received.
    """
    if signum not in SIGNALS_TO_RUN:
        SIGNALS_TO_RUN[signum] = []
        signal.signal(signum, run_signals)
        logger.info(f"Registered signal number {signum}.")
    if not callable(signal_func):
        raise ValueError(f"Signal function {signal_func} is not callable.")
    
    SIGNALS_TO_RUN[signum].append( signal_func)
    logger.info(f"Signal function for `{signal_func.__name__}` added to signal number {signum}.")

def run_signals(signum, frame):
    """
    Run all registered signals.
    """
    if signum in SIGNALS_TO_RUN:
        for signal_func in SIGNALS_TO_RUN[signum]:
            try:
                logger.info(f"Running signal function {signal_func.__name__} for signal {signum}.")
                signal_func()
            except Exception as e:
                logger.info(f"Error running signal function {signal_func.__name__}: {e}")
    else:
        logger.info(f"No registered signals for signal number {signum}.")
            
def get_signals_to_run():
    """
    Get the registered signals to run.
    """
    return SIGNALS_TO_RUN


def set_pool_enabled(value: bool):
    """
    Set the pool enabled flag.
    """
    global POOL_ENABLED
    POOL_ENABLED = value

def get_pool_enabled():
    """
    Get the pool enabled flag.
    """
    return POOL_ENABLED

set_pool_enabled(bool(os.environ.get('POOL_ENABLED')))

## Universal Holidays set
## Get Business days FOR NYSE (Some days are still trading days)
nyse = mcal.get_calendar('NYSE')
schedule = nyse.schedule(start_date='2000-01-01', end_date=datetime.today())
all_trading_days = mcal.date_range(schedule, frequency='1D').date
all_days = pd.date_range(start='2000-01-01', end=datetime.today(), freq='B')
holidays = set(all_days.difference(all_trading_days).strftime('%Y-%m-%d').to_list())
HOLIDAY_SET = set(holidays)

## Additional holidays
HOLIDAY_SET.update({
    '2025-01-09', ## Jimmy Carter's Death
}) 


## Import Pricing Config
with open(f"{os.environ['WORK_DIR']}/pricingConfig.json") as f:
    PRICING_CONFIG = json.load(f)

