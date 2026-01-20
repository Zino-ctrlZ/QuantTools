## Initialisation of the trade package
import os
import signal
import json
import warnings
import atexit
from zoneinfo import ZoneInfo
import pandas as pd
import pandas_market_calendars as mcal
from dotenv import load_dotenv
from trade.helpers.clear_cache import cleanup_expired_caches
from .helpers.Logging import setup_logger

warnings.filterwarnings("ignore")


USER = str(os.environ.get("USER", "unknown_user")).lower()  ## Temporary fix to allow only chidi utilize some features
POOL_ENABLED = None
SIGNALS_TO_RUN = {}
EXIT_HANDLERS = []  # Handlers for normal program exit
_ATEXIT_REGISTERED = False
OWNER_PID = os.getpid()
logger = setup_logger("trade.__init__", stream_log_level="WARNING")
cleanup_expired_caches()


## Universal Holidays set
## Get Business days FOR NYSE (Some days are still trading days)
##TODO: Make this more dynamic, so it can be used for other exchanges as well. And end date should be dynamic as well.
NY = ZoneInfo("America/New_York")
nyse = mcal.get_calendar("NYSE")
schedule = nyse.schedule(start_date="2000-01-01", end_date="2040-01-01", tz=NY)
# pylint: disable=E1101
all_trading_days = mcal.date_range(schedule, frequency="1D").date  ## type: ignore
all_days = pd.date_range(start="2000-01-01", end="2040-01-01", freq="B")
holidays = set(all_days.difference(all_trading_days).strftime("%Y-%m-%d").to_list())
HOLIDAY_SET = set(holidays)

## Additional holidays
HOLIDAY_SET.update(
    {
        "2025-01-09",  ## Jimmy Carter's Death
    }
)


def get_current_user() -> str:
    """
    Get the current user's name from the USER environment variable.

    Returns:
    -------
    str
        The current user's name (lowercase).
    """
    user = str(os.environ.get("USER", "unknown_user")).lower()
    if user == "unknown_user":
        logger.warning("USER environment variable is not set. Please set it for proper user identification.")
    return user


def is_allowed_user(allowed_users: list) -> bool:
    """
    Check if the current user is in the list of allowed users.

    Parameters:
    ----------
    allowed_users : list
        List of allowed usernames.

    Returns:
    -------
    bool
        True if the current user is allowed, False otherwise.
    """
    allowed_users = [user.lower() for user in allowed_users]
    if get_current_user() in allowed_users:
        return True
    else:
        return False


def _run_exit_handlers():
    """Run all registered exit handlers."""
    global EXIT_HANDLERS
    for handler in EXIT_HANDLERS:
        try:
            logger.info("Running exit handler %s.", handler.__name__)
            handler()
        except Exception as e:
            logger.error("Error running exit handler %s: %s", handler.__name__, e)


def register_signal(signum, signal_func):
    """
    Register a signal to be run when the process is interrupted or exits.

    Parameters:
    ----------
    signum : int or str
        The signal number (e.g., signal.SIGINT, signal.SIGTERM) or 'exit' for normal program exit.
    signal_func : callable
        The function to execute when the signal is received or program exits.

    Examples:
    --------
    >>> register_signal(signal.SIGTERM, cleanup_function)
    >>> register_signal(signal.SIGINT, cleanup_function)
    >>> register_signal('exit', save_data_function)  # For normal program exit
    """
    global _ATEXIT_REGISTERED

    if not callable(signal_func):
        raise ValueError(f"Signal function {signal_func} is not callable.")

    # Handle normal program exit
    if signum == "exit" or signum == 0:
        EXIT_HANDLERS.append(signal_func)
        # Register atexit handler only once
        if not _ATEXIT_REGISTERED:
            atexit.register(_run_exit_handlers)
            _ATEXIT_REGISTERED = True
            logger.info("Registered atexit handler for normal program exit.")
        logger.info("Exit handler `%s` registered for normal program exit.", signal_func.__name__)
        return

    # Handle signal-based interrupts
    if signum not in SIGNALS_TO_RUN:
        SIGNALS_TO_RUN[signum] = []
        signal.signal(signum, run_signals)
        logger.info("Registered signal number %d.", signum)

    SIGNALS_TO_RUN[signum].append(signal_func)
    logger.info("Signal function for `%s` added to signal number %d.", signal_func.__name__, signum)


def run_signals(signum, frame):
    """
    Run all registered signals.
    """
    if os.getpid() != OWNER_PID:
        logger.info("Signal received in child process (PID: %d). Ignoring signal %d.", os.getpid(), signum)
        return
    if signum in SIGNALS_TO_RUN:
        for signal_func in SIGNALS_TO_RUN[signum]:
            try:
                logger.info("Running signal function %s for signal %d.", signal_func.__name__, signum)
                signal_func()
            except (ValueError, TypeError) as e:
                logger.info("Error running signal function %s: %s", signal_func.__name__, e)
    else:
        logger.info("No registered signals for signal number %d.", signum)


def str_to_bool(value: str) -> bool:
    """
    Convert a string to a boolean value.
    Args:
        value (str): The string to convert.
    Returns:
        bool: True if the string is 'True', '1', or 'yes' (case-insensitive), False otherwise.
    """
    if value.lower() in ["true", "1", "yes"]:
        return True
    elif value.lower() in ["false", "0", "no"]:
        return False
    else:
        raise ValueError("Invalid boolean string. Expected 'True', 'False', '1', '0', 'yes', or 'no'.")


def get_signals_to_run():
    """
    Get the registered signals to run.
    """
    return SIGNALS_TO_RUN


def set_pool_enabled(value: bool):
    """
    Set the pool enabled flag.
    """
    logger.info(f"Setting POOL_ENABLED to {value}")
    global POOL_ENABLED
    POOL_ENABLED = value


def get_pool_enabled():
    """
    Get the pool enabled flag.
    """
    return POOL_ENABLED


def reset_pool_enabled():
    """
    Reset the pool enabled flag to None.
    """
    load_dotenv(f"{os.environ['WORK_DIR']}/.env")
    set_pool_enabled(str_to_bool(os.environ.get("POOL_ENABLED", "False")))


reset_pool_enabled()


## Import Pricing Config
with open(f"{os.environ['WORK_DIR']}/pricingConfig.json", encoding="utf-8") as f:
    PRICING_CONFIG = json.load(f)


def get_pricing_config() -> dict:
    """
    Get the pricing configuration.
    """
    MISSING_DEFAULTS = {
        "VOL_SURFACE_MAX_DTE_THRESHOLD": 365,
        "VOL_SURFACE_MIN_DTE_THRESHOLD": 0,
        "VOL_SURFACE_MAX_MONEYNESS_THRESHOLD": 1,
        "VOL_SURFACE_MIN_MONEYNESS_THRESHOLD": 0,
    }
    with open(f"{os.environ['WORK_DIR']}/pricingConfig.json", encoding="utf-8") as f:
        PRICING_CONFIG = json.load(f)

    for key, value in MISSING_DEFAULTS.items():
        if key not in PRICING_CONFIG:
            PRICING_CONFIG[key] = value
            logger.warning(f"Missing key {key} in pricing config. Setting default value {value}.")
    return PRICING_CONFIG


def reload_pricing_config():
    """
    Reload the pricing configuration from the file.
    """

    global PRICING_CONFIG
    with open(f"{os.environ['WORK_DIR']}/pricingConfig.json", encoding="utf-8") as pricing_file:
        PRICING_CONFIG = json.load(pricing_file)

    logger.info("Pricing configuration reloaded.")
