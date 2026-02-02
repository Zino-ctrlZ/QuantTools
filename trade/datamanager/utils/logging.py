import logging

from git import List
from trade.helpers.Logging import setup_logger, find_loggers_by_pattern, change_logger_stream_level
LOGGING_LEVEL = "DEBUG"
logger = setup_logger("trade.datamanager.utils", stream_log_level=LOGGING_LEVEL)

FACTOR_DMS = {
    "trade.datamanager.spot",
    "trade.datamanager.rates",
    "trade.datamanager.dividends",
    "trade.datamanager.forward",
    "trade.datamanager.vol",
    "trade.datamanager.option_spot",
    "trade.datamanager.greeks",
    "trade.datamanager.base"
}

VARS = [
    "trade.datamanager.vars",
]

UTILS_LOGGER_NAME = "trade.datamanager.utils"

def set_logging_level(level: str):
    global LOGGING_LEVEL
    LOGGING_LEVEL = level

def get_logging_level() -> str:
    return LOGGING_LEVEL

def get_datamanager_loggers() -> List[logging.Logger]:
    """Retrieve all loggers under 'trade.datamanager'"""
    return find_loggers_by_pattern("trade.datamanager")

def change_logging_in_all_datamanager_loggers(level: str = None):
    """Change logging level for all loggers under 'trade.datamanager'."""
    if level is None:
        level = LOGGING_LEVEL
    loggers = find_loggers_by_pattern("trade.datamanager")
    for logger in loggers:
        change_logger_stream_level(logger, getattr(logging, level.upper(), logging.INFO))

def change_datamanager_utils_logging_level(level: str = None):
    """Change logging level for 'trade.datamanager.utils' logger."""
    if level is None:
        level = LOGGING_LEVEL
    logger = logging.getLogger("trade.datamanager.utils")
    change_logger_stream_level(logger, getattr(logging, level.upper(), logging.INFO))

def change_datamanager_factor_loggers_level(level: str = None):
    """Change logging level for all factor loggers under 'trade.datamanager'."""
    if level is None:
        level = LOGGING_LEVEL
    for factor in FACTOR_DMS:
        loggers = find_loggers_by_pattern(factor)
        for logger in loggers:
            change_logger_stream_level(logger, getattr(logging, level.upper(), logging.INFO))


def change_all_optionlib_loggers_level(level: str = None):
    """Change logging level for all loggers under 'trade.optionlib'"""
    if level is None:
        level = LOGGING_LEVEL
    loggers = find_loggers_by_pattern("trade.optionlib")
    for logger in loggers:
        change_logger_stream_level(logger, getattr(logging, level.upper(), logging.INFO))


def register_to_factor_list(name:str):
    FACTOR_DMS.add(name)
    
