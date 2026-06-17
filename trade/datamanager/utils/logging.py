"""Central logging configuration for the datamanager package.

Defines the shared default log level, logger discovery helpers, and runtime
level mutation for all ``trade.datamanager`` loggers.

Core Functions:
    get_logging_level: Return the current datamanager stream log level.
    set_logging_level: Update the shared datamanager stream log level.
    change_logging_in_all_datamanager_loggers: Apply a level to all datamanager loggers.
"""

import logging
from typing import List

from trade.helpers.Logging import setup_logger, find_loggers_by_pattern, change_logger_stream_level

LOGGING_LEVEL = "WARNING"


def get_logging_level() -> str:
    """Return the current datamanager stream log level."""
    return LOGGING_LEVEL


logger = setup_logger("trade.datamanager.utils", stream_log_level=get_logging_level())

FACTOR_DMS = {
    "trade.datamanager.spot",
    "trade.datamanager.rates",
    "trade.datamanager.dividends",
    "trade.datamanager.forward",
    "trade.datamanager.vol",
    "trade.datamanager.option_spot",
    "trade.datamanager.greeks",
    "trade.datamanager.base",
    "trade.datamanager.market_data_helpers.dividends",
}

VARS = [
    "trade.datamanager.vars",
]

UTILS_LOGGER_NAME = "trade.datamanager.utils"
MODEL_NA_LOGGER_NAME = "trade.datamanager.utils.model_na"


def set_logging_level(level: str) -> None:
    """Update the shared datamanager stream log level."""
    global LOGGING_LEVEL
    LOGGING_LEVEL = level


def get_datamanager_loggers() -> List[logging.Logger]:
    """Retrieve all loggers under 'trade.datamanager'."""
    return find_loggers_by_pattern("trade.datamanager")


def change_logging_in_all_datamanager_loggers(level: str = None) -> None:
    """Change logging level for all loggers under 'trade.datamanager'."""
    if level is None:
        level = get_logging_level()
    loggers = find_loggers_by_pattern("trade.datamanager")
    for logger in loggers:
        change_logger_stream_level(logger, getattr(logging, level.upper(), logging.INFO))


def change_datamanager_utils_logging_level(level: str = None) -> None:
    """Change logging level for 'trade.datamanager.utils' logger."""
    if level is None:
        level = get_logging_level()
    utils_logger = logging.getLogger("trade.datamanager.utils")
    change_logger_stream_level(utils_logger, getattr(logging, level.upper(), logging.INFO))


def change_datamanager_factor_loggers_level(level: str = None) -> None:
    """Change logging level for all factor loggers under 'trade.datamanager'."""
    if level is None:
        level = get_logging_level()
    for factor in FACTOR_DMS:
        loggers = find_loggers_by_pattern(factor)
        for logger in loggers:
            change_logger_stream_level(logger, getattr(logging, level.upper(), logging.INFO))


def change_all_optionlib_loggers_level(level: str = None) -> None:
    """Change logging level for all loggers under 'trade.optionlib'."""
    if level is None:
        level = get_logging_level()
    loggers = find_loggers_by_pattern("trade.optionlib")
    for logger in loggers:
        change_logger_stream_level(logger, getattr(logging, level.upper(), logging.INFO))


def change_all_logger(level: str = None) -> None:
    """Change logging level for all loggers under 'trade'."""
    if level is None:
        level = get_logging_level()
    change_all_optionlib_loggers_level(level)
    change_logging_in_all_datamanager_loggers(level)


def register_to_factor_list(name: str) -> None:
    """Register a datamanager module name for factor logger level changes."""
    FACTOR_DMS.add(name)
