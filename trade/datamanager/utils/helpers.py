"""Runtime helpers for toggling datamanager option spot endpoint sources."""

from typing import Optional

from trade.datamanager.config import OptionDataConfig, OptionSpotEndpointSource
from trade.datamanager.utils.logging import get_logging_level
from trade.helpers.Logging import setup_logger

logger = setup_logger("trade.datamanager.utils.helpers", stream_log_level=get_logging_level())
__PREVIOUS_OPTION_SPOT_ENDPOINT_SOURCE: Optional[OptionSpotEndpointSource] = None

def enable_quotes() -> None:
    """Enable quotes for the data manager."""
    global __PREVIOUS_OPTION_SPOT_ENDPOINT_SOURCE
    __PREVIOUS_OPTION_SPOT_ENDPOINT_SOURCE = OptionDataConfig().option_spot_endpoint_source
    OptionDataConfig().option_spot_endpoint_source = OptionSpotEndpointSource.QUOTE

def disable_quotes() -> None:
    """Disable quotes for the data manager."""
    global __PREVIOUS_OPTION_SPOT_ENDPOINT_SOURCE
    if __PREVIOUS_OPTION_SPOT_ENDPOINT_SOURCE is not None:
        OptionDataConfig().option_spot_endpoint_source = __PREVIOUS_OPTION_SPOT_ENDPOINT_SOURCE
    else:
        logger.warning("Quotes are not enabled. No previous source to restore.")

def enable_eod() -> None:
    """Enable EOD for the data manager."""
    global __PREVIOUS_OPTION_SPOT_ENDPOINT_SOURCE
    if __PREVIOUS_OPTION_SPOT_ENDPOINT_SOURCE is not None:
        OptionDataConfig().option_spot_endpoint_source = __PREVIOUS_OPTION_SPOT_ENDPOINT_SOURCE
    else:
        logger.warning("EOD is not enabled. No previous source to restore.")

def disable_eod() -> None:
    """Disable EOD for the data manager."""
    global __PREVIOUS_OPTION_SPOT_ENDPOINT_SOURCE
    if __PREVIOUS_OPTION_SPOT_ENDPOINT_SOURCE is not None:
        OptionDataConfig().option_spot_endpoint_source = __PREVIOUS_OPTION_SPOT_ENDPOINT_SOURCE
    else:
        logger.warning("EOD is not enabled. No previous source to restore.")