from typing import List, Optional, Union, Iterable
import numpy as np
from trade.helpers.helper_types import DATE_HINT
from trade.datamanager._enums import (GreekType, ModelPrice, OptionPricingModel, OptionSpotEndpointSource, VolatilityModel)
from trade.datamanager.result import GreekResultSet
from trade.optionlib.config.types import DivType
from trade.helpers.Logging import setup_logger
from trade.datamanager.utils.logging import get_logging_level, UTILS_LOGGER_NAME
logger = setup_logger(UTILS_LOGGER_NAME, stream_log_level=get_logging_level())

def _get_prefilled_greek_result_set(
    key: str,
    symbol: str,
    strike: float,
    expiration: DATE_HINT,
    right: str,
    endpoint_source: OptionSpotEndpointSource,
    market_model: OptionPricingModel,
    vol_model: VolatilityModel,
    dividend_type: DivType,
    undo_adjust: bool = True,
    model_price: Optional[ModelPrice] = None,
) -> GreekResultSet:
    """Utility to create prefilled GreekResultSet with metadata."""
    result = GreekResultSet(
        key=key,
        symbol=symbol,
        strike=strike,
        expiration=expiration,
        right=right,
        endpoint_source=endpoint_source,
        market_model=market_model,
        vol_model=vol_model,
        dividend_type=dividend_type,
        undo_adjust=undo_adjust,
        model_price=model_price
    )
    return result


def _prepare_greeks_to_compute(
    greeks_to_compute: Optional[Union[GreekType, Iterable[GreekType]]] = None,
) -> List[GreekType]:
    
    ## If None, set to all greeks
    if greeks_to_compute is None:
        greeks_to_compute = GreekType.GREEKS
    
    ## Expand GREEKS to all greek types (sentinel only excluded)
    if greeks_to_compute == GreekType.GREEKS:
        greeks_to_compute = list(set(GreekType) - {GreekType.GREEKS})
    
    ## Validate greek_to_compute is list/tuple/set of GreekType
    if not isinstance(greeks_to_compute, (list, np.ndarray, set, tuple)):
        greeks_to_compute = [greeks_to_compute]

    ## Validate all elements are GreekType
    if not all(isinstance(greek, GreekType) for greek in greeks_to_compute):
        raise ValueError(f"greeks_to_compute must be a GreekType or list of GreekType. Found: {greeks_to_compute}")
    
    ## Validate no duplicates
    greeks_to_compute = list(set(greeks_to_compute))

    return list(greeks_to_compute)


def _greek_column_names(greeks_to_compute: Iterable[GreekType]) -> List[str]:
    """Normalize ``GreekType`` values to DataFrame column names."""
    return [g.value if isinstance(g, GreekType) else str(g) for g in greeks_to_compute]


def _missing_greek_columns(cached_data, greeks_to_compute: Iterable[GreekType]) -> List[str]:
    """Return requested greek columns absent from a cached frame.

    Args:
        cached_data: Cached greeks DataFrame (or None).
        greeks_to_compute: Requested greek types.

    Returns:
        Column names that are missing; empty if the cache covers the request.
    """
    if cached_data is None or getattr(cached_data, "empty", True):
        return _greek_column_names(greeks_to_compute)
    have = {str(c) for c in cached_data.columns}
    return [c for c in _greek_column_names(greeks_to_compute) if c not in have]
