from datetime import datetime
from typing import Any, Optional, Tuple, List
import pandas as pd
from trade.datamanager.result import (
    DividendsResult,
    VolatilityResult,
    SpotResult,
    ForwardResult,
    RatesResult,
    OptionSpotResult,
    ModelResultPack,
)
from trade.datamanager.utils.date import sync_date_index, time_distance_helper, _sync_date
from trade.datamanager.base import BaseDataManager
from trade.datamanager._enums import OptionSpotEndpointSource
from trade.datamanager.utils.cache import (
    _check_cache_for_timeseries_data_structure,
    _data_structure_cache_it,
)
from trade.helpers.helper import to_datetime
from trade.helpers.Logging import setup_logger
from trade.datamanager.utils.data_structure import _data_structure_sanitize
from trade.optionlib.config.types import DivType
from trade.optionlib.assets.dividend import vector_convert_to_time_frac

logger = setup_logger("trade.datamanager.utils", stream_log_level="INFO")


def _prepare_vol_calculation_setup(
    manager: BaseDataManager,
    start_date: str,
    end_date: str,
    expiration: str,
    strike: float,
    right: str,
    dividend_type: Optional[DivType],
    endpoint_source: Optional[OptionSpotEndpointSource],
    result: Optional[VolatilityResult] = None,
) -> Tuple[VolatilityResult, DivType, OptionSpotEndpointSource, str, str, datetime, datetime]:
    """Prepare common setup for volatility calculations."""
    result = VolatilityResult() if result is None else result
    dividend_type = dividend_type or manager.CONFIG.dividend_type
    endpoint_source = endpoint_source or manager.CONFIG.option_spot_endpoint_source

    start_date, end_date = _sync_date(
        symbol=manager.symbol,
        start_date=start_date,
        end_date=end_date,
        expiration=expiration,
        right=right,
        strike=strike,
        endpoint_source=endpoint_source,
    )

    start_str = to_datetime(start_date).strftime("%Y-%m-%d")
    end_str = to_datetime(end_date).strftime("%Y-%m-%d")

    return result, dividend_type, endpoint_source, start_str, end_str, start_date, end_date


def _handle_cache_for_vol(
    manager: BaseDataManager,
    key: str,
    start_date: datetime,
    end_date: datetime,
    result: VolatilityResult,
) -> Tuple[Optional[pd.Series], bool, datetime, datetime, Optional[VolatilityResult]]:
    """Handle cache checking logic for volatility calculations.

    Returns:
        Tuple of (cached_data, is_partial, adjusted_start, adjusted_end, result_or_none)
        If result_or_none is not None, caller should return it immediately (full cache hit)
    """
    cached_data, is_partial, start_date, end_date = _check_cache_for_timeseries_data_structure(
        key=key, self=manager, start_dt=start_date, end_dt=end_date
    )

    if cached_data is not None and not is_partial:
        logger.info(f"Cache hit for vol timeseries key: {key}")
        result.timeseries = cached_data
        return cached_data, is_partial, start_date, end_date, result
    elif is_partial:
        logger.info(f"Cache partially covers requested date range. Key: {key}. Fetching missing dates.")
    else:
        logger.info(f"No cache found for key: {key}. Fetching from source.")

    return cached_data, is_partial, start_date, end_date, None


def _merge_and_cache_vol_result(
    manager: BaseDataManager,
    iv_timeseries: pd.Series,
    cached_data: Optional[pd.Series],
    is_partial: bool,
    key: str,
    start_str: str,
    end_str: str,
) -> pd.Series:
    """Merge with cache if partial, cache result, and sanitize."""
    # Merge with cached data if partial
    if cached_data is not None and is_partial:
        merged = pd.concat([cached_data, iv_timeseries])
        iv_timeseries = merged[~merged.index.duplicated(keep="last")].sort_index()

    # Cache the fetched data
    _data_structure_cache_it(manager, key, iv_timeseries)

    # Sanitize before returning
    iv_timeseries = _data_structure_sanitize(
        iv_timeseries,
        start=start_str,
        end=end_str,
    )

    return iv_timeseries


def _merge_provided_with_loaded_data(
    model_data: "ModelResultPack",
    S: Optional[SpotResult] = None,
    F: Optional[ForwardResult] = None,
    r: Optional[RatesResult] = None,
    dividends: Optional[DividendsResult] = None,
    market_price: Optional[OptionSpotResult] = None,
) -> Tuple[
    Optional[SpotResult], Optional[ForwardResult], RatesResult, Optional[DividendsResult], Optional[OptionSpotResult]
]:
    """Merge user-provided data with loaded data, prioritizing provided data."""
    S = S if S is not None else model_data.spot
    F = F if F is not None else model_data.forward
    r = r if r is not None else model_data.rates
    dividends = dividends if dividends is not None else model_data.dividend
    market_price = market_price if market_price is not None else model_data.option_spot

    # Update model_data for consistency
    if S is not None:
        model_data.spot = S
    if F is not None:
        model_data.forward = F
    if r is not None:
        model_data.rates = r
    if dividends is not None:
        model_data.dividend = dividends
    if market_price is not None:
        model_data.option_spot = market_price

    return S, F, r, dividends, market_price


def _prepare_dividend_data_for_pricing(
    dividends: DividendsResult,
    dividend_type: DivType,
    expiration: str,
    *data_to_sync: pd.Series,
) -> Tuple[Any, ...]:
    """Prepare dividend data and synchronize all series.

    Returns:
        Tuple of synchronized series (including prepared dividends as last element)
    """
    if dividend_type == DivType.DISCRETE:
        dividends_ts = dividends.daily_discrete_dividends
        synced = sync_date_index(*data_to_sync, dividends_ts)

        # Convert to time fractions
        dividends_prepared = vector_convert_to_time_frac(
            schedules=synced[-1],
            valuation_dates=synced[0].index,
            end_dates=[to_datetime(expiration)] * len(synced[0].index),
        )
        return (*synced[:-1], dividends_prepared)

    elif dividend_type == DivType.CONTINUOUS:
        dividends_ts = dividends.daily_continuous_dividends
        synced = sync_date_index(*data_to_sync, dividends_ts)
        return synced


def _prepare_time_to_expiration(
    date_index: pd.DatetimeIndex,
    expiration: str,
) -> List[float]:
    """Calculate time to expiration for each date in the index."""
    return [time_distance_helper(x, expiration) for x in date_index]
