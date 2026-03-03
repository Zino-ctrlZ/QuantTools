"""Convenient loader functions for comprehensive option data retrieval.

This module provides high-level loader functions that simplify fetching complete
option data packages including spot, forward, dividend, vol, greeks, and rates.
Functions handle parameter validation, date conversion, and coordinate data loading
across multiple DataManagers.

Key Features:
    - One-call option data loading (all dependencies included)
    - Automatic parameter validation and date conversion
    - Support for timeseries, single-date, and real-time modes
    - Configurable pricing models and dividend treatments
    - Returns unified ModelResultPack with all data components

Typical Usage:
    >>> from trade.datamanager.loaders import load_full_option_data
    >>> from trade.datamanager._enums import DivType
    >>>
    >>> # Load historical option data with all dependencies
    >>> pack = load_full_option_data(
    ...     symbol="AAPL",
    ...     strike=150.0,
    ...     expiration="2025-06-20",
    ...     right="call",
    ...     start_date="2025-01-01",
    ...     end_date="2025-01-31",
    ...     dividend_type=DivType.DISCRETE
    ... )
    >>>
    >>> # Access individual components
    >>> greeks = pack.greek.timeseries
    >>> vol = pack.vol.timeseries
    >>> spot = pack.spot.timeseries
    >>>
    >>> # Real-time mode
    >>> rt_pack = load_full_option_data(
    ...     symbol="AAPL",
    ...     strike=150.0,
    ...     expiration="2025-06-20",
    ...     right="call",
    ...     rt=True
    ... )
"""

from typing import Optional
from trade.datamanager.result import DividendsResult, SpotResult
from trade.datamanager.utils.model import LoadRequest, _load_model_data_timeseries, ModelResultPack
from trade.datamanager.utils.date import DATE_HINT
from trade.datamanager._enums import (
    SeriesId,
    OptionSpotEndpointSource,
    VolatilityModel,
    OptionPricingModel,
    ModelPrice,
    DivType,
)
from trade.helpers.helper import to_datetime
from trade.helpers.Logging import setup_logger
from trade.datamanager.utils.logging import get_logging_level, register_to_factor_list

logger = setup_logger("trade.datamanager.loaders", stream_log_level=get_logging_level())
register_to_factor_list("trade.datamanager.loaders")


def load_full_option_data(
    symbol: str,
    *,
    expiration: DATE_HINT,
    strike: float,
    right: str,
    start_date: DATE_HINT = None,
    end_date: DATE_HINT = None,
    as_of: DATE_HINT = None,
    rt: bool = False,
    ## Optional parameters. If not passed will refer to global defaults found in OptionConfig
    series_id: SeriesId = None,
    dividend_type: DivType = None,
    endpoint_source: OptionSpotEndpointSource = None,
    vol_model: VolatilityModel = None,
    market_model: OptionPricingModel = None,
    model_price: ModelPrice = None,

    ## Optional data for modelling.
    spot_timeseries: Optional[SpotResult] = None,
    dividend_timeseries: Optional[DividendsResult] = None,
    forward_timeseries: Optional[SpotResult] = None,
    option_spot_timeseries: Optional[SpotResult] = None,
    vol_timeseries: Optional[SpotResult] = None,
    greek_timeseries: Optional[SpotResult] = None,
    rates_timeseries: Optional[SpotResult] = None,
) -> ModelResultPack:
    """Load comprehensive option data including spot, forward, vol, greeks, and rates.

    Convenience function that loads all required data for option analysis in a single
    call. Automatically handles data dependencies, caching, and model selection. Supports
    three modes: timeseries (start/end dates), single date (as_of), and real-time (rt).

    Args:
        symbol: Equity symbol (e.g., "AAPL", "MSFT")
        expiration: Option expiration date (YYYY-MM-DD string or datetime)
        strike: Option strike price
        right: Option type - "call"/"c" or "put"/"p"
        start_date: Start of timeseries range (YYYY-MM-DD string or datetime)
        end_date: End of timeseries range (YYYY-MM-DD string or datetime)
        as_of: Single date for historical snapshot (YYYY-MM-DD string or datetime)
        rt: If True, load real-time data
        series_id: Option series identifier (default from OptionConfig)
        dividend_type: DivType.DISCRETE or DivType.CONTINUOUS (default from OptionConfig)
        endpoint_source: Data source for option prices (default from OptionConfig)
        vol_model: Volatility calculation model (default from OptionConfig)
        market_model: Option pricing model (BSM, CRR, etc., default from OptionConfig)
        model_price: Model price type (default from OptionConfig)

    Returns:
        ModelResultPack containing:
            - spot: SpotResult with underlying prices
            - forward: ForwardResult with forward prices
            - dividend: DividendsResult with dividend schedules
            - rates: RatesResult with risk-free rates
            - option_spot: OptionSpotResult with market option prices
            - vol: VolatilityResult with implied volatilities
            - greek: GreekResultSet with option sensitivities

    Raises:
        ValueError: If mode specification is ambiguous (e.g., both start_date and as_of provided)

    Examples:
        >>> # Historical timeseries
        >>> pack = load_full_option_data(
        ...     symbol="AAPL",
        ...     strike=150.0,
        ...     expiration="2025-06-20",
        ...     right="call",
        ...     start_date="2025-01-01",
        ...     end_date="2025-01-31"
        ... )
        >>> print(pack.greek.timeseries.delta.head())
        datetime
        2025-01-02    0.5234
        2025-01-03    0.5301
        ...

        >>> # Single date snapshot
        >>> pack = load_full_option_data(
        ...     symbol="AAPL",
        ...     strike=150.0,
        ...     expiration="2025-06-20",
        ...     right="call",
        ...     as_of="2025-01-15"
        ... )
        >>> print(f"Vol on 2025-01-15: {pack.vol.as_of_value:.4f}")

        >>> # Real-time
        >>> pack = load_full_option_data(
        ...     symbol="AAPL",
        ...     strike=150.0,
        ...     expiration="2025-06-20",
        ...     right="call",
        ...     rt=True
        ... )
        >>> print(f"Current delta: {pack.greek.rt_value.delta:.4f}")

    Notes:
        - Only one mode should be specified: (start_date, end_date), as_of, or rt
        - All optional parameters default to values in OptionConfig
        - Data is automatically cached for efficient repeated access
        - Uses split-adjusted prices (undo_adjust=True) by default
    """
    if start_date and end_date:
        ts_start = to_datetime(start_date)
        ts_end = to_datetime(end_date)
        as_of = None
        rt = False

    elif as_of:
        ts_start = None
        ts_end = None
        as_of = to_datetime(as_of)
        rt = False

    elif rt:
        ts_start = None
        ts_end = None
        as_of = None
        rt = True

    request = LoadRequest(
        symbol=symbol,
        start_date=ts_start,
        end_date=ts_end,
        as_of=as_of,
        expiration=expiration,
        strike=strike,
        right=right,
        series_id=series_id,
        dividend_type=dividend_type,
        endpoint_source=endpoint_source,
        vol_model=vol_model,
        market_model=market_model,
        model_price=model_price,
        load_spot=True,
        load_dividend=True,
        load_forward=True,
        load_option_spot=True,
        load_vol=True,
        load_greek=True,
        load_rates=True,
        undo_adjust=True,
        rt=rt,

        ## Provided data (if any)
        spot_timeseries=spot_timeseries,
        dividend_timeseries=dividend_timeseries,
        forward_timeseries=forward_timeseries,
        option_spot_timeseries=option_spot_timeseries,
        vol_timeseries=vol_timeseries,
        greek_timeseries=greek_timeseries,
        rates_timeseries=rates_timeseries,
    )

    data_packet = _load_model_data_timeseries(request)
    return data_packet
