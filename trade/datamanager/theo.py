"""Theoretical option pricing module for computing fair values and scenario analysis.

This module provides functions for calculating theoretical option prices using various
pricing models (Black-Scholes-Merton, Cox-Ross-Rubinstein binomial) and performing
scenario analysis across different spot and volatility levels. It handles the complete
workflow including data loading, model selection, and result formatting.

Key Features:
    - Multiple pricing models: BSM, CRR binomial
    - Support for American and European exercise styles
    - Discrete and continuous dividend treatments
    - Automatic data loading
    - Scenario analysis (spot and volatility stress testing)
    - P&L analysis capabilities

Typical Usage:
    >>> from trade.datamanager.theo import get_option_theoretical_price, calculate_scenarios
    >>> from trade.optionlib.config.types import DivType
    >>> from trade.datamanager._enums import OptionPricingModel
    >>>
    >>> # Get theoretical prices for an option over time
    >>> result = get_option_theoretical_price(
    ...     symbol="AAPL",
    ...     start_date="2025-01-01",
    ...     end_date="2025-01-31",
    ...     strike=150.0,
    ...     expiration="2025-06-20",
    ...     right="c",
    ...     market_model=OptionPricingModel.BSM,
    ...     dividend_type=DivType.DISCRETE
    ... )
    >>> print(result.timeseries.head())
    >>>
    >>> # Run scenario analysis for risk management
    >>> scenarios = calculate_scenarios(
    ...     symbol="AAPL",
    ...     as_of="2025-01-15",
    ...     strike=150.0,
    ...     expiration="2025-06-20",
    ...     right="c",
    ...     spot_scenarios=[0.9, 0.95, 1.0, 1.05, 1.1],
    ...     vol_scenarios=[-0.05, 0.0, 0.05],
    ...     return_pnl=True
    ... )
    >>> print(scenarios.grid)
"""
from datetime import datetime
from itertools import product
import pandas as pd
from typing import Optional, Literal, Dict, List
from trade.datamanager.utils.model import (
    LoadRequest,
    _load_model_data,
    DivType,
    VolatilityModel,
    OptionPricingModel,
)
from trade.helpers.helper import time_distance_helper
from trade.datamanager.config import OptionDataConfig
from trade.datamanager.result import (
    VolatilityResult,
    ForwardResult,
    RatesResult,
    OptionSpotResult,
    SpotResult,
    DividendsResult,
    TheoreticalPriceResult,
    ScenariosResult,
)
from trade.datamanager._enums import (
    OptionSpotEndpointSource,
    ModelPrice,
)
from trade.datamanager.utils.model import _adjust_div_yield_for_spot_shock
from trade.datamanager.utils.date import DATE_HINT
from trade.optionlib.assets.dividend import (
    vectorized_discrete_pv,
    get_vectorized_continuous_dividends,
    vector_convert_to_time_frac,
)
from trade.datamanager.utils.date import sync_date_index, to_datetime
from trade.helpers.Logging import setup_logger
from trade.optionlib.pricing.binomial import vector_crr_binomial_pricing
from trade.optionlib.pricing.black_scholes import black_scholes_vectorized
from trade.optionlib.assets.forward import vectorized_forward_continuous, vectorized_forward_discrete
from trade.datamanager.utils.logging import get_logging_level, register_to_factor_list
from trade.datamanager.vars import DEFAULT_SCENARIOS, DEFAULT_VOL_SCENARIOS

logger = setup_logger("trade.datamanager.theo", stream_log_level=get_logging_level())
register_to_factor_list("trade.datamanager.theo")
CONFIG = OptionDataConfig()


def _create_load_request(
    ## Requied parameters to ensure correct data is loaded
    symbol: str,
    expiration: DATE_HINT,
    strike: float,
    right: str,
    dividend_type: DivType,
    market_model: OptionPricingModel,
    endpoint_source: OptionSpotEndpointSource,
    model_price: ModelPrice,
    is_scenario_load: bool = False,
    *,
    ## Optional pre-loaded data. If not provided, will be loaded.
    start_date: Optional[DATE_HINT] = None,
    end_date: Optional[DATE_HINT] = None,
    as_of: Optional[DATE_HINT] = None,
    rt: Optional[bool] = False,
    s: Optional[SpotResult] = None,
    r: Optional[RatesResult] = None,
    f: Optional[ForwardResult] = None,
    d: Optional[DividendsResult] = None,
    vol: Optional[VolatilityResult] = None,
    option_spot: Optional[OptionSpotResult] = None,
    undo_adjust: bool = True,
) -> LoadRequest:
    """Create a LoadRequest specifying which market data to load for theoretical pricing.

    Internal utility that determines which data sources need to be loaded based on:
    1. Which data is already provided (pre-loaded)
    2. Which pricing model is being used (BSM needs forwards, binomial needs spot)
    3. Whether this is a scenario load (requires additional data)

    Args:
        start_date: First valuation date (YYYY-MM-DD string or datetime).
        end_date: Last valuation date (YYYY-MM-DD string or datetime).
        symbol: Ticker symbol for the underlying asset.
        expiration: Option expiration date (YYYY-MM-DD string or datetime).
        strike: Strike price of the option.
        right: Option type ('c' for call, 'p' for put).
        dividend_type: Dividend treatment type (DISCRETE or CONTINUOUS).
        market_model: Pricing model (BSM or BINOMIAL).
        endpoint_source: Option data source (ORATS, HIST, QUOTE).
        model_price: Which price to use (CLOSE, OPEN, MIDPOINT).
        is_scenario_load: If True, loads option_spot for base price comparison.
        s: Optional pre-loaded spot data. If None, will be loaded.
        r: Optional pre-loaded rates data. If None, will be loaded.
        f: Optional pre-loaded forward data. If None, will be loaded (BSM only).
        d: Optional pre-loaded dividend data. If None, will be loaded.
        vol: Optional pre-loaded volatility data. If None, will be loaded.
        option_spot: Optional pre-loaded option market prices. If None, loaded for scenarios.
        undo_adjust: If True, uses split-adjusted prices.

    Returns:
        LoadRequest object with flags indicating which data sources to load.

    Examples:
        >>> # Internal usage - creates request for theoretical pricing
        >>> request = _create_load_request(
        ...     start_date="2025-01-01",
        ...     end_date="2025-01-31",
        ...     symbol="AAPL",
        ...     expiration="2025-06-20",
        ...     strike=150.0,
        ...     right="c",
        ...     dividend_type=DivType.DISCRETE,
        ...     market_model=OptionPricingModel.BSM,
        ...     endpoint_source=OptionSpotEndpointSource.HIST,
        ...     model_price=ModelPrice.CLOSE
        ... )
        >>> # request.load_spot = True (BSM needs spot for forward calc)
        >>> # request.load_vol = True (no vol provided)
    """
    if is_scenario_load:
        ## For scenario loads, always load all data to ensure completeness.
        load_spot = s is None
        load_vol = vol is None
        load_dividend = d is None
        load_rates = r is None
        option_spot = option_spot is None
        load_forward = False  ## Not needed for scenario load
    else:
        ## For regular loads, determine based on provided data and model needs.
        load_spot = (s is None) and (market_model == OptionPricingModel.BINOMIAL)
        load_vol = vol is None
        load_dividend = d is None
        load_rates = r is None
        option_spot = False  ## Not needed for greek calculation
        load_forward = (market_model == OptionPricingModel.BSM) and (f is None)

    req = LoadRequest(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        expiration=expiration,
        strike=strike,
        right=right,
        dividend_type=dividend_type,
        endpoint_source=endpoint_source,
        vol_model=VolatilityModel.MARKET,
        model_price=model_price,
        market_model=market_model,
        
        ## Load spot only if missing.
        load_spot=load_spot,
        
        ## Load forward only if missing and using BSM model. Binomial uses spot price.
        load_forward=load_forward,
        load_vol=load_vol,
        load_dividend=load_dividend,
        load_rates=load_rates,

        ## Not needed for greek calculation
        load_option_spot=option_spot,
        undo_adjust=undo_adjust,

        ## Real-time/Date flag
        rt=rt,
        as_of=as_of,
    )
    return req


def get_option_theoretical_price(
    symbol: str,
    strike: float,
    expiration: DATE_HINT,
    right: Literal["c", "p"],
    *,
    start_date: Optional[DATE_HINT] = None,
    end_date: Optional[DATE_HINT] = None,
    as_of: Optional[DATE_HINT] = None,
    market_model: Optional[OptionPricingModel] = None,
    endpoint_source: OptionSpotEndpointSource = None,
    dividend_type: Optional[DivType] = None,
    vol: Optional[VolatilityResult] = None,
    model_price: Optional[ModelPrice] = None,
    spot: Optional[SpotResult] = None,
    f: Optional[ForwardResult] = None,
    r: Optional[RatesResult] = None,
    d: Optional[DividendsResult] = None,
    undo_adjust: bool = True,
    n_steps: Optional[int] = None,
    rt: Optional[bool] = False,
) -> TheoreticalPriceResult:
    """Calculate theoretical option prices over a date range using specified pricing model.

    Computes fair value option prices for each business day in [start_date, end_date]
    using either BSM or binomial pricing models. Automatically loads required market
    data (spot, volatility, rates, dividends) if not provided.

    Args:
        symbol: Ticker symbol for the underlying asset (e.g., "AAPL", "MSFT").
        start_date: First valuation date (YYYY-MM-DD string or datetime).
        end_date: Last valuation date (YYYY-MM-DD string or datetime).
        as_of: Specific date for single-date pricing (YYYY-MM-DD string or datetime).
        strike: Strike price of the option.
        expiration: Option expiration date (YYYY-MM-DD string or datetime).
        right: Option type ('c' for call, 'p' for put).
        market_model: OptionPricingModel.BSM or BINOMIAL. Defaults to CONFIG setting.
        endpoint_source: Option data source for volatility (ORATS, HIST, QUOTE).
            Defaults to CONFIG setting.
        dividend_type: DivType.DISCRETE or DivType.CONTINUOUS. Defaults to CONFIG setting.
        vol: Optional pre-computed implied volatilities. If None, loads automatically.
        model_price: Which price to use (CLOSE, OPEN, MIDPOINT). Defaults to CONFIG setting.
        spot: Optional pre-computed spot prices. If None, loads automatically.
        f: Optional pre-computed forward prices. If None, loads automatically (BSM only).
        r: Optional pre-computed risk-free rates. If None, loads automatically.
        d: Optional pre-computed dividend data. If None, loads automatically.
        undo_adjust: If True, uses split-adjusted prices.
        n_steps: Number of time steps for binomial tree. Defaults to CONFIG.n_steps.
        as_of: Specific date for single-date pricing (YYYY-MM-DD string or datetime). = None,
        rt: If True, prices as of current real-time data.

    Returns:
        TheoreticalPriceResult containing daily theoretical prices as Series with
        DatetimeIndex, plus model metadata.

    Examples:
        >>> # Basic usage with BSM model
        >>> result = get_option_theoretical_price(
        ...     symbol="AAPL",
        ...     start_date="2025-01-01",
        ...     end_date="2025-01-31",
        ...     strike=150.0,
        ...     expiration="2025-06-20",
        ...     right="c",
        ...     market_model=OptionPricingModel.BSM,
        ...     dividend_type=DivType.DISCRETE
        ... )
        >>> print(result.timeseries.head())

        >>> # American option with binomial model
        >>> result = get_option_theoretical_price(
        ...     symbol="AAPL",
        ...     start_date="2025-01-01",
        ...     end_date="2025-01-31",
        ...     strike=150.0,
        ...     expiration="2025-06-20",
        ...     right="p",
        ...     market_model=OptionPricingModel.BINOMIAL,
        ...     n_steps=200
        ... )

        >>> # Provide pre-computed volatility
        >>> from trade.datamanager.vol import VolDataManager
        >>> vol_mgr = VolDataManager("AAPL")
        >>> vol_result = vol_mgr.get_implied_volatility_timeseries(
        ...     start_date="2025-01-01",
        ...     end_date="2025-01-31",
        ...     expiration="2025-06-20",
        ...     strike=150.0,
        ...     right="c"
        ... )
        >>> theo_result = get_option_theoretical_price(
        ...     symbol="AAPL",
        ...     start_date="2025-01-01",
        ...     end_date="2025-01-31",
        ...     strike=150.0,
        ...     expiration="2025-06-20",
        ...     right="c",
        ...     vol=vol_result
        ... )
    """

    if not as_of and not rt and (not start_date or not end_date):
        raise ValueError("Either 'as_of', rt=True, or both 'start_date' and 'end_date' must be provided.")
    
    market_model = market_model or CONFIG.option_model
    endpoint_source = endpoint_source or CONFIG.option_spot_endpoint_source
    dividend_type = dividend_type or CONFIG.dividend_type
    vol_model = CONFIG.volatility_model
    model_price = model_price or CONFIG.model_price
    n_steps = n_steps or CONFIG.n_steps
    result = TheoreticalPriceResult()
    result.dividend_type = dividend_type
    result.market_model = market_model
    result.model_price = model_price
    result.vol_model = vol_model
    result.endpoint_source = endpoint_source
    result.expiration = to_datetime(expiration)
    result.right = right
    result.strike = strike
    result.symbol = symbol
    result.rt = rt
    result.undo_adjust = undo_adjust

    # Create load request to determine which data to load
    load_request = _create_load_request(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        expiration=expiration,
        strike=strike,
        right=right,
        dividend_type=dividend_type,
        market_model=market_model,
        endpoint_source=endpoint_source,
        model_price=model_price,
        s=spot,
        f=f,
        r=r,
        vol=vol,
        is_scenario_load=False,
        rt=rt,
        as_of=as_of,
    )

    ## Router clips to a single anchor row for rt/as_of; full window for hist.
    packet = _load_model_data(load_request)

    # Extract time series data, using provided data if available
    s, r, vol, d, f = (
        packet.spot.timeseries
        if not packet.spot.is_empty()
        else spot.timeseries
        if spot is not None
        else pd.Series(dtype=float),
        packet.rates.timeseries
        if not packet.rates.is_empty()
        else r.timeseries
        if r is not None
        else pd.Series(dtype=float),
        packet.vol.timeseries
        if not packet.vol.is_empty()
        else vol.timeseries
        if vol is not None
        else pd.Series(dtype=float),
        packet.dividend.timeseries
        if not packet.dividend.is_empty()
        else d.timeseries
        if d is not None
        else pd.Series(dtype=float),
        packet.forward.timeseries
        if not packet.forward.is_empty()
        else f.timeseries
        if f is not None
        else pd.Series(dtype=float),
    )

    # Use loaded data to calculate theoretical prices
    if market_model == OptionPricingModel.BINOMIAL:
        s, vol, r, d = sync_date_index(s, vol, r, d)
        t = time_distance_helper(start=s.index, end=[expiration] * len(s))
        if dividend_type == DivType.DISCRETE:
            discrete = vector_convert_to_time_frac(
                schedules=d.values,
                valuation_dates=d.index,
                end_dates=[to_datetime(expiration)] * len(s),
            )
            dividend_yield = [0.0] * len(s)
        else:
            discrete = [()] * len(s)
            dividend_yield = d.values

        prices = vector_crr_binomial_pricing(
            K=[strike] * len(s),
            T=t,
            sigma=vol.values,
            r=r.values,
            N=[n_steps] * len(s),
            S0=s.values,
            right=[right] * len(s),
            american=[True] * len(s),
            dividend_yield=dividend_yield,
            dividends=discrete,
            dividend_type=[dividend_type.value] * len(s),
        )
        result.timeseries = pd.Series(data=prices, index=s.index, name="theoretical_price", dtype=float)
        return result

    elif market_model == OptionPricingModel.BSM:
        f, vol, r, d = (
            packet.forward.timeseries,
            packet.vol.timeseries,
            packet.rates.timeseries,
            packet.dividend.timeseries,
        )
        f, vol, r, d = sync_date_index(f, vol, r, d)
        t = time_distance_helper(start=f.index, end=[expiration] * len(f))
        prices = black_scholes_vectorized(
            F=f.values,
            K=[strike] * len(f),
            T=t,
            r=r.values,
            sigma=vol.values,
            option_type=[right] * len(f),
        )
        result.timeseries = pd.Series(data=prices, index=f.index, name="theoretical_price", dtype=float)
        return result


def _calculate_binomial_scenarios(
    base_prices: pd.Series,
    s: pd.Series,
    strike: float,
    expiration: DATE_HINT,
    right: Literal["c", "p"],
    vol: pd.Series,
    r: pd.Series,
    dividend_type: DivType,
    dividends: pd.Series,
    spot_scenarios: List[float] = None,
    vol_scenarios: List[float] = None,
    return_pnl: bool = False,
    return_pnl_in_pct: bool = False,
    n_steps: int = None,
    prettify_columns: bool = False,
) -> pd.DataFrame:
    """Calculate option price scenarios using Cox-Ross-Rubinstein binomial model.

    Internal function that computes option prices across a grid of spot and volatility
    scenarios. Spot scenarios are multiplicative (e.g., 0.9 = 10% down, 1.1 = 10% up).
    Volatility scenarios are additive (e.g., 0.05 = +5% vol, -0.05 = -5% vol).

    Args:
        base_prices: Current market prices of the option (single-date Series).
        s: Current spot prices (single-date Series).
        strike: Strike price of the option.
        expiration: Option expiration date (YYYY-MM-DD string or datetime).
        right: Option type ('c' for call, 'p' for put).
        vol: Current implied volatilities (single-date Series).
        r: Risk-free interest rates (single-date Series).
        dividend_type: Dividend treatment type (DISCRETE or CONTINUOUS).
        dividends: Dividend data (schedules for DISCRETE, yields for CONTINUOUS).
        spot_scenarios: List of spot price multipliers. E.g., [0.9, 1.0, 1.1] tests
            spot -10%, unchanged, +10%. Defaults to [1.0].
        vol_scenarios: List of volatility adjustments. E.g., [-0.05, 0.0, 0.05] tests
            vol -5%, unchanged, +5%. Defaults to [0.0].
        return_pnl: If True, returns P&L relative to base_prices instead of absolute prices.
        return_pnl_in_pct: If True (with return_pnl=True), returns P&L as percentage.
        n_steps: Number of time steps in binomial tree.
        prettify_columns: If True, formats column/index labels for display.

    Returns:
        DataFrame with volatility scenarios as rows, spot scenarios as columns, and
        option prices (or P&L) as values.

    Raises:
        AssertionError: If neither spot_scenarios nor vol_scenarios provided.
        AssertionError: If input series contain more than one date.

    Examples:
        >>> # Internal usage - calculate scenario grid
        >>> scenarios_df = _calculate_binomial_scenarios(
        ...     base_prices=pd.Series([10.5]),
        ...     s=pd.Series([150.0]),
        ...     strike=150.0,
        ...     expiration="2025-06-20",
        ...     right="c",
        ...     vol=pd.Series([0.25]),
        ...     r=pd.Series([0.05]),
        ...     dividend_type=DivType.DISCRETE,
        ...     dividends=pd.Series([Schedule()]),
        ...     spot_scenarios=[0.9, 0.95, 1.0, 1.05, 1.1],
        ...     vol_scenarios=[-0.05, 0.0, 0.05],
        ...     n_steps=100,
        ...     prettify_columns=True
        ... )
    """
    assert any([spot_scenarios, vol_scenarios]), "At least one of spot_scenarios or vol_scenarios must be provided."
    assert len(vol.index) == 1, "Spot scenarios calculation only supports single-date series."

    ## Default scenarios
    if spot_scenarios is None:
        spot_scenarios = [1.0]
    if vol_scenarios is None:
        vol_scenarios = [0.0]

    ## Sync all data to same index
    s, vol, r, dividends, base_prices = sync_date_index(s, vol, r, dividends, base_prices)
    scenario_prices: Dict[str, pd.Series] = {}

    ## Define pricing function for reuse
    def price_func(
        scenario_spot: pd.Series,
        scenario_vol: pd.Series,
        expiration: DATE_HINT,
        right: Literal["c", "p"],
        strike: float,
        dividend_type: DivType,
        dividends: pd.Series,
        n_steps: int,
        r: pd.Series,
    ) -> pd.Series:
        t = time_distance_helper(start=scenario_spot.index, end=[expiration] * len(scenario_spot))
        if dividend_type == DivType.DISCRETE:
            discrete = vector_convert_to_time_frac(
                schedules=dividends.values,
                valuation_dates=scenario_spot.index,
                end_dates=[to_datetime(expiration)] * len(scenario_spot),
            )
            dividend_yield = [0.0] * len(scenario_spot)
        else:
            discrete = [()] * len(scenario_spot)
            dividend_yield = dividends.values

        prices = vector_crr_binomial_pricing(
            K=[strike] * len(scenario_spot),
            T=t,
            sigma=scenario_vol.values,
            r=r.values,
            N=[n_steps] * len(scenario_spot),
            S0=scenario_spot.values,
            right=[right] * len(scenario_spot),
            american=[True] * len(scenario_spot),
            dividend_yield=dividend_yield,
            dividends=discrete,
            dividend_type=[dividend_type.value] * len(scenario_spot),
        )
        return pd.Series(data=prices, index=scenario_spot.index, name="theoretical_price", dtype=float)

    ## Calculate prices for each scenario
    scenarios = list(product(spot_scenarios, vol_scenarios))
    for spot_mult, vol_add in scenarios:
        scenario_spot = s * spot_mult
        scenario_vol = vol + vol_add
        if dividend_type == DivType.CONTINUOUS:
            adjusted_dividends = _adjust_div_yield_for_spot_shock(spot_mult, dividends)
        else:
            adjusted_dividends = dividends

        prices = price_func(
            scenario_spot, scenario_vol, expiration, right, strike, dividend_type, adjusted_dividends, n_steps, r
        )
        prices = prices[0]
        if return_pnl:
            prices = prices - base_prices[0]
            if return_pnl_in_pct:
                prices = prices / base_prices[0]
        scenario_prices.setdefault(spot_mult, []).append(prices)

    df = pd.DataFrame(scenario_prices, index=vol_scenarios)
    if prettify_columns:
        df.columns = [f"Spot x{col:.2f}" for col in df.columns]
        df.index = [f"Vol {'+' if idx > 0 else ''}{idx:.2%}" for idx in df.index]
    return df


def _calculate_bsm_scenarios(
    base_prices: pd.Series,
    s: pd.Series,
    strike: float,
    expiration: DATE_HINT,
    right: Literal["c", "p"],
    vol: pd.Series,
    r: pd.Series,
    dividend_type: DivType,
    pv_divs: pd.Series = None,
    q_factor: pd.Series = None,
    spot_scenarios: List[float] = None,
    vol_scenarios: List[float] = None,
    return_pnl: bool = False,
    return_pnl_in_pct: bool = False,
    prettify_columns: bool = False,
) -> pd.DataFrame:
    """Calculate option price scenarios using Black-Scholes-Merton model.

    Internal function that computes European-style option prices across a grid of spot
    and volatility scenarios. Spot scenarios are multiplicative (e.g., 0.9 = 10% down,
    1.1 = 10% up). Volatility scenarios are additive (e.g., 0.05 = +5% vol, -0.05 = -5% vol).

    Args:
        base_prices: Current market prices of the option (single-date Series).
        s: Current spot prices (single-date Series).
        strike: Strike price of the option.
        expiration: Option expiration date (YYYY-MM-DD string or datetime).
        right: Option type ('c' for call, 'p' for put).
        vol: Current implied volatilities (single-date Series).
        r: Risk-free interest rates (single-date Series).
        dividend_type: Dividend treatment type (DISCRETE or CONTINUOUS).
        pv_divs: Present value of discrete dividends (required if dividend_type=DISCRETE).
        q_factor: Continuous dividend yield factor (required if dividend_type=CONTINUOUS).
        spot_scenarios: List of spot price multipliers. E.g., [0.9, 1.0, 1.1] tests
            spot -10%, unchanged, +10%. Defaults to [1.0].
        vol_scenarios: List of volatility adjustments. E.g., [-0.05, 0.0, 0.05] tests
            vol -5%, unchanged, +5%. Defaults to [0.0].
        return_pnl: If True, returns P&L relative to base_prices instead of absolute prices.
        return_pnl_in_pct: If True (with return_pnl=True), returns P&L as percentage.
        prettify_columns: If True, formats column/index labels for display.

    Returns:
        DataFrame with volatility scenarios as rows, spot scenarios as columns, and
        option prices (or P&L) as values.

    Raises:
        AssertionError: If neither spot_scenarios nor vol_scenarios provided.
        AssertionError: If input series contain more than one date.
        AssertionError: If pv_divs not provided when dividend_type=DISCRETE.
        AssertionError: If q_factor not provided when dividend_type=CONTINUOUS.

    Examples:
        >>> # Internal usage - calculate scenario grid
        >>> scenarios_df = _calculate_bsm_scenarios(
        ...     base_prices=pd.Series([10.5]),
        ...     s=pd.Series([150.0]),
        ...     strike=150.0,
        ...     expiration="2025-06-20",
        ...     right="c",
        ...     vol=pd.Series([0.25]),
        ...     r=pd.Series([0.05]),
        ...     dividend_type=DivType.DISCRETE,
        ...     pv_divs=pd.Series([2.5]),
        ...     spot_scenarios=[0.9, 0.95, 1.0, 1.05, 1.1],
        ...     vol_scenarios=[-0.05, 0.0, 0.05],
        ...     prettify_columns=True
        ... )
    """
    assert any([spot_scenarios, vol_scenarios]), "At least one of spot_scenarios or vol_scenarios must be provided."
    assert len(vol.index) == 1, "Spot scenarios calculation only supports single-date series."

    ## Default scenarios
    if spot_scenarios is None:
        spot_scenarios = [1.0]
    if vol_scenarios is None:
        vol_scenarios = [0.0]

    if dividend_type == DivType.CONTINUOUS:
        assert q_factor is not None, "For continuous dividends, q_factor must be provided."
        dividends = q_factor
    else:
        assert pv_divs is not None, "For discrete dividends, pv_divs must be provided."
        dividends = pv_divs

    ## Sync all data to same index
    s, vol, r, dividends, base_prices = sync_date_index(s, vol, r, dividends, base_prices)
    scenario_prices: Dict[str, pd.Series] = {}

    ## Define pricing function for reuse
    def price_func(
        scenario_spot: pd.Series,
        scenario_vol: pd.Series,
        expiration: DATE_HINT,
        right: Literal["c", "p"],
        strike: float,
    ) -> pd.Series:
        t = time_distance_helper(start=scenario_spot.index, end=[expiration] * len(scenario_spot))
        if dividend_type == DivType.CONTINUOUS:
            F = vectorized_forward_continuous(
                S=scenario_spot.values,
                r=r.values,
                q_factor=dividends.values,
                T=t,
            )
        else:
            F = vectorized_forward_discrete(
                S=scenario_spot.values,
                r=r.values,
                pv_divs=dividends.values,
                T=t,
            )
        prices = black_scholes_vectorized(
            F=F,
            K=[strike] * len(scenario_spot),
            T=t,
            r=r.values,
            sigma=scenario_vol.values,
            option_type=[right] * len(scenario_spot),
        )
        return pd.Series(data=prices, index=scenario_spot.index, name="theoretical_price", dtype=float)

    ## Calculate prices for each scenarios
    scenarios = list(product(spot_scenarios, vol_scenarios))
    for spot_mult, vol_add in scenarios:
        scenario_spot = s * spot_mult
        scenario_vol = vol + vol_add

        prices = price_func(scenario_spot, scenario_vol, expiration, right, strike)
        prices = prices[0]
        if return_pnl:
            prices = prices - base_prices[0]
            if return_pnl_in_pct:
                prices = prices / base_prices[0]
        scenario_prices.setdefault(spot_mult, []).append(prices)

    df = pd.DataFrame(scenario_prices, index=vol_scenarios)
    if prettify_columns:
        df.columns = [f"Spot x{col:.2f}" for col in df.columns]
        df.index = [f"Vol {'+' if idx > 0 else ''}{idx:.2%}" for idx in df.index]
    return df


def calculate_scenarios(
    symbol: str,
    strike: float,
    expiration: DATE_HINT,
    right: Literal["c", "p"],
    as_of: Optional[DATE_HINT] = None,
    spot_scenarios: Optional[List[float]] = None,
    vol_scenarios: Optional[List[float]] = None,
    *,
    rt: Optional[bool] = False,
    market_model: Optional[OptionPricingModel] = None,
    endpoint_source: OptionSpotEndpointSource = None,
    dividend_type: Optional[DivType] = None,
    vol: Optional[VolatilityResult] = None,
    model_price: Optional[ModelPrice] = None,
    spot: Optional[SpotResult] = None,
    option_spot: Optional[OptionSpotResult] = None,
    r: Optional[RatesResult] = None,
    d: Optional[DividendsResult] = None,
    undo_adjust: bool = True,
    n_steps: Optional[int] = None,
    prettify_columns: bool = False,
    return_pnl: bool = False,
    return_pnl_in_pct: bool = False,
) -> ScenariosResult:
    """Calculate option price scenarios across spot and volatility stress levels.

    Performs scenario analysis by computing option prices across a grid of spot price
    and volatility levels. Useful for risk management, stress testing, and understanding
    option P&L sensitivity. Can return absolute prices or P&L relative to current market.

    Args:
        symbol: Ticker symbol for the underlying asset (e.g., "AAPL", "MSFT").
        as_of: Valuation date for scenario analysis (YYYY-MM-DD string or datetime).
        strike: Strike price of the option.
        expiration: Option expiration date (YYYY-MM-DD string or datetime).
        right: Option type ('c' for call, 'p' for put).
        spot_scenarios: List of spot price multipliers. E.g., [0.9, 0.95, 1.0, 1.05, 1.1]
            tests spot at -10%, -5%, unchanged, +5%, +10%. Defaults to DEFAULT_SCENARIOS.
        vol_scenarios: List of volatility adjustments (absolute). E.g., [-0.1, -0.05, 0.0, 0.05, 0.1]
            tests vol at -10%, -5%, unchanged, +5%, +10%. Defaults to DEFAULT_VOL_SCENARIOS.
        market_model: OptionPricingModel.BSM or BINOMIAL. Defaults to CONFIG setting.
        endpoint_source: Option data source for volatility (ORATS, HIST, QUOTE).
            Defaults to CONFIG setting.
        dividend_type: DivType.DISCRETE or DivType.CONTINUOUS. Defaults to CONFIG setting.
        vol: Optional pre-computed implied volatilities. If None, loads automatically.
        model_price: Which price to use (CLOSE, OPEN, MIDPOINT). Defaults to CONFIG setting.
        spot: Optional pre-computed spot prices. If None, loads automatically.
        option_spot: Optional pre-computed option market prices. If None, loads automatically.
        r: Optional pre-computed risk-free rates. If None, loads automatically.
        d: Optional pre-computed dividend data. If None, loads automatically.
        undo_adjust: If True, uses split-adjusted prices.
        n_steps: Number of time steps for binomial tree. Defaults to CONFIG.n_steps.
        prettify_columns: If True, formats grid labels for display ("Spot x0.95", "Vol +5.00%").
        return_pnl: If True, returns P&L relative to current market price.
        return_pnl_in_pct: If True (with return_pnl=True), returns P&L as percentage of market price.
        rt: If True, uses real-time data where available (default False).

    Returns:
        ScenariosResult containing DataFrame grid with volatility scenarios as rows,
        spot scenarios as columns, and prices/P&L as values, plus model metadata.

    Examples:
        >>> # Basic scenario analysis
        >>> result = calculate_scenarios(
        ...     symbol="AAPL",
        ...     as_of="2025-01-15",
        ...     strike=150.0,
        ...     expiration="2025-06-20",
        ...     right="c",
        ...     spot_scenarios=[0.9, 0.95, 1.0, 1.05, 1.1],
        ...     vol_scenarios=[-0.05, 0.0, 0.05],
        ...     prettify_columns=True
        ... )
        >>> print(result.grid)
        >>>
        >>> # P&L analysis for risk management
        >>> pnl_result = calculate_scenarios(
        ...     symbol="AAPL",
        ...     as_of="2025-01-15",
        ...     strike=150.0,
        ...     expiration="2025-06-20",
        ...     right="p",
        ...     spot_scenarios=[0.8, 0.9, 1.0, 1.1, 1.2],
        ...     vol_scenarios=[-0.1, 0.0, 0.1],
        ...     return_pnl=True,
        ...     return_pnl_in_pct=True,
        ...     prettify_columns=True
        ... )
        >>> print(f"Worst case: {pnl_result.grid.min().min():.2%}")
        >>>
        >>> # Custom stress scenarios with binomial model
        >>> result = calculate_scenarios(
        ...     symbol="AAPL",
        ...     as_of="2025-01-15",
        ...     strike=150.0,
        ...     expiration="2025-06-20",
        ...     right="c",
        ...     spot_scenarios=[0.7, 0.85, 1.0, 1.15, 1.3],
        ...     vol_scenarios=[-0.15, -0.075, 0.0, 0.075, 0.15],
        ...     market_model=OptionPricingModel.BINOMIAL,
        ...     n_steps=200,
        ...     return_pnl=True
        ... )
    """
    if not as_of and not rt:
        raise ValueError("Either as_of date must be provided or rt=True for real-time data.")
    
    market_model = market_model or CONFIG.option_model
    endpoint_source = endpoint_source or CONFIG.option_spot_endpoint_source
    dividend_type = dividend_type or CONFIG.dividend_type
    vol_model = vol or CONFIG.volatility_model
    model_price = model_price or CONFIG.model_price
    n_steps = n_steps or CONFIG.n_steps
    spot_scenarios = spot_scenarios or DEFAULT_SCENARIOS
    vol_scenarios = vol_scenarios or DEFAULT_VOL_SCENARIOS
    result = ScenariosResult()

    result.dividend_type = dividend_type
    result.market_model = market_model
    result.model_price = model_price
    result.vol_model = vol_model
    result.endpoint_source = endpoint_source
    result.expiration = to_datetime(expiration)
    result.right = right
    result.strike = strike
    result.symbol = symbol
    result.rt = False
    result.undo_adjust = undo_adjust
    result.spot_scenarios = spot_scenarios
    result.vol_scenarios = vol_scenarios
    result.as_of = to_datetime(as_of) if not rt else datetime.now()
    result.rt = rt

    # Create load request to determine which data to load
    load_request = _create_load_request(
        symbol=symbol,
        expiration=expiration,
        strike=strike,
        right=right,
        dividend_type=dividend_type,
        market_model=market_model,
        endpoint_source=endpoint_source,
        model_price=model_price,
        s=spot,
        r=r,
        d=d,
        vol=vol,
        option_spot=option_spot,
        is_scenario_load=True,
        undo_adjust=undo_adjust,
        as_of=as_of,
        rt=rt,
    )

    ## Router clips to a single anchor row for rt/as_of; full window for hist.
    packet = _load_model_data(load_request)

    s, r, vol, base_prices, d = (
        packet.spot.timeseries if not packet.spot.is_empty() else spot.timeseries,
        packet.rates.timeseries if not packet.rates.is_empty() else r.timeseries,
        packet.vol.timeseries if not packet.vol.is_empty() else vol.timeseries,
        packet.option_spot.price if not packet.option_spot.is_empty() else option_spot.price,
        packet.dividend.timeseries if not packet.dividend.is_empty() else d.timeseries,
    )
    # Use loaded data to calculate theoretical prices
    if market_model == OptionPricingModel.BINOMIAL:
        s, vol, r, d, base_prices = sync_date_index(s, vol, r, d, base_prices)
        df = _calculate_binomial_scenarios(
            base_prices=base_prices,
            s=s,
            strike=strike,
            expiration=expiration,
            right=right,
            vol=vol,
            r=r,
            dividend_type=dividend_type,
            dividends=d,
            spot_scenarios=spot_scenarios,
            vol_scenarios=vol_scenarios,
            n_steps=n_steps,
            prettify_columns=prettify_columns,
            return_pnl=return_pnl,
            return_pnl_in_pct=return_pnl_in_pct,
        )
        result.grid = df
        return result

    ## BSM model
    elif market_model == OptionPricingModel.BSM:
        s, vol, r, d, base_prices = sync_date_index(s, vol, r, d, base_prices)
        if dividend_type == DivType.DISCRETE:
            pv_divs = vectorized_discrete_pv(
                schedules=d.values,
                _valuation_dates=s.index,
                _end_dates=[to_datetime(expiration)] * len(s),
                r=r.values,
            )
            pv_divs = pd.Series(data=pv_divs, index=s.index, name="pv_dividends", dtype=float)
            q_factor = None
        else:
            pv_divs = None
            q_factor = get_vectorized_continuous_dividends(
                div_rates=d.values,
                _valuation_dates=s.index,
                _end_dates=[to_datetime(expiration)] * len(s),
            )
            q_factor = pd.Series(data=q_factor, index=s.index, name="q_factor", dtype=float)
        df = _calculate_bsm_scenarios(
            base_prices=base_prices,
            s=s,
            strike=strike,
            expiration=expiration,
            right=right,
            vol=vol,
            r=r,
            dividend_type=dividend_type,
            pv_divs=pv_divs,
            q_factor=q_factor,
            spot_scenarios=spot_scenarios,
            vol_scenarios=vol_scenarios,
            prettify_columns=prettify_columns,
            return_pnl=return_pnl,
            return_pnl_in_pct=return_pnl_in_pct,
        )
        result.grid = df
        return result
