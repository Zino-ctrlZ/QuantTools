"""Greek data manager for computing option sensitivities (delta, gamma, vega, theta, rho).

This module provides the GreekDataManager class for calculating option greeks using
various pricing models (Black-Scholes-Merton, Cox-Ross-Rubinstein binomial). It handles
the complete workflow including data loading, caching, model selection, and result
formatting.

Key Features:
    - Multiple pricing models: BSM, CRR binomial
    - Support for American and European exercise styles
    - Discrete and continuous dividend treatments
    - Automatic data loading and caching
    - Real-time and historical greek calculation
    - Configurable greek selection (compute only needed greeks)

Typical Usage:
    >>> from trade.datamanager.greeks import GreekDataManager
    >>> from trade.datamanager._enums import GreekType
    >>> from trade.optionlib.config.types import DivType
    >>>
    >>> # Initialize manager for AAPL
    >>> greek_mgr = GreekDataManager("AAPL")
    >>>
    >>> # Get all greeks for an option
    >>> result = greek_mgr.get_greeks_timeseries(
    ...     start_date="2025-01-01",
    ...     end_date="2025-01-31",
    ...     expiration="2025-06-20",
    ...     strike=150.0,
    ...     right="c",
    ...     dividend_type=DivType.DISCRETE,
    ... )
    >>> print(result.timeseries[["delta", "gamma", "vega"]].head())
    >>>
    >>> # Get only specific greeks
    >>> result = greek_mgr.get_greeks_timeseries(
    ...     start_date="2025-01-01",
    ...     end_date="2025-01-31",
    ...     expiration="2025-06-20",
    ...     strike=150.0,
    ...     right="c",
    ...     greeks_to_compute=[GreekType.DELTA, GreekType.GAMMA]
    ... )
"""

from datetime import datetime
from typing import Optional, Union, List
import pandas as pd
from trade.datamanager.result import (
    GreekResultSet,
    SpotResult,
    RatesResult,
    DividendsResult,
    VolatilityResult,
    ForwardResult,
)
from trade.datamanager.utils.vol_helpers import (
    _handle_cache_for_vol,
    _merge_and_cache_vol_result,
    _prepare_vol_calculation_setup,
    reconcile_checked_missing_dates_for_option_artifact,
    _certify_option_model_result,
)
from trade.datamanager.utils.date import sync_date_index
from trade.datamanager.utils.model import _load_model_data_timeseries, LoadRequest
from trade.datamanager.utils.greeks_helpers import _prepare_greeks_to_compute, _get_prefilled_greek_result_set
from trade.datamanager._enums import (
    GreekType,
    ModelPrice,
    OptionPricingModel,
    OptionSpotEndpointSource,
    VolatilityModel,
    RealTimeFallbackOption,
    ArtifactType,
    CertificationLevel,
    Interval,
)
from trade.optionlib.greeks.numerical.binomial import binomial_tree_greeks
from trade.optionlib.greeks.numerical.black_scholes import vectorized_black_scholes_greeks
from trade.optionlib.assets.dividend import (
    vectorized_discrete_pv,
    get_vectorized_continuous_dividends,
    vector_convert_to_time_frac,
)
from trade.helpers.helper import to_datetime
from trade.datamanager._enums import SeriesId
from trade.datamanager.base import BaseDataManager, CacheSpec
from trade.datamanager.config import OptionDataConfig
from trade.helpers.helper_types import DATE_HINT
from trade.optionlib.config.types import DivType
from trade.helpers.Logging import setup_logger
from trade.datamanager.utils.logging import get_logging_level, UTILS_LOGGER_NAME
from trade.datamanager.utils.point_in_time import resolve_value_at_date
from trade import MARKET_CLOSE

logger = setup_logger(UTILS_LOGGER_NAME, stream_log_level=get_logging_level())


class GreekDataManager(BaseDataManager):
    """Manager for computing and caching option greeks (delta, gamma, vega, theta, rho).

    Class that orchestrates the computation of option sensitivities (greeks) using
    various option pricing models. Automatically loads required market data (spot,
    forward, rates, dividends, implied volatilities) and caches results for efficient reuse.

    Supports two pricing approaches:
        1. Black-Scholes-Merton (BSM) - Fast, European-style greeks
        2. Cox-Ross-Rubinstein (CRR) - Binomial tree, supports American exercise

    Attributes:
        CONFIG: Configuration object with default settings for pricing models.
        CACHE_NAME: Cache identifier for greek data.
        CACHE_SPEC: Cache specification for data persistence.
        DEFAULT_SERIES_ID: Default series identifier (historical data).
        symbol: Ticker symbol for the underlying asset.

    Examples:
        >>> # Basic usage with BSM model
        >>> greek_mgr = GreekDataManager("AAPL")
        >>> result = greek_mgr.get_greeks_timeseries(
        ...     start_date="2025-01-01",
        ...     end_date="2025-01-31",
        ...     expiration="2025-06-20",
        ...     strike=150.0,
        ...     right="c"
        ... )

        >>> # Get only delta and gamma
        >>> from trade.datamanager._enums import GreekType
        >>> result = greek_mgr.get_greeks_timeseries(
        ...     start_date="2025-01-01",
        ...     end_date="2025-01-31",
        ...     expiration="2025-06-20",
        ...     strike=150.0,
        ...     right="c",
        ...     greeks_to_compute=[GreekType.DELTA, GreekType.GAMMA]
        ... )

        >>> # Real-time greeks
        >>> rt_greeks = greek_mgr.rt(
        ...     expiration="2025-06-20",
        ...     strike=150.0,
        ...     right="c"
        ... )
    """

    CONFIG: OptionDataConfig = OptionDataConfig()
    CACHE_NAME: str = "greek_datamanager_cache"
    CACHE_SPEC: CacheSpec = CacheSpec(cache_fname=CACHE_NAME)
    DEFAULT_SERIES_ID: SeriesId = SeriesId.HIST

    def __init__(self, symbol: str):
        """Initialize GreekDataManager with symbol-specific configuration.

        Args:
            symbol: Ticker symbol for the underlying asset (e.g., "AAPL", "MSFT").

        Examples:
            >>> greek_mgr = GreekDataManager("AAPL")
        """
        super().__init__(symbol=symbol)

    def get_greeks_timeseries(
        self,
        start_date: DATE_HINT,
        end_date: DATE_HINT,
        expiration: DATE_HINT,
        strike: float,
        right: str,
        dividend_type: Optional[DivType] = None,
        *,
        greeks_to_compute: Optional[Union[List[GreekType], GreekType]] = GreekType.GREEKS,
        f: Optional[ForwardResult] = None,
        S: Optional[SpotResult] = None,
        r: Optional[RatesResult] = None,
        d: Optional[DividendsResult] = None,
        vol: Optional[VolatilityResult] = None,
        endpoint_source: Optional[OptionSpotEndpointSource] = None,
        market_model: Optional[OptionPricingModel] = None,
        model_price: Optional[ModelPrice] = None,
        undo_adjust: bool = True,
        certification_level: Optional[CertificationLevel] = None,
    ) -> GreekResultSet:
        """Returns daily option greeks timeseries using specified pricing model.

        Computes option sensitivities (delta, gamma, vega, theta, rho) for each business day
        in [start_date, end_date]. Automatically selects appropriate pricing model (BSM or
        binomial) and loads required market data. Uses caching to avoid redundant computations.

        Args:
            start_date: First valuation date (YYYY-MM-DD string or datetime).
            end_date: Last valuation date (YYYY-MM-DD string or datetime).
            expiration: Option expiration date (YYYY-MM-DD string or datetime).
            strike: Strike price of the option.
            right: Option type ('c' for call, 'p' for put).
            dividend_type: DivType.DISCRETE or DivType.CONTINUOUS. Defaults to CONFIG setting.
            greeks_to_compute: Which greeks to compute. Single GreekType or list of GreekTypes.
                Defaults to GreekType.GREEKS (all standard greeks). Available: DELTA, GAMMA,
                VEGA, THETA, RHO, CHARM, VANNA, SPEED, ZOMMA, COLOR, ULTIMA.
            f: Optional pre-computed forward prices. If None, loads automatically.
            S: Optional pre-computed spot prices. If None, loads automatically.
            r: Optional pre-computed risk-free rates. If None, loads automatically.
            d: Optional pre-computed dividend data. If None, loads automatically.
            vol: Optional pre-computed implied volatilities. If None, loads automatically.
            endpoint_source: Option data source (ORATS, HIST, QUOTE). Defaults to CONFIG setting.
            market_model: OptionPricingModel.BSM or BINOMIAL. Defaults to CONFIG setting.
            model_price: Which price to use (CLOSE, OPEN, MIDPOINT). Defaults to CONFIG setting.
            undo_adjust: If True, uses split-adjusted prices.

        Returns:
            GreekResultSet containing DataFrame with computed greeks as columns and
            DatetimeIndex, plus model metadata and cache key.

        Raises:
            ValueError: If unsupported market model is specified.

        Examples:
            >>> # Basic usage - compute all greeks
            >>> greek_mgr = GreekDataManager("AAPL")
            >>> result = greek_mgr.get_greeks_timeseries(
            ...     start_date="2025-01-01",
            ...     end_date="2025-01-31",
            ...     expiration="2025-06-20",
            ...     strike=150.0,
            ...     right="c",
            ...     dividend_type=DivType.DISCRETE
            ... )
            >>> print(result.timeseries[["delta", "gamma", "vega"]].head())

            >>> # Compute only delta and gamma with binomial model
            >>> from trade.datamanager._enums import GreekType, OptionPricingModel
            >>> result = greek_mgr.get_greeks_timeseries(
            ...     start_date="2025-01-01",
            ...     end_date="2025-01-31",
            ...     expiration="2025-06-20",
            ...     strike=150.0,
            ...     right="p",
            ...     greeks_to_compute=[GreekType.DELTA, GreekType.GAMMA],
            ...     market_model=OptionPricingModel.BINOMIAL
            ... )

            >>> # Provide pre-computed volatility data
            >>> vol_mgr = VolDataManager("AAPL")
            >>> vol_result = vol_mgr.get_implied_volatility_timeseries(
            ...     start_date="2025-01-01",
            ...     end_date="2025-01-31",
            ...     expiration="2025-06-20",
            ...     strike=150.0,
            ...     right="c"
            ... )
            >>> greek_result = greek_mgr.get_greeks_timeseries(
            ...     start_date="2025-01-01",
            ...     end_date="2025-01-31",
            ...     expiration="2025-06-20",
            ...     strike=150.0,
            ...     right="c",
            ...     vol=vol_result
            ... )
        """
        dividend_type = dividend_type or self.CONFIG.dividend_type
        endpoint_source = endpoint_source or self.CONFIG.option_spot_endpoint_source
        market_model = market_model or self.CONFIG.option_model
        vol_model = VolatilityModel.MARKET

        result = _get_prefilled_greek_result_set(
            key=None,
            symbol=self.symbol,
            strike=strike,
            expiration=expiration,
            right=right,
            endpoint_source=endpoint_source,
            market_model=market_model,
            vol_model=vol_model,
            dividend_type=dividend_type,
            model_price=model_price,
            undo_adjust=undo_adjust,
        )
        if market_model == OptionPricingModel.BINOMIAL:
            return self._get_binomial_greeks(
                start_date=start_date,
                end_date=end_date,
                expiration=expiration,
                strike=strike,
                right=right,
                dividend_type=dividend_type,
                result=result,
                greeks_to_compute=greeks_to_compute,
                S=S,
                r=r,
                d=d,
                vol=vol,
                endpoint_source=endpoint_source,
                undo_adjust=undo_adjust,
                model_price=model_price,
                certification_level=certification_level,
            )
        elif market_model == OptionPricingModel.BSM or market_model == OptionPricingModel.EURO_EQIV:
            return self._get_bsm_greeks(
                start_date=start_date,
                end_date=end_date,
                expiration=expiration,
                strike=strike,
                right=right,
                dividend_type=dividend_type,
                result=result,
                greeks_to_compute=greeks_to_compute,
                f=f,
                S=S,
                r=r,
                d=d,
                vol=vol,
                endpoint_source=endpoint_source,
                undo_adjust=undo_adjust,
                model_price=model_price,
                certification_level=certification_level,
            )
        else:
            raise ValueError(f"Unsupported market model: {market_model}")

    def _get_binomial_greeks(
        self,
        start_date: DATE_HINT,
        end_date: DATE_HINT,
        expiration: DATE_HINT,
        strike: float,
        right: str,
        dividend_type: Optional[DivType] = None,
        *,
        result: Optional[GreekResultSet] = None,
        greeks_to_compute: Optional[Union[List[GreekType], GreekType]] = None,
        S: Optional[SpotResult] = None,
        r: Optional[RatesResult] = None,
        d: Optional[DividendsResult] = None,
        vol: Optional[VolatilityResult] = None,
        endpoint_source: Optional[OptionSpotEndpointSource] = None,
        model_price: Optional[ModelPrice] = None,
        undo_adjust: bool = True,
        certification_level: Optional[CertificationLevel] = None,
    ) -> GreekResultSet:
        """Compute option greeks using Cox-Ross-Rubinstein binomial tree model.

        Internal method that calculates daily option sensitivities using CRR binomial trees.
        Supports American exercise. Automatically loads required data (spot, rates, dividends,
        implied volatilities) if not provided. Uses caching for efficient reuse.

        Note: Binomial tree model computes all greeks simultaneously, so caching stores the
        complete set even if only specific greeks are requested.

        Args:
            start_date: First valuation date (YYYY-MM-DD string or datetime).
            end_date: Last valuation date (YYYY-MM-DD string or datetime).
            expiration: Option expiration date (YYYY-MM-DD string or datetime).
            strike: Strike price of the option.
            right: Option type ('c' for call, 'p' for put).
            dividend_type: Dividend treatment type (DISCRETE or CONTINUOUS).
            result: Optional pre-initialized GreekResultSet container.
            greeks_to_compute: Which greeks to return (all are computed regardless).
            S: Optional pre-computed spot prices. If None, loads automatically.
            r: Optional pre-computed risk-free rates. If None, loads automatically.
            d: Optional pre-computed dividend data. If None, loads automatically.
            vol: Optional pre-computed implied volatilities. If None, loads automatically.
            endpoint_source: Option data source for volatility calculation.
            model_price: Which price to use (CLOSE, OPEN, MIDPOINT).
            undo_adjust: If True, uses split-adjusted prices.

        Returns:
            GreekResultSet containing DataFrame with computed greeks as columns and
            DatetimeIndex, plus model metadata and cache key.

        Examples:
            >>> # Internal usage - typically called via get_greeks_timeseries
            >>> result = greek_mgr._get_binomial_greeks(
            ...     start_date="2025-01-01",
            ...     end_date="2025-01-31",
            ...     expiration="2025-06-20",
            ...     strike=150.0,
            ...     right="c",
            ...     dividend_type=DivType.DISCRETE
            ... )
        """

        ## biomial tree greeks calculation function calculates all greeks at once. So I'll check cache
        ## for a greek and if missing, compute all and store in cache.
        ## endpoint_source & div_type will resolved at `get_timeseries` level; the frontend function.

        endpoint_source = endpoint_source or self.CONFIG.option_spot_endpoint_source
        model_price = model_price or self.CONFIG.model_price
        result = result or GreekResultSet()
        result, dividend_type, endpoint_source, start_str, end_str, start_date, end_date = (
            _prepare_vol_calculation_setup(
                self, start_date, end_date, expiration, strike, right, dividend_type, endpoint_source, result
            )
        )
        ## Using self.CONFIG allows frontend to override default settings for greeks_to_compute.
        ## Also allows user to specify greeks_to_compute at function call level which can get hidden as calls become nested.
        greeks_to_compute = greeks_to_compute or self.CONFIG.greeks_to_compute
        greeks_to_compute = _prepare_greeks_to_compute(greeks_to_compute)
        key = self.make_key(
            symbol=self.symbol,
            interval=Interval.EOD,
            artifact_type=ArtifactType.GREEKS,
            series_id=SeriesId.HIST,
            option_pricing_model=OptionPricingModel.BINOMIAL,
            volatility_model=VolatilityModel.MARKET,
            model_price=model_price,
            dividend_type=dividend_type,
            endpoint_source=endpoint_source,
            expiration=expiration,
            strike=strike,
            right=right,
        )
        result.key = key
        result.model_price = model_price

        cached_data, is_partial, start_date, end_date, early_return, checked_missing_dates = _handle_cache_for_vol(
            self, key, start_date, end_date, result, optional_name="greeks"
        )
        if early_return:
            result.timeseries = cached_data[greeks_to_compute]
            return _certify_option_model_result(
                result,
                start_date,
                end_date,
                cache_key=key,
                checked_missing_dates=checked_missing_dates,
                certification_level=certification_level,
            )

        expiration_ts = to_datetime(expiration)
        expiration_ts = expiration_ts.replace(
            hour=MARKET_CLOSE.hour, minute=MARKET_CLOSE.minute
        )  # Set to end of day for accurate T calculation
        request = self._create_load_request(
            start_date=start_date,
            end_date=end_date,
            expiration=expiration_ts,
            strike=strike,
            right=right,
            dividend_type=dividend_type,
            market_model=OptionPricingModel.BINOMIAL,
            model_price=model_price,
            endpoint_source=endpoint_source,
            s=S,
            r=r,
            d=d,
            vol=vol,
            undo_adjust=undo_adjust,
        )
        model_data = _load_model_data_timeseries(request)
        checked_missing_dates = reconcile_checked_missing_dates_for_option_artifact(
            symbol=self.symbol,
            strike=strike,
            right=right,
            expiration=expiration,
            start_date=start_date,
            end_date=end_date,
            cached_checked_missing=checked_missing_dates,
        )
        S = model_data.spot.timeseries if request.load_spot else S.timeseries
        r = model_data.rates.timeseries if request.load_rates else r.timeseries
        d = model_data.dividend.timeseries if request.load_dividend else d.timeseries
        vol = model_data.vol.timeseries if request.load_vol else vol.timeseries
        S, r, d, vol = sync_date_index(S, r, d, vol)

        if dividend_type == DivType.DISCRETE:
            d = vector_convert_to_time_frac(
                schedules=d,
                valuation_dates=to_datetime(S.index.tolist(), format="%Y-%m-%d"),
                end_dates=to_datetime([expiration_ts] * len(S), format="%Y-%m-%d"),
            )

        ## Now compute greeks
        greeks_res_dict = binomial_tree_greeks(
            K=[strike] * len(S),
            expiration=[expiration_ts] * len(S),
            sigma=vol,
            S=S,
            r=r,
            N=[100] * len(S),
            dividend_type=[dividend_type.value] * len(S),
            div_amount=d,
            option_type=[right] * len(S),
            start_date=to_datetime(S.index.tolist(), format="%Y-%m-%d"),
            valuation_date=to_datetime(S.index.tolist(), format="%Y-%m-%d"),
            american=[True] * len(S),
        )

        ## Remove "models" key if exists
        if "models" in greeks_res_dict:
            del greeks_res_dict["models"]

        greeks_df = pd.DataFrame(greeks_res_dict, index=S.index)

        ## Use utility: Merge and cache
        greeks_df = _merge_and_cache_vol_result(
            self,
            greeks_df,
            cached_data,
            is_partial,
            key,
            start_str,
            end_str,
            checked_missing_dates=checked_missing_dates,
        )
        result.timeseries = greeks_df[greeks_to_compute]

        return _certify_option_model_result(
            result,
            start_date,
            end_date,
            cache_key=key,
            checked_missing_dates=checked_missing_dates,
            certification_level=certification_level,
        )

    def _get_bsm_greeks(
        self,
        start_date: DATE_HINT,
        end_date: DATE_HINT,
        expiration: DATE_HINT,
        strike: float,
        right: str,
        dividend_type: Optional[DivType] = None,
        *,
        result: Optional[GreekResultSet] = None,
        greeks_to_compute: Optional[Union[List[GreekType], GreekType]] = GreekType.GREEKS,
        f: Optional[ForwardResult] = None,
        S: Optional[SpotResult] = None,
        r: Optional[RatesResult] = None,
        d: Optional[DividendsResult] = None,
        vol: Optional[VolatilityResult] = None,
        endpoint_source: Optional[OptionSpotEndpointSource] = None,
        model_price: Optional[ModelPrice] = None,
        undo_adjust: bool = True,
        certification_level: Optional[CertificationLevel] = None,
    ) -> GreekResultSet:
        """Compute option greeks using Black-Scholes-Merton model.

        Internal method that calculates daily option sensitivities using closed-form BSM
        formulas. Only supports European-style greeks. Automatically loads required data
        (forward, spot, rates, dividends, implied volatilities) if not provided. Uses
        caching for efficient reuse.

        Note: BSM model computes all greeks simultaneously, so caching stores the complete
        set even if only specific greeks are requested.

        Args:
            start_date: First valuation date (YYYY-MM-DD string or datetime).
            end_date: Last valuation date (YYYY-MM-DD string or datetime).
            expiration: Option expiration date (YYYY-MM-DD string or datetime).
            strike: Strike price of the option.
            right: Option type ('c' for call, 'p' for put).
            dividend_type: Dividend treatment type (DISCRETE or CONTINUOUS).
            result: Optional pre-initialized GreekResultSet container.
            greeks_to_compute: Which greeks to return (all are computed regardless).
            f: Optional pre-computed forward prices. If None, loads automatically.
            S: Optional pre-computed spot prices. If None, loads automatically.
            r: Optional pre-computed risk-free rates. If None, loads automatically.
            d: Optional pre-computed dividend data. If None, loads automatically.
            vol: Optional pre-computed implied volatilities. If None, loads automatically.
            endpoint_source: Option data source for volatility calculation.
            model_price: Which price to use (CLOSE, OPEN, MIDPOINT).
            undo_adjust: If True, uses split-adjusted prices.

        Returns:
            GreekResultSet containing DataFrame with computed greeks as columns and
            DatetimeIndex, plus model metadata and cache key.

        Examples:
            >>> # Internal usage - typically called via get_greeks_timeseries
            >>> result = greek_mgr._get_bsm_greeks(
            ...     start_date="2025-01-01",
            ...     end_date="2025-01-31",
            ...     expiration="2025-06-20",
            ...     strike=150.0,
            ...     right="c",
            ...     dividend_type=DivType.DISCRETE
            ... )
        """

        ## biomial tree greeks calculation function calculates all greeks at once. So I'll check cache
        ## for a greek and if missing, compute all and store in cache.
        ## endpoint_source & div_type will resolved at `get_timeseries` level; the frontend function.
        endpoint_source = endpoint_source or self.CONFIG.option_spot_endpoint_source
        model_price = model_price or self.CONFIG.model_price
        result = result or GreekResultSet()
        result, dividend_type, endpoint_source, start_str, end_str, start_date, end_date = (
            _prepare_vol_calculation_setup(
                self, start_date, end_date, expiration, strike, right, dividend_type, endpoint_source, result
            )
        )

        greeks_to_compute = _prepare_greeks_to_compute(greeks_to_compute)
        key = self.make_key(
            symbol=self.symbol,
            interval=Interval.EOD,
            artifact_type=ArtifactType.GREEKS,
            series_id=SeriesId.HIST,
            option_pricing_model=OptionPricingModel.BSM,
            volatility_model=VolatilityModel.MARKET,
            model_price=model_price,
            dividend_type=dividend_type,
            endpoint_source=endpoint_source,
            expiration=expiration,
            strike=strike,
            right=right,
        )
        result.key = key
        result.model_price = model_price
        result.endpoint_source = endpoint_source

        cached_data, is_partial, start_date, end_date, early_return, checked_missing_dates = _handle_cache_for_vol(
            self, key, start_date, end_date, result, optional_name="greeks"
        )
        if early_return:
            result.timeseries = cached_data[greeks_to_compute]
            return _certify_option_model_result(
                result,
                start_date,
                end_date,
                cache_key=key,
                checked_missing_dates=checked_missing_dates,
                certification_level=certification_level,
            )

        request = self._create_load_request(
            start_date=start_date,
            end_date=end_date,
            expiration=expiration,
            strike=strike,
            right=right,
            dividend_type=dividend_type,
            market_model=OptionPricingModel.BSM,
            endpoint_source=endpoint_source,
            s=S,
            f=f,
            r=r,
            d=d,
            vol=vol,
            undo_adjust=undo_adjust,
            model_price=model_price,
        )
        model_data = _load_model_data_timeseries(request)
        checked_missing_dates = reconcile_checked_missing_dates_for_option_artifact(
            symbol=self.symbol,
            strike=strike,
            right=right,
            expiration=expiration,
            start_date=start_date,
            end_date=end_date,
            cached_checked_missing=checked_missing_dates,
        )
        S = model_data.spot.timeseries if request.load_spot else S.timeseries
        r = model_data.rates.timeseries if request.load_rates else r.timeseries
        d = model_data.dividend.timeseries if request.load_dividend else d.timeseries
        vol = model_data.vol.timeseries if request.load_vol else vol.timeseries
        f = model_data.forward.timeseries if request.load_forward else f.timeseries
        s, f, r, d, vol = sync_date_index(S, f, r, d, vol)
        expiration_ts = to_datetime(expiration)
        expiration_ts = expiration_ts.replace(
            hour=MARKET_CLOSE.hour, minute=MARKET_CLOSE.minute
        )  # Set to end of day for accurate T calculation

        ## Convert dividends to present value amounts
        if dividend_type == DivType.DISCRETE:
            pv_divs = vectorized_discrete_pv(
                schedules=d,
                _valuation_dates=f.index.tolist(),
                _end_dates=[expiration_ts] * len(f),
                r=r,
            )

        ## Continuous dividends. Discount dividend rates to present value amounts
        else:
            pv_divs = get_vectorized_continuous_dividends(
                div_rates=d.values, _valuation_dates=f.index.tolist(), _end_dates=[expiration_ts] * len(f)
            )

        ## Now compute greeks
        greeks_res_dict = vectorized_black_scholes_greeks(
            S=s,
            K=[strike] * len(s),
            F=f,
            r=r,
            sigma=vol,
            valuation_dates=s.index.tolist(),
            end_dates=[expiration_ts] * len(s),
            option_type=[right.lower()] * len(s),
            dividend_type=dividend_type.value,
            div_amount=pv_divs,
        )
        ## Remove "models" key if exists
        if "models" in greeks_res_dict:
            del greeks_res_dict["models"]

        greeks_df = pd.DataFrame(greeks_res_dict, index=s.index)

        ## Use utility: Merge and cache
        greeks_df = _merge_and_cache_vol_result(
            self,
            greeks_df,
            cached_data,
            is_partial,
            key,
            start_str,
            end_str,
            checked_missing_dates=checked_missing_dates,
        )
        result.timeseries = greeks_df[greeks_to_compute]

        return _certify_option_model_result(
            result,
            start_date,
            end_date,
            cache_key=key,
            checked_missing_dates=checked_missing_dates,
            certification_level=certification_level,
        )

    def get_at_time_greeks(
        self,
        as_of: DATE_HINT,
        expiration: DATE_HINT,
        strike: float,
        right: str,
        dividend_type: Optional[DivType] = None,
        *,
        greeks_to_compute: Optional[Union[List[GreekType], GreekType]] = GreekType.GREEKS,
        S: Optional[SpotResult] = None,
        f: Optional[ForwardResult] = None,
        r: Optional[RatesResult] = None,
        d: Optional[DividendsResult] = None,
        vol: Optional[VolatilityResult] = None,
        endpoint_source: Optional[OptionSpotEndpointSource] = None,
        market_model: Optional[OptionPricingModel] = None,
        undo_adjust: bool = True,
        fallback_option: Optional[RealTimeFallbackOption] = None,
        model_price: Optional[ModelPrice] = None,
    ) -> GreekResultSet:
        """Get option greeks at a specific point in time.

        Fetches a 10-business-day lookback window certified at L1, then resolves
        greeks per ``fallback_option``.
        """
        vol_model = VolatilityModel.MARKET
        dividend_type = dividend_type or self.CONFIG.dividend_type
        endpoint_source = endpoint_source or self.CONFIG.option_spot_endpoint_source
        market_model = market_model or self.CONFIG.option_model
        fallback_option = fallback_option or self.CONFIG.real_time_fallback_option
        model_price = model_price or self.CONFIG.model_price

        def _fetch(start: str, end: str) -> GreekResultSet:
            return self.get_greeks_timeseries(
                start_date=start,
                end_date=end,
                expiration=expiration,
                strike=strike,
                right=right,
                dividend_type=dividend_type,
                greeks_to_compute=greeks_to_compute,
                S=S,
                r=r,
                d=d,
                vol=vol,
                f=f,
                endpoint_source=endpoint_source,
                market_model=market_model,
                undo_adjust=undo_adjust,
                model_price=model_price,
                certification_level=CertificationLevel.L1,
            )

        row, meta = resolve_value_at_date(
            as_of,
            fetch_timeseries=_fetch,
            extract_timeseries=lambda r: r.timeseries if r.timeseries is not None else pd.Series(dtype=float),
            fallback_option=fallback_option,
        )

        result = meta.source_result if meta.source_result is not None else GreekResultSet()
        if (
            meta.source_result is None
            and fallback_option in (RealTimeFallbackOption.NAN, RealTimeFallbackOption.ZEROED)
            and isinstance(row, pd.Series)
        ):
            fill = row.iloc[0]
            cols = _prepare_greeks_to_compute(greeks_to_compute)
            row = pd.DataFrame({g: [fill] for g in cols}, index=row.index)
        result.timeseries = row
        result.fallback_option = meta.fallback_option
        if meta.source_result is None:
            result.vol_model = vol_model
            result.market_model = market_model
            result.expiration = to_datetime(expiration)
            result.right = right
            result.strike = strike
            result.endpoint_source = endpoint_source
            result.dividend_type = dividend_type
            result.symbol = self.symbol
            result.model_price = model_price
        return result

    def rt(
        self,
        expiration: DATE_HINT,
        strike: float,
        right: str,
        dividend_type: Optional[DivType] = None,
        *,
        greeks_to_compute: Optional[Union[List[GreekType], GreekType]] = GreekType.GREEKS,
        S: Optional[SpotResult] = None,
        f: Optional[ForwardResult] = None,
        r: Optional[RatesResult] = None,
        d: Optional[DividendsResult] = None,
        vol: Optional[VolatilityResult] = None,
        market_model: Optional[OptionPricingModel] = None,
        undo_adjust: bool = True,
        fallback_option: Optional[RealTimeFallbackOption] = None,
        model_price: Optional[ModelPrice] = None,
    ) -> GreekResultSet:
        """Get real-time option greeks using current market data.

        Convenience method that computes greeks as of current datetime using QUOTE endpoint
        for live market prices. Ideal for real-time trading systems and live option monitoring.

        Args:
            expiration: Option expiration date (YYYY-MM-DD string or datetime).
            strike: Strike price of the option.
            right: Option type ('c' for call, 'p' for put).
            dividend_type: DivType.DISCRETE or DivType.CONTINUOUS. Defaults to CONFIG setting.
            greeks_to_compute: Which greeks to compute. Single GreekType or list of GreekTypes.
            S: Optional pre-computed spot prices. If None, loads real-time data.
            f: Optional pre-computed forward prices. If None, loads real-time data.
            r: Optional pre-computed risk-free rates. If None, loads real-time data.
            d: Optional pre-computed dividend data. If None, loads real-time data.
            vol: Optional pre-computed implied volatilities. If None, loads real-time data.
            market_model: OptionPricingModel.BSM or BINOMIAL. Defaults to CONFIG setting.
            undo_adjust: If True, uses split-adjusted prices.
            fallback_option: How to handle market closed (USE_LAST_AVAILABLE, NAN, ZERO).
                Defaults to CONFIG setting.
            model_price: Which price to use (CLOSE, OPEN, MIDPOINT). Defaults to CONFIG setting.

        Returns:
            GreekResultSet containing single-row DataFrame with computed greeks as columns,
            plus model metadata and cache key.

        Examples:
            >>> # Get real-time greeks during market hours
            >>> greek_mgr = GreekDataManager("AAPL")
            >>> result = greek_mgr.rt(
            ...     expiration="2025-06-20",
            ...     strike=150.0,
            ...     right="c"
            ... )
            >>> print(f"Delta: {result.timeseries['delta'].iloc[0]:.4f}")

            >>> # Get only delta and vega in real-time
            >>> from trade.datamanager._enums import GreekType
            >>> result = greek_mgr.rt(
            ...     expiration="2025-06-20",
            ...     strike=150.0,
            ...     right="c",
            ...     greeks_to_compute=[GreekType.DELTA, GreekType.VEGA]
            ... )

            >>> # Use last available if market closed
            >>> from trade.datamanager._enums import RealTimeFallbackOption
            >>> result = greek_mgr.rt(
            ...     expiration="2025-06-20",
            ...     strike=150.0,
            ...     right="c",
            ...     fallback_option=RealTimeFallbackOption.USE_LAST_AVAILABLE
            ... )
        """

        res = self.get_at_time_greeks(
            as_of=datetime.now(),
            expiration=expiration,
            strike=strike,
            right=right,
            dividend_type=dividend_type,
            greeks_to_compute=greeks_to_compute,
            S=S,
            r=r,
            d=d,
            f=f,
            vol=vol,
            endpoint_source=OptionSpotEndpointSource.QUOTE,
            market_model=market_model,
            undo_adjust=undo_adjust,
            fallback_option=fallback_option,
            model_price=model_price,
        )
        res.rt = True
        return res

    def _create_load_request(
        ## Requied parameters to ensure correct data is loaded
        self,
        start_date: DATE_HINT,
        end_date: DATE_HINT,
        expiration: DATE_HINT,
        strike: float,
        right: str,
        dividend_type: DivType,
        market_model: OptionPricingModel,
        endpoint_source: OptionSpotEndpointSource,
        model_price: ModelPrice,
        *,
        ## Optional pre-loaded data. If not provided, will be loaded.
        s: Optional[SpotResult] = None,
        r: Optional[RatesResult] = None,
        f: Optional[ForwardResult] = None,
        d: Optional[DividendsResult] = None,
        vol: Optional[VolatilityResult] = None,
        undo_adjust: bool = True,
    ) -> LoadRequest:
        """Create a LoadRequest specifying which market data to load for greek calculation.

        Internal utility that determines which data sources need to be loaded based on:
        1. Which data is already provided (pre-loaded)
        2. Which pricing model is being used (BSM needs forwards, binomial needs spot)

        Args:
            start_date: First valuation date (YYYY-MM-DD string or datetime).
            end_date: Last valuation date (YYYY-MM-DD string or datetime).
            expiration: Option expiration date (YYYY-MM-DD string or datetime).
            strike: Strike price of the option.
            right: Option type ('c' for call, 'p' for put).
            dividend_type: Dividend treatment type (DISCRETE or CONTINUOUS).
            market_model: Pricing model (BSM or BINOMIAL).
            endpoint_source: Option data source (ORATS, HIST, QUOTE).
            model_price: Which price to use (CLOSE, OPEN, MIDPOINT).
            s: Optional pre-loaded spot data. If None, will be loaded.
            r: Optional pre-loaded rates data. If None, will be loaded.
            f: Optional pre-loaded forward data. If None, will be loaded (BSM only).
            d: Optional pre-loaded dividend data. If None, will be loaded.
            vol: Optional pre-loaded volatility data. If None, will be loaded.
            undo_adjust: If True, uses split-adjusted prices.

        Returns:
            LoadRequest object with flags indicating which data sources to load.

        Examples:
            >>> # Internal usage - creates request to load all data
            >>> request = greek_mgr._create_load_request(
            ...     start_date="2025-01-01",
            ...     end_date="2025-01-31",
            ...     expiration="2025-06-20",
            ...     strike=150.0,
            ...     right="c",
            ...     dividend_type=DivType.DISCRETE,
            ...     market_model=OptionPricingModel.BSM,
            ...     endpoint_source=OptionSpotEndpointSource.HIST,
            ...     model_price=ModelPrice.CLOSE
            ... )
            >>> # request.load_forward = True (BSM needs forwards)
            >>> # request.load_spot = True (no spot provided)
            >>> # request.load_vol = True (no vol provided)
        """

        req = LoadRequest(
            symbol=self.symbol,
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
            load_spot=(s is None),
            ## Load forward only if missing and using BSM model. Binomial uses spot price.
            load_forward=(market_model == OptionPricingModel.BSM) and (f is None),
            load_vol=(vol is None),
            load_dividend=(d is None),
            load_rates=(r is None),
            ## Not needed for greek calculation
            load_option_spot=False,
            undo_adjust=undo_adjust,
        )
        return req
