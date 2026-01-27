"""Volatility data manager for computing implied volatilities from option market prices.

This module provides the VolDataManager class for calculating implied volatilities using
various pricing models (Black-Scholes-Merton, Cox-Ross-Rubinstein binomial, European
equivalent). It handles the complete workflow including data loading, caching, model
selection, and result formatting.

Key Features:
    - Multiple pricing models: BSM, CRR binomial, European equivalent
    - Support for American and European exercise styles
    - Discrete and continuous dividend treatments
    - Automatic data loading and caching
    - Real-time and historical volatility calculation
    - Singleton pattern per symbol for efficient resource management

Typical Usage:
    >>> from trade.datamanager.vol import VolDataManager
    >>> from trade.optionlib.config.types import DivType
    >>>
    >>> # Initialize manager for AAPL
    >>> vol_mgr = VolDataManager("AAPL")
    >>>
    >>> # Get implied volatilities for an option
    >>> result = vol_mgr.get_implied_volatility_timeseries(
    ...     start_date="2025-01-01",
    ...     end_date="2025-01-31",
    ...     expiration="2025-06-20",
    ...     strike=150.0,
    ...     right="c",
    ...     dividend_type=DivType.DISCRETE,
    ...     american=True
    ... )
    >>> print(result.timeseries.head())
"""
from datetime import datetime
from typing import Any, ClassVar, Optional
import pandas as pd
from trade.datamanager._enums import (
    ArtifactType,
    Interval,
    OptionPricingModel,
    RealTimeFallbackOption,
    VolatilityModel,
    SeriesId,
    OptionSpotEndpointSource,
)
from trade.datamanager.base import BaseDataManager, CacheSpec
from trade.datamanager.config import OptionDataConfig
from trade.datamanager.requests import LoadRequest
from trade.datamanager.result import (
    VolatilityResult,
    ForwardResult,
    RatesResult,
    OptionSpotResult,
    SpotResult,
    DividendsResult,
)
from trade.datamanager.utils.vol_helpers import (
    _prepare_time_to_expiration,
    _handle_cache_for_vol,
    _merge_provided_with_loaded_data,
    _prepare_dividend_data_for_pricing,
    _merge_and_cache_vol_result,
    _prepare_vol_calculation_setup,
)
from trade.datamanager.utils.model import _load_model_data_timeseries
from trade.optionlib.vol.implied_vol import vector_bsm_iv_estimation, vector_crr_iv_estimation
from trade.optionlib.pricing.binomial import vector_crr_binomial_pricing
from trade.optionlib.config.types import DivType
from trade.helpers.helper import change_to_last_busday, to_datetime
from trade.helpers.Logging import setup_logger
from trade.datamanager.utils.date import is_available_on_date, sync_date_index
from trade.optionlib.assets.dividend import vector_convert_to_time_frac

logger = setup_logger("trade.datamanager.vol", stream_log_level="INFO")

class VolDataManager(BaseDataManager):
    """Manager for computing and caching implied volatilities from option market prices.

    Singleton class (per symbol) that orchestrates the computation of implied volatilities
    using various option pricing models. Automatically loads required market data (spot,
    forward, rates, dividends, option prices) and caches results for efficient reuse.

    Supports three pricing approaches:
        1. Black-Scholes-Merton (BSM) - Fast, European options only
        2. Cox-Ross-Rubinstein (CRR) - Binomial tree, supports American exercise
        3. European Equivalent (EURO_EQIV) - Converts American IVs to European equivalent

    Attributes:
        CACHE_NAME: Cache identifier for volatility data.
        DEFAULT_SERIES_ID: Default series identifier (historical data).
        CONFIG: Configuration object with default settings for pricing models.
        INSTANCES: Class-level dict maintaining singleton instances per symbol.
        symbol: Ticker symbol for the underlying asset.

    Examples:
        >>> # Basic usage with BSM model
        >>> vol_mgr = VolDataManager("AAPL")
        >>> result = vol_mgr.get_implied_volatility_timeseries(
        ...     start_date="2025-01-01",
        ...     end_date="2025-01-31",
        ...     expiration="2025-06-20",
        ...     strike=150.0,
        ...     right="c",
        ...     model=OptionPricingModel.BSM
        ... )

        >>> # American option with CRR binomial
        >>> result = vol_mgr.get_implied_volatility_timeseries(
        ...     start_date="2025-01-01",
        ...     end_date="2025-01-31",
        ...     expiration="2025-06-20",
        ...     strike=150.0,
        ...     right="p",
        ...     american=True,
        ...     model=OptionPricingModel.BINOMIAL,
        ...     n_steps=200
        ... )

        >>> # Real-time volatility
        >>> rt_vol = vol_mgr.rt(
        ...     expiration="2025-06-20",
        ...     strike=150.0,
        ...     right="c"
        ... )
    """
    CACHE_NAME: ClassVar[str] = "vol_data_manager_cache"
    DEFAULT_SERIES_ID: ClassVar["SeriesId"] = SeriesId.HIST
    CONFIG = OptionDataConfig()
    INSTANCES = {}

    def __new__(cls, symbol: str, *args: Any, **kwargs: Any) -> "VolDataManager":
        if symbol not in cls.INSTANCES:
            instance = object.__new__(cls)
            cls.INSTANCES[symbol] = instance
        return cls.INSTANCES[symbol]

    def __init__(
        self, symbol: str, *, cache_spec: Optional[CacheSpec] = None, enable_namespacing: bool = False
    ) -> None:
        """Initialize VolDataManager with symbol-specific configuration.

        Args:
            symbol: Ticker symbol for the underlying asset (e.g., "AAPL", "MSFT").
            cache_spec: Optional cache configuration. If None, uses default cache settings.
            enable_namespacing: If True, enables namespace prefixing for cache keys.

        Examples:
            >>> # Basic initialization
            >>> vol_mgr = VolDataManager("AAPL")

            >>> # With custom cache settings
            >>> from trade.datamanager.base import CacheSpec
            >>> cache_spec = CacheSpec(
            ...     default_expire_days=365,
            ...     cache_fname="custom_vol_cache"
            ... )
            >>> vol_mgr = VolDataManager("AAPL", cache_spec=cache_spec)
        """
        self.symbol = symbol

        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        super().__init__(
            cache_spec=cache_spec,
            enable_namespacing=enable_namespacing,
        )

    def _get_bsm_implied_volatility_timeseries(
        self,
        start_date: str,
        end_date: str,
        expiration: str,
        strike: float,
        right: str,
        dividend_type: Optional[DivType] = DivType.DISCRETE,
        *,
        result: Optional[VolatilityResult] = None,
        F: Optional[ForwardResult] = None,
        r: Optional[RatesResult] = None,
        market_price: Optional[OptionSpotResult] = None,
        undo_adjust: bool = True,
    ) -> VolatilityResult:
        """Compute implied volatilities using Black-Scholes-Merton model.

        Internal method that calculates daily implied volatilities by matching market prices
        to BSM prices. Automatically loads required data (forward, rates, option prices) if
        not provided. Uses caching to avoid redundant computations.

        Args:
            start_date: First valuation date (YYYY-MM-DD string or datetime).
            end_date: Last valuation date (YYYY-MM-DD string or datetime).
            expiration: Option expiration date (YYYY-MM-DD string or datetime).
            strike: Strike price of the option.
            right: Option type ('c' for call, 'p' for put).
            dividend_type: Dividend treatment type (DISCRETE or CONTINUOUS).
            result: Optional pre-initialized VolatilityResult container.
            F: Optional pre-computed forward prices. If None, loads automatically.
            r: Optional pre-computed risk-free rates. If None, loads automatically.
            market_price: Optional pre-computed option market prices. If None, loads automatically.
            undo_adjust: If True, uses split-adjusted prices.

        Returns:
            VolatilityResult containing daily implied volatilities with DatetimeIndex,
            model metadata, and cache key.

        Examples:
            >>> # Internal usage - typically called via get_implied_volatility_timeseries
            >>> result = vol_mgr._get_bsm_implied_volatility_timeseries(
            ...     start_date="2025-01-01",
            ...     end_date="2025-01-31",
            ...     expiration="2025-06-20",
            ...     strike=150.0,
            ...     right="c",
            ...     dividend_type=DivType.DISCRETE
            ... )
        """

        # Use utility: Prepare setup
        endpoint_source = result.endpoint_source if result is not None else self.CONFIG.option_spot_endpoint_source
        result, dividend_type, endpoint_source, start_str, end_str, start_date, end_date = _prepare_vol_calculation_setup(
            self, start_date, end_date, expiration, strike, right, dividend_type, endpoint_source, result
        )

        # Make key for caching
        key = self.make_key(
            symbol=self.symbol,
            interval=Interval.EOD,
            artifact_type=ArtifactType.IV,
            series_id=SeriesId.HIST,
            option_pricing_model=OptionPricingModel.BSM,
            volatility_model=VolatilityModel.MARKET,
            dividend_type=dividend_type,
            endpoint_source=endpoint_source,
            expiration=expiration,
            strike=strike,
            right=right,
        )

        # Use utility: Handle cache
        cached_data, is_partial, start_date, end_date, early_return = _handle_cache_for_vol(
            self, key, start_date, end_date, result
        )
        if early_return is not None:
            return early_return

        # Load model data
        load_request = LoadRequest(
            symbol=self.symbol,
            start_date=start_date,
            end_date=end_date,
            expiration=expiration,
            dividend_type=dividend_type,
            load_spot=False,
            load_forward=F is None,
            load_rates=r is None,
            load_option_spot=market_price is None,
            load_dividend=False, ## Not needed for BSM IV. Already handled in forward.
            load_vol=False,
            strike=strike,
            right=right,
            undo_adjust=undo_adjust,
            endpoint_source=endpoint_source,
        )
        model_data = _load_model_data_timeseries(load_request)

        # Use utility: Merge provided data
        _, F, r, _, market_price = _merge_provided_with_loaded_data(model_data, F=F, r=r, market_price=market_price)

        # Extract data
        forward = F.daily_continuous_forward if dividend_type == DivType.CONTINUOUS else F.daily_discrete_forward
        rates = r.daily_risk_free_rates
        option_spot = market_price.daily_option_spot.midpoint
        forward, rates, option_spot = sync_date_index(forward, rates, option_spot)

        # Use utility: Prepare T
        T = _prepare_time_to_expiration(forward.index, expiration)

        # Calculate IV
        iv_timeseries = vector_bsm_iv_estimation(
            F=forward.values,
            K=[strike] * len(forward),
            T=T,
            r=rates.values,
            market_price=option_spot.values,
            right=[right.lower()] * len(forward),
        )
        iv_timeseries = pd.Series(data=iv_timeseries, index=forward.index)

        # Use utility: Merge and cache
        iv_timeseries = _merge_and_cache_vol_result(
            self, iv_timeseries, cached_data, is_partial, key, start_str, end_str
        )

        # Prepare result
        result.timeseries = iv_timeseries
        return result

    def _get_crr_implied_volatility_timeseries(
        self,
        start_date: str,
        end_date: str,
        expiration: str,
        strike: float,
        right: str,
        dividend_type: Optional[DivType] = DivType.DISCRETE,
        american: bool = True,
        result: Optional[VolatilityResult] = None,
        *,
        S: Optional[SpotResult] = None,
        r: Optional[RatesResult] = None,
        dividends: Optional[DividendsResult] = None,
        market_price: Optional[OptionSpotResult] = None,
        undo_adjust: bool = True,
        n_steps: Optional[int] = None,
    ) -> VolatilityResult:
        """Compute implied volatilities using Cox-Ross-Rubinstein binomial model.

        Internal method that calculates daily implied volatilities using CRR binomial trees.
        Supports both American and European exercise styles. Automatically loads required
        data (spot, rates, dividends, option prices) if not provided. Uses caching for
        efficient reuse.

        Args:
            start_date: First valuation date (YYYY-MM-DD string or datetime).
            end_date: Last valuation date (YYYY-MM-DD string or datetime).
            expiration: Option expiration date (YYYY-MM-DD string or datetime).
            strike: Strike price of the option.
            right: Option type ('c' for call, 'p' for put).
            dividend_type: Dividend treatment type (DISCRETE or CONTINUOUS).
            american: If True, prices American exercise; if False, European.
            result: Optional pre-initialized VolatilityResult container.
            S: Optional pre-computed spot prices. If None, loads automatically.
            r: Optional pre-computed risk-free rates. If None, loads automatically.
            dividends: Optional pre-computed dividend data. If None, loads automatically.
            market_price: Optional pre-computed option market prices. If None, loads automatically.
            undo_adjust: If True, uses split-adjusted prices.
            n_steps: Number of time steps in binomial tree. Defaults to CONFIG.n_steps.

        Returns:
            VolatilityResult containing daily implied volatilities with DatetimeIndex,
            model metadata (BINOMIAL), and cache key.

        Examples:
            >>> # Internal usage - typically called via get_implied_volatility_timeseries
            >>> result = vol_mgr._get_crr_implied_volatility_timeseries(
            ...     start_date="2025-01-01",
            ...     end_date="2025-01-31",
            ...     expiration="2025-06-20",
            ...     strike=150.0,
            ...     right="p",
            ...     american=True,
            ...     n_steps=200
            ... )
        """

        # Use utility: Prepare setup
        endpoint_source = market_price.endpoint_source if market_price is not None else None
        result, dividend_type, endpoint_source, start_str, end_str, start_date, end_date = _prepare_vol_calculation_setup(
            self, start_date, end_date, expiration, strike, right, dividend_type, endpoint_source, result
        )
        n_steps = n_steps or self.CONFIG.n_steps

        # Make key for caching
        key = self.make_key(
            symbol=self.symbol,
            interval=Interval.EOD,
            artifact_type=ArtifactType.IV,
            series_id=SeriesId.HIST,
            option_pricing_model=OptionPricingModel.BINOMIAL,
            volatility_model=VolatilityModel.MARKET,
            dividend_type=dividend_type,
            endpoint_source=endpoint_source,
            expiration=expiration,
            strike=strike,
            right=right,
            american=american,
            n_steps=n_steps,
        )
        result.key = key

        # Use utility: Handle cache
        cached_data, is_partial, start_date, end_date, early_return = _handle_cache_for_vol(
            self, key, start_date, end_date, result
        )
        if early_return is not None:
            return early_return

        # Load model data
        load_request = LoadRequest(
            symbol=self.symbol,
            start_date=start_date,
            end_date=end_date,
            expiration=expiration,
            dividend_type=dividend_type,
            load_spot=S is None,
            load_rates=r is None,
            load_dividend=dividends is None,
            load_option_spot=market_price is None,
            load_forward= False, ## Not needed for CRR IV. Spot used directly.
            load_vol=False,
            strike=strike,
            right=right,
            undo_adjust=undo_adjust,
            endpoint_source=endpoint_source,
        )
        model_data = _load_model_data_timeseries(load_request)

        # Use utility: Merge provided data
        S, _, r, dividends, market_price = _merge_provided_with_loaded_data(
            model_data, S=S, r=r, dividends=dividends, market_price=market_price
        )

        # Extract data
        spot = S.daily_spot
        rates = r.daily_risk_free_rates
        option_spot = market_price.midpoint

        # Use utility: Prepare dividends and sync
        spot, rates, option_spot, dividends_ts = _prepare_dividend_data_for_pricing(
            dividends, dividend_type, expiration, spot, rates, option_spot
        )

        # Use utility: Prepare T
        T = _prepare_time_to_expiration(option_spot.index, expiration)

        # Calculate IV
        iv_timeseries = vector_crr_iv_estimation(
            S=spot.values,
            K=[strike] * len(spot),
            T=T,
            r=rates.values,
            market_price=option_spot.values,
            dividends=dividends_ts,
            option_type=[right.lower()] * len(spot),
            dividend_type=[dividend_type.name.lower()] * len(spot),
            american=[american] * len(spot),
            N=[n_steps] * len(spot),
        )
        iv_timeseries = pd.Series(data=iv_timeseries, index=spot.index)

        # Use utility: Merge and cache
        iv_timeseries = _merge_and_cache_vol_result(
            self, iv_timeseries, cached_data, is_partial, key, start_str, end_str
        )

        # Prepare result
        result.timeseries = iv_timeseries
        return result

    def _get_european_equivalent_volatility_timeseries(
        self,
        start_date: str,
        end_date: str,
        expiration: str,
        strike: float,
        right: str,
        *,
        result: Optional[VolatilityResult] = None,
        crr_american_vols: VolatilityResult,
        F: Optional[ForwardResult] = None,
        r: Optional[RatesResult] = None,
        dividends: Optional[DividendsResult] = None,
        dividend_type: Optional[DivType] = DivType.DISCRETE,
        undo_adjust: bool = True,
        n_steps: Optional[int] = None,
    ) -> VolatilityResult:
        """Convert American implied volatilities to European-equivalent BSM volatilities.

        Internal method that takes CRR American implied volatilities and converts them to
        European-equivalent Black-Scholes volatilities. This is done by:
        1. Pricing European options using CRR with American IVs
        2. Solving for BSM volatilities that match those European CRR prices

        This conversion is useful for comparing American option volatilities to European
        benchmarks or for further analysis requiring BSM framework.

        Args:
            start_date: First valuation date (YYYY-MM-DD string or datetime).
            end_date: Last valuation date (YYYY-MM-DD string or datetime).
            expiration: Option expiration date (YYYY-MM-DD string or datetime).
            strike: Strike price of the option.
            right: Option type ('c' for call, 'p' for put).
            result: Optional pre-initialized VolatilityResult container.
            crr_american_vols: Pre-computed American implied volatilities from CRR model.
            F: Optional pre-computed forward prices. If None, loads automatically.
            r: Optional pre-computed risk-free rates. If None, loads automatically.
            dividends: Optional pre-computed dividend data. If None, loads automatically.
            dividend_type: Dividend treatment type (DISCRETE or CONTINUOUS).
            undo_adjust: If True, uses split-adjusted prices.
            n_steps: Number of time steps in binomial tree. Defaults to CONFIG.n_steps.

        Returns:
            VolatilityResult containing daily European-equivalent implied volatilities
            with DatetimeIndex, model metadata (EURO_EQIV), and cache key.

        Examples:
            >>> # Internal usage - typically called via get_implied_volatility_timeseries
            >>> # First get American IVs
            >>> american_vols = vol_mgr._get_crr_implied_volatility_timeseries(...)
            >>> # Convert to European equivalent
            >>> euro_vols = vol_mgr._get_european_equivalent_volatility_timeseries(
            ...     start_date="2025-01-01",
            ...     end_date="2025-01-31",
            ...     expiration="2025-06-20",
            ...     strike=150.0,
            ...     right="p",
            ...     crr_american_vols=american_vols
            ... )
        """

        # Use utility: Prepare setup
        endpoint_source = crr_american_vols.endpoint_source
        result, dividend_type, endpoint_source, start_str, end_str, start_date, end_date = _prepare_vol_calculation_setup(
            self, start_date, end_date, expiration, strike, right, dividend_type, endpoint_source, result
        )

        # Make key for caching
        key = self.make_key(
            symbol=self.symbol,
            interval=Interval.EOD,
            artifact_type=ArtifactType.IV,
            series_id=SeriesId.HIST,
            option_pricing_model=OptionPricingModel.EURO_EQIV,
            volatility_model=VolatilityModel.MARKET,
            dividend_type=dividend_type,
            endpoint_source=endpoint_source,
            expiration=expiration,
            strike=strike,
            right=right,
        )

        # Use utility: Handle cache
        cached_data, is_partial, start_date, end_date, early_return = _handle_cache_for_vol(
            self, key, start_date, end_date, result
        )
        if early_return is not None:
            return early_return

        # Load model data
        load_request = LoadRequest(
            symbol=self.symbol,
            start_date=start_date,
            end_date=end_date,
            expiration=expiration,
            dividend_type=dividend_type,
            load_spot=True,
            load_forward=F is None,
            load_rates=r is None,
            load_dividend=dividends is None,
            strike=strike,
            right=right,
            undo_adjust=undo_adjust,
            endpoint_source=endpoint_source,
        )
        model_data = _load_model_data_timeseries(load_request)

        # Use utility: Merge provided data
        S, F, r, dividends, _ = _merge_provided_with_loaded_data(
            model_data, S=model_data.spot, F=F, r=r, dividends=dividends
        )

        # Extract data
        spot = S.daily_spot
        forward = F.daily_continuous_forward if dividend_type == DivType.CONTINUOUS else F.daily_discrete_forward
        rates = r.daily_risk_free_rates

        # Prepare dividends based on type
        if dividend_type == DivType.DISCRETE:
            dividends_ts = dividends.daily_discrete_dividends
            spot, forward, rates, dividends_ts, crr_american_iv = sync_date_index(
                spot, forward, rates, dividends_ts, crr_american_vols.timeseries
            )
            dividends_ts = vector_convert_to_time_frac(
                schedules=dividends_ts,
                valuation_dates=spot.index,
                end_dates=[to_datetime(expiration)] * len(spot.index),
            )
            dividend_yield = pd.Series(data=0.0, index=spot.index)
        elif dividend_type == DivType.CONTINUOUS:
            dividends_yield = dividends.daily_continuous_dividends
            spot, forward, rates, dividend_yield, crr_american_iv = sync_date_index(
                spot, forward, rates, dividends_yield, crr_american_vols.timeseries
            )
            dividends_ts = [()] * len(spot)

        # Price with CRR using American IVs in European mode
        european_crr_price = vector_crr_binomial_pricing(
            S0=spot.values,
            K=[strike] * len(spot),
            T=_prepare_time_to_expiration(spot.index, expiration),
            r=rates.values,
            sigma=crr_american_iv.values,
            dividend_yield=dividend_yield.values,
            dividends=dividends_ts,
            right=[right.lower()] * len(spot),
            N=[n_steps or self.CONFIG.n_steps] * len(spot),
            dividend_type=[dividend_type.name.lower()] * len(spot),
            american=[False] * len(spot),
        )

        # Convert to BSM equivalent IV
        european_equiv_iv = vector_bsm_iv_estimation(
            F=forward.values,
            K=[strike] * len(spot),
            T=_prepare_time_to_expiration(spot.index, expiration),
            r=rates.values,
            market_price=european_crr_price,
            right=[right.lower()] * len(spot),
        )
        european_equiv_iv = pd.Series(data=european_equiv_iv, index=spot.index)

        # Use utility: Merge and cache
        european_equiv_iv = _merge_and_cache_vol_result(
            self, european_equiv_iv, cached_data, is_partial, key, start_str, end_str
        )

        # Prepare result
        result.timeseries = european_equiv_iv
        return result

    def get_implied_volatility_timeseries(
        self,
        start_date: str,
        end_date: str,
        expiration: str,
        strike: float,
        right: str,
        dividend_type: Optional[DivType] = None,
        american: bool = True,
        *,
        market_model: Optional[OptionPricingModel] = None,
        S: Optional[SpotResult] = None,
        F: Optional[ForwardResult] = None,
        dividends: Optional[DividendsResult] = None,
        r: Optional[RatesResult] = None,
        market_price: Optional[OptionSpotResult] = None,
        undo_adjust: bool = True,
        n_steps: Optional[int] = None,
        endpoint_source: Optional[OptionSpotEndpointSource] = None,
        vol_model: Optional[VolatilityModel] = None,
    ) -> VolatilityResult:
        """Compute daily implied volatilities for a specific option across a date range.

        Main public method for calculating implied volatility timeseries. Automatically
        selects the appropriate pricing model (BSM, CRR, or European equivalent) and
        orchestrates data loading, computation, and caching.

        Args:
            start_date: First valuation date (YYYY-MM-DD string or datetime).
            end_date: Last valuation date (YYYY-MM-DD string or datetime).
            expiration: Option expiration date (YYYY-MM-DD string or datetime).
            strike: Strike price of the option.
            right: Option type ('c' for call, 'p' for put).
            dividend_type: Dividend treatment (DISCRETE or CONTINUOUS). Defaults to DISCRETE.
            american: If True, uses American exercise; if False, European.
            model: Pricing model to use (BSM, BINOMIAL, EURO_EQIV). Defaults to CONFIG.option_model.
            S: Optional pre-computed spot prices. If None, loads automatically.
            F: Optional pre-computed forward prices. If None, loads automatically.
            dividends: Optional pre-computed dividend data. If None, loads automatically.
            r: Optional pre-computed risk-free rates. If None, loads automatically.
            market_price: Optional pre-computed option prices. If None, loads automatically.
            undo_adjust: If True, uses split-adjusted prices.
            n_steps: Number of binomial tree steps. Only used for BINOMIAL/EURO_EQIV models.
            endpoint_source: Data source for option prices (e.g., CHAIN, QUOTE).
            vol_model: Volatility model to use (MARKET). Defaults to CONFIG.volatility_model.
        Returns:
            VolatilityResult containing:
                - timeseries: Daily implied volatilities as pandas Series
                - model: Volatility model type (MARKET)
                - market_model: Pricing model used (BSM, BINOMIAL, or EURO_EQIV)
                - dividend_type: Dividend treatment used
                - key: Cache key for result

        Raises:
            ValueError: If unsupported pricing model is specified.

        Examples:
            >>> # Basic European call with BSM
            >>> vol_mgr = VolDataManager("AAPL")
            >>> result = vol_mgr.get_implied_volatility_timeseries(
            ...     start_date="2025-01-01",
            ...     end_date="2025-01-31",
            ...     expiration="2025-06-20",
            ...     strike=150.0,
            ...     right="c",
            ...     american=False,
            ...     model=OptionPricingModel.BSM
            ... )
            >>> print(result.timeseries.head())

            >>> # American put with CRR binomial
            >>> result = vol_mgr.get_implied_volatility_timeseries(
            ...     start_date="2025-01-01",
            ...     end_date="2025-01-31",
            ...     expiration="2025-06-20",
            ...     strike=150.0,
            ...     right="p",
            ...     american=True,
            ...     model=OptionPricingModel.BINOMIAL,
            ...     n_steps=200,
            ...     dividend_type=DivType.DISCRETE
            ... )

            >>> # European equivalent from American
            >>> result = vol_mgr.get_implied_volatility_timeseries(
            ...     start_date="2025-01-01",
            ...     end_date="2025-01-31",
            ...     expiration="2025-06-20",
            ...     strike=150.0,
            ...     right="c",
            ...     model=OptionPricingModel.EURO_EQIV
            ... )
        """
        # Volatility model (currently only MARKET supported)
        vol_model = vol_model or self.CONFIG.volatility_model

        # Load model information
        market_model = market_model or self.CONFIG.option_model
        if dividend_type is None:
            logger.info(f"VolDm Using default dividend type from config: {self.CONFIG.dividend_type}")
        else:
            logger.info(f"VolDm Using specified dividend type: {dividend_type}")
        dividend_type = dividend_type or self.CONFIG.dividend_type
        endpoint_source = endpoint_source or self.CONFIG.option_spot_endpoint_source

        # Prepare result container
        result = VolatilityResult()
        result.symbol = self.symbol
        result.expiration = to_datetime(expiration)
        result.right = right
        result.strike = strike
        result.dividend_type = dividend_type
        result.vol_model = vol_model
        result.endpoint_source = endpoint_source
        result.market_model = market_model

        if market_model == OptionPricingModel.BSM:
            return self._get_bsm_implied_volatility_timeseries(
                start_date=start_date,
                end_date=end_date,
                expiration=expiration,
                strike=strike,
                right=right,
                dividend_type=dividend_type,
                F=F,
                r=r,
                market_price=market_price,
                undo_adjust=undo_adjust,
                result=result,
            )
        elif market_model == OptionPricingModel.BINOMIAL:
            return self._get_crr_implied_volatility_timeseries(
                start_date=start_date,
                end_date=end_date,
                expiration=expiration,
                strike=strike,
                right=right,
                dividend_type=dividend_type,
                S=S,
                r=r,
                dividends=dividends,
                market_price=market_price,
                undo_adjust=undo_adjust,
                american=american,
                n_steps=n_steps,
                result=result,
            )
        elif market_model == OptionPricingModel.EURO_EQIV:
            # First get the CRR American implied volatilities
            crr_american_vols = self._get_crr_implied_volatility_timeseries(
                start_date=start_date,
                end_date=end_date,
                expiration=expiration,
                strike=strike,
                right=right,
                dividend_type=dividend_type,
                S=S,
                r=r,
                dividends=dividends,
                market_price=market_price,
                undo_adjust=undo_adjust,
                american=True,
                n_steps=n_steps,
            )
            return self._get_european_equivalent_volatility_timeseries(
                start_date=start_date,
                end_date=end_date,
                expiration=expiration,
                strike=strike,
                right=right,
                crr_american_vols=crr_american_vols,
                F=F,
                r=r,
                dividends=dividends,
                dividend_type=dividend_type,
                undo_adjust=undo_adjust,
                n_steps=n_steps,
                result=result,
            )
        else:
            raise ValueError(f"Unsupported option pricing model: {market_model}")

    def get_at_time_implied_volatility(
        self,
        as_of: str,
        expiration: str,
        strike: float,
        right: str,
        dividend_type: Optional[DivType] = DivType.DISCRETE,
        american: bool = True,
        *,
        vol_model: Optional[VolatilityModel] = None,
        fallback_option: Optional[RealTimeFallbackOption] = None,
        market_model: Optional[OptionPricingModel] = None,
        S: Optional[SpotResult] = None,
        F: Optional[ForwardResult] = None,
        dividends: Optional[DividendsResult] = None,
        r: Optional[RatesResult] = None,
        market_price: Optional[OptionSpotResult] = None,
        undo_adjust: bool = True,
        n_steps: Optional[int] = None,
        endpoint_source: Optional[OptionSpotEndpointSource] = None,
    ) -> VolatilityResult:
        """Compute implied volatility at a specific point in time.

        Convenience method that retrieves implied volatility for a single date by calling
        get_implied_volatility_timeseries with start_date=end_date=as_of. Useful for
        historical backtesting or analysis at specific dates.

        Args:
            as_of: Specific valuation date (YYYY-MM-DD string or datetime).
            expiration: Option expiration date (YYYY-MM-DD string or datetime).
            strike: Strike price of the option.
            right: Option type ('c' for call, 'p' for put).
            dividend_type: Dividend treatment (DISCRETE or CONTINUOUS). Defaults to DISCRETE.
            american: If True, uses American exercise; if False, European.
            market_model: Pricing model to use (BSM, BINOMIAL, EURO_EQIV). Defaults to CONFIG.option_model.
            S: Optional pre-computed spot prices. If None, loads automatically.
            F: Optional pre-computed forward prices. If None, loads automatically.
            dividends: Optional pre-computed dividend data. If None, loads automatically.
            r: Optional pre-computed risk-free rates. If None, loads automatically.
            market_price: Optional pre-computed option prices. If None, loads automatically.
            undo_adjust: If True, uses split-adjusted prices.
            n_steps: Number of binomial tree steps. Only used for BINOMIAL/EURO_EQIV models.
            endpoint_source: Data source for option prices (e.g., CHAIN, QUOTE).

        Returns:
            VolatilityResult with single-row timeseries containing the implied volatility
            at the specified date.

        Examples:
            >>> # Get IV on a specific historical date
            >>> vol_mgr = VolDataManager("AAPL")
            >>> result = vol_mgr.get_at_time_implied_volatility(
            ...     as_of="2025-01-15",
            ...     expiration="2025-06-20",
            ...     strike=150.0,
            ...     right="c",
            ...     american=True
            ... )
            >>> print(f"IV on 2025-01-15: {result.timeseries.iloc[0]:.4f}")

            >>> # Use in backtesting loop
            >>> for date in backtest_dates:
            ...     vol_result = vol_mgr.get_at_time_implied_volatility(
            ...         as_of=date,
            ...         expiration=expiration,
            ...         strike=strike,
            ...         right="p"
            ...     )
            ...     iv_value = vol_result.timeseries.iloc[0]
        """
        fallback_option = fallback_option or self.CONFIG.real_time_fallback_option
        if not is_available_on_date(as_of):
            logger.warning(
                f"Valuation date {as_of} is not a business day or holiday. Resolving using fallback options {fallback_option}."
            )
            if fallback_option == RealTimeFallbackOption.RAISE_ERROR:
                raise ValueError(f"Valuation date {as_of} is not a business day or holiday.")
            if fallback_option == RealTimeFallbackOption.USE_LAST_AVAILABLE:
                as_of = change_to_last_busday(as_of, eod_time=False)
            else:
                result = VolatilityResult()
                result.timeseries = pd.Series(dtype=float,
                                              index=pd.DatetimeIndex([to_datetime(as_of)]),
                                              values = [float('nan') if fallback_option == RealTimeFallbackOption.NAN else 0.0])
                
                result.key = None
                result.vol_model = vol_model or self.CONFIG.volatility_model
                result.market_model = market_model or self.CONFIG.option_model
                result.expiration = to_datetime(expiration)
                result.right = right
                result.strike = strike
                result.endpoint_source = endpoint_source or self.CONFIG.option_spot_endpoint_source
                result.dividend_type = dividend_type or self.CONFIG.dividend_type
                result.symbol = self.symbol
                return result

        iv_timeseries = self.get_implied_volatility_timeseries(
            start_date=as_of,
            end_date=as_of,
            expiration=expiration,
            strike=strike,
            right=right,
            dividend_type=dividend_type,
            american=american,
            market_model=market_model,
            S=S,
            F=F,
            dividends=dividends,
            r=r,
            market_price=market_price,
            undo_adjust=undo_adjust,
            n_steps=n_steps,
            endpoint_source=endpoint_source,
            vol_model=vol_model,
        )
        iv_timeseries.timeseries = iv_timeseries.timeseries.loc[to_datetime(as_of) : to_datetime(as_of)]
        return iv_timeseries

    def rt(
        self,
        expiration: str,
        strike: float,
        right: str,
        dividend_type: Optional[DivType] = DivType.DISCRETE,
        american: bool = True,
        *,
        fallback_option: Optional[RealTimeFallbackOption] = None,
        market_model: Optional[OptionPricingModel] = None,
        S: Optional[SpotResult] = None,
        F: Optional[ForwardResult] = None,
        dividends: Optional[DividendsResult] = None,
        r: Optional[RatesResult] = None,
        market_price: Optional[OptionSpotResult] = None,
        undo_adjust: bool = True,
        n_steps: Optional[int] = None,
    ) -> VolatilityResult:
        """Compute current real-time implied volatility using latest market data.

        Convenience method for real-time volatility calculation. Automatically uses today's
        date and QUOTE endpoint source for live market prices. Useful for live trading,
        monitoring, and real-time analytics.

        Args:
            expiration: Option expiration date (YYYY-MM-DD string or datetime).
            strike: Strike price of the option.
            right: Option type ('c' for call, 'p' for put).
            dividend_type: Dividend treatment (DISCRETE or CONTINUOUS). Defaults to DISCRETE.
            american: If True, uses American exercise; if False, European.
            market_model: Pricing model to use (BSM, BINOMIAL, EURO_EQIV). Defaults to CONFIG.option_model.
            S: Optional pre-computed spot prices. If None, loads automatically.
            F: Optional pre-computed forward prices. If None, loads automatically.
            dividends: Optional pre-computed dividend data. If None, loads automatically.
            r: Optional pre-computed risk-free rates. If None, loads automatically.
            market_price: Optional pre-computed option prices. If None, fetches live quotes.
            undo_adjust: If True, uses split-adjusted prices.
            n_steps: Number of binomial tree steps. Only used for BINOMIAL/EURO_EQIV models.

        Returns:
            VolatilityResult with single-row timeseries containing the current implied
            volatility. Uses OptionSpotEndpointSource.QUOTE for real-time data.

        Examples:
            >>> # Get current IV for a call option
            >>> vol_mgr = VolDataManager("AAPL")
            >>> rt_vol = vol_mgr.rt(
            ...     expiration="2025-06-20",
            ...     strike=150.0,
            ...     right="c"
            ... )
            >>> print(f"Current IV: {rt_vol.timeseries.iloc[0]:.4f}")

            >>> # Monitor IV throughout the trading day
            >>> import time
            >>> while market_open:
            ...     vol = vol_mgr.rt(
            ...         expiration="2025-06-20",
            ...         strike=150.0,
            ...         right="p",
            ...         american=True
            ...     )
            ...     print(f"Current IV: {vol.timeseries.iloc[0]:.4f}")
            ...     time.sleep(60)  # Update every minute

            >>> # Use with specific model
            >>> rt_vol = vol_mgr.rt(
            ...     expiration="2025-06-20",
            ...     strike=150.0,
            ...     right="c",
            ...     market_model=OptionPricingModel.BSM
            ... )
        """

        res = self.get_at_time_implied_volatility(
            as_of=datetime.now().strftime("%Y-%m-%d"),
            expiration=expiration,
            strike=strike,
            right=right,
            dividend_type=dividend_type,
            american=american,
            market_model=market_model,
            S=S,
            F=F,
            dividends=dividends,
            r=r,
            market_price=market_price,
            undo_adjust=undo_adjust,
            n_steps=n_steps,
            endpoint_source=OptionSpotEndpointSource.QUOTE,
        )
        return res