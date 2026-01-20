"""Forward price computation and caching for options pricing models.

This module provides the ForwardDataManager class for computing and caching forward
prices using spot prices, risk-free rates, and dividends. Supports both discrete
(schedule-based) and continuous (yield-based) dividend models with intelligent caching.

Typical usage:
    >>> fwd_mgr = ForwardDataManager("AAPL")
    >>> result = fwd_mgr.get_forward_timeseries(
    ...     start_date="2025-01-01",
    ...     end_date="2025-01-31",
    ...     maturity_date="2025-06-20",
    ...     div_type=DivType.DISCRETE,
    ...     use_chain_spot=True
    ... )
    >>> forwards = result.daily_discrete_forward
"""

from datetime import datetime, date
from typing import Any, ClassVar, Optional, Tuple, Union
import pandas as pd
from EventDriven.riskmanager.market_data import TimeseriesData
from trade.helpers.Logging import setup_logger
from trade.helpers.helper import get_missing_dates
from trade.datamanager.utils.data_structure import _data_structure_sanitize
from trade.datamanager.base import BaseDataManager, CacheSpec
from trade.datamanager.dividend import DividendDataManager
from trade.datamanager.result import ForwardResult
from trade.datamanager.rates import RatesDataManager
from trade.datamanager.config import OptionDataConfig
from trade.datamanager.result import DividendsResult, RatesResult
from trade.datamanager._enums import ArtifactType, Interval, SeriesId
from trade.datamanager.utils.cache import _data_structure_cache_it
from trade.datamanager.vars import TS
from trade.optionlib.config.types import DivType
from trade.optionlib.config.defaults import OPTION_TIMESERIES_START_DATE
from trade.optionlib.assets.dividend import (
    vectorized_discrete_pv,
    SECONDS_IN_DAY,
    SECONDS_IN_YEAR,
)
from trade.optionlib.assets.forward import (
    vectorized_forward_discrete,
    vectorized_forward_continuous,
    get_vectorized_continuous_dividends,
)

logger = setup_logger("trade.datamanager.forward")


class ForwardDataManager(BaseDataManager):
    """Manages forward price computation and caching for a specific symbol using spot, rates, and dividends.

    Computes forward prices using cost-of-carry models with discrete or continuous dividends.
    Implements singleton pattern per symbol to avoid redundant timeseries loading. Supports
    both split-adjusted (chain_spot) and unadjusted (spot) price bases.

    Attributes:
        CACHE_NAME: Class-level cache identifier for this manager type.
        DEFAULT_SERIES_ID: Default historical series identifier.
        INSTANCES: Class-level cache of manager instances per symbol.
        CONFIG: Configuration object for forward computation settings.
        symbol: The equity ticker symbol this manager handles.

    Examples:
        >>> # Singleton access - same instance returned for same symbol
        >>> fwd_mgr1 = ForwardDataManager("AAPL")
        >>> fwd_mgr2 = ForwardDataManager("AAPL")
        >>> assert fwd_mgr1 is fwd_mgr2

        >>> # Get forward price time-series with discrete dividends
        >>> result = fwd_mgr1.get_forward_timeseries(
        ...     start_date="2025-01-01",
        ...     end_date="2025-01-31",
        ...     maturity_date="2025-06-20",
        ...     div_type=DivType.DISCRETE,
        ...     use_chain_spot=True
        ... )
        >>> forwards = result.daily_discrete_forward

        >>> # Get single forward price for a date
        >>> result = fwd_mgr1.get_forward(
        ...     date="2025-01-15",
        ...     maturity_date="2025-06-20",
        ...     div_type=DivType.DISCRETE
        ... )
    """

    CACHE_NAME: ClassVar[str] = "forward_data_manager"
    DEFAULT_SERIES_ID: ClassVar["SeriesId"] = SeriesId.HIST
    INSTANCES = {}
    CONFIG = OptionDataConfig()

    def __new__(cls, symbol: str, *args: Any, **kwargs: Any) -> "ForwardDataManager":
        """Returns cached instance for symbol, creating new one if needed.

        Implements singleton pattern per symbol to ensure timeseries are loaded only once.
        Automatically loads market timeseries data on first instantiation.

        Args:
            symbol: Equity ticker symbol (e.g., "AAPL", "MSFT").
            *args: Additional positional arguments passed to __init__.
            **kwargs: Additional keyword arguments passed to __init__.

        Returns:
            Singleton ForwardDataManager instance for the given symbol.

        Examples:
            >>> mgr1 = ForwardDataManager("AAPL")
            >>> mgr2 = ForwardDataManager("AAPL")
            >>> assert mgr1 is mgr2  # Same instance
        """
        if symbol not in cls.INSTANCES:
            TS.load_timeseries(symbol, start_date=OPTION_TIMESERIES_START_DATE, end_date=datetime.now())
            instance = super(ForwardDataManager, cls).__new__(cls)
            cls.INSTANCES[symbol] = instance
        return cls.INSTANCES[symbol]

    def __init__(
        self,
        symbol: str,
        *,
        cache_spec: Optional[CacheSpec] = None,
        enable_namespacing: bool = False,
    ) -> None:
        """Initializes manager once per symbol instance.

        Sets up persistent cache for forward price data. Only executes once per
        symbol due to singleton pattern.

        Args:
            symbol: Equity ticker symbol.
            cache_spec: Optional cache configuration. Uses default if None.
            enable_namespacing: If True, enables namespace isolation in cache keys.

        Examples:
            >>> mgr = ForwardDataManager("AAPL")
            >>> mgr = ForwardDataManager("AAPL", cache_spec=CacheSpec(expire_days=30))
        """
        if getattr(self, "_initialized", False):
            return

        self._initialized = True
        super().__init__(cache_spec=cache_spec, enable_namespacing=enable_namespacing)
        self.symbol = symbol

    def _normalize_inputs(
        self,
        start_date: Union[datetime, str],
        end_date: Union[datetime, str],
        maturity_date: Union[datetime, str],
        div_type: Optional[DivType],
    ) -> Tuple[DivType, date, date, date, str, str, str]:
        """Converts date inputs to both date objects and strings.
        
        Normalizes various date input formats to consistent datetime objects and
        YYYY-MM-DD strings. Sets default dividend type to DISCRETE if not specified.
        
        Args:
            start_date: Start date (YYYY-MM-DD string or datetime).
            end_date: End date (YYYY-MM-DD string or datetime).
            maturity_date: Maturity date (YYYY-MM-DD string or datetime).
            div_type: Optional DivType. Defaults to DISCRETE if None.
        
        Returns:
            Tuple containing:
                - DivType: Dividend type (DISCRETE or CONTINUOUS)
                - date: Start date object
                - date: End date object
                - date: Maturity date object
                - str: Start date string (YYYY-MM-DD)
                - str: End date string (YYYY-MM-DD)
                - str: Maturity date string (YYYY-MM-DD)
        
        Examples:
            >>> fwd_mgr = ForwardDataManager("AAPL")
            >>> div_type, start_dt, end_dt, mat_dt, start_str, end_str, mat_str = \
            ...     fwd_mgr._normalize_inputs(
            ...         start_date="2025-01-01",
            ...         end_date="2025-01-31",
            ...         maturity_date="2025-06-20",
            ...         div_type=None
            ...     )
            >>> print(div_type)  # DivType.DISCRETE
        """
        div_type = DivType(div_type) if div_type is not None else DivType.DISCRETE

        start_dt = datetime.strptime(start_date, "%Y-%m-%d") if isinstance(start_date, str) else start_date
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") if isinstance(end_date, str) else end_date
        mat_dt = datetime.strptime(maturity_date, "%Y-%m-%d") if isinstance(maturity_date, str) else maturity_date

        start_str = datetime.strftime(start_dt, "%Y-%m-%d")
        end_str = datetime.strftime(end_dt, "%Y-%m-%d")
        mat_str = datetime.strftime(mat_dt, "%Y-%m-%d")
        return div_type, start_dt, end_dt, mat_dt, start_str, end_str, mat_str

    def _build_key(self, *, mat_str: str, div_type: DivType, use_chain_spot: bool) -> str:
        """Constructs cache key from maturity, dividend type, and spot type.

        Creates unique cache identifier incorporating symbol, maturity date, dividend type,
        and whether split-adjusted prices are used.

        Args:
            mat_str: Maturity date string (YYYY-MM-DD).
            div_type: DivType.DISCRETE or DivType.CONTINUOUS.
            use_chain_spot: If True, uses split-adjusted chain_spot prices.

        Returns:
            Unique cache key string.

        Examples:
            >>> fwd_mgr = ForwardDataManager("AAPL")
            >>> key = fwd_mgr._build_key(
            ...     mat_str="2025-06-20",
            ...     div_type=DivType.DISCRETE,
            ...     use_chain_spot=True
            ... )
        """
        return self.make_key(
            symbol=self.symbol,
            artifact_type=ArtifactType.FWD,
            series_id=SeriesId.HIST,
            maturity=mat_str,
            div_type=div_type.value,
            use_chain_spot=use_chain_spot,
            interval=Interval.EOD,
        )

    def _try_get_cached(
        self,
        *,
        key: str,
        start_str: str,
        end_str: str,
        start_date: Union[datetime, str],
        end_date: Union[datetime, str],
        div_type: DivType,
    ) -> Tuple[Optional[pd.Series], bool, str, str, Optional[ForwardResult]]:
        """Checks cache for existing data and identifies missing dates.

        Attempts to retrieve forward prices from cache. If found, checks if the cached
        data fully covers the requested date range. Returns cached result directly if
        complete, or identifies missing dates that need computation.

        Args:
            key: Cache key identifier.
            start_str: Start date string (YYYY-MM-DD).
            end_str: End date string (YYYY-MM-DD).
            start_date: Start date (string or datetime).
            end_date: End date (string or datetime).
            div_type: DivType.DISCRETE or DivType.CONTINUOUS.

        Returns:
            Tuple containing:
                - Optional[pd.Series]: Cached series if partial hit, None otherwise
                - bool: True if partial cache hit (need to fetch missing dates)
                - str: Start date for fetching (adjusted if partial hit)
                - str: End date for fetching (adjusted if partial hit)
                - Optional[ForwardResult]: Complete result if full cache hit, None otherwise

        Examples:
            >>> fwd_mgr = ForwardDataManager("AAPL")
            >>> cached, partial, start, end, result = fwd_mgr._try_get_cached(
            ...     key="my_key",
            ...     start_str="2025-01-01",
            ...     end_str="2025-01-31",
            ...     start_date="2025-01-01",
            ...     end_date="2025-01-31",
            ...     div_type=DivType.DISCRETE
            ... )
        """
        cached_series = self.get(key, default=None)
        if cached_series is None:
            return None, False, start_str, end_str, None

        missing = get_missing_dates(cached_series, _start=start_str, _end=end_str)
        if not missing:
            logger.info(f"Cache hit for forward timeseries key: {key}")
            cached_series = _data_structure_sanitize(
                cached_series,
                start=start_str,
                end=end_str,
            )

            result = ForwardResult()
            if div_type == DivType.DISCRETE:
                result.daily_discrete_forward = cached_series
            else:
                result.daily_continuous_forward = cached_series
            result.dividend_type = div_type
            result.key = key
            return cached_series, False, start_str, end_str, result

        logger.info(
            f"Cache partially covers requested date range for forward timeseries. "
            f"Key: {key}. Fetching missing dates: {missing}"
        )
        return cached_series, True, min(missing), max(missing), None

    def _get_dividend_result(
        self,
        *,
        start_str: str,
        end_str: str,
        mat_str: str,
        div_type: DivType,
        dividend_result: Optional[DividendsResult],
        use_chain_spot: bool,
    ) -> DividendsResult:
        """Fetches or validates dividend data with adjustment consistency checks.

        Retrieves dividend data from DividendDataManager if not provided. Validates
        that dividend adjustments match the spot price basis (undo_adjust must equal
        use_chain_spot for consistency).

        Args:
            start_str: Start date string (YYYY-MM-DD).
            end_str: End date string (YYYY-MM-DD).
            mat_str: Maturity date string (YYYY-MM-DD).
            div_type: DivType.DISCRETE or DivType.CONTINUOUS.
            dividend_result: Optional pre-computed dividend data. Fetched if None.
            use_chain_spot: If True, uses split-adjusted chain_spot prices.

        Returns:
            DividendsResult containing dividend schedules or yields.

        Raises:
            ValueError: If dividend_result is empty.
            ValueError: If dividend_result.undo_adjust != use_chain_spot.

        Examples:
            >>> fwd_mgr = ForwardDataManager("AAPL")
            >>> div_result = fwd_mgr._get_dividend_result(
            ...     start_str="2025-01-01",
            ...     end_str="2025-01-31",
            ...     mat_str="2025-06-20",
            ...     div_type=DivType.DISCRETE,
            ...     dividend_result=None,
            ...     use_chain_spot=True
            ... )

        Notes:
            - If using chain_spot (split-adjusted), dividends must be back-adjusted
            - Ensures consistency between spot and dividend adjustment methods
        """
        if dividend_result is None:
            dividend_result = DividendDataManager(symbol=self.symbol).get_schedule_timeseries(
                start_date=start_str,
                end_date=end_str,
                maturity_date=mat_str,
                div_type=div_type,
                undo_adjust=use_chain_spot,  # If using chain spot, back adjust dividends
            )

        if dividend_result.is_empty():
            raise ValueError("Dividend result is empty. Cannot compute forward prices without dividend information.")

        if dividend_result.undo_adjust != use_chain_spot:
            raise ValueError("Mismatch between dividend_result.undo_adjust and use_chain_spot. They must be the same.")

        return dividend_result

    def _load_spot(self, *, use_chain_spot: bool, spot: Optional[TimeseriesData] = None) -> pd.Series:
        """Loads spot or chain_spot price series.

        Retrieves either split-adjusted (chain_spot) or unadjusted (spot) closing prices
        from timeseries data.

        Args:
            use_chain_spot: If True, returns split-adjusted chain_spot prices.
            spot: Optional pre-loaded TimeseriesData. Fetched from TS if None.

        Returns:
            Series of closing prices indexed by datetime.

        Examples:
            >>> fwd_mgr = ForwardDataManager("AAPL")
            >>> spot_prices = fwd_mgr._load_spot(use_chain_spot=True)
            >>> print(spot_prices.head())
            datetime
            2025-01-02    155.32
            2025-01-03    156.01
            ...

        Notes:
            - chain_spot: Split-adjusted prices (use with undo_adjust=True dividends)
            - spot: Unadjusted prices (use with undo_adjust=False dividends)
        """
        if spot is None:
            spot = TS.get_timeseries(self.symbol, skip_preload_check=True)
        if use_chain_spot:
            return spot.chain_spot["close"]
        return spot.spot["close"]

    def _load_rates(self, *, start_str: str, end_str: str, rates: Optional[RatesResult] = None) -> pd.Series:
        """Loads risk-free rates for date range.

        Retrieves risk-free interest rates from RatesDataManager if not provided.
        Filters to exact date range requested.

        Args:
            start_str: Start date string (YYYY-MM-DD).
            end_str: End date string (YYYY-MM-DD).
            rates: Optional pre-computed rates data. Fetched if None.

        Returns:
            Series of annualized risk-free rates indexed by datetime.

        Examples:
            >>> fwd_mgr = ForwardDataManager("AAPL")
            >>> rates = fwd_mgr._load_rates(
            ...     start_str="2025-01-01",
            ...     end_str="2025-01-31"
            ... )
            >>> print(rates.head())
            Datetime
            2025-01-02    0.0485
            2025-01-03    0.0487
            ...
        """
        if rates is None:
            rates_data = RatesDataManager().get_risk_free_rate_timeseries(
                start_date=start_str,
                end_date=end_str,
                interval=Interval.EOD,
            )
            rates = rates_data.daily_risk_free_rates
        else:
            rates = rates.daily_risk_free_rates
        rates = rates[(rates.index >= pd.to_datetime(start_str)) & (rates.index <= pd.to_datetime(end_str))]
        return rates

    def _align_3(
        self, spot: pd.Series, rates: pd.Series, third: pd.Series, *, third_name: str
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Aligns three series to common dates and validates no NaNs.

        Synchronizes spot prices, risk-free rates, and dividend data to a common
        date index. Validates that rates and dividend data have no missing values.

        Args:
            spot: Series of spot prices.
            rates: Series of risk-free rates.
            third: Series of dividend data (schedules or yields).
            third_name: Descriptive name for third series (for error messages).

        Returns:
            Tuple of three aligned Series with common index.

        Raises:
            ValueError: If rates contain NaNs after alignment.
            ValueError: If third series contains NaNs after alignment.

        Examples:
            >>> fwd_mgr = ForwardDataManager("AAPL")
            >>> spot_aligned, rates_aligned, divs_aligned = fwd_mgr._align_3(
            ...     spot=spot_series,
            ...     rates=rates_series,
            ...     third=dividend_series,
            ...     third_name="discrete dividend schedules"
            ... )

        Notes:
            - Only dates present in all three series are retained
            - Spot prices may have NaNs (will be handled by vectorized functions)
            - Rates and dividend data must be complete (no NaNs allowed)
        """
        idx = spot.index.intersection(rates.index).intersection(third.index)

        spot = spot.reindex(idx)
        rates = rates.reindex(idx)
        third = third.reindex(idx)

        if rates.isna().any():
            raise ValueError("NaNs in rates after alignment.")
        if third.isna().any():
            raise ValueError(f"NaNs in {third_name} after alignment.")

        return spot, rates, third

    def _compute_forward_discrete(
        self,
        *,
        spot: pd.Series,
        rates: pd.Series,
        discrete_divs: pd.Series,  # series of Schedule objects
        mat_dt: date,
    ) -> pd.Series:
        """Computes forward prices using discrete dividend schedules.

        Calculates forward prices using the discrete dividend model:
        F = S * exp(r*T) - PV(divs)

        Where PV(divs) is the present value of all dividends between valuation
        date and maturity.

        Args:
            spot: Series of spot prices.
            rates: Series of annualized risk-free rates.
            discrete_divs: Series of Schedule objects (dividend events).
            mat_dt: Maturity date (e.g., option expiry).

        Returns:
            Series of forward prices indexed by valuation dates.

        Examples:
            >>> fwd_mgr = ForwardDataManager("AAPL")
            >>> forwards = fwd_mgr._compute_forward_discrete(
            ...     spot=spot_prices,
            ...     rates=risk_free_rates,
            ...     discrete_divs=dividend_schedules,
            ...     mat_dt=date(2025, 6, 20)
            ... )
            >>> print(forwards.head())
            datetime
            2025-01-02    156.45
            2025-01-03    157.12
            ...

        Notes:
            - Uses vectorized computation for efficiency
            - Discounts each dividend in schedule to valuation date
            - Time to maturity calculated as (mat_dt - val_dt) in years
        """

        pv_divs = vectorized_discrete_pv(
            schedules=discrete_divs.to_list(),
            r=rates.tolist(),
            _valuation_dates=discrete_divs.index.tolist(),
            _end_dates=[mat_dt] * len(discrete_divs),
        )
        pv_divs = [pv_divs] if isinstance(pv_divs, (int, float)) else pv_divs

        second_vector = [(mat_dt - val).days * SECONDS_IN_DAY for val in discrete_divs.index]
        t = [val / SECONDS_IN_YEAR for val in second_vector]

        forwards = vectorized_forward_discrete(
            S=spot.tolist(),
            r=rates.tolist(),
            T=t,
            pv_divs=pv_divs,
        )
        return pd.Series(data=forwards, index=discrete_divs.index)

    def _compute_forward_continuous(
        self,
        *,
        spot: pd.Series,
        rates: pd.Series,
        continuous_divs: pd.Series,  # series of dividend yields
        mat_dt: date,
    ) -> pd.Series:
        """Computes forward prices using continuous dividend yields.

        Calculates forward prices using the continuous dividend model:
        F = S * exp((r - q) * T)

        Where q is the continuous dividend yield.

        Args:
            spot: Series of spot prices.
            rates: Series of annualized risk-free rates.
            continuous_divs: Series of annualized dividend yields.
            mat_dt: Maturity date (e.g., option expiry).

        Returns:
            Series of forward prices indexed by valuation dates.

        Examples:
            >>> fwd_mgr = ForwardDataManager("AAPL")
            >>> forwards = fwd_mgr._compute_forward_continuous(
            ...     spot=spot_prices,
            ...     rates=risk_free_rates,
            ...     continuous_divs=dividend_yields,
            ...     mat_dt=date(2025, 6, 20)
            ... )
            >>> print(forwards.head())
            datetime
            2025-01-02    156.28
            2025-01-03    156.95
            ...

        Notes:
            - Uses vectorized computation for efficiency
            - Assumes constant dividend yield between valuation and maturity
            - Time to maturity calculated as (mat_dt - val_dt) in years
        """
        q_factor = get_vectorized_continuous_dividends(
            div_rates=continuous_divs.tolist(),
            _valuation_dates=continuous_divs.index.tolist(),
            _end_dates=[mat_dt] * len(continuous_divs),
        )

        second_vector = [(mat_dt - val).days * SECONDS_IN_DAY for val in continuous_divs.index]
        t = [val / SECONDS_IN_YEAR for val in second_vector]

        forwards = vectorized_forward_continuous(
            S=spot.tolist(),
            r=rates.tolist(),
            T=t,
            q_factor=q_factor,
        )
        return pd.Series(data=forwards, index=continuous_divs.index)

    def _merge_partial(self, cached_series: pd.Series, forward_series: pd.Series) -> pd.Series:
        """Merges newly computed data with cached data, keeping latest values.

        Combines partial cache hits with newly computed forward prices, deduplicating
        by index and keeping the most recent values.

        Args:
            cached_series: Existing cached forward prices.
            forward_series: Newly computed forward prices.

        Returns:
            Merged Series with deduplicated index.

        Examples:
            >>> fwd_mgr = ForwardDataManager("AAPL")
            >>> merged = fwd_mgr._merge_partial(
            ...     cached_series=old_forwards,
            ...     forward_series=new_forwards
            ... )

        Notes:
            - Duplicates are removed, keeping 'last' (newest) values
            - Useful when cache partially covers requested date range
        """
        merged = pd.concat([cached_series, forward_series])
        forward_series = merged[~merged.index.duplicated(keep="last")]
        return forward_series

    def get_forward_timeseries(
        self,
        start_date: Union[datetime, str],
        end_date: Union[datetime, str],
        maturity_date: Union[datetime, str],
        div_type: Optional[DivType] = None,
        spot: Optional[TimeseriesData] = None,
        rates: Optional[RatesResult] = None,
        *,
        dividend_result: Optional[DividendsResult] = None,
        use_chain_spot: bool = True,
    ) -> ForwardResult:
        """Returns daily forward price time-series from valuation dates to maturity.

        Computes forward prices for each business day in [start_date, end_date],
        where each forward is valued to the fixed maturity_date. Uses discrete
        dividends (Schedule objects) or continuous yields depending on div_type.

        Args:
            start_date: First valuation date (YYYY-MM-DD string or datetime).
            end_date: Last valuation date (YYYY-MM-DD string or datetime).
            maturity_date: Fixed horizon date for all forwards (e.g., option expiry).
            div_type: DivType.DISCRETE or DivType.CONTINUOUS. Defaults to DISCRETE.
            spot: Optional pre-loaded TimeseriesData. Fetched if None.
            rates: Optional pre-computed rates data. Fetched if None.
            dividend_result: Pre-computed dividend data. If None, fetches internally.
            use_chain_spot: If True, uses split-adjusted chain_spot prices.

        Returns:
            ForwardResult containing daily_discrete_forward or daily_continuous_forward
            Series with DatetimeIndex, plus the dividend_result used and cache key.

        Raises:
            ValueError: If maturity_date < start_date.
            ValueError: If dividend_result.undo_adjust != use_chain_spot.

        Examples:
            >>> # Basic usage with automatic data fetching
            >>> fwd_mgr = ForwardDataManager("AAPL")
            >>> result = fwd_mgr.get_forward_timeseries(
            ...     start_date="2025-01-01",
            ...     end_date="2025-01-31",
            ...     maturity_date="2025-06-20",
            ...     div_type=DivType.DISCRETE,
            ...     use_chain_spot=True
            ... )
            >>> print(result.daily_discrete_forward.head())
            datetime
            2025-01-02    155.32
            2025-01-03    156.01
            ...

            >>> # Provide pre-computed data for efficiency
            >>> div_mgr = DividendDataManager("AAPL")
            >>> div_result = div_mgr.get_schedule_timeseries(
            ...     start_date="2025-01-01",
            ...     end_date="2025-01-31",
            ...     maturity_date="2025-06-20",
            ...     undo_adjust=True
            ... )
            >>> fwd_result = fwd_mgr.get_forward_timeseries(
            ...     start_date="2025-01-01",
            ...     end_date="2025-01-31",
            ...     maturity_date="2025-06-20",
            ...     dividend_result=div_result,
            ...     use_chain_spot=True
            ... )

        Notes:
            - Partial cache hits only compute missing dates
            - Cache expires after 12 hours
            - Spot, rates, and dividends are aligned to common dates
            - start_date/end_date define valuation date range
            - maturity_date is the fixed horizon (e.g., option expiry)
        """
        result = ForwardResult()
        og_start_date = start_date
        og_end_date = end_date
        div_type, start_dt, end_dt, mat_dt, start_str, end_str, mat_str = self._normalize_inputs(
            start_date=start_date,
            end_date=end_date,
            maturity_date=maturity_date,
            div_type=div_type,
        )

        if mat_dt < start_dt:
            raise ValueError("maturity_date must be >= start_date")

        key = self._build_key(mat_str=mat_str, div_type=div_type, use_chain_spot=use_chain_spot)

        cached_series, partial_hit, start_str, end_str, cached_result = self._try_get_cached(
            key=key,
            start_str=start_str,
            end_str=end_str,
            start_date=start_date,
            end_date=end_date,
            div_type=div_type,
        )
        if cached_result is not None:
            return cached_result

        dividend_result = self._get_dividend_result(
            start_str=start_str,
            end_str=end_str,
            mat_str=mat_str,
            div_type=div_type,
            dividend_result=dividend_result,
            use_chain_spot=use_chain_spot,
        )

        spot = self._load_spot(use_chain_spot=use_chain_spot, spot=spot)
        rates = self._load_rates(start_str=start_str, end_str=end_str, rates=rates)

        if div_type == DivType.DISCRETE:
            discrete_divs = dividend_result.daily_discrete_dividends

            spot, rates, discrete_divs = self._align_3(
                spot=spot,
                rates=rates,
                third=discrete_divs,
                third_name="discrete dividend schedules",
            )

            forward_series = self._compute_forward_discrete(
                spot=spot,
                rates=rates,
                discrete_divs=discrete_divs,
                mat_dt=mat_dt,
            )

            result.daily_discrete_forward = forward_series
            result.dividend_result = dividend_result

        elif div_type == DivType.CONTINUOUS:
            continuous_divs = dividend_result.daily_continuous_dividends

            spot, rates, continuous_divs = self._align_3(
                spot=spot,
                rates=rates,
                third=continuous_divs,
                third_name="div yields",
            )

            forward_series = self._compute_forward_continuous(
                spot=spot,
                rates=rates,
                continuous_divs=continuous_divs,
                mat_dt=mat_dt,
            )

            result.daily_continuous_forward = forward_series
            result.dividend_result = dividend_result

        else:
            raise ValueError(f"Unsupported dividend type: {div_type}")

        result.dividend_type = div_type
        result.key = key

        if partial_hit:
            forward_series = self._merge_partial(cached_series=cached_series, forward_series=forward_series)

        self.cache_it(key, forward_series, expire=86400 / 2)  # 12 hours expiry

        forward_series = _data_structure_sanitize(
            forward_series,
            start=og_start_date,
            end=og_end_date,
        )

        if div_type == DivType.DISCRETE:
            result.daily_discrete_forward = forward_series
        else:
            result.daily_continuous_forward = forward_series

        result.undo_adjust = use_chain_spot

        return result

    def make_key(self, *, symbol: str, interval=None, artifact_type=None, series_id=None, **extra_parts) -> str:
        """Delegates to BaseDataManager key construction.

        Constructs cache key by forwarding to parent class method.

        Args:
            symbol: Ticker symbol.
            interval: Time interval (e.g., Interval.EOD).
            artifact_type: Type of artifact (e.g., ArtifactType.FWD).
            series_id: Series identifier (e.g., SeriesId.HIST).
            **extra_parts: Additional key components (maturity, div_type, etc.).

        Returns:
            Unique cache key string.

        Examples:
            >>> fwd_mgr = ForwardDataManager("AAPL")
            >>> key = fwd_mgr.make_key(
            ...     symbol="AAPL",
            ...     artifact_type=ArtifactType.FWD,
            ...     maturity="2025-06-20"
            ... )
        """
        return super().make_key(
            symbol=symbol, interval=interval, artifact_type=artifact_type, series_id=series_id, **extra_parts
        )

    def cache_it(self, key: str, value: pd.Series, *, expire: Optional[int] = None) -> None:
        """Merges and caches forward time-series, excluding today's partial data.

        Appends new forward price data to existing cached time-series if cache entry exists.
        Filters out today's data to avoid caching incomplete/changing values.

        Args:
            key: Cache key identifier.
            value: Series of forward prices to cache (indexed by datetime).
            expire: Optional expiration time in seconds. Uses cache default if None.

        Examples:
            >>> fwd_mgr = ForwardDataManager("AAPL")
            >>> forwards = pd.Series([156.45, 157.12], index=pd.date_range("2025-01-01", periods=2))
            >>> fwd_mgr.cache_it("my_key", forwards, expire=43200)  # 12 hours

        Notes:
            - Existing cache entries are merged with new data
            - Duplicates are removed, keeping latest values
            - Today's data excluded to avoid caching incomplete values
        """
        ## Since it is a timeseries, we will append to existing if exists
        _data_structure_cache_it(self, key, value, expire=expire)
        return

    def get_forward(
        self,
        date: Union[datetime, str],
        maturity_date: Union[datetime, str],
        div_type: Optional[DivType] = None,
        dividend_result: Optional[DividendsResult] = None,
        spot: Optional[TimeseriesData] = None,
        rates: Optional[RatesResult] = None,
        *,
        use_chain_spot: bool = True,
    ) -> ForwardResult:
        """Returns the forward price at a specific valuation date.

        Computes forward price for a single valuation date to maturity. Wrapper around
        get_forward_timeseries with single-date range.

        Args:
            date: Valuation date (YYYY-MM-DD string or datetime).
            maturity_date: Horizon date (e.g., option expiry).
            div_type: DivType.DISCRETE or DivType.CONTINUOUS. Defaults to DISCRETE.
            dividend_result: Optional pre-computed dividend data.
            spot: Optional pre-loaded TimeseriesData.
            rates: Optional pre-computed rates data.
            use_chain_spot: If True, uses split-adjusted chain_spot prices.

        Returns:
            ForwardResult containing single forward price in daily_discrete_forward
            or daily_continuous_forward Series.

        Examples:
            >>> fwd_mgr = ForwardDataManager("AAPL")
            >>> result = fwd_mgr.get_forward(
            ...     date="2025-01-15",
            ...     maturity_date="2025-06-20",
            ...     div_type=DivType.DISCRETE,
            ...     use_chain_spot=True
            ... )
            >>> forward_price = result.daily_discrete_forward.iloc[0]
            >>> print(f"Forward: ${forward_price:.2f}")
            Forward: $156.45

        Notes:
            - Suitable for real-time pricing scenarios
            - Internally calls get_forward_timeseries with date as both start and end
        """
        div_type = DivType(div_type) if div_type is not None else DivType.DISCRETE
        date_str = date.strftime("%Y-%m-%d") if isinstance(date, datetime) else date
        mat_str = maturity_date.strftime("%Y-%m-%d") if isinstance(maturity_date, datetime) else maturity_date
        start = date_str
        end = date_str

        result = self.get_forward_timeseries(
            start_date=start,
            end_date=end,
            maturity_date=mat_str,
            div_type=div_type,
            use_chain_spot=use_chain_spot,
            dividend_result=dividend_result,
            spot=spot,
            rates=rates,
        )
        return result

    def offload(self, *args: Any, **kwargs: Any) -> None:
        """Placeholder for offload logic (not implemented).

        Reserved for future implementation of cache offloading or cleanup operations.
        Currently performs no action.

        Args:
            *args: Arbitrary positional arguments.
            **kwargs: Arbitrary keyword arguments.

        Examples:
            >>> fwd_mgr = ForwardDataManager("AAPL")
            >>> fwd_mgr.offload()  # No-op
        """
        print(f"No offload logic implemented for {self.CACHE_NAME}")
