"""Market Data Management and Timeseries Infrastructure.

This module provides comprehensive market data loading, caching, and retrieval
infrastructure for options backtesting and live trading. It manages spot prices,
chain data, dividends, risk-free rates, and custom market indicators with
intelligent caching strategies for performance optimization.

Core Classes:
    MarketTimeseries: Main container for all market data with lazy loading
    TimeseriesData: Structured holder for symbol-specific timeseries
    AtIndexResult: Point-in-time snapshot of market data for a symbol

Key Features:
    - Multi-source data retrieval (OpenBB, ThetaData, YFinance)
    - Hierarchical caching system (memory, disk, persistent)
    - Automatic data refresh with configurable intervals
    - Corporate action awareness (splits, dividends)
    - Custom data integration via user-defined callables
    - Thread-safe access with proper locking
    - Signal handlers for cleanup on exit

Data Types Managed:
    Spot Prices (Equity OHLCV):
        - Open, high, low, close prices
        - Volume and trading activity
        - Adjusted for splits and dividends
        - Sourced from OpenBB/YFinance

    Chain Spot Prices:
        - Underlying prices from option chain data
        - Used for option pricing consistency
        - May differ from equity spot due to timing
        - Sourced from ThetaData

    Dividends:
        - Regular dividend timeseries
        - Special dividends with ex-dates
        - Used for American option pricing
        - Affects early exercise decisions

    Risk-Free Rates:
        - Treasury yield curve (multiple tenors)
        - Interpolated rates for option pricing
        - Daily updates from Fed data
        - Annualized rate convention

    Additional Data (Custom):
        - User-defined indicators
        - Market regime indicators
        - Volatility surfaces
        - Sentiment data

Caching Architecture:
    Three-Tier System:
        1. Memory Cache (Fastest):
           - In-memory dictionaries
           - No expiration during session
           - Cleared on exit

        2. Disk Cache (Fast):
           - CustomCache with pickle serialization
           - 30-minute to 45-day expiration
           - Per-symbol and per-data-type

        3. Persistent Cache:
           - Long-term storage for historical data
           - Survives process restarts
           - Used for backtesting data

    Cache Keys:
        - Spot: SPOT_CACHE (45-day expiration)
        - Chain Spot: CHAIN_SPOT_CACHE (30-day expiration)
        - Dividends: DIVIDEND_CACHE (60-day expiration)

Data Retrieval Flow:
    1. Check memory cache → return if hit
    2. Check disk cache → populate memory if hit
    3. Query data source (OpenBB/ThetaData)
    4. Process and validate data
    5. Store in all cache levels
    6. Return to caller

AtIndexResult Structure:
    Point-in-time market data snapshot:
        - sym: Ticker symbol
        - date: Query date (pd.Timestamp)
        - spot: OHLCV data (pd.Series)
        - chain_spot: Chain-derived spot (pd.Series)
        - rates: Risk-free rates (pd.Series)
        - dividends: Dividend timeseries (pd.Series)
        - additional: Custom data dict

TimeseriesData Structure:
    Complete timeseries for a symbol:
        - spot: Full OHLCV DataFrame
        - chain_spot: Full chain spot DataFrame
        - dividends: Full dividend Series
        - additional_data: Dict of custom Series/DataFrames

MarketTimeseries Features:
    Lazy Loading:
        - Data loaded on first access
        - Avoids memory bloat for unused symbols
        - Transparent to caller

    Auto-Refresh:
        - Configurable refresh interval (default 30 min)
        - Checks last refresh timestamp
        - Updates stale data automatically
        - Disabled for historical backtests

    Property Protection:
        - Direct property access raises UnaccessiblePropertyError
        - Forces use of get_timeseries() or get_at_index()
        - Prevents inconsistent data states
        - Clear error messages guide users

    Signal Handling:
        - Registers SIGTERM and SIGINT handlers
        - Flushes caches on exit
        - Prevents data corruption
        - Ensures cleanup in all exit scenarios

Usage:
    # Initialize market timeseries
    market_data = MarketTimeseries(
        start='2024-01-01',
        end='2024-12-31'
    )

    # Get full timeseries for a symbol
    ts_data = market_data.get_timeseries(
        sym='AAPL',
        data_type='spot'
    )

    # Get point-in-time snapshot
    snapshot = market_data.get_at_index(
        sym='AAPL',
        date=pd.Timestamp('2024-06-15')
    )

    # Add custom data
    market_data.add_additional_data(
        sym='AAPL',
        name='custom_indicator',
        data=custom_series,
        callable_func=lambda df: process(df)
    )

Integration:
    - BacktestTimeseries extends this for backtest-specific needs
    - RiskManager uses for all market data access
    - OrderPicker queries for chain data
    - Position analysis uses for Greek calculations

Performance Considerations:
    - Caching dramatically reduces API calls
    - Memory usage grows with symbol count
    - Refresh interval trades freshness for performance
    - Disk cache speeds up repeated backtests

Data Sources:
    OpenBB:
        - Primary source for spot prices
        - Dividend data
        - Wide symbol coverage
        - Free tier available

    ThetaData:
        - Option chain data
        - Chain-derived spot prices
        - High-quality historical data
        - Requires subscription

    YFinance (Fallback):
        - Backup for spot prices
        - Free but rate-limited
        - Used when OpenBB fails

Error Handling:
    - YFinanceEmptyData: Raised when no data available
    - UnaccessiblePropertyError: Raised on direct property access
    - Automatic fallback to alternative sources
    - Logging of all data retrieval failures

Notes:
    - All dates handled as pandas Timestamps
    - Business day calendar used for date arithmetic
    - Data resampled to daily frequency
    - Missing data handled via forward-fill
    - Thread-safe via proper locking mechanisms
"""

import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Literal, Optional
import pandas as pd
from pandas.tseries.offsets import BDay
from dbase.DataAPI.ThetaData import resample  # noqa
from trade.helpers.helper import retrieve_timeseries, ny_now, CustomCache, YFinanceEmptyData, to_datetime
from trade.helpers.Logging import setup_logger
from trade.assets.rates import get_risk_free_rate_helper_v2
from EventDriven._vars import load_riskmanager_cache
from trade.optionlib.config.defaults import OPTION_TIMESERIES_START_DATE
from EventDriven.exceptions import UnaccessiblePropertyError
from trade.datamanager.utils.cache import (
    _cache_it_timeseries_data_structure,
    _data_structure_cache_check_missing,
    _CachedData,  # noqa
    _extract_data,  # noqa
    _simple_extract_from_cache,  # noqa
)
from trade.datamanager.utils.data_structure import _data_structure_sanitize
from trade.datamanager.utils.date import _sync_equity_date
from trade.datamanager.exceptions import EmptyDataException
from trade import SIGNALS_TO_RUN
from trade.datamanager.utils.logging import get_logging_level
from trade.datamanager.certification.market_timeseries import certify_market_factor_payload
from trade.datamanager._enums import CertificationLevel, RealTimeFallbackOption
from trade.datamanager.utils.point_in_time import resolve_value_at_date


logger = setup_logger("trade.datamanager.market_data", stream_log_level=get_logging_level())

## TODO: This var is from optionlib. Once ready, import from there.
## TODO: Implement interval handling to have multiple intervals

OPTIMESERIES: Optional["MarketTimeseries"] = None
DIVIDEND_CACHE: CustomCache = load_riskmanager_cache(target="dividend_timeseries")
SPOT_CACHE: CustomCache = load_riskmanager_cache(target="spot_timeseries")
CHAIN_SPOT_CACHE: CustomCache = load_riskmanager_cache(target="chain_spot_timeseries")
SPLIT_FACTOR_CACHE: CustomCache = load_riskmanager_cache(
    target="split_factor_timeseries", create_on_missing=True, clear_on_exit=False
)


@dataclass
class AtIndexResult:
    """Point-in-time market data snapshot for a symbol at a specific date.

    Container for all market data retrieved at a single date/timestamp. Used for
    accessing complete market state at a specific point in time for pricing, risk
    analysis, or strategy decisions.

    Attributes:
        sym: Equity ticker symbol (e.g., "AAPL", "MSFT").
        date: Query date as pd.Timestamp.
        spot: OHLCV data series with keys ['open', 'high', 'low', 'close', 'volume'].
        chain_spot: Chain-derived spot series (split-adjusted from ThetaData).
        rates: Risk-free rate series (currently np.nan, reserved for future use).
        dividends: Dividend amount paid on this date (0 if no dividend).
        dividend_yield: Calculated yield (dividend / spot close price).
        split_factor: Cumulative split adjustment factor (1.0 = no adjustment).
        additional: Dictionary of custom/additional data computed for this date.

    Examples:
        >>> mts = MarketTimeseries()
        >>> result = mts.get_at_index("AAPL", "2025-06-15")
        >>> print(f"Close: ${result.spot['close']:.2f}")
        >>> print(f"Dividend: ${result.dividends:.2f}")
        >>> print(f"Split Factor: {result.split_factor}")
        >>> if result.dividends > 0:
        ...     print(f"Ex-dividend date with yield: {result.dividend_yield:.2%}")
    """

    sym: str
    date: pd.Timestamp
    spot: pd.Series
    chain_spot: pd.Series
    rates: pd.Series
    dividends: int | float
    dividend_yield: int | float
    split_factor: float | int
    additional: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"AtIndexResult(sym={self.sym}, date={self.date})"


@dataclass
class TimeseriesData:
    """Complete timeseries data container for a specific symbol.

    Holds all market data types for a symbol as full timeseries (DataFrames or Series).
    Returned by MarketTimeseries.get_timeseries() with requested factors populated and
    non-requested factors set to None. Used for bulk analysis, backtesting, and
    vectorized calculations.

    Attributes:
        spot: OHLCV DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
            and DatetimeIndex. None if not requested.
        chain_spot: Chain-derived spot DataFrame (split-adjusted) with same structure
            as spot plus 'split_factor' column. None if not requested.
        dividends: Daily dividend amounts as Series with DatetimeIndex. Values are 0
            on non-dividend dates. None if not requested.
        dividend_yield: Calculated yield series (dividends / spot close). None if not
            requested or cannot be calculated.
        split_factor: Cumulative split adjustment factors as Series with DatetimeIndex.
            None if not requested.
        rates: Risk-free rate series (annualized). None if not requested.
        additional_data: Dictionary mapping custom data names to their Series/DataFrames.
            Empty dict if no additional data.

    Examples:
        >>> mts = MarketTimeseries()
        >>> # Get all data
        >>> ts_data = mts.get_timeseries("AAPL")
        >>> print(ts_data.spot.head())
        >>> print(f"Total dividends: ${ts_data.dividends.sum():.2f}")

        >>> # Get specific factor only
        >>> spot_only = mts.get_timeseries("AAPL", factor="spot")
        >>> assert spot_only.dividends is None  # Not requested
        >>> print(spot_only.spot['close'].mean())

        >>> # Work with additional custom data
        >>> custom = mts.get_timeseries("AAPL", factor="additional",
        ...                             additional_data_name="sma_20")
        >>> print(custom.additional_data['sma_20'].tail())
    """

    spot: pd.DataFrame
    chain_spot: pd.DataFrame
    dividends: pd.Series
    dividend_yield: pd.Series
    split_factor: pd.Series
    rates: Optional[pd.Series] = None
    additional_data: Dict[str, pd.Series] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"TimeseriesData(spot={self.spot is not None}, chain_spot={self.chain_spot is not None}, dividends={self.dividends is not None}, additional_data_keys={list(self.additional_data.keys())})"


@dataclass
class MarketTimeseries:
    """Comprehensive market data manager with multi-tier caching and lazy loading.

    Central hub for retrieving equity market data (spot prices, dividends, splits, rates)
    with intelligent caching at memory and disk levels. Implements lazy loading to minimize
    memory footprint and API calls. Prevents direct property access to ensure consistent
    data retrieval patterns.

    Architecture:
        - Three-tier caching: memory (instant), disk (fast), source (slow)
        - Lazy loading: data loaded only when accessed
        - Partial cache support: loads missing date ranges incrementally
        - Property protection: forces use of get_timeseries() or get_at_index()
        - Custom data support: user-defined transformations via callables

    Data Sources:
        - Spot prices: OpenBB/YFinance (equity OHLCV)
        - Chain spot: ThetaData (option chain underlying prices)
        - Dividends: OpenBB (regular and special dividends)
        - Rates: Federal Reserve (treasury yield curve)

    Attributes:
        additional_data: Dict of custom computed data {name: {symbol: Series}}.
        rates: DataFrame of risk-free rates with annualized yields.
        DEFAULT_NAMES: Class constant listing standard data types.
        _refresh_delta: Time interval for auto-refresh (None = disabled).
        _last_refresh: Timestamp of last data refresh.
        _start: Default start date for data retrieval (YYYY-MM-DD).
        _end: Default end date for data retrieval (YYYY-MM-DD).
        _today: Current date string (YYYY-MM-DD).
        should_refresh: Enable/disable auto-refresh behavior.

    Protected Properties:
        spot, chain_spot, dividends, split_factor: Direct access raises
        UnaccessiblePropertyError. Use get_timeseries() or get_at_index() instead.

    Cache Management:
        Uses module-level CustomCache instances:
        - SPOT_CACHE: 45-day expiration
        - CHAIN_SPOT_CACHE: 30-day expiration
        - DIVIDEND_CACHE: 60-day expiration
        - SPLIT_FACTOR_CACHE: Persistent (no expiration)

    Examples:
        >>> # Initialize with custom date range
        >>> mts = MarketTimeseries(
        ...     _start="2025-01-01",
        ...     _end="2025-12-31"
        ... )

        >>> # Get complete timeseries for a symbol
        >>> ts_data = mts.get_timeseries("AAPL")
        >>> print(ts_data.spot['close'].mean())

        >>> # Get point-in-time snapshot
        >>> snapshot = mts.get_at_index("AAPL", "2025-06-15")
        >>> print(f"Close: ${snapshot.spot['close']:.2f}")

        >>> # Add custom indicator
        >>> mts.calculate_additional_data(
        ...     factor="spot",
        ...     sym="AAPL",
        ...     additional_data_name="sma_50",
        ...     _callable=lambda s: s.rolling(50).mean(),
        ...     column="close"
        ... )

        >>> # Preload data for multiple symbols
        >>> for sym in ["AAPL", "MSFT", "GOOGL"]:
        ...     mts.load_timeseries(sym)

        >>> # Clear all caches
        >>> MarketTimeseries.clear_caches()

    Integration:
        Used by:
        - RiskManager for all market data access
        - BacktestTimeseries for historical simulations
        - OrderPicker for option chain data
        - Position analysis for Greek calculations

    Thread Safety:
        Cache operations are thread-safe via CustomCache locking mechanisms.
        Multiple readers can access cached data concurrently.
    """

    additional_data: Dict[str, Any] = field(default_factory=dict)
    DEFAULT_NAMES: ClassVar[List[str]] = ["spot", "chain_spot", "dividends", "split_factor", "dividend_yield"]
    _refresh_delta: Optional[timedelta] = timedelta(minutes=30)
    _last_refresh: Optional[datetime] = field(default_factory=ny_now)
    _start: str = OPTION_TIMESERIES_START_DATE
    _end: str = (datetime.now() - BDay(1)).strftime("%Y-%m-%d")
    _today: str = datetime.now().strftime("%Y-%m-%d")
    should_refresh: bool = True
    _rates: ClassVar[pd.DataFrame] = None

    @property
    def rates(self) -> pd.DataFrame:
        """Risk-free rates DataFrame with annualized yields.

        Retrieves and caches the risk-free rate curve from the Federal Reserve. Rates
        are annualized and indexed by date and tenor (e.g., 1M, 3M, 6M, 1Y). Used for
        option pricing and discounting cash flows.

        Returns:
            DataFrame with columns ['date', 'tenor', 'rate'] where 'rate' is the
            annualized yield for that tenor on that date.
        Examples:
            >>> mts = MarketTimeseries()
            >>> rates = mts.rates
            >>> print(rates.head())
        """
        if self._rates is None:
            self._rates = get_risk_free_rate_helper_v2()
        return self._rates

    @property
    def spot(self) -> dict:
        raise UnaccessiblePropertyError(
            "The 'spot' property is not accessible directly. Use 'get_timeseries' method instead. Or access via 'get_at_index' method."
        )

    @property
    def split_factor(self) -> dict:
        raise UnaccessiblePropertyError(
            "The 'split_factor' property is not accessible directly. Use 'get_timeseries' method instead. Or access via 'get_at_index' method."
        )

    @property
    def chain_spot(self) -> dict:
        raise UnaccessiblePropertyError(
            "The 'chain_spot' property is not accessible directly. Use 'get_timeseries' method instead. Or access via 'get_at_index' method."
        )

    @property
    def dividends(self) -> dict:
        raise UnaccessiblePropertyError(
            "The 'dividends' property is not accessible directly. Use 'get_timeseries' method instead. Or access via 'get_at_index' method."
        )

    @property
    def _spot(self) -> CustomCache:
        return SPOT_CACHE

    @property
    def _chain_spot(self) -> CustomCache:
        return CHAIN_SPOT_CACHE

    @property
    def _dividends(self) -> CustomCache:
        return DIVIDEND_CACHE

    @property
    def _split_factor(self) -> CustomCache:
        return SPLIT_FACTOR_CACHE

    @classmethod
    def clear_caches(cls) -> None:
        """Clear all caches used by MarketTimeseries.

        Removes all cached data from spot, chain_spot, dividend, and split_factor caches.
        Useful for forcing fresh data retrieval or reducing memory usage.

        Examples:
            >>> MarketTimeseries.clear_caches()
            >>> # All caches cleared, next data access will reload from source
        """
        SPOT_CACHE.clear()
        CHAIN_SPOT_CACHE.clear()
        DIVIDEND_CACHE.clear()
        SPLIT_FACTOR_CACHE.clear()
        logger.info("All MarketTimeseries caches have been cleared.")

    def _sanitize_market_factor_for_cache(
        self,
        data: pd.DataFrame | pd.Series,
        sym: str,
        start: str,
        end: str,
        factor: str,
    ) -> pd.DataFrame | pd.Series | None:
        """Structural sanitize before writing a market factor into MT internal cache.

        Args:
            data: Raw vendor payload.
            sym: Equity ticker.
            start: Inclusive cache window start (YYYY-MM-DD).
            end: Inclusive cache window end (YYYY-MM-DD).
            factor: Factor label for logging (spot, chain_spot, dividends, split_factor).

        Returns:
            Sanitized payload, or None when sanitize yields no rows.
        """
        try:
            return _data_structure_sanitize(
                data,
                start=start,
                end=end,
                source_name=f"market_timeseries {factor} for {sym}",
            )
        except EmptyDataException:
            logger.warning(
                "No %s data for symbol %s after sanitize for window %s to %s.",
                factor,
                sym,
                start,
                end,
            )
            return None

    def _load_spot_into_cache(self, sym: str, start: str, end: str) -> pd.DataFrame | None:
        """Load spot OHLCV data for a symbol into the cache.

        Retrieves equity spot prices from data source (OpenBB/YFinance) and stores in
        the spot cache with intelligent merge logic for existing data. Handles missing
        data gracefully with warning logging.

        Args:
            sym: Stock ticker symbol (e.g., "AAPL", "MSFT").
            start: Start date string (YYYY-MM-DD format).
            end: End date string (YYYY-MM-DD format).

        Examples:
            >>> mts = MarketTimeseries()
            >>> mts._load_spot_into_cache("AAPL", "2025-01-01", "2025-01-31")
            >>> # Spot data now cached and available for retrieval
        """

        try:
            spot_data = retrieve_timeseries(
                tick=sym,
                start=start,
                end=end,
            )
            spot_data = self._sanitize_market_factor_for_cache(
                spot_data, sym, start, end, "spot"
            )
            if spot_data is None:
                return None
            _cache_it_timeseries_data_structure(
                existing=self._spot.get(sym),
                key=sym,
                value=spot_data,
                expire=None,
                cache=self._spot,
            )
            logger.info("Loaded spot data for symbol %s into cache.", sym)
            return spot_data
        except YFinanceEmptyData:
            logger.warning("No spot data found for symbol %s from data source. Will skip caching.", sym)
            return None

    def _load_chain_spot_into_cache(self, sym: str, start: str, end: str) -> pd.DataFrame | None:
        """Load chain-derived spot data for a symbol into the cache.

        Retrieves underlying prices from option chain data (ThetaData) and stores in
        the chain_spot cache. Chain spot is split-adjusted and may differ from equity
        spot due to timing. Used for consistent option pricing.

        Args:
            sym: Stock ticker symbol (e.g., "AAPL", "MSFT").
            start: Start date string (YYYY-MM-DD format).
            end: End date string (YYYY-MM-DD format).

        Examples:
            >>> mts = MarketTimeseries()
            >>> mts._load_chain_spot_into_cache("AAPL", "2025-01-01", "2025-01-31")
            >>> # Chain spot data now cached with split adjustments
        """
        try:
            chain_spot_data = retrieve_timeseries(
                tick=sym,
                start=start,
                end=end,
                spot_type="chain_spot",
            )
            chain_spot_data = self._sanitize_market_factor_for_cache(
                chain_spot_data, sym, start, end, "chain_spot"
            )
            if chain_spot_data is None:
                return None

            _cache_it_timeseries_data_structure(
                existing=self._chain_spot.get(sym),
                key=sym,
                value=chain_spot_data,
                expire=None,
                cache=self._chain_spot,
            )
            logger.info("Loaded chain spot data for symbol %s into cache.", sym)
            return chain_spot_data
        except YFinanceEmptyData:
            logger.warning("No chain spot data found for symbol %s from data source. Will skip caching.", sym)
            return None

    def _load_dividends_into_cache(self, sym: str, start: str = None, end: str = None) -> pd.DataFrame | None:
        """Load daily dividend timeseries for a symbol into the cache.

        Retrieves regular and special dividends with ex-dates from data source and stores
        in the dividends cache. Used for American option pricing and forward calculations.
        Defaults to instance start/end dates if not provided.

        Args:
            sym: Stock ticker symbol (e.g., "AAPL", "MSFT").
            start: Optional start date string (YYYY-MM-DD). Defaults to self._start.
            end: Optional end date string (YYYY-MM-DD). Defaults to self._end.

        Examples:
            >>> mts = MarketTimeseries()
            >>> mts._load_dividends_into_cache("AAPL")
            >>> # Loads dividends for instance's full date range
            >>> mts._load_dividends_into_cache("MSFT", "2025-01-01", "2025-06-30")
            >>> # Loads dividends for specific date range
        """
        from trade.datamanager.market_data_helpers.dividends import get_daily_dividends_timeseries

        try:
            divs = get_daily_dividends_timeseries(sym, start=start or self._start, end=end or self._end)
            load_start = start or self._start
            load_end = end or self._end
            divs = self._sanitize_market_factor_for_cache(
                divs, sym, load_start, load_end, "dividends"
            )
            if divs is None:
                return None
            _cache_it_timeseries_data_structure(
                existing=self._dividends.get(sym),
                key=sym,
                value=divs,
                expire=None,
                cache=self._dividends,
                skip_today_check=True,  # Dividends don't change intraday, so skip today check
            )
            logger.info("Loaded dividend data for symbol %s into cache.", sym)
            return divs
        except YFinanceEmptyData:
            logger.warning("No dividend data found for symbol %s from data source. Will skip caching.", sym)
            return None

    def _load_split_factor_into_cache(self, sym: str, start: str, *args, **kwargs) -> pd.DataFrame | None:
        """Load split factor timeseries for a symbol into the cache.

        Extracts split factors from chain spot data and stores in the split_factor cache.
        Split factors are cumulative multipliers for historical price adjustment. Skips
        today check since splits don't change intraday.

        Args:
            sym: Stock ticker symbol (e.g., "AAPL", "MSFT").
            start: Start date string (YYYY-MM-DD format). End date uses instance _end.

        Examples:
            >>> mts = MarketTimeseries()
            >>> mts._load_split_factor_into_cache("AAPL", "2025-01-01")
            >>> # Split factors loaded from chain spot data
        """
        try:
            chain_spot = self._load_chain_spot_into_cache(sym, start, self._end)
            chain_spot = _extract_data(chain_spot)
            # if isinstance(chain_spot, _CachedData) or chain_spot.__class__.__name__ == "_CachedData":
            #     chain_spot = chain_spot.data

            split_factor = chain_spot["split_factor"]
            split_factor = self._sanitize_market_factor_for_cache(
                split_factor, sym, start, self._end, "split_factor"
            )
            if split_factor is None:
                return None
            _cache_it_timeseries_data_structure(
                existing=self._split_factor.get(sym),
                key=sym,
                value=split_factor,
                expire=None,
                cache=self._split_factor,
                ## Cutting out today check as split factors don't change intraday
                skip_today_check=True,
            )
            logger.info("Loaded split factor data for symbol %s into cache.", sym)
            return split_factor
        except YFinanceEmptyData:
            logger.warning("No split factor data found for symbol %s from data source. Will skip caching.", sym)
            return None

    def _clip_to_date_range(
        self, df: pd.DataFrame | pd.Series, start: str, end: str, *args, **kwargs
    ) -> pd.DataFrame | pd.Series:
        """Clip a DataFrame or Series to the specified date range.

        Filters timeseries data to only include dates within [start, end] inclusive.
        Uses date objects for comparison to handle datetime vs date mismatches.

        Args:
            df: DataFrame or Series with DatetimeIndex to filter.
            start: Start date string (YYYY-MM-DD format).
            end: End date string (YYYY-MM-DD format).

        Returns:
            Filtered DataFrame or Series with only dates in range.

        Examples:
            >>> mts = MarketTimeseries()
            >>> spot_full = mts._get_spot_timeseries("AAPL")
            >>> spot_q1 = mts._clip_to_date_range(spot_full, "2025-01-01", "2025-03-31")
            >>> # Returns only Q1 2025 data
        """
        df = _extract_data(df)  # Unwrap from _CachedData if needed
        # if isinstance(df, _CachedData) or df.__class__.__name__ == "_CachedData":
        #     df = df.data
        clipped = df[(df.index.date >= to_datetime(start).date()) & (df.index.date <= to_datetime(end).date())]
        return clipped

    def _sync_equity_window(self, sym: str, start: str, end: str) -> tuple[str, str]:
        """Clamp a factor request window to runtime EOD availability.

        Args:
            sym: Equity ticker.
            start: Requested window start (YYYY-MM-DD).
            end: Requested window end (YYYY-MM-DD).

        Returns:
            Synced ``(start, end)`` strings for cache, clip, and certification.
        """
        start_dt, end_dt = _sync_equity_date(start, end, symbol=sym.upper())
        return start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")

    def _get_spot_timeseries(self, sym: str, start: str = None, end: str = None, *args, **kwargs) -> pd.DataFrame:
        """Retrieve spot OHLCV timeseries for a symbol with automatic cache management.

        Checks cache for existing data, loads from source if missing, and handles partial
        cache hits by loading only missing dates. Automatically clips to requested range.

        Args:
            sym: Stock ticker symbol (e.g., "AAPL", "MSFT").
            start: Optional start date string (YYYY-MM-DD). Defaults to self._start.
            end: Optional end date string (YYYY-MM-DD). Defaults to self._end.
            **kwargs: Additional arguments (currently unused, for extensibility).

        Returns:
            DataFrame with OHLCV columns (open, high, low, close, volume) and DatetimeIndex.

        Examples:
            >>> mts = MarketTimeseries()
            >>> spot = mts._get_spot_timeseries("AAPL", "2025-01-01", "2025-01-31")
            >>> print(spot.columns)  # ['open', 'high', 'low', 'close', 'volume']
        """
        start = start or self._start
        end = end or self._end
        start, end = self._sync_equity_window(sym, start, end)
        cached_data = self._spot.get(sym)
        if cached_data is None:
            cached_data = self._load_spot_into_cache(sym, start, end)
        cached_data, is_partial, missing_start_date, missing_end_date = _data_structure_cache_check_missing(
            cached_data=cached_data,
            key=sym,
            start_dt=start,
            end_dt=end,
        )
        if is_partial:
            miss_start = missing_start_date.strftime("%Y-%m-%d")
            miss_end = missing_end_date.strftime("%Y-%m-%d")
            data = self._load_spot_into_cache(sym, miss_start, miss_end)
            ## Compare tip reload request vs what the vendor load actually returned.
            if data is not None and len(data):
                logger.info(
                    "Partial reload vendor span for key=%s factor=spot: "
                    "requested=%s..%s returned=%s..%s rows=%s",
                    sym,
                    miss_start,
                    miss_end,
                    data.index.min().date(),
                    data.index.max().date(),
                    len(data),
                )
            else:
                logger.info(
                    "Partial reload vendor span for key=%s factor=spot: "
                    "requested=%s..%s returned=empty",
                    sym,
                    miss_start,
                    miss_end,
                )
            ## Keep concat so tip rows cut from disk cache still reach certify;
            ## dedupe so overlapping reload (wide miss / full-history clip) cannot
            ## flood L3 with DUPLICATE_INDEX.
            cached_data = pd.concat([cached_data, data]).sort_index()
            cached_data = cached_data[~cached_data.index.duplicated(keep="last")]

        clipped = self._clip_to_date_range(cached_data, start, end).copy()
        return certify_market_factor_payload(
            sym,
            "spot",
            clipped,
            start,
            end,
            method="_get_spot_timeseries",
            certification_level=kwargs.get("certification_level"),
        )

    def _get_chain_spot_timeseries(self, sym: str, start: str = None, end: str = None, *args, **kwargs) -> pd.DataFrame:
        """Retrieve chain-derived spot timeseries for a symbol with automatic cache management.

        Checks cache for existing chain spot data, loads from ThetaData if missing, and
        handles partial cache hits. Chain spot is split-adjusted and used for option pricing.

        Args:
            sym: Stock ticker symbol (e.g., "AAPL", "MSFT").
            start: Optional start date string (YYYY-MM-DD). Defaults to self._start.
            end: Optional end date string (YYYY-MM-DD). Defaults to self._end.
            **kwargs: Additional arguments (currently unused, for extensibility).

        Returns:
            DataFrame with chain spot columns including split_factor and DatetimeIndex.

        Examples:
            >>> mts = MarketTimeseries()
            >>> chain_spot = mts._get_chain_spot_timeseries("AAPL", "2025-01-01", "2025-01-31")
            >>> print(chain_spot['split_factor'])  # Cumulative split adjustments
        """

        start = start or self._start
        end = end or self._end
        start, end = self._sync_equity_window(sym, start, end)
        cached_data = self._chain_spot.get(sym)
        if cached_data is None:
            cached_data = self._load_chain_spot_into_cache(sym, start, end)
        cached_data, is_partial, missing_start_date, missing_end_date = _data_structure_cache_check_missing(
            cached_data=cached_data,
            key=sym,
            start_dt=start,
            end_dt=end,
        )
        if is_partial:
            miss_start = missing_start_date.strftime("%Y-%m-%d")
            miss_end = missing_end_date.strftime("%Y-%m-%d")
            data = self._load_chain_spot_into_cache(sym, miss_start, miss_end)
            ## chain_spot may download full history internally; log post-load span here.
            if data is not None and len(data):
                logger.info(
                    "Partial reload vendor span for key=%s factor=chain_spot: "
                    "requested=%s..%s returned=%s..%s rows=%s",
                    sym,
                    miss_start,
                    miss_end,
                    data.index.min().date(),
                    data.index.max().date(),
                    len(data),
                )
            else:
                logger.info(
                    "Partial reload vendor span for key=%s factor=chain_spot: "
                    "requested=%s..%s returned=empty",
                    sym,
                    miss_start,
                    miss_end,
                )
            ## Keep concat so tip rows cut from disk cache still reach certify;
            ## dedupe so overlapping reload cannot flood L3 with DUPLICATE_INDEX.
            cached_data = pd.concat([cached_data, data]).sort_index()
            cached_data = cached_data[~cached_data.index.duplicated(keep="last")]
        clipped = self._clip_to_date_range(cached_data, start, end)
        return certify_market_factor_payload(
            sym,
            "chain_spot",
            clipped,
            start,
            end,
            method="_get_chain_spot_timeseries",
            certification_level=kwargs.get("certification_level"),
        )

    def _get_dividends_timeseries(self, sym: str, start: str = None, end: str = None, *args, **kwargs) -> pd.Series:
        """Retrieve daily dividend timeseries for a symbol with automatic cache management.

        Checks cache for existing dividend data, loads from source if missing, and handles
        partial cache hits. Returns daily dividend amounts with ex-dates as index.

        Args:
            sym: Stock ticker symbol (e.g., "AAPL", "MSFT").
            start: Optional start date string (YYYY-MM-DD). Defaults to self._start.
            end: Optional end date string (YYYY-MM-DD). Defaults to self._end.
            **kwargs: Additional arguments (currently unused, for extensibility).

        Returns:
            Series with dividend amounts and DatetimeIndex of ex-dates.

        Examples:
            >>> mts = MarketTimeseries()
            >>> divs = mts._get_dividends_timeseries("AAPL", "2025-01-01", "2025-12-31")
            >>> print(divs[divs > 0])  # Show only dividend payment dates
        """
        if start is None:
            start = self._start
        if end is None:
            end = self._end
        start, end = self._sync_equity_window(sym, start, end)
        cached_data = self._dividends.get(sym)
        if cached_data is None:
            cached_data = self._load_dividends_into_cache(sym, start, end)
        cached_data, is_partial, missing_start_date, missing_end_date = _data_structure_cache_check_missing(
            cached_data=cached_data,
            key=sym,
            start_dt=start,
            end_dt=end,
        )
        if is_partial:
            miss_start = missing_start_date.strftime("%Y-%m-%d")
            miss_end = missing_end_date.strftime("%Y-%m-%d")
            data = self._load_dividends_into_cache(sym, miss_start, miss_end)
            ## Compare tip reload request vs dividend vendor payload span.
            if data is not None and len(data):
                logger.info(
                    "Partial reload vendor span for key=%s factor=dividends: "
                    "requested=%s..%s returned=%s..%s rows=%s",
                    sym,
                    miss_start,
                    miss_end,
                    data.index.min().date(),
                    data.index.max().date(),
                    len(data),
                )
            else:
                logger.info(
                    "Partial reload vendor span for key=%s factor=dividends: "
                    "requested=%s..%s returned=empty",
                    sym,
                    miss_start,
                    miss_end,
                )
            ## Keep concat so tip rows cut from disk cache still reach certify;
            ## dedupe so overlapping reload cannot flood L3 with DUPLICATE_INDEX.
            cached_data = pd.concat([cached_data, data]).sort_index()
            cached_data = cached_data[~cached_data.index.duplicated(keep="last")]

        clipped = self._clip_to_date_range(cached_data, start, end, *args, **kwargs)
        return certify_market_factor_payload(
            sym,
            "dividends",
            clipped,
            start,
            end,
            method="_get_dividends_timeseries",
            certification_level=kwargs.get("certification_level"),
        )

    def _get_split_factor_timeseries(self, sym: str, start: str = None, end: str = None, *args, **kwargs) -> pd.Series:
        """Retrieve split factor timeseries for a symbol with automatic cache management.

        Checks cache for existing split factor data, extracts from chain spot if missing,
        and handles partial cache hits. Split factors are cumulative multipliers for
        historical price adjustment.

        Args:
            sym: Stock ticker symbol (e.g., "AAPL", "MSFT").
            start: Optional start date string (YYYY-MM-DD). Defaults to self._start.
            end: Optional end date string (YYYY-MM-DD). Defaults to self._end.
            **kwargs: Additional arguments (currently unused, for extensibility).

        Returns:
            Series with cumulative split factors and DatetimeIndex.

        Examples:
            >>> mts = MarketTimeseries()
            >>> splits = mts._get_split_factor_timeseries("AAPL", "2025-01-01", "2025-12-31")
            >>> print(splits[splits != 1.0])  # Show dates with splits
        """

        ## Decide dates
        start = start or self._start
        end = end or self._end
        start, end = self._sync_equity_window(sym, start, end)
        cached_data = self._split_factor.get(sym)

        ## If no data, load from chain spot and extract split factor
        if cached_data is None:
            cached_data = self._load_split_factor_into_cache(sym, start)

        ## No need to access data from _CachedData yet
        ## Data structure checks this data
        cached_data, is_partial, missing_start_date, missing_end_date = _data_structure_cache_check_missing(
            cached_data=cached_data,
            key=sym,
            start_dt=start,
            end_dt=self._end,
        )
        if is_partial:
            self._load_split_factor_into_cache(
                sym, missing_start_date.strftime("%Y-%m-%d"), end=missing_end_date.strftime("%Y-%m-%d")
            )
            cached_data = self._split_factor.get(sym)

            ## If it is _CachedData, get the data out of it for the next steps
            cached_data = _extract_data(cached_data)
            # if isinstance(cached_data, _CachedData):
            #     cached_data = cached_data.data

        clipped = self._clip_to_date_range(cached_data, start, end, *args, **kwargs)
        return certify_market_factor_payload(
            sym,
            "split_factor",
            clipped,
            start,
            end,
            method="_get_split_factor_timeseries",
            certification_level=kwargs.get("certification_level"),
        )

    def _get_dividend_yield_timeseries(
        self, sym: str, start: str = None, end: str = None, *args, **kwargs
    ) -> pd.Series:
        """Calculate and retrieve dividend yield timeseries for a symbol.

        Computes daily dividend yield by dividing dividend amounts by spot close prices.
        Automatically retrieves spot and dividend data from cache or loads if needed.

        Args:
            sym: Stock ticker symbol (e.g., "AAPL", "MSFT").
            start: Optional start date string (YYYY-MM-DD). Defaults to self._start.
            end: Optional end date string (YYYY-MM-DD). Defaults to self._end.
            **kwargs: Additional arguments passed to date clipping.

        Returns:
            Series with dividend yields (as decimals, not percentages) and DatetimeIndex.

        Examples:
            >>> mts = MarketTimeseries()
            >>> yield_ts = mts._get_dividend_yield_timeseries("AAPL")
            >>> print(yield_ts.mean() * 100)  # Average yield as percentage
        """
        start = start or self._start
        end = end or self._end
        start, end = self._sync_equity_window(sym, start, end)
        spot = self._get_spot_timeseries(sym, start=start, end=end)
        dividends = self._get_dividends_timeseries(sym, start=start, end=end)

        dividend_yield = dividends / spot["close"]
        # Fill non-dividend dates with 0 yield. I believe it should be fine
        # TODO: Pay close attention to this. Maybe find an alternative way to handle non-dividend dates if it causes issues.
        dividend_yield.fillna(0.0, inplace=True)

        clipped = self._clip_to_date_range(dividend_yield, start, end, *args, **kwargs)
        return certify_market_factor_payload(
            sym, "dividend_yield", clipped, start, end, method="_get_dividend_yield_timeseries"
        )

    def get_split_factor_at_index(
        self,
        sym: str,
        index: pd.Timestamp,
        fallback_option: RealTimeFallbackOption = RealTimeFallbackOption.USE_LAST_AVAILABLE,
        *args,
        **kwargs,
    ) -> float | int:
        """Retrieve the split factor for a symbol at a specific date with forward-fill logic.

        Fetches a 10-business-day lookback window certified at L1, then resolves the
        split factor per ``fallback_option``.

        Args:
            sym: Stock ticker symbol (e.g., "AAPL", "MSFT").
            index: Date for split factor lookup (pd.Timestamp or date string).
            fallback_option: Policy when exact date is missing or non-trading.

        Returns:
            Cumulative split factor at the specified date (1.0 = no adjustment).
        """
        sym = sym.upper()

        def _fetch(start: str, end: str) -> pd.Series:
            series = self._get_split_factor_timeseries(
                sym,
                start=start,
                end=end,
                certification_level=CertificationLevel.L1,
            )
            return series if series is not None else pd.Series(dtype=float)

        row, _meta = resolve_value_at_date(
            index,
            fetch_timeseries=_fetch,
            extract_timeseries=lambda s: s,
            fallback_option=fallback_option,
        )
        if row.empty:
            return 1.0
        return row.iloc[0]

    def get_at_index(
        self,
        sym: str,
        index: pd.Timestamp,
        fallback_option: RealTimeFallbackOption = RealTimeFallbackOption.USE_LAST_AVAILABLE,
        *args,
        **kwargs,
    ) -> AtIndexResult:
        """Retrieve point-in-time market data snapshot for a symbol at a specific date.

        Returns a complete snapshot of market data (spot, chain_spot, dividends, rates,
        split_factor, dividend_yield) for a single date. Ensures all necessary data is
        loaded before retrieval.

        Args:
            sym: Stock ticker symbol (e.g., "AAPL", "MSFT").
            index: Date for data snapshot (pd.Timestamp or date string YYYY-MM-DD).
            interval: Time interval (currently only "1d" supported).

        Returns:
            AtIndexResult containing spot (Series), chain_spot (Series), dividends (float),
            rates (float), dividend_yield (float), split_factor (float), and metadata.

        Examples:
            >>> mts = MarketTimeseries()
            >>> snapshot = mts.get_at_index("AAPL", "2025-06-15")
            >>> print(snapshot.spot['close'])  # Closing price
            >>> print(snapshot.dividends)  # Dividend amount on this date
            >>> print(snapshot.split_factor)  # Cumulative split adjustment
        """

        sym = sym.upper()
        index = to_datetime(index, format="%Y-%m-%d")
        cert_kw = {"certification_level": CertificationLevel.L1}

        spot_row, _ = resolve_value_at_date(
            index,
            fetch_timeseries=lambda s, e: self._get_spot_timeseries(sym, start=s, end=e, **cert_kw),
            extract_timeseries=lambda df: df,
            fallback_option=fallback_option,
        )
        chain_row, _ = resolve_value_at_date(
            index,
            fetch_timeseries=lambda s, e: self._get_chain_spot_timeseries(sym, start=s, end=e, **cert_kw),
            extract_timeseries=lambda df: df,
            fallback_option=fallback_option,
        )
        div_row, _ = resolve_value_at_date(
            index,
            fetch_timeseries=lambda s, e: self._get_dividends_timeseries(sym, start=s, end=e, **cert_kw),
            extract_timeseries=lambda s: s,
            fallback_option=fallback_option,
        )
        split_row, _ = resolve_value_at_date(
            index,
            fetch_timeseries=lambda s, e: self._get_split_factor_timeseries(sym, start=s, end=e, **cert_kw),
            extract_timeseries=lambda s: s,
            fallback_option=fallback_option,
        )

        spot = spot_row.iloc[0] if not spot_row.empty else None
        chain_spot = chain_row.iloc[0] if not chain_row.empty else None
        dividends = float(div_row.iloc[0]) if not div_row.empty and pd.notna(div_row.iloc[0]) else 0.0
        split_factor = float(split_row.iloc[0]) if not split_row.empty and pd.notna(split_row.iloc[0]) else 1.0
        index_str = index.strftime("%Y-%m-%d")
        dividend_yield = (
            dividends / spot["close"]
            if spot is not None and dividends is not None and spot.get("close")
            else None
        )

        return AtIndexResult(
            spot=spot,
            chain_spot=chain_spot,
            dividends=dividends,
            sym=sym,
            date=index_str,
            rates=np.nan,
            dividend_yield=dividend_yield,
            split_factor=split_factor,
        )

    def calculate_additional_data(
        self,
        factor: Literal["spot", "chain_spot", "dividends", "split_factor"],
        sym: str,
        additional_data_name: str,
        _callable: Any,
        column: Optional[str] = "close",
        force_add: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """Load additional data for a factor using a custom transformation function.

        Applies a user-defined callable to existing market data to create custom indicators
        or derived timeseries. The callable receives a pd.Series and must return a pd.Series.
        Results are stored in the additional_data dictionary for later retrieval.

        Storage Schema:
            additional_data = {additional_data_name: {sym: pd.Series}}

        Args:
            factor: Base data type to transform ('spot', 'chain_spot', 'dividends', 'split_factor').
            sym: Stock ticker symbol (e.g., "AAPL", "MSFT").
            additional_data_name: Identifier for storing the computed data.
            _callable: Function that takes pd.Series and returns pd.Series.
            column: Column to extract from DataFrame factors (e.g., 'close', 'volume').
            force_add: If True, overwrites existing data for this name and symbol.

        Raises:
            ValueError: If factor not recognized.
            ValueError: If symbol data not found for the specified factor.
            ValueError: If column not found in factor DataFrame.

        Examples:
            >>> mts = MarketTimeseries()
            >>> # Calculate 20-day moving average of close prices
            >>> mts.calculate_additional_data(
            ...     factor="spot",
            ...     sym="AAPL",
            ...     additional_data_name="sma_20",
            ...     _callable=lambda s: s.rolling(20).mean(),
            ...     column="close"
            ... )
            >>> # Calculate RSI from dividends
            >>> mts.calculate_additional_data(
            ...     factor="dividends",
            ...     sym="MSFT",
            ...     additional_data_name="div_rsi",
            ...     _callable=lambda s: compute_rsi(s, period=14)
            ... )
        """

        ## Raise error if factor not recognized
        if factor not in self.DEFAULT_NAMES:
            raise ValueError(f"Factor {factor} not recognized. Must be one of ['spot', 'chain_spot', 'dividends'].")

        ## Get the data for the specified factor and symbol
        factor_data = getattr(self, factor).get(sym)

        ## Raise error if symbol not found
        if factor_data is None:
            raise ValueError(f"No data found for factor {factor} and symbol {sym}.")

        ## If column specified, ensure it exists in the DataFrame
        if column and isinstance(factor_data, (pd.DataFrame, pd.Series)):
            if column not in factor_data.columns:
                raise ValueError(f"Column {column} not found in data for factor {factor} and symbol {sym}.")
            factor_data = factor_data[column]

        ## Process the data using the provided callable
        processed_data = _callable(factor_data)
        if additional_data_name not in self.additional_data:
            self.additional_data[additional_data_name] = {}

        ## Check if data already exists and force_add is not set
        exists = sym in self.additional_data.get(additional_data_name, {})
        if exists and not force_add:
            logger.info(
                "Additional data for %s and symbol %s already exists. Use force_add=True to overwrite.",
                additional_data_name,
                sym,
            )
            return

        self.additional_data[additional_data_name][sym] = processed_data

    def get_timeseries(
        self,
        sym: str,
        factor: Literal["spot", "chain_spot", "dividends", "split_factor", "additional"] = None,
        additional_data_name: Optional[str] = None,
        start_date: str | datetime = None,
        end_date: str | datetime = None,
        *args,
        **kwargs,
    ) -> TimeseriesData:
        """Retrieve timeseries data for a symbol with optional factor and date filtering.

        Main method for accessing market data. Can return specific factors (spot, chain_spot,
        dividends, split_factor), additional custom data, or all factors combined. Automatically
        handles caching, data loading, and date range filtering.

        Args:
            sym: Stock ticker symbol (e.g., "AAPL", "MSFT").
            factor: Data type to retrieve. If None, returns all factors.
            additional_data_name: Required when factor='additional'. Identifies custom data.
            start_date: Optional start date for filtering (YYYY-MM-DD string or datetime).
            end_date: Optional end date for filtering (YYYY-MM-DD string or datetime).

        Returns:
            TimeseriesData containing requested data. Non-requested fields are None.

        Raises:
            ValueError: If factor not recognized.
            ValueError: If factor='additional' but additional_data_name not provided.
            ValueError: If additional_data_name not found in cached additional data.
            ValueError: If no data found for requested factor and symbol.

        Examples:
            >>> mts = MarketTimeseries()
            >>> # Get all data for a symbol
            >>> all_data = mts.get_timeseries("AAPL")
            >>> print(all_data.spot.head())
            >>> print(all_data.dividends.sum())

            >>> # Get specific factor with date range
            >>> spot_q1 = mts.get_timeseries(
            ...     "AAPL",
            ...     factor="spot",
            ...     start_date="2025-01-01",
            ...     end_date="2025-03-31"
            ... )

            >>> # Get custom additional data
            >>> sma_data = mts.get_timeseries(
            ...     "AAPL",
            ...     factor="additional",
            ...     additional_data_name="sma_20"
            ... )

            >>> # Calculate dividend yield on the fly
            >>> yield_data = mts.get_timeseries("MSFT", factor="dividend_yield")
        """

        data_funcs = {
            "spot": self._get_spot_timeseries,
            "chain_spot": self._get_chain_spot_timeseries,
            "dividends": self._get_dividends_timeseries,
            "split_factor": self._get_split_factor_timeseries,
            "dividend_yield": self._get_dividend_yield_timeseries,
        }
        sym = sym.upper()
        end_date = end_date or self._end

        if factor not in self.DEFAULT_NAMES + ["additional", None]:
            raise ValueError(f"Factor {factor} not recognized. Must be one of {self.DEFAULT_NAMES + ['additional']}.")
        if factor == "additional":
            if additional_data_name is None:
                raise ValueError("additional_data_name must be provided when factor is 'additional'.")
            data = self.additional_data.get(additional_data_name, {}).get(sym)
            if data is None:
                raise ValueError(f"No additional data found for name {additional_data_name} and symbol {sym}.")
            return TimeseriesData(
                spot=None,
                chain_spot=None,
                dividends=None,
                additional_data={additional_data_name: data},
                split_factor=None,
                dividend_yield=None,
            )

        elif factor in self.DEFAULT_NAMES:
            data = data_funcs[factor](sym, start=start_date, end=end_date)

            if data is None:
                raise ValueError(f"No data found for factor {factor} and symbol {sym}.")
            if factor == "spot":
                ts = TimeseriesData(spot=data, chain_spot=None, dividends=None, dividend_yield=None, split_factor=None)
            elif factor == "chain_spot":
                ts = TimeseriesData(spot=None, chain_spot=data, dividends=None, dividend_yield=None, split_factor=None)
            elif factor == "dividends":
                ts = TimeseriesData(spot=None, chain_spot=None, dividends=data, dividend_yield=None, split_factor=None)
            elif factor == "dividend_yield":
                ts = TimeseriesData(spot=None, chain_spot=None, dividends=None, dividend_yield=data, split_factor=None)
            elif factor == "split_factor":
                ts = TimeseriesData(spot=None, chain_spot=None, dividends=None, split_factor=data, dividend_yield=None)
            else:
                raise ValueError(f"Unhandled factor {factor}.")

        ## If no factor specified, return all data
        elif factor is None:
            spot = self._get_spot_timeseries(sym, start=start_date, end=end_date)
            chain_spot = self._get_chain_spot_timeseries(sym, start=start_date, end=end_date)
            dividends = self._get_dividends_timeseries(sym, start=start_date, end=end_date)
            dividend_yield = self._get_dividend_yield_timeseries(sym, start=start_date, end=end_date)
            split_factor = self._get_split_factor_timeseries(sym, start=start_date, end=end_date)

            ## Filter data by start_date and end_date if provided
            if start_date is not None or end_date is not None:
                start_date = pd.to_datetime(start_date).strftime("%Y-%m-%d") if start_date is not None else None
                end_date = pd.to_datetime(end_date).strftime("%Y-%m-%d") if end_date is not None else None

                ## Start date filter
                if start_date is not None:
                    spot = spot[spot.index >= start_date]
                    chain_spot = chain_spot[chain_spot.index >= start_date]
                    dividends = dividends[dividends.index >= start_date]
                    dividend_yield = dividend_yield[dividend_yield.index >= start_date]
                    split_factor = split_factor[split_factor.index >= start_date]

                ## End date filter
                if end_date is not None:
                    spot = spot[spot.index <= end_date]
                    chain_spot = chain_spot[chain_spot.index <= end_date]
                    dividends = dividends[dividends.index <= end_date]
                    dividend_yield = dividend_yield[dividend_yield.index <= end_date]
                    split_factor = split_factor[split_factor.index <= end_date]

            ## Construct TimeseriesData with all data
            ts = TimeseriesData(
                spot=spot,
                chain_spot=chain_spot,
                dividends=dividends,
                dividend_yield=dividend_yield,
                split_factor=split_factor,
                rates=self.rates["annualized"],
            )

        return ts

    def load_timeseries(self, sym: str, start_date: str = None, end_date: str = None, *args, **kwargs) -> None:
        """Preload all market data timeseries for a symbol into cache.

        Eagerly loads spot, chain_spot, dividends, and split_factor data into their
        respective caches. Useful for warming cache before intensive operations or
        reducing latency for first access. Uses instance date range if not specified.

        Args:
            sym: Stock ticker symbol (e.g., "AAPL", "MSFT").
            start_date: Optional start date string (YYYY-MM-DD). Defaults to self._start.
            end_date: Optional end date string (YYYY-MM-DD). Defaults to self._end.

        Examples:
            >>> mts = MarketTimeseries()
            >>> # Preload full date range for a symbol
            >>> mts.load_timeseries("AAPL")
            >>> # Now all subsequent access for AAPL will be instant

            >>> # Preload specific date range
            >>> mts.load_timeseries("MSFT", "2025-01-01", "2025-12-31")

            >>> # Batch preload multiple symbols
            >>> for sym in ["AAPL", "MSFT", "GOOGL"]:
            ...     mts.load_timeseries(sym)
        """
        sym = sym.upper()
        start_date = start_date or self._start
        end_date = end_date or self._end
        self._load_spot_into_cache(sym, start_date, end_date)
        self._load_chain_spot_into_cache(sym, start_date, end_date)
        self._load_dividends_into_cache(sym, start_date, end_date)
        self._load_split_factor_into_cache(sym, start_date, end_date)

    def __repr__(self) -> str:
        return f"MarketTimeseries(symbols: {list(self._spot.keys())}, intervals: {list(self._spot.keys())})"


def get_timeseries_obj(live: bool = False, *args, **kwargs) -> MarketTimeseries:
    """Get or create the singleton MarketTimeseries instance.

    Returns the global OPTIMESERIES instance, creating it if necessary. Implements
    singleton pattern to ensure only one MarketTimeseries exists per session, sharing
    caches across all callers for optimal performance.

    Args:
        live: If True, sets end date to today. If False, sets to last business day.

    Returns:
        Global MarketTimeseries singleton instance.

    Examples:
        >>> # Get singleton instance for backtesting (end = yesterday)
        >>> mts = get_timeseries_obj(live=False)
        >>> data = mts.get_timeseries("AAPL")

        >>> # Get singleton for live trading (end = today)
        >>> mts_live = get_timeseries_obj(live=True)
        >>> # Same instance if called again
        >>> assert get_timeseries_obj() is mts_live
    """
    global OPTIMESERIES
    if OPTIMESERIES is None:
        OPTIMESERIES = MarketTimeseries(
            _end=(datetime.now() - BDay(1)).strftime("%Y-%m-%d") if not live else datetime.now().strftime("%Y-%m-%d")
        )

    return OPTIMESERIES


def reset_timeseries_obj(*args, **kwargs) -> None:
    """Reset the singleton MarketTimeseries instance to None.

    Clears the global OPTIMESERIES variable, forcing the next call to
    get_timeseries_obj() to create a fresh instance. Useful for testing or
    when switching between live and backtest modes. Does not clear caches.

    Examples:
        >>> mts = get_timeseries_obj(live=False)
        >>> # ... use mts ...
        >>> reset_timeseries_obj()  # Clear singleton
        >>> mts_live = get_timeseries_obj(live=True)  # New instance
        >>> assert mts is not mts_live
    """
    global OPTIMESERIES
    OPTIMESERIES = None


if __name__ == "__main__":
    mts = get_timeseries_obj()
    mts.load_timeseries("BA", force=True)
    ts = mts.get_timeseries("BA")
    print(ts)
    print(SIGNALS_TO_RUN)
