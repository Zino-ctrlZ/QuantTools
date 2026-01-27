"""Spot price data management for options pricing with split adjustment support.

This module provides the SpotDataManager class for retrieving spot (or split-adjusted
chain_spot) prices for equity symbols. Implements singleton pattern per symbol to
avoid redundant timeseries loading.

Typical usage:
    >>> spot_mgr = SpotDataManager("AAPL")
    >>> result = spot_mgr.get_spot_timeseries(
    ...     start_date="2025-01-01",
    ...     end_date="2025-01-31",
    ...     undo_adjust=True
    ... )
    >>> prices = result.daily_spot
"""

from datetime import datetime
from typing import Any, ClassVar, Optional, Union
from EventDriven.riskmanager.market_data import AtIndexResult
from trade.datamanager.utils.date import is_available_on_date
from trade.helpers.Logging import setup_logger

from trade.datamanager.base import BaseDataManager, CacheSpec
from trade.datamanager.result import SpotResult
from trade.datamanager.vars import TS, load_name
from trade.helpers.helper import change_to_last_busday, to_datetime
from trade.datamanager._enums import RealTimeFallbackOption, SeriesId
from trade.datamanager.utils.data_structure import _data_structure_sanitize


logger = setup_logger("trade.datamanager.spot", stream_log_level="INFO")


class SpotDataManager(BaseDataManager):
    """Manages spot price retrieval for a specific symbol with split adjustment support.

    Provides access to spot prices (unadjusted) or chain_spot prices (split-adjusted)
    from the global MarketTimeseries cache. Implements singleton pattern per symbol
    to ensure efficient data access.

    Attributes:
        CACHE_NAME: Class-level cache identifier for this manager type.
        DEFAULT_SERIES_ID: Default historical series identifier.
        INSTANCES: Class-level cache of manager instances per symbol.
        symbol: The equity ticker symbol this manager handles.

    Examples:
        >>> # Singleton access - same instance returned for same symbol
        >>> spot_mgr1 = SpotDataManager("AAPL")
        >>> spot_mgr2 = SpotDataManager("AAPL")
        >>> assert spot_mgr1 is spot_mgr2

        >>> # Get split-adjusted prices (chain_spot)
        >>> result = spot_mgr1.get_spot_timeseries(
        ...     start_date="2025-01-01",
        ...     end_date="2025-01-31",
        ...     undo_adjust=True
        ... )
        >>> chain_spot = result.daily_spot

        >>> # Get unadjusted prices (spot)
        >>> result = spot_mgr1.get_spot_timeseries(
        ...     start_date="2025-01-01",
        ...     end_date="2025-01-31",
        ...     undo_adjust=False
        ... )
        >>> spot = result.daily_spot

        >>> # Get price at specific datetime
        >>> at_time_result = spot_mgr1.get_at_time("2025-01-15")
        >>> price = at_time_result.close
    """

    CACHE_NAME: ClassVar[str] = "spot_data_manager"
    DEFAULT_SERIES_ID: ClassVar["SeriesId"] = SeriesId.HIST
    INSTANCES = {}

    def __new__(cls, symbol: str, *args: Any, **kwargs: Any) -> "SpotDataManager":
        """Returns cached instance for symbol, creating new one if needed.

        Implements singleton pattern per symbol to ensure timeseries are loaded only once.
        Automatically loads market timeseries data on first instantiation.

        Args:
            symbol: Equity ticker symbol (e.g., "AAPL", "MSFT").
            *args: Additional positional arguments passed to __init__.
            **kwargs: Additional keyword arguments passed to __init__.

        Returns:
            Singleton SpotDataManager instance for the given symbol.

        Examples:
            >>> mgr1 = SpotDataManager("AAPL")
            >>> mgr2 = SpotDataManager("AAPL")
            >>> assert mgr1 is mgr2  # Same instance
        """
        if symbol not in cls.INSTANCES:
            instance = super(SpotDataManager, cls).__new__(cls)
            cls.INSTANCES[symbol] = instance
        return cls.INSTANCES[symbol]

    def __init__(
        self, symbol: str, *, cache_spec: Optional[CacheSpec] = None, enable_namespacing: bool = False
    ) -> None:
        """Initializes manager once per symbol instance.

        Sets up the data manager for the symbol. Only executes initialization logic
        on first instantiation due to singleton pattern.

        Args:
            symbol: Equity ticker symbol.
            cache_spec: Optional cache configuration. Uses default if None.
            enable_namespacing: If True, enables namespace isolation in cache keys.

        Examples:
            >>> mgr = SpotDataManager("AAPL")
            >>> mgr = SpotDataManager("AAPL", cache_spec=CacheSpec(expire_days=30))
        """
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        super().__init__(cache_spec=cache_spec, enable_namespacing=enable_namespacing)
        self.symbol = symbol

    def get_spot_timeseries(
        self,
        start_date: Union[datetime, str],
        end_date: Union[datetime, str],
        undo_adjust: bool = True,
    ) -> SpotResult:
        """Returns spot or chain_spot price series for date range from MarketTimeseries.

        Retrieves closing prices from the global MarketTimeseries cache. Returns either
        split-adjusted (chain_spot) or unadjusted (spot) prices based on undo_adjust flag.

        Args:
            start_date: Start of date range (YYYY-MM-DD string or datetime).
            end_date: End of date range (YYYY-MM-DD string or datetime).
            undo_adjust: If True, returns split-adjusted chain_spot prices.
                If False, returns unadjusted spot prices.

        Returns:
            SpotResult containing daily_spot Series indexed by datetime, plus metadata
            (undo_adjust flag and cache key).

        Examples:
            >>> spot_mgr = SpotDataManager("AAPL")
            >>> # Get split-adjusted prices (recommended for backtesting)
            >>> result = spot_mgr.get_spot_timeseries(
            ...     start_date="2025-01-01",
            ...     end_date="2025-01-31",
            ...     undo_adjust=True
            ... )
            >>> chain_spot = result.daily_spot
            >>> print(chain_spot.head())
            datetime
            2025-01-02    155.32
            2025-01-03    156.01
            ...

            >>> # Get unadjusted prices (for real-time pricing)
            >>> result = spot_mgr.get_spot_timeseries(
            ...     start_date="2025-01-01",
            ...     end_date="2025-01-31",
            ...     undo_adjust=False
            ... )
            >>> spot = result.daily_spot

        Notes:
            - chain_spot: Split-adjusted prices (use with undo_adjust=True dividends)
            - spot: Unadjusted prices (use with undo_adjust=False dividends)
            - Data loaded directly from global TS cache (no additional caching)
            - Automatically filters to business days (excludes weekends/holidays)
        """
        ## Load first
        load_name(self.symbol)
        
        timeseries = TS.get_timeseries(self.symbol, skip_preload_check=True, start_date=start_date, end_date=end_date)
        if undo_adjust:
            spot_series = timeseries.chain_spot["close"]
        else:
            spot_series = timeseries.spot["close"]

        spot_series = _data_structure_sanitize(
            spot_series,
            start=start_date,
            end=end_date,
        )
        result = SpotResult()
        key = None  # No caching key for now
        result.daily_spot = spot_series
        result.undo_adjust = undo_adjust
        result.key = key
        result.symbol = self.symbol

        return result

    def get_at_time(
        self,
        date: Union[datetime, str],
    ) -> AtIndexResult:
        """Returns spot data at a specific datetime from MarketTimeseries.

        Retrieves comprehensive market data (OHLCV + other fields) for a specific date
        or datetime. Useful for point-in-time lookups.

        Args:
            date: Target date or datetime (YYYY-MM-DD string or datetime object).

        Returns:
            AtIndexResult containing OHLCV data and other market fields at the
            specified datetime.

        Examples:
            >>> spot_mgr = SpotDataManager("AAPL")
            >>> result = spot_mgr.get_at_time("2025-01-15")
            >>> print(f"Close: ${result.close:.2f}")
            Close: $156.45
            >>> print(f"Volume: {result.volume:,.0f}")
            Volume: 45,123,000

            >>> # Using datetime object
            >>> from datetime import datetime
            >>> result = spot_mgr.get_at_time(datetime(2025, 1, 15))
            >>> print(f"Open: ${result.open:.2f}, High: ${result.high:.2f}")
            Open: $155.20, High: $157.80

        Notes:
            - Returns data as of market close for the specified date
            - Delegates to global TS.get_at_index method
            - Result includes open, high, low, close, volume, and other fields
        """
        ## Load first
        load_name(self.symbol)
        return TS.get_at_index(sym=self.symbol, index=date)
    
    def rt(
        self,
        fallback_option: Optional[RealTimeFallbackOption] = None,
    ) -> AtIndexResult:
        """Returns the most recent spot price for the symbol.

        Retrieves the latest available spot price from the MarketTimeseries cache.
        Useful for real-time pricing scenarios.

        Returns:
            Most recent spot price as a float.
        Examples:
            >>> spot_mgr = SpotDataManager("AAPL")
            >>> latest_price = spot_mgr.rt()
            >>> print(f"Latest AAPL Price: ${latest_price:.2f}")    
            Latest AAPL Price: $158.23
        """
        fallback_option = fallback_option or self.CONFIG.real_time_fallback_option
        date = datetime.now()
        if not is_available_on_date(to_datetime(date).date()):
            logger.warning(
                f"Requested date {date} is not a business day or is a US holiday. Resorting to fallback option `{fallback_option}`."
            )
            if fallback_option == RealTimeFallbackOption.RAISE_ERROR:
                raise ValueError(f"Date {date} is not available for risk-free rate data.")
            
            if fallback_option == RealTimeFallbackOption.USE_LAST_AVAILABLE:
                date = change_to_last_busday(date)
            else:
                raise ValueError(f"Unsupported fallback option: {fallback_option}")
            
        at_index_result = TS.get_at_index(sym=self.symbol, index=date)
        return at_index_result
