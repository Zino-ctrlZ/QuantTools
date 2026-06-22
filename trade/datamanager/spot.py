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
from trade.helpers.Logging import setup_logger
from trade.datamanager.base import BaseDataManager, CacheSpec
from trade.datamanager.result import SpotResult
from trade.datamanager.vars import get_times_series, load_name
from trade.datamanager._enums import ArtifactType, CertificationLevel, Interval, RealTimeFallbackOption, SeriesId
from trade.datamanager.utils.logging import get_logging_level
from trade.datamanager.utils.na_logging import log_na_after_retrieval
from trade.datamanager.utils.date import _sync_equity_date
from trade.datamanager.utils.point_in_time import resolve_value_at_date


logger = setup_logger("trade.datamanager.spot", stream_log_level=get_logging_level())
TS = get_times_series()  # Load market timeseries data on module import to avoid circular imports
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
    CACHE_SPEC: CacheSpec = CacheSpec(cache_fname=CACHE_NAME)

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
        self, symbol: str, *, enable_namespacing: bool = False
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
        super().__init__(enable_namespacing=enable_namespacing, symbol=symbol)
        self.symbol = symbol

    def get_spot_timeseries(
        self,
        start_date: Union[datetime, str],
        end_date: Union[datetime, str],
        undo_adjust: bool = True,
        certification_level: Optional[CertificationLevel] = None,
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
            - MarketTimeseries getters certify before return; this manager does not re-certify
            - Automatically filters to business days (excludes weekends/holidays)
        """
        ## Load first
        load_name(self.symbol)
        start_date, end_date = _sync_equity_date(start_date, end_date, symbol=self.symbol)

        if undo_adjust:
            spot_frame = TS._get_chain_spot_timeseries(
                sym=self.symbol, start=start_date, end=end_date, certification_level=certification_level
            )
        else:
            spot_frame = TS._get_spot_timeseries(
                sym=self.symbol, start=start_date, end=end_date, certification_level=certification_level
            )
        spot_series = spot_frame["close"]

        key = self.make_key(
            symbol=self.symbol,
            artifact_type=ArtifactType.SPOT,
            series_id=SeriesId.HIST,
            interval=Interval.EOD,
            undo_adjust=undo_adjust,
            price_source="chain_spot" if undo_adjust else "spot",
        )
        result = SpotResult()
        result.daily_spot = spot_series
        result.undo_adjust = undo_adjust
        result.key = key
        result.symbol = self.symbol
        ## MT ``_get_*_timeseries`` already ran sanitize → NA log → certify on the frame.
        result.is_certified = True
        return result

    @log_na_after_retrieval("spot")
    def get_at_time(
        self,
        date: Union[datetime, str],
        undo_adjust: bool = True,
        fallback_option: Optional[RealTimeFallbackOption] = None,
    ) -> SpotResult:
        """Returns spot close at a specific date from MarketTimeseries.

        Fetches a 10-business-day lookback window certified at L1, then resolves
        the close price per ``fallback_option``.

        Args:
            date: Target date or datetime (YYYY-MM-DD string or datetime object).
            undo_adjust: If True, uses split-adjusted chain_spot; else raw spot.
            fallback_option: Policy when exact date is missing or non-trading.

        Returns:
            SpotResult with a single-row ``timeseries`` (close only).
        """
        fallback_option = fallback_option or self.CONFIG.real_time_fallback_option
        load_name(self.symbol)

        def _fetch(start: str, end: str) -> SpotResult:
            return self.get_spot_timeseries(
                start_date=start,
                end_date=end,
                undo_adjust=undo_adjust,
                certification_level=CertificationLevel.L1,
            )

        row, meta = resolve_value_at_date(
            date,
            fetch_timeseries=_fetch,
            extract_timeseries=lambda r: r.daily_spot,
            fallback_option=fallback_option,
        )

        container = SpotResult()
        container.symbol = self.symbol
        container.undo_adjust = undo_adjust
        container.fallback_option = meta.fallback_option
        if meta.source_result is not None:
            container.key = meta.source_result.key
            container.is_certified = meta.source_result.is_certified
        container.timeseries = row
        return container
    
    @log_na_after_retrieval("spot")
    def rt(
        self,
        fallback_option: Optional[RealTimeFallbackOption] = None,
        undo_adjust: bool = True,
    ) -> SpotResult:
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

        date = datetime.now()
        at_index_result = self.get_at_time(date=date, undo_adjust=undo_adjust, fallback_option=fallback_option)
        at_index_result.rt = True
        return at_index_result
