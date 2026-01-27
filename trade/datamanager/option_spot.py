"""Option spot price data management with Thetadata API integration.

This module provides the OptionSpotDataManager class for retrieving and caching
option contract spot prices from Thetadata API. Supports both EOD (end-of-day)
and Quote endpoints with intelligent partial caching.

Typical usage:
    >>> opt_spot_mgr = OptionSpotDataManager("AAPL")
    >>> result = opt_spot_mgr.get_option_spot_timeseries(
    ...     start_date="2025-01-01",
    ...     end_date="2025-01-31",
    ...     strike=150.0,
    ...     expiration="2025-06-20",
    ...     right="C"
    ... )
    >>> prices = result.daily_option_spot
"""

from datetime import datetime
from typing import Optional, Tuple, Union
import pandas as pd
from trade.helpers.Logging import setup_logger
from trade.datamanager.base import BaseDataManager, CacheSpec
from trade.datamanager.result import OptionSpotResult
from trade.datamanager._enums import ArtifactType, Interval, SeriesId, OptionSpotEndpointSource
from trade.datamanager.utils.data_structure import _data_structure_sanitize
from trade.datamanager.utils.cache import _data_structure_cache_it, _check_cache_for_timeseries_data_structure
from trade.datamanager.config import OptionDataConfig
from trade.datamanager.utils.date import DateRangePacket, DATE_HINT, _sync_date
from dbase.DataAPI.ThetaData import retrieve_eod_ohlc, quote_to_eod_patch, retrieve_quote_rt
from dbase.utils import default_timestamp
from dbase.DataAPI.ThetaData.utils import _handle_opttick_param

logger = setup_logger("trade.datamanager.option_spot", stream_log_level="INFO")

class OptionSpotDataManager(BaseDataManager):
    """Manages option spot price retrieval for a specific symbol from Thetadata API.

    Retrieves historical and real-time option contract prices (OHLC data) from
    Thetadata's EOD or Quote endpoints. Implements intelligent caching with partial
    cache support to minimize API calls.

    Attributes:
        CACHE_NAME: Class-level cache identifier for this manager type.
        DEFAULT_SERIES_ID: Default historical series identifier.
        CONFIG: Configuration object for option data settings.
        INSTANCES: Class-level cache of manager instances per symbol.
        symbol: The underlying equity ticker symbol.

    Examples:
        >>> # Get option spot price for single date
        >>> opt_mgr = OptionSpotDataManager("AAPL")
        >>> result = opt_mgr.get_option_spot(
        ...     date="2025-01-15",
        ...     strike=150.0,
        ...     expiration="2025-06-20",
        ...     right="C"
        ... )
        >>> price = result.daily_option_spot["close"].iloc[0]

        >>> # Get time-series of option prices
        >>> result = opt_mgr.get_option_spot_timeseries(
        ...     start_date="2025-01-01",
        ...     end_date="2025-01-31",
        ...     strike=150.0,
        ...     expiration="2025-06-20",
        ...     right="C",
        ...     endpoint_source=OptionSpotEndpointSource.EOD
        ... )
        >>> ohlc_data = result.daily_option_spot
    """

    CACHE_NAME: str = "option_spot_manager"
    DEFAULT_SERIES_ID: SeriesId = SeriesId.HIST
    CONFIG = OptionDataConfig()

    def __init__(
        self,
        symbol: str,
        *,
        cache_spec: Optional[CacheSpec] = None,
        enable_namespacing: bool = False,
    ) -> None:
        """Initializes manager for a specific symbol.

        Sets up the option spot data manager with persistent cache for API responses.

        Args:
            symbol: Underlying equity ticker symbol (e.g., "AAPL", "SPY").
            cache_spec: Optional cache configuration. Uses default if None.
            enable_namespacing: If True, enables namespace isolation in cache keys.

        Examples:
            >>> opt_mgr = OptionSpotDataManager("AAPL")
            >>> opt_mgr = OptionSpotDataManager("AAPL", cache_spec=CacheSpec(expire_days=7))
        """
        super().__init__(cache_spec=cache_spec, enable_namespacing=enable_namespacing)
        self.symbol = symbol

    def _sync_date(
        self,
        start_date: DATE_HINT,
        end_date: DATE_HINT,
        strike: Optional[float] = None,
        expiration: Optional[Union[datetime, str]] = None,
        right: Optional[str] = None,
        endpoint_source: Optional[OptionSpotEndpointSource] = OptionSpotEndpointSource.EOD
    ) -> Tuple[DATE_HINT, DATE_HINT]:
        """Synchronizes requested dates with available data range from Thetadata.

        Queries Thetadata for available dates for the specified option contract and
        adjusts the requested date range to fit within available data bounds.

        Args:
            start_date: Requested start date.
            end_date: Requested end date.
            strike: Option strike price.
            expiration: Option expiration date.
            right: Option type ("C" for call, "P" for put).

        Returns:
            Tuple of (adjusted_start_date, adjusted_end_date) constrained to
            available data range.

        Examples:
            >>> opt_mgr = OptionSpotDataManager("AAPL")
            >>> start, end = opt_mgr._sync_date(
            ...     start_date="2025-01-01",
            ...     end_date="2025-12-31",
            ...     strike=150.0,
            ...     expiration="2025-06-20",
            ...     right="C"
            ... )

        Notes:
            - Constrains start_date to max(requested_start, min_available_date)
            - Constrains end_date to min(requested_end, max_available_date)
            - Prevents requesting dates outside available data range
        """
        return _sync_date(
            symbol=self.symbol,
            start_date=start_date,
            end_date=end_date,
            strike=strike,
            expiration=expiration,
            right=right,
            endpoint_source=endpoint_source
        )
    
    def get_option_spot(
        self,
        date: Union[datetime, str],
        *,
        strike: Optional[float] = None,
        expiration: Optional[Union[datetime, str]] = None,
        right: Optional[str] = None,
        opttick: Optional[str] = None,
        endpoint_source: Optional[OptionSpotEndpointSource] = None,
    ) -> OptionSpotResult:
        """Fetches option spot price for a single date from Thetadata API.

        Retrieves OHLC data for a specific option contract on a single date.
        Wrapper around get_option_spot_timeseries with single-date range.

        Args:
            date: Target date (YYYY-MM-DD string or datetime).
            strike: Option strike price. Required unless opttick provided.
            expiration: Option expiration date. Required unless opttick provided.
            right: Option type ("C" for call, "P" for put). Required unless opttick provided.
            opttick: Optional ticker string (e.g., "AAPL250620C00150000"). If provided,
                overrides strike, expiration, and right parameters.
            endpoint_source: API endpoint to use (EOD or QUOTE). Uses config default if None.

        Returns:
            OptionSpotResult containing daily_option_spot DataFrame with OHLC data,
            plus metadata (key, endpoint_source).

        Examples:
            >>> opt_mgr = OptionSpotDataManager("AAPL")
            >>> # Using strike/expiration/right
            >>> result = opt_mgr.get_option_spot(
            ...     date="2025-01-15",
            ...     strike=150.0,
            ...     expiration="2025-06-20",
            ...     right="C"
            ... )
            >>> close_price = result.daily_option_spot["close"].iloc[0]

            >>> # Using opttick
            >>> result = opt_mgr.get_option_spot(
            ...     date="2025-01-15",
            ...     opttick="AAPL250620C00150000"
            ... )

        Notes:
            - Returns DataFrame with columns: open, high, low, close, volume
            - Uses EOD endpoint by default for historical data
            - Quote endpoint available for more recent data
        """
        date_str = pd.to_datetime(date).strftime("%Y-%m-%d") if isinstance(date, datetime) else date
        result = self.get_option_spot_timeseries(
            start_date=date_str,
            end_date=date_str,
            strike=strike,
            expiration=expiration,
            right=right,
            opttick=opttick,
            endpoint_source=endpoint_source,
        )
        return result

    def get_option_spot_timeseries(
        self,
        start_date: Union[datetime, str],
        end_date: Union[datetime, str],
        *,
        strike: Optional[float] = None,
        expiration: Optional[Union[datetime, str]] = None,
        right: Optional[str] = None,
        opttick: Optional[str] = None,
        endpoint_source: Optional[OptionSpotEndpointSource] = None,
    ) -> OptionSpotResult:
        """Fetches option spot price time-series from Thetadata API.

        Retrieves daily OHLC data for a specific option contract over a date range.
        Implements intelligent caching with partial cache support to minimize API calls.

        Args:
            start_date: Start of date range (YYYY-MM-DD string or datetime).
            end_date: End of date range (YYYY-MM-DD string or datetime).
            strike: Option strike price. Required unless opttick provided.
            expiration: Option expiration date. Required unless opttick provided.
            right: Option type ("C" for call, "P" for put). Required unless opttick provided.
            opttick: Optional ticker string (e.g., "AAPL250620C00150000"). If provided,
                overrides strike, expiration, and right parameters.
            endpoint_source: API endpoint to use (EOD or QUOTE). Uses config default if None.

        Returns:
            OptionSpotResult containing daily_option_spot DataFrame indexed by datetime
            with OHLC data, plus metadata (key, endpoint_source).

        Examples:
            >>> opt_mgr = OptionSpotDataManager("AAPL")
            >>> # Get historical option prices
            >>> result = opt_mgr.get_option_spot_timeseries(
            ...     start_date="2025-01-01",
            ...     end_date="2025-01-31",
            ...     strike=150.0,
            ...     expiration="2025-06-20",
            ...     right="C",
            ...     endpoint_source=OptionSpotEndpointSource.EOD
            ... )
            >>> ohlc = result.daily_option_spot
            >>> print(ohlc.head())
            datetime         open   high    low  close   volume
            2025-01-02     5.25   5.50   5.20   5.45   12500
            2025-01-03     5.50   5.75   5.45   5.70   15300
            ...

            >>> # Using opttick format
            >>> result = opt_mgr.get_option_spot_timeseries(
            ...     start_date="2025-01-01",
            ...     end_date="2025-01-31",
            ...     opttick="AAPL250620C00150000"
            ... )

        Notes:
            - Partial cache hits only fetch missing dates from API
            - Full cache hits return immediately without API calls
            - Automatically adjusts date range to available data bounds
            - EOD endpoint: Historical end-of-day data
            - QUOTE endpoint: Constructed from quote data (fallback for recent dates)
        """
        if endpoint_source is None:
            endpoint_source = self.CONFIG.option_spot_endpoint_source

        strike, right, symbol, expiration = _handle_opttick_param(
            strike=strike,
            right=right,
            symbol=self.symbol,
            exp=expiration,
            opttick=opttick,
            enforce_single_option=True,
        )

        ## Sync requested dates with available data range
        start_date, end_date = self._sync_date(
            start_date=start_date,
            end_date=end_date,
            strike=float(strike),
            expiration=expiration,
            right=right,
            endpoint_source=endpoint_source,
        )

        date_packet = DateRangePacket(start_date=start_date, end_date=end_date, maturity_date=expiration)
        start_date, end_date = date_packet.start_date, date_packet.end_date
        start_str, end_str = date_packet.start_str, date_packet.end_str
        expiration = date_packet.maturity_date

        # Construct cache key
        key = self.make_key(
            symbol=self.symbol,
            artifact_type=ArtifactType.OPTION_SPOT,
            series_id=SeriesId.HIST,
            endpoint_source=endpoint_source.value,
            interval=Interval.EOD,
            strike=strike,
            right=right,
            expiration=expiration,
        )

        # Check cache
        cached_data, is_partial, start_date, end_date = _check_cache_for_timeseries_data_structure(
            key=key,
            self=self,
            start_dt=start_date,
            end_dt=end_date,
        )

        if cached_data is not None and not is_partial:
            logger.info(f"Cache hit for option spot timeseries key: {key}")
            result = OptionSpotResult()
            result.daily_option_spot = cached_data
            result.key = key
            result.endpoint_source = endpoint_source
            return result
        elif is_partial:
            logger.info(
                f"Cache partially covers requested date range for option spot timeseries. Key: {key}. Fetching missing dates."
            )
        else:
            logger.info(f"No cache found for option spot timeseries key: {key}. Fetching from source.")

        # Fetch data from Thetadata API (placeholder logic)
        fetched_data = self._query_thetadata_api(
            start_date=start_date,
            end_date=end_date,
            endpoint_source=endpoint_source,
            strike=strike,
            expiration=expiration,
            right=right,
        )

        # Merge with cached data if partial
        if cached_data is not None and is_partial:
            merged = pd.concat([cached_data, fetched_data])
            fetched_data = merged[~merged.index.duplicated(keep="last")]

        fetched_data.index = default_timestamp(fetched_data.index)

        # Cache the fetched data
        _data_structure_cache_it(self, key, fetched_data)

        # Sanitize before returning
        fetched_data = _data_structure_sanitize(
            fetched_data,
            start=start_str,
            end=end_str,
        )

        result = OptionSpotResult()
        result.daily_option_spot = fetched_data
        result.key = key
        result.endpoint_source = endpoint_source
        return result

    def _query_thetadata_api(
        self,
        start_date: Union[datetime, str],
        end_date: Union[datetime, str],
        endpoint_source: OptionSpotEndpointSource,
        strike: Optional[float] = None,
        expiration: Optional[Union[datetime, str]] = None,
        right: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetches option spot data from Thetadata API using specified endpoint.

        Makes HTTP requests to Thetadata's EOD or Quote endpoints to retrieve
        option contract OHLC data.

        Args:
            start_date: Start of date range (YYYY-MM-DD string or datetime).
            end_date: End of date range (YYYY-MM-DD string or datetime).
            endpoint_source: API endpoint (OptionSpotEndpointSource.EOD or QUOTE).
            strike: Option strike price.
            expiration: Option expiration date.
            right: Option type ("C" for call, "P" for put).

        Returns:
            DataFrame indexed by datetime with columns:
                - open: Opening price
                - high: Highest price
                - low: Lowest price
                - close: Closing price
                - volume: Trading volume

        Examples:
            >>> opt_mgr = OptionSpotDataManager("AAPL")
            >>> df = opt_mgr._query_thetadata_api(
            ...     start_date="2025-01-01",
            ...     end_date="2025-01-31",
            ...     endpoint_source=OptionSpotEndpointSource.EOD,
            ...     strike=150.0,
            ...     expiration="2025-06-20",
            ...     right="C"
            ... )

        Notes:
            - EOD endpoint: Uses retrieve_eod_ohlc for historical data
            - QUOTE endpoint: Uses quote_to_eod_patch (constructs OHLC from quotes)
            - Quote endpoint useful when EOD data not yet available
        """
        # In a real implementation, this method would make HTTP requests to Thetadata's API.
        if endpoint_source == OptionSpotEndpointSource.EOD:
            return retrieve_eod_ohlc(
                symbol=self.symbol,
                start_date=start_date,
                end_date=end_date,
                strike=float(strike),
                exp=expiration,
                right=right,
            )

        else:
            logger.info(
                f"Fetching option spot data from Thetadata Quote endpoint for {self.symbol} from {start_date} to {end_date}."
            )
        return quote_to_eod_patch(
            symbol=self.symbol,
            start_date=start_date,
            end_date=end_date,
            strike=float(strike),
            exp=expiration,
            right=right,
            ohlc_format=True,
        )
    
    def rt(
        self,
        strike: float,
        right: str,
        expiration: Union[datetime, str],
    ) -> OptionSpotResult:
        """
        Fetches real-time option spot price from Thetadata Quote endpoint.

        Retrieves the most recent OHLC data for a specific option contract using
        Thetadata's Quote endpoint.

        Args:
            strike: Option strike price.
            right: Option type ("C" for call, "P" for put).
            expiration: Option expiration date.

        Returns:
            OptionSpotResult containing daily_option_spot DataFrame with OHLC data,
            plus metadata (key, endpoint_source).
    """
        rt = retrieve_quote_rt(
            symbol=self.symbol,
            exp=expiration,
            strike=strike,
            right=right,
        )
        rt.index = default_timestamp(rt.index)
        rt.columns = rt.columns.str.lower()
        result = OptionSpotResult()
        result.daily_option_spot = rt
        result.key = self.make_key(
            symbol=self.symbol,
            time = datetime.now().time(),
            date = datetime.now(),
            artifact_type=ArtifactType.OPTION_SPOT,
            series_id=SeriesId.AT_TIME,
            endpoint_source=OptionSpotEndpointSource.QUOTE,
        )
        result.symbol = self.symbol
        result.strike = strike
        result.right = right
        result.expiration = pd.to_datetime(expiration)
        result.endpoint_source = OptionSpotEndpointSource.QUOTE
        return result
    