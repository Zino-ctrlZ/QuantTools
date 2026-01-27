"""Risk-free rate data management for options pricing with caching.

This module provides the RatesDataManager class for retrieving and caching
risk-free interest rates from US Treasury bills (^IRX - 13 Week Treasury Bill).
Implements singleton pattern with intelligent partial caching.

Typical usage:
    >>> rates_mgr = RatesDataManager()
    >>> result = rates_mgr.get_risk_free_rate_timeseries(
    ...     start_date="2025-01-01",
    ...     end_date="2025-01-31"
    ... )
    >>> rates = result.daily_risk_free_rates
"""

from datetime import datetime
from typing import ClassVar, Optional, Union
import pandas as pd
import yfinance as yf
import numpy as np
from trade.helpers.Logging import setup_logger
from trade.helpers.helper import get_missing_dates, change_to_last_busday
from .utils.cache import _data_structure_cache_it
from .utils.date import is_available_on_date, to_datetime
from .utils.data_structure import _data_structure_sanitize
from .config import OptionDataConfig
from ._enums import ArtifactType, Interval, SeriesId, RealTimeFallbackOption
from .result import RatesResult
from .base import BaseDataManager, CacheSpec

logger = setup_logger("trade.datamanager.rates", stream_log_level="INFO")


def deannualize(annual_rate: float, periods: int = 365) -> float:
    """Converts annualized interest rate to per-period rate.

    Uses compound interest formula to convert annual rate to daily rate
    or other period-based rate.

    Args:
        annual_rate: Annualized interest rate (e.g., 0.05 for 5%).
        periods: Number of periods per year. Defaults to 365 for daily rate.

    Returns:
        Per-period interest rate (e.g., daily rate if periods=365).

    Examples:
        >>> # Convert 5% annual to daily rate
        >>> daily_rate = deannualize(0.05, periods=365)
        >>> print(f"{daily_rate:.6f}")
        0.000134

        >>> # Convert 5% annual to weekly rate
        >>> weekly_rate = deannualize(0.05, periods=52)
        >>> print(f"{weekly_rate:.6f}")
        0.000942
    """
    return (1 + annual_rate) ** (1 / periods) - 1


class RatesDataManager(BaseDataManager):
    """Singleton manager for risk-free rate data from treasury bills (^IRX).

    Manages retrieval and caching of US Treasury Bill rates (13-week T-Bill) used as
    risk-free rates in options pricing. Implements singleton pattern to ensure single
    instance across application. Supports partial caching with automatic cache merging.

    Attributes:
        CACHE_NAME: Class-level cache identifier for this manager type.
        DEFAULT_SERIES_ID: Default historical series identifier.
        INSTANCE: Singleton instance reference.
        DEFAULT_YFINANCE_TICKER: Yahoo Finance ticker for 13-week T-Bill (^IRX).
        CONFIG: Configuration object for rates data settings.

    Examples:
        >>> # Singleton access - same instance returned
        >>> rates_mgr1 = RatesDataManager()
        >>> rates_mgr2 = RatesDataManager()
        >>> assert rates_mgr1 is rates_mgr2

        >>> # Get rate for a single date
        >>> result = rates_mgr1.get_rate(date="2025-01-15")
        >>> rate = result.daily_risk_free_rates.iloc[0]

        >>> # Get time-series of rates
        >>> result = rates_mgr1.get_risk_free_rate_timeseries(
        ...     start_date="2025-01-01",
        ...     end_date="2025-01-31"
        ... )
        >>> rates = result.daily_risk_free_rates
    """

    CACHE_NAME: ClassVar[str] = "rates_data_manager"
    DEFAULT_SERIES_ID: ClassVar["SeriesId"] = SeriesId.HIST
    INSTANCE = None
    DEFAULT_YFINANCE_TICKER = "^IRX"  # 13 WEEK TREASURY BILL
    CONFIG: OptionDataConfig = OptionDataConfig()

    def __new__(
        cls,
        *,
        cache_spec: Optional[CacheSpec] = None,
        enable_namespacing: bool = False,
    ) -> "RatesDataManager":
        """Ensures only one instance exists (singleton pattern).

        Returns the existing singleton instance if available, otherwise creates
        a new one. Ensures risk-free rate data is managed globally.

        Args:
            cache_spec: Optional cache configuration. Uses default if None.
            enable_namespacing: If True, enables namespace isolation in cache keys.

        Returns:
            Singleton RatesDataManager instance.

        Examples:
            >>> mgr1 = RatesDataManager()
            >>> mgr2 = RatesDataManager()
            >>> assert mgr1 is mgr2  # Same instance
        """

        if cls.INSTANCE is not None:
            return cls.INSTANCE
        instance = object.__new__(cls)
        cls.INSTANCE = instance
        return instance

    def __init__(self, *, cache_spec: Optional[CacheSpec] = None, enable_namespacing: bool = False) -> None:
        """Initializes singleton instance once, skipping subsequent calls.

        Sets up persistent cache for rates data. Only executes initialization logic
        on first instantiation due to singleton pattern.

        Args:
            cache_spec: Optional cache configuration. Uses default if None.
            enable_namespacing: If True, enables namespace isolation in cache keys.

        Examples:
            >>> mgr = RatesDataManager()
            >>> mgr = RatesDataManager(cache_spec=CacheSpec(expire_days=30))
        """
        if getattr(self, "_init_called", False):
            return
        self._init_called = True
        super().__init__(cache_spec=cache_spec, enable_namespacing=enable_namespacing)

    def get_rate(
        self,
        date: Union[datetime, str],
        interval: Interval = Interval.EOD,
        str_interval: Optional[str] = None,
        fallback_option: Optional[RealTimeFallbackOption] = None,
    ) -> RatesResult:
        """Returns risk-free rate for a single date.

        Fetches the risk-free interest rate (from 13-week T-Bill) for a specific date.
        Returns empty result if date is not a business day or is a US holiday.

        Args:
            date: Target date for rate lookup (YYYY-MM-DD string or datetime).
            interval: Time interval resolution. Defaults to Interval.EOD (end-of-day).
            str_interval: Optional yfinance interval string (e.g., "1d", "30m").
                Overrides interval parameter if provided.

        Returns:
            RatesResult containing daily_risk_free_rates Series with single value,
            or empty Series if date is invalid.

        Examples:
            >>> rates_mgr = RatesDataManager()
            >>> result = rates_mgr.get_rate(date="2025-01-15")
            >>> if not result.daily_risk_free_rates.empty:
            ...     rate = result.daily_risk_free_rates.iloc[0]
            ...     print(f"Rate: {rate:.4f}")
            Rate: 0.0485

            >>> # Weekend date returns empty result
            >>> result = rates_mgr.get_rate(date="2025-01-18")  # Saturday
            >>> print(result.daily_risk_free_rates.empty)
            True

        Notes:
            - Validates date is a business day (not weekend or US holiday)
            - Uses internal timeseries method with single-date range
            - Returns annualized rate (e.g., 0.0485 = 4.85%)
        """
        fallback_option = fallback_option or self.CONFIG.real_time_fallback_option

        if not is_available_on_date(to_datetime(date).date()):
            logger.warning(
                f"Requested date {date} is not a business day or is a US holiday. Resorting to fallback option `{fallback_option}`."
            )
            if fallback_option == RealTimeFallbackOption.RAISE_ERROR:
                raise ValueError(f"Date {date} is not available for risk-free rate data.")
            
            if fallback_option == RealTimeFallbackOption.USE_LAST_AVAILABLE:
                date = change_to_last_busday(date)
            else:
                value = pd.Series(dtype=float,
                                  index = [pd.to_datetime(date)],
                                  value = [np.nan if fallback_option == RealTimeFallbackOption.NAN else 0.0])
                
                return RatesResult(timeseries=value, symbol=self.DEFAULT_YFINANCE_TICKER)
        date_str = pd.to_datetime(date).strftime("%Y-%m-%d") if isinstance(date, datetime) else date

        rates_data = self.get_risk_free_rate_timeseries(
            start_date=date_str,
            end_date=date_str,
            interval=interval,
            str_interval=str_interval,
        )
        rate = rates_data.timeseries
        if rate is not None and not rate.empty:
            rate = rate.iloc[0:1]

        return RatesResult(timeseries=rate, symbol=self.DEFAULT_YFINANCE_TICKER)

    def get_risk_free_rate_timeseries(
        self,
        start_date: Union[datetime, str],
        end_date: Union[datetime, str],
        interval: Interval = Interval.EOD,
        str_interval: Optional[str] = None,
    ) -> RatesResult:
        """Returns risk-free rate time-series with partial cache support.

        Fetches daily risk-free interest rates from 13-week Treasury Bills for the
        specified date range. Intelligently uses cache when available and only fetches
        missing dates from yfinance.

        Args:
            start_date: Start of date range (YYYY-MM-DD string or datetime).
            end_date: End of date range (YYYY-MM-DD string or datetime).
            interval: Time interval resolution. Defaults to Interval.EOD (end-of-day).
            str_interval: Optional yfinance interval string (e.g., "1d", "30m").
                Overrides interval parameter if provided.

        Returns:
            RatesResult containing daily_risk_free_rates Series indexed by datetime
            with annualized rates.

        Examples:
            >>> rates_mgr = RatesDataManager()
            >>> result = rates_mgr.get_risk_free_rate_timeseries(
            ...     start_date="2025-01-01",
            ...     end_date="2025-01-31"
            ... )
            >>> rates = result.daily_risk_free_rates
            >>> print(rates.head())
            Datetime
            2025-01-02    0.0485
            2025-01-03    0.0487
            2025-01-06    0.0486
            ...

            >>> # High-frequency intraday rates
            >>> result = rates_mgr.get_risk_free_rate_timeseries(
            ...     start_date="2025-01-15",
            ...     end_date="2025-01-15",
            ...     str_interval="30m"
            ... )

        Notes:
            - Partial cache hits fetch only missing dates and merge with cache
            - Full cache hits return immediately without external calls
            - Automatically filters to business days (excludes weekends/holidays)
            - Returns annualized rates (e.g., 0.0485 = 4.85%)
            - Cache automatically merges and deduplicates by date
        """

        if str_interval is not None:
            # normalize common yfinance strings
            intraday_tokens = ["m", "h"]
            if any(t in str_interval for t in intraday_tokens):
                raise ValueError(
                    "RatesDataManager supports daily-or-higher intervals only. " f"Received str_interval={str_interval}."
                )

        if interval != Interval.EOD:
            raise ValueError("RatesDataManager is EOD-only.")


        start_str = pd.to_datetime(start_date).strftime("%Y-%m-%d") if isinstance(start_date, datetime) else start_date
        end_str = pd.to_datetime(end_date).strftime("%Y-%m-%d") if isinstance(end_date, datetime) else end_date
        
        ## Determine yfinance interval
        if not str_interval:
            fn_interval = "1d" if interval == Interval.EOD else "30m"
        else:
            fn_interval = str_interval

        ## Make cache key
        key = self.make_key(
            symbol=self.DEFAULT_YFINANCE_TICKER,
            artifact_type=ArtifactType.RATES,
            series_id=SeriesId.HIST,
            interval=interval,
            fn_interval=fn_interval,
        )



        ## Check cache
        series = self.get(key, default=None)

        ## Check if cache covers requested date range
        if series is not None:
            logger.info(f"Cache hit for risk-free rate timeseries key: {key}")
            missing = get_missing_dates(
                series,
                pd.to_datetime(start_date).strftime("%Y-%m-%d"),
                pd.to_datetime(end_date).strftime("%Y-%m-%d"),
            )

            ## If no missing dates, return cached series
            if not missing:
                logger.info(f"Cache fully covers requested date range for risk-free rate timeseries. Key: {key}")
                series = _data_structure_sanitize(
                    series,
                    start=start_str,
                    end=end_str,
                )
                return RatesResult(timeseries=series, symbol=self.DEFAULT_YFINANCE_TICKER)
            else:
                ## Fetch missing dates
                start_date = min(missing)
                end_date = max(missing)
                logger.info(
                    f"Cache partially covers requested date range for risk-free rate timeseries. Key: {key}. Fetching missing dates: {missing}"
                )
        else:
            logger.info(f"No cache found for risk-free rate timeseries key: {key}. Fetching from source.")

        # Fetch rates data
        rates_data = self._query_yfinance(
            start_date=start_date,
            end_date=end_date,
            interval=fn_interval,
        )["annualized"]

        if series is not None:
            # Merge with existing cached series
            merged = pd.concat([series, rates_data])
            rates_data = merged[~merged.index.duplicated(keep="last")]

        ## Cache the updated series
        self.cache_it(key, rates_data)

        ## Sanitize before returning
        rates_data = _data_structure_sanitize(
            rates_data,
            start=start_str,  # Ensure only requested range
            end=end_str,
        )

        return RatesResult(symbol=self.DEFAULT_YFINANCE_TICKER, timeseries=rates_data)

    def cache_it(self, key: str, value: pd.Series, *, expire: Optional[int] = None) -> None:
        """Merges and caches rate time-series, excluding today's partial data.

        Appends new rate data to existing cached time-series if cache entry exists.
        Filters out today's data to avoid caching incomplete/changing values.

        Args:
            key: Cache key identifier.
            value: Series of rates to cache (indexed by datetime).
            expire: Optional expiration time in seconds. Uses cache default if None.

        Examples:
            >>> rates_mgr = RatesDataManager()
            >>> rates = pd.Series([0.048, 0.049], index=pd.date_range("2025-01-01", periods=2))
            >>> rates_mgr.cache_it("my_key", rates, expire=86400)

        Notes:
            - Existing cache entries are merged with new data
            - Duplicates are removed, keeping latest values
            - Today's data excluded to avoid caching incomplete values
        """
        ## Since it is a timeseries, we will append to existing if exists
        _data_structure_cache_it(self, key, value, expire=expire)

    def _query_yfinance(
        self,
        start_date: Union[datetime, str],
        end_date: Union[datetime, str],
        interval: str,
    ) -> pd.DataFrame:
        """Fetches ^IRX treasury bill rates from yfinance and formats output.

        Downloads 13-week Treasury Bill data from Yahoo Finance, processes it,
        and returns formatted DataFrame with annualized and daily rates. Adds
        5-day buffer to date range to ensure complete data retrieval.

        Args:
            start_date: Start of date range (YYYY-MM-DD string or datetime).
            end_date: End of date range (YYYY-MM-DD string or datetime).
            interval: yfinance interval string (e.g., "1d" for daily, "30m" for 30-minute).

        Returns:
            DataFrame indexed by Datetime with columns:
                - name: Ticker symbol (^IRX)
                - description: "13 WEEK TREASURY BILL"
                - daily: Daily rate (deannualized from annual rate)
                - annualized: Annualized rate (as decimal, e.g., 0.0485 = 4.85%)

        Examples:
            >>> rates_mgr = RatesDataManager()
            >>> df = rates_mgr._query_yfinance(
            ...     start_date="2025-01-01",
            ...     end_date="2025-01-31",
            ...     interval="1d"
            ... )
            >>> print(df.head())
                        name                 description     daily  annualized
            Datetime
            2025-01-02  ^IRX  13 WEEK TREASURY BILL  0.000129      0.0485
            2025-01-03  ^IRX  13 WEEK TREASURY BILL  0.000130      0.0487
            ...

        Notes:
            - Uses 5-day buffer before/after date range for data completeness
            - Converts yfinance percentage (4.85) to decimal (0.0485)
            - Calculates daily rate using compound interest formula
            - Filters final output to exact date range requested
        """

        ## Date buffer to ensure we get all data
        start_date = to_datetime(start_date) - pd.Timedelta(days=5)
        end_date = to_datetime(end_date) + pd.Timedelta(days=5)

        data_min = yf.download(
            "^IRX",
            start=start_date,
            end=end_date,
            interval=interval,
            progress=False,
            multi_level_index=False,
        )

        data_min.columns = data_min.columns.str.lower()
        data_min["annualized"] = data_min["close"] / 100
        data_min["daily"] = (data_min["annualized"]).apply(deannualize)
        data_min["name"] = "^IRX"
        data_min["description"] = "13 WEEK TREASURY BILL"
        data_min.index.name = "Datetime"
        data_min = data_min[["name", "description", "daily", "annualized"]]
        data_min = data_min[
            (data_min.index >= pd.to_datetime(start_date)) & (data_min.index <= pd.to_datetime(end_date))
        ]
        return data_min
    
    def rt(self, fallback_option: Optional[RealTimeFallbackOption] = None) -> RatesResult:
        """Shortcut for get_rate method.

        Provides a concise alias for retrieving risk-free rate at the current date.

        Returns:
            RatesResult containing daily_risk_free_rates Series with single value
            for today's date.
        Examples:
            >>> rates_mgr = RatesDataManager()
            >>> result = rates_mgr.rt()
            >>> rate = result.daily_risk_free_rates.iloc[0]
            >>> print(f"Today's Rate: {rate:.4f}")  
            Today's Rate: 0.0485
        """
        return self.get_rate(date=datetime.now(), fallback_option=fallback_option)
