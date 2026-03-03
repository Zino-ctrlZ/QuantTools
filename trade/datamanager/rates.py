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
from trade.helpers.helper import change_to_last_busday
from curl_cffi.requests.exceptions import SSLError
import backoff
from .utils.cache import _data_structure_cache_it, _check_cache_for_timeseries_data_structure
from .utils.date import is_available_on_date, to_datetime
from .utils.data_structure import _data_structure_sanitize
from .config import OptionDataConfig
from ._enums import ArtifactType, Interval, SeriesId, RealTimeFallbackOption
from .result import RatesResult
from .base import BaseDataManager, CacheSpec
from .utils.logging import get_logging_level

logger = setup_logger("trade.datamanager.rates", stream_log_level=get_logging_level())


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
    INSTANCE: ClassVar[Optional["RatesDataManager"]] = None
    DEFAULT_YFINANCE_TICKER :str = "^IRX"  # 13 WEEK TREASURY BILL
    CONFIG: OptionDataConfig = OptionDataConfig()
    CACHE_SPEC: CacheSpec = CacheSpec(cache_fname=CACHE_NAME)

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

    def __init__(self, *, enable_namespacing: bool = False) -> None:
        """Initializes singleton instance once, skipping subsequent calls.

        Sets up persistent cache for rates data. Only executes initialization logic
        on first instantiation due to singleton pattern.

        Args:
            enable_namespacing: If True, enables namespace isolation in cache keys.

        Examples:
            >>> mgr = RatesDataManager()
            >>> mgr = RatesDataManager(cache_spec=CacheSpec(expire_days=30))
        """
        if getattr(self, "_init_called", False):
            return
        self._init_called = True
        super().__init__(enable_namespacing=enable_namespacing)

    @property
    def symbol(self) -> str:
        """Returns the symbol associated with this RatesDataManager."""
        return self.DEFAULT_YFINANCE_TICKER
    
    @symbol.setter
    def symbol(self, value: str) -> None:
        """Sets the symbol associated with this RatesDataManager."""
        pass

    def get_rate(
        self,
        date: Union[datetime, str],
        interval: Interval = Interval.EOD,
        str_interval: Optional[str] = None,
        fallback_option: Optional[RealTimeFallbackOption] = None,
        *,
        force_fail_n: int = 0,
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
        ## To avoid infinite recursion in fallback
        if force_fail_n > 7:
            raise ValueError("Exceeded maximum recursion attempts in get_rate fallback handling.")
        force_fail_n += 1 
        fallback_option = fallback_option or self.CONFIG.real_time_fallback_option
        date = to_datetime(date)
        
        ## Resolving date availability
        if not is_available_on_date(date.date()):
            logger.warning(
                f"Requested date {date} is not a business day or is a US holiday. Resorting to fallback option `{fallback_option}`."
            )
            if fallback_option == RealTimeFallbackOption.RAISE_ERROR:
                raise ValueError(f"Date {date} is not available for risk-free rate data.")
            
            if fallback_option == RealTimeFallbackOption.USE_LAST_AVAILABLE:
                ## Move date back to last business day
                ## Using only change_to_last_busday assumes input date is not business day or is holiday
                ## Which the function would roll back
                ## But there's a possibility input date is today's date but before market open
                ## In that case we need to move back one more business day
                date = change_to_last_busday(date - pd.tseries.offsets.BDay(1), time_of_day_aware=False)
                return self.get_rate(
                    date=date,
                    interval=interval,
                    str_interval=str_interval,
                    fallback_option=RealTimeFallbackOption.USE_LAST_AVAILABLE,
                    force_fail_n=force_fail_n,
                )
            else:
                value = pd.Series(dtype=float,
                                  index = [pd.to_datetime(date)],
                                  data = [np.nan if fallback_option == RealTimeFallbackOption.NAN else 0.0])
                
                res = RatesResult(timeseries=value, symbol=self.DEFAULT_YFINANCE_TICKER)
                res.fallback_option = fallback_option
                return res
        date_str = pd.to_datetime(date).strftime("%Y-%m-%d") if isinstance(date, datetime) else date

        rates_data = self.get_risk_free_rate_timeseries(
            start_date=date_str,
            end_date=date_str,
            interval=interval,
            str_interval=str_interval,
        )

        ## Date is available, but resolving no data found
        rate = rates_data.timeseries
        if rate is not None and not rate.empty:
            rate = rate.iloc[0:1]
        else:
            logger.warning(
                f"No rate data found for date {date}. Resorting to fallback option `{fallback_option}`."
            )
            if fallback_option == RealTimeFallbackOption.RAISE_ERROR:
                raise ValueError(f"No rate data found for date {date}.")
            elif fallback_option == RealTimeFallbackOption.USE_LAST_AVAILABLE:
                ## Move date back to last business day
                ## We are retaining the original date for logging and index purposes.
                ## Because the returned rate should still be indexed by the requested date, which would be used in further calculations.
                original_date = to_datetime(date).date()
                date = change_to_last_busday(date - pd.tseries.offsets.BDay(1), time_of_day_aware=False)
                res = self.get_rate(
                    date=date,
                    interval=interval,
                    str_interval=str_interval,
                    fallback_option=RealTimeFallbackOption.USE_LAST_AVAILABLE,
                    force_fail_n=force_fail_n,
                )
                rate = res.timeseries
                rate = rate.iloc[0:1]
                logger.warning(
                    f"Using last available rate for date {date} for {original_date} as fallback."
                )
                rate.index = [pd.to_datetime(original_date)]
            else:
                rate = pd.Series(dtype=float,
                                 index = [to_datetime(date)],
                                 data = [np.nan if fallback_option == RealTimeFallbackOption.NAN else 0.0])

        res = RatesResult(timeseries=rate, symbol=self.DEFAULT_YFINANCE_TICKER)
        res.fallback_option = fallback_option
        return res

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
        series, is_partial, start_date, end_date = _check_cache_for_timeseries_data_structure(
            self=self, key=key, start_dt=start_str, end_dt=end_str
        )

            ## If no missing dates, return cached series
        if series is not None and not is_partial:
            logger.info(f"Cache fully covers requested date range for risk-free rate timeseries. Key: {key}")
            series = _data_structure_sanitize(
                series,
                start=start_str,
                end=end_str,
                source_name=f"cached risk-free rate timeseries for {self.DEFAULT_YFINANCE_TICKER}",
            )
            return RatesResult(timeseries=series, symbol=self.DEFAULT_YFINANCE_TICKER)
        else:
            ## Fetch overriding date range for missing data
            start_date = start_str
            end_date = end_str
            logger.info(
                f"Cache partially covers requested date range for risk-free rate timeseries. Key: {key}"
            )
            
        # Fetch rates data
        rates_data = self._query_yfinance(
            start_date=start_date,
            end_date=end_date,
            interval=fn_interval,
        )["annualized"]
        rates_data = rates_data[(rates_data.index >= pd.to_datetime(start_str)) & (rates_data.index <= pd.to_datetime(end_str))]
        
        # Merge with existing cached series
        if series is not None:
            merged = pd.concat([series, rates_data])
            rates_data = merged[~merged.index.duplicated(keep="last")]

        # If data is empty, return empty result
        if rates_data.empty:
            logger.warning(
                f"No risk-free rate data found for date range {start_date} to {end_date}."
            )
            return RatesResult(symbol=self.DEFAULT_YFINANCE_TICKER, timeseries=pd.Series(dtype=float))


        ## Cache the updated series. This is allowed cause `cache_it` uses the utility function from
        ## trade.datamanager.utils.cache which wraps into a _CacheData object.
        self.cache_it(key, rates_data)

        ## Sanitize before returning
        rates_data = _data_structure_sanitize(
            rates_data,
            start=start_str,  # Ensure only requested range
            end=end_str,
            source_name=f"final risk-free rate timeseries for {self.DEFAULT_YFINANCE_TICKER} after merging cache and fetched data",
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

    @backoff.on_exception(
        backoff.expo,
        (SSLError, Exception),  # Catching general Exception as yfinance can raise various exceptions
        max_tries=5,
        logger=logger,
    )
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
        buffered_start = to_datetime(start_date) - pd.Timedelta(days=5)
        buffered_end = to_datetime(end_date) + pd.Timedelta(days=1)
        yf_ticker = yf.Ticker(self.DEFAULT_YFINANCE_TICKER)


        try:
            data_min = yf_ticker.history(
                start=buffered_start,
                end=buffered_end,
                interval=interval,
            )
            data_min.index = data_min.index.tz_localize(None)

        ## Fallback in case of yfinance issues
        except Exception as e: # noqa
            data_min = yf.download(
                yf_ticker.ticker,
                start=buffered_start,
                end=buffered_end,
                interval=interval,
                progress=False,
                multi_level_index=False,
            )

        data_min.columns = data_min.columns.str.lower()
        data_min["annualized"] = data_min["close"] / 100
        data_min["daily"] = (data_min["annualized"]).apply(deannualize)
        data_min["name"] = self.DEFAULT_YFINANCE_TICKER
        data_min["description"] = yf_ticker.info.get("shortName", "UNKNOWN")
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
        res = self.get_rate(date=datetime.now(), fallback_option=fallback_option)
        res.rt = True
        return res