"""Dividend data management for options pricing with caching and schedule construction.

This module provides the DividendDataManager class for retrieving, caching, and
constructing dividend schedules (discrete or continuous) for equity symbols. Supports
backtest-style time-series construction with split adjustments and partial caching.

Typical usage:
    >>> div_mgr = DividendDataManager("AAPL")
    >>> result = div_mgr.get_schedule_timeseries(
    ...     start_date="2025-01-01",
    ...     end_date="2025-01-31",
    ...     maturity_date="2025-06-20",
    ...     dividend_type=DivType.DISCRETE,
    ...     undo_adjust=True
    ... )
    >>> schedules = result.daily_discrete_dividends
"""

from datetime import datetime
from typing import Any, ClassVar, Optional, Tuple, Union, List
import pandas as pd
from trade.helpers.Logging import setup_logger
from trade.optionlib.assets.dividend import Schedule, ScheduleEntry
from trade.datamanager.vars import TS, DM_GEN_PATH, load_name
from trade.datamanager.config import OptionDataConfig
from trade.datamanager.result import DividendsResult
from trade.datamanager.base import BaseDataManager, CacheSpec
from trade.datamanager._enums import ArtifactType, SeriesId, Interval, RealTimeFallbackOption
from trade.datamanager.utils import slice_schedule
from trade.datamanager.utils.date import DateRangePacket, is_available_on_date
from trade.datamanager.utils.cache import _data_structure_cache_it
from trade.helpers.helper import CustomCache, get_missing_dates, change_to_last_busday
from trade.optionlib.config.types import DivType
from trade.optionlib.assets.dividend import get_vectorized_dividend_scehdule

from trade import HOLIDAY_SET
from .utils.data_structure import _data_structure_sanitize

logger = setup_logger("trade.datamanager.dividend", stream_log_level="DEBUG")


class DividendDataManager(BaseDataManager):
    """Manages dividend data retrieval, caching, and schedule construction for a specific symbol.

    This manager handles both discrete and continuous dividends with intelligent caching,
    partial cache merging, and split adjustment logic. Implements singleton pattern per symbol
    to avoid redundant timeseries loading.

    Attributes:
        CACHE_NAME: Class-level cache identifier for this manager type.
        DEFAULT_SERIES_ID: Default historical series identifier.
        CONFIG: Configuration object for dividend data settings.
        INSTANCES: Class-level cache of manager instances per symbol.
        symbol: The equity ticker symbol this manager handles.
        temp_cache: Short-lived cache for temporary dividend data.

    Examples:
        >>> # Singleton access - same instance returned for same symbol
        >>> div_mgr1 = DividendDataManager("AAPL")
        >>> div_mgr2 = DividendDataManager("AAPL")
        >>> assert div_mgr1 is div_mgr2

        >>> # Get discrete dividend schedule for a date range
        >>> schedule, key = div_mgr1.get_discrete_dividend_schedule(
        ...     start_date="2025-01-01",
        ...     end_date="2025-06-20"
        ... )

        >>> # Get daily time-series of schedules
        >>> result = div_mgr1.get_schedule_timeseries(
        ...     start_date="2025-01-01",
        ...     end_date="2025-01-31",
        ...     maturity_date="2025-06-20"
        ... )
    """

    CACHE_NAME: ClassVar[str] = "dividend_data_manager"
    DEFAULT_SERIES_ID: ClassVar["SeriesId"] = SeriesId.HIST
    CONFIG = OptionDataConfig()
    INSTANCES = {}

    def __new__(cls, symbol: str, *args: Any, **kwargs: Any) -> "DividendDataManager":
        """Returns cached instance for symbol, creating new one if needed.

        Implements singleton pattern per symbol to ensure timeseries are loaded only once.
        Automatically loads market timeseries data on first instantiation.

        Args:
            symbol: Equity ticker symbol (e.g., "AAPL", "MSFT").
            *args: Additional positional arguments passed to __init__.
            **kwargs: Additional keyword arguments passed to __init__.

        Returns:
            Singleton DividendDataManager instance for the given symbol.

        Examples:
            >>> mgr1 = DividendDataManager("AAPL")
            >>> mgr2 = DividendDataManager("AAPL")
            >>> assert mgr1 is mgr2  # Same instance
        """
        if symbol not in cls.INSTANCES:
            instance = object.__new__(cls)
            cls.INSTANCES[symbol] = instance
        return cls.INSTANCES[symbol]

    def __init__(
        self, symbol: str, *, cache_spec: Optional[CacheSpec] = None, enable_namespacing: bool = False
    ) -> None:
        """Initializes manager for a symbol with cache and temp cache for short-lived data.

        Sets up persistent cache for dividend schedules and temporary cache for short-lived
        data. Only executes once per symbol due to singleton pattern.

        Args:
            symbol: Equity ticker symbol.
            cache_spec: Optional cache configuration. Uses default if None.
            enable_namespacing: If True, enables namespace isolation in cache keys.

        Examples:
            >>> mgr = DividendDataManager("AAPL")
            >>> mgr = DividendDataManager("AAPL", cache_spec=CacheSpec(expire_days=30))
        """

        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        super().__init__(cache_spec=cache_spec, enable_namespacing=enable_namespacing)
        self.symbol = symbol
        self.temp_cache: CustomCache = CustomCache(
            location=DM_GEN_PATH.as_posix(), fname="dividend_temp_cache", expire_days=1, clear_on_exit=True
        )

    ## General caching logic
    def cache_it(self, key: str, value: Any, *, expire: Optional[int] = None, _type: str = "discrete") -> None:
        """Caches dividend data with merge logic for discrete dividends (no future dates).

        For discrete dividends, implements smart merging: filters out future dates (> today)
        and merges with existing cache by date, keeping unique entries. For other types,
        performs direct cache storage.

        Args:
            key: Cache key identifier.
            value: Data to cache (typically List[ScheduleEntry] for discrete).
            expire: Optional expiration time in seconds. Uses cache default if None.
            _type: Type of dividend data ("discrete" or other). Affects merge logic.

        Examples:
            >>> div_mgr = DividendDataManager("AAPL")
            >>> schedule = [ScheduleEntry(date=date(2025, 3, 15), amount=0.25)]
            >>> div_mgr.cache_it("my_key", schedule, expire=86400, _type="discrete")

        Notes:
            - Discrete dividends are filtered to exclude future dates (> today)
            - Existing cache entries are merged and deduplicated by date
            - Non-discrete types bypass merge logic
        """

        ## If discrete dividends, we first check if key exists
        ## If it does, we add to it. Only values <= today.
        ## If it does not, we create new entry
        if _type == "discrete":
            existing = self.get(key, default=None)
            today = datetime.today().date()
            allowed = [e for e in value if e.date <= today]

            if existing is not None:
                # Merge existing and new values. We're expecting lists of ScheduleEntry
                merged = existing + allowed

                ## Unique by date
                merged = {entry.date: entry for entry in merged}
                uniques = sorted(merged.values(), key=lambda e: e.date)
                self.set(key, uniques, expire=expire)
                return
            else:
                self.set(key, allowed, expire=expire)
                return

        # For other types or if no existing, just setattr
        self.set(key, value, expire=expire)

    ## Dividend yield history retrieval for continuous dividends. Already cached in MarketTimeseries.
    def get_div_yield_history(self, symbol: str, skip_preload_check: bool = False) -> pd.Series:
        """Retrieves continuous dividend yield history from MarketTimeseries.

        Fetches annual dividend yield as a percentage time-series from the global
        MarketTimeseries cache (TS). Used for continuous dividend modeling.

        Args:
            symbol: Equity ticker symbol.
            skip_preload_check: If True, skips validation that timeseries is preloaded.

        Returns:
            Time-indexed Series of annualized dividend yields (e.g., 0.025 = 2.5%).

        Examples:
            >>> div_mgr = DividendDataManager("AAPL")
            >>> yields = div_mgr.get_div_yield_history("AAPL")
            >>> logger.info(yields.head())
            datetime
            2020-01-02    0.0124
            2020-01-03    0.0125
            ...
        """
        div_history = TS.get_timeseries(symbol, skip_preload_check=skip_preload_check)
        return div_history.dividend_yield

    ## Discrete dividend schedule retrieval with caching.
    def get_discrete_dividend_schedule(
        self,
        *,
        end_date: Union[str, datetime, pd.Timestamp],
        start_date: Union[str, datetime, pd.Timestamp],
        valuation_date: Optional[Union[str, datetime, pd.Timestamp]] = None,
    ) -> Tuple[List[ScheduleEntry], str]:
        """Returns discrete dividend schedule between dates with partial cache support.

        Fetches individual dividend payment events (ex-dates and amounts) expected between
        start_date and end_date. Intelligently uses cache if available and fetches missing
        data only when needed.

        Args:
            start_date: Start of date range for dividend events (YYYY-MM-DD string or datetime).
            end_date: End of date range for dividend events (YYYY-MM-DD string or datetime).
            valuation_date: Optional reference date for forecasting. Defaults to start_date.

        Returns:
            Tuple containing:
                - List[ScheduleEntry]: Dividend events with dates and amounts
                - str: Cache key used for this data

        Examples:
            >>> div_mgr = DividendDataManager("AAPL")
            >>> schedule, key = div_mgr.get_discrete_dividend_schedule(
            ...     start_date="2025-01-01",
            ...     end_date="2025-06-20"
            ... )
            >>> for entry in schedule:
            ...     logger.info(f"{entry.date}: ${entry.amount:.2f}")
            2025-02-14: $0.25
            2025-05-16: $0.25

        Notes:
            - Uses vectorized dividend schedule fetching from optionlib
            - Partial cache hits trigger fetches for missing date ranges only
            - Cache stores raw ScheduleEntry lists without splits applied
        """

        ## Load first
        load_name(self.symbol)

        ## Dates
        packet = DateRangePacket(start_date, end_date)
        start_date = packet.start_date
        end_date = packet.end_date
        start_str = packet.start_str
        end_str = packet.end_str

        ticker = self.symbol
        method = self.CONFIG.default_forecast_method.value
        lookback_years = self.CONFIG.default_lookback_years
        key = self.make_key(
            symbol=ticker,
            artifact_type=ArtifactType.DIVS,
            series_id=SeriesId.HIST,
            method=method,
            lookback_years=lookback_years,
            current_state="schedule",
            interval=Interval.NA,
            vendor="yfinance",
        )

        available_schedule = self.get(key, default=None)
        if available_schedule:
            logger.info(f"Cache hit for key: {key}")
            ## If max date in available schedule >= end_date, we can use cache
            max_cached_date = max(entry.date for entry in available_schedule)
            min_cached_date = min(entry.date for entry in available_schedule)
            fully_covered = (min_cached_date <= datetime.strptime(start_str, "%Y-%m-%d").date()) and (
                max_cached_date >= datetime.strptime(end_str, "%Y-%m-%d").date()
            )
            if fully_covered:
                logger.info(f"Cache fully covers requested date range. Key: {key}")

                ## Filter to requested date range
                start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
                end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
                filtered_schedule = [e for e in available_schedule if start_dt <= e.date <= end_dt]
                return filtered_schedule, key
            else:
                logger.info(f"Cache partially covers requested date range. Key: {key}. Fetching missing data.")

        schedule = get_vectorized_dividend_scehdule(
            tickers=[ticker],
            end_dates=[end_date],
            start_dates=[start_date],
            method=method,
            lookback_years=lookback_years,
            valuation_dates=[valuation_date] if valuation_date else None,
        )

        raw_schedule = schedule[0].schedule
        self.cache_it(key, raw_schedule, _type="discrete")

        return raw_schedule, key

    ## Switcher to choose between constructing all the way or using cached pieces
    def _get_discrete_schedule_timeseries(
        self,
        start_date: Union[datetime, str],
        end_date: Union[datetime, str],
        maturity_date: Union[datetime, str],
        dividend_type: Optional[DivType] = None,
        undo_adjust: bool = True,
    ) -> Tuple[pd.Series, str]:
        """Builds daily dividend schedule series with partial cache merging and split adjustment.

        Constructs a time-series where each business day gets its own Schedule object representing
        dividends from that valuation date to maturity. Optimizes by fetching dividend events once
        and slicing for each date. Optionally applies split adjustments.

        Args:
            start_date: First valuation date (YYYY-MM-DD string or datetime).
            end_date: Last valuation date (YYYY-MM-DD string or datetime).
            maturity_date: Fixed horizon date for all schedules (e.g., option expiry).
            dividend_type: DivType.DISCRETE or DivType.CONTINUOUS. Uses config default if None.
            undo_adjust: If True, adjusts dividends for splits as of valuation date.

        Returns:
            Tuple containing:
                - pd.Series: DatetimeIndex with Schedule objects as values
                - str: Cache key used

        Raises:
            ValueError: If maturity_date < start_date.

        Examples:
            >>> div_mgr = DividendDataManager("AAPL")
            >>> series, key = div_mgr._get_discrete_schedule_timeseries(
            ...     start_date="2025-01-01",
            ...     end_date="2025-01-31",
            ...     maturity_date="2025-06-20",
            ...     undo_adjust=True
            ... )
            >>> logger.info(series.head())
            datetime
            2025-01-02    Schedule([ScheduleEntry(...), ...])
            2025-01-03    Schedule([ScheduleEntry(...), ...])
            ...

        Notes:
            - Fetches full schedule from start_date to maturity_date once
            - Builds daily schedules by slicing based on valuation date
            - Split adjustments multiply dividend amounts by cumulative split factor
            - Partial cache hits merge with existing data
            - Cache expires after 12 hours
        """
        logger.info(
            f"Fetching discrete dividend schedule timeseries for {self.symbol} from {start_date} to {end_date} with maturity {maturity_date}"
        )
        packet = DateRangePacket(start_date, end_date, maturity_date=maturity_date)
        dividend_type = DivType(dividend_type) if dividend_type is not None else self.CONFIG.dividend_type
        is_partial = False
        start_dt = packet.start_date.date()
        end_dt = packet.end_date.date()
        mat_dt = packet.maturity_date.date()
        start_str = packet.start_str
        end_str = packet.end_str
        mat_str = packet.maturity_str

        if mat_dt < start_dt:
            logger.info(f"Maturity date {mat_dt} is before start date {start_dt}")
            raise ValueError("maturity_date must be >= start_date")

        key = self.make_key(
            symbol=self.symbol,
            artifact_type=ArtifactType.DIVS,
            series_id=SeriesId.HIST,
            method=self.CONFIG.default_forecast_method.value,
            lookback_years=self.CONFIG.default_lookback_years,
            current_state="schedule_timeseries",
            interval=Interval.EOD,
            undo_adjust=undo_adjust,
            maturity=mat_str,
        )

        cached_series = self.get(key, default=None)
        if cached_series is not None:
            logger.info(f"Cache hit for discrete schedule timeseries key: {key}")
            missing_dates = get_missing_dates(cached_series, start_str, end_str)
            if not missing_dates:
                logger.info(f"Cache fully covers requested date range for timeseries. Key: {key}")
                cached_series = cached_series[
                    (cached_series.index >= pd.to_datetime(start_date))
                    & (cached_series.index <= pd.to_datetime(end_date))
                ]
                return cached_series, key
            else:
                logger.info(
                    f"Cache partially covers requested date range for timeseries. Key: {key}. Fetching missing dates: {missing_dates}"
                )
                start_str, end_str = min(missing_dates), max(missing_dates)
                is_partial = True
        else:
            logger.info(f"No cache found for discrete schedule timeseries key: {key}. Building from scratch.")

        # Build from scratch for missing dates
        # Fetch ONCE: all events from start_date to maturity_date
        full_schedule, _ = self.get_discrete_dividend_schedule(
            start_date=start_str,
            end_date=mat_str,
            valuation_date=start_str,
        )

        # Build daily schedules efficiently using a moving pointer
        series = {}
        date_range = pd.date_range(start=start_dt, end=end_dt, freq="B").strftime("%Y-%m-%d")
        for d in date_range:
            if d in HOLIDAY_SET:
                # Skip holidays
                continue
            d_date = datetime.strptime(d, "%Y-%m-%d").date()

            ## Simple filter approach
            series[d_date] = Schedule(slice_schedule(full_schedule, d_date, mat_dt))
        data = pd.Series(series, name="dividend_schedule")

        # Back-adjust to represent cashflows as of valuation date. Ie undoing splits
        if undo_adjust:
            data = data.to_frame()
            split_factors = TS._split_factor[self.symbol].copy()
            data["split_factor"] = split_factors
            data["dividend_schedule"] = data["dividend_schedule"] * data["split_factor"]
            data = data["dividend_schedule"]

        # Cache the constructed timeseries
        if is_partial:
            # Merge with existing cached series
            merged = pd.concat([cached_series, data])
            data = merged[~merged.index.duplicated(keep="last")]

        data = _data_structure_sanitize(data, start_date, end_date)

        _data_structure_cache_it(self, key, data, expire=86400)
        return data, key

    def get_schedule_timeseries(
        self,
        start_date: Union[datetime, str],
        end_date: Union[datetime, str],
        maturity_date: Union[datetime, str],
        dividend_type: Optional[DivType] = None,
        undo_adjust: bool = True,
    ) -> DividendsResult:
        """Returns daily dividend schedule time-series from valuation dates to maturity.

        Constructs a daily series where each business day has its own Schedule representing
        dividends from that valuation date to the fixed maturity date. Supports both discrete
        and continuous dividend models.

        Args:
            start_date: First valuation date (YYYY-MM-DD string or datetime).
            end_date: Last valuation date (YYYY-MM-DD string or datetime).
            maturity_date: Fixed horizon date for all schedules (e.g., option expiry).
            dividend_type: DivType.DISCRETE or DivType.CONTINUOUS. Uses config default if None.
            undo_adjust: If True, adjusts dividends for splits as of valuation date.

        Returns:
            DividendsResult containing daily_discrete_dividends or daily_continuous_dividends
            Series, plus metadata (key, dividend_type, undo_adjust).

        Examples:
            >>> div_mgr = DividendDataManager("AAPL")
            >>> result = div_mgr.get_schedule_timeseries(
            ...     start_date="2025-01-01",
            ...     end_date="2025-01-31",
            ...     maturity_date="2025-06-20",
            ...     dividend_type=DivType.DISCRETE,
            ...     undo_adjust=True
            ... )
            >>> schedules = result.daily_discrete_dividends
            >>> logger.info(schedules.iloc[0])  # First day's schedule
            Schedule([ScheduleEntry(date=..., amount=...)])

        Notes:
            - For DISCRETE: Returns Series of Schedule objects (one per day)
            - For CONTINUOUS: Returns Series of annual yield percentages
            - start_date/end_date define valuation date range
            - maturity_date is the fixed horizon (e.g., option expiry)
        """
        load_name(self.symbol)
        if dividend_type:
            logger.info(f"Using provided dividend_type: {dividend_type}")
        else:
            logger.info(f"Using config default dividend_type: {self.CONFIG.dividend_type}")

        dividend_type = DivType(dividend_type) if dividend_type is not None else self.CONFIG.dividend_type
        result = DividendsResult()
        result.symbol = self.symbol
        result.dividend_type = dividend_type
        result.undo_adjust = undo_adjust

        if dividend_type == DivType.DISCRETE:
            data, key = self._get_discrete_schedule_timeseries(
                start_date=start_date,
                end_date=end_date,
                maturity_date=maturity_date,
                dividend_type=dividend_type,
                undo_adjust=undo_adjust,
            )
            data.index = pd.to_datetime(data.index)
            data.index.name = "datetime"
            data = data[(data.index >= pd.to_datetime(start_date)) & (data.index <= pd.to_datetime(end_date))]
            data = data.sort_index()
            data = data.drop_duplicates()
            result.daily_discrete_dividends = data
            result.key = key

        elif dividend_type == DivType.CONTINUOUS:
            start_str = (
                pd.to_datetime(start_date).strftime("%Y-%m-%d") if isinstance(start_date, datetime) else start_date
            )
            end_str = pd.to_datetime(end_date).strftime("%Y-%m-%d") if isinstance(end_date, datetime) else end_date
            yield_history = self.get_div_yield_history(self.symbol, skip_preload_check=True)
            filtered = yield_history[(yield_history.index >= start_str) & (yield_history.index <= end_str)]
            result.daily_continuous_dividends = filtered
            result.key = None
        return result

    def get_schedule(
        self,
        valuation_date: Union[datetime, str],
        maturity_date: Union[datetime, str],
        dividend_type: Optional[DivType] = None,
        undo_adjust: bool = True,
        fallback_option: Optional[RealTimeFallbackOption] = None,
    ) -> DividendsResult:
        """Returns dividend schedule for a single valuation date to maturity.

        Fetches dividend data (discrete events or continuous yields) from a single
        valuation date to maturity date. Suitable for real-time pricing scenarios.

        Args:
            valuation_date: Reference date for valuation (YYYY-MM-DD string or datetime).
            maturity_date: Horizon date for dividends (YYYY-MM-DD string or datetime).
            dividend_type: DivType.DISCRETE or DivType.CONTINUOUS. Uses config default if None.
            undo_adjust: If True, adjusts dividends for splits as of valuation date.

        Returns:
            DividendsResult with daily_discrete_dividends or daily_continuous_dividends,
            plus metadata.

        Examples:
            >>> div_mgr = DividendDataManager("AAPL")
            >>> result = div_mgr.get_schedule(
            ...     valuation_date="2025-01-15",
            ...     maturity_date="2025-06-20",
            ...     dividend_type=DivType.DISCRETE,
            ...     undo_adjust=True
            ... )
            >>> schedule = result.daily_discrete_dividends.iloc[0]
            >>> logger.info(schedule.schedule)  # List of ScheduleEntry objects

        Notes:
            - For DISCRETE: Returns Series with single entry containing Schedule object
            - For CONTINUOUS: Returns filtered yield history between dates
            - Split adjustments applied if undo_adjust=True
        """
        load_name(self.symbol)
        fallback_option = fallback_option or self.CONFIG.real_time_fallback_option
        dividend_type = DivType(dividend_type) if dividend_type is not None else self.CONFIG.dividend_type

        if not is_available_on_date(valuation_date):
            logger.warning(f"Valuation date {valuation_date} is not a business day or holiday. No dividends available. Resolution: {fallback_option}")
            if fallback_option == RealTimeFallbackOption.RAISE_ERROR:
                raise ValueError(f"Valuation date {valuation_date} is not a business day or holiday.")
            if fallback_option == RealTimeFallbackOption.USE_LAST_AVAILABLE:
                valuation_date = change_to_last_busday(valuation_date)
            else:
                result = DividendsResult()
                dividend = pd.Series(dtype=float)
                if dividend_type == DivType.DISCRETE:
                    result.daily_discrete_dividends = dividend
                else:
                    result.daily_continuous_dividends = dividend
                result.key = None
                result.undo_adjust = undo_adjust
                result.dividend_type = dividend_type
                result.symbol = self.symbol
                return result
            

        

        val_str = valuation_date.strftime("%Y-%m-%d") if isinstance(valuation_date, datetime) else valuation_date
        mat_str = maturity_date.strftime("%Y-%m-%d") if isinstance(maturity_date, datetime) else maturity_date

        if dividend_type == DivType.DISCRETE:
            data, key = self.get_discrete_dividend_schedule(
                start_date=val_str,
                end_date=mat_str,
                valuation_date=val_str,  # optional, but consistent
            )
            if undo_adjust:
                split_factor = TS.get_split_factor_at_index(self.symbol, pd.to_datetime(valuation_date))
            else:
                split_factor = 1.0
            data = Schedule(schedule=[entry * split_factor for entry in data])
            data = pd.Series({val_str: data})
        elif dividend_type == DivType.CONTINUOUS:
            data = self.get_div_yield_history(self.symbol)
            data = data[(data.index.date >= pd.to_datetime(valuation_date).date()) & (data.index.date <= pd.to_datetime(maturity_date).date())]
            key = None
        else:
            raise ValueError(f"Unsupported dividend type: {dividend_type}")

        result = DividendsResult()

        if dividend_type == DivType.DISCRETE:
            result.daily_discrete_dividends = data
        else:
            result.daily_continuous_dividends = data
        result.key = key
        result.undo_adjust = undo_adjust
        result.dividend_type = dividend_type
        result.symbol = self.symbol

        return result

    def rt(
        self,
        maturity_date: Union[datetime, str],
        dividend_type: Optional[DivType] = None,
        undo_adjust: bool = True,
        fallback_option: Optional[RealTimeFallbackOption] = None,
    ) -> DividendsResult:
        """Real-time enabled method to get dividend schedule for a single valuation date.

        Wrapper around get_schedule with real-time fallback handling. If data is missing
        for the valuation date, applies the specified fallback strategy.

        Args:
            valuation_date: Reference date for valuation (YYYY-MM-DD string or datetime).
            maturity_date: Horizon date for dividends (YYYY-MM-DD string or datetime).
            dividend_type: DivType.DISCRETE or DivType.CONTINUOUS. Uses config default if None.
            undo_adjust: If True, adjusts dividends for splits as of valuation date.
            fallback_option: Strategy for handling missing data. Uses config default if None.
        Returns:
            DividendsResult with dividend schedule or fallback data.
        """
        load_name(self.symbol)

        if fallback_option is None:
            fallback_option = self.CONFIG.real_time_fallback_option

        result = self.get_schedule(
            valuation_date=datetime.now(),
            maturity_date=maturity_date,
            dividend_type=dividend_type,
            undo_adjust=undo_adjust,
            fallback_option=fallback_option,
        )
        return result
        
    def offload(self, *args: Any, **kwargs: Any) -> None:

        """
        Placeholder for offload logic (not implemented).

        Reserved for future implementation of cache offloading or cleanup operations.
        Currently performs no action.

        Args:
            *args: Arbitrary positional arguments.
            **kwargs: Arbitrary keyword arguments.

        Examples:
            >>> div_mgr = DividendDataManager("AAPL")
            >>> div_mgr.offload()  # No-op
        """
        logger.info(f"No offload logic implemented for {self.CACHE_NAME}")

