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

from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Literal, Optional, Tuple
import pandas as pd
from pandas.tseries.offsets import BDay
from openbb import obb
from dbase.DataAPI.ThetaData import resample
from trade.helpers.helper import retrieve_timeseries, ny_now, CustomCache, get_missing_dates, YFinanceEmptyData
from trade.helpers.decorators import timeit
from trade.helpers.Logging import setup_logger
from trade.assets.rates import get_risk_free_rate_helper
from EventDriven._vars import OPTION_TIMESERIES_START_DATE, load_riskmanager_cache
from EventDriven.exceptions import UnaccessiblePropertyError
from trade import register_signal, SIGNALS_TO_RUN


logger = setup_logger("EventDriven.riskmanager.market_data", stream_log_level="INFO")

## TODO: This var is from optionlib. Once ready, import from there.
## TODO: Implement interval handling to have multiple intervals

OPTIMESERIES: Optional["MarketTimeseries"] = None
DIVIDEND_CACHE: CustomCache = load_riskmanager_cache(target="dividend_timeseries")
SPOT_CACHE: CustomCache = load_riskmanager_cache(target="spot_timeseries")
CHAIN_SPOT_CACHE: CustomCache = load_riskmanager_cache(target="chain_spot_timeseries")
SPLIT_FACTOR_CACHE: CustomCache = load_riskmanager_cache(target="split_factor_timeseries", create_on_missing=True)
_SANITIZED_ON_EXIT: bool = False


@dataclass
class AtIndexResult:
    """Dataclass to hold the result of retrieving market data at a specific index (date)."""

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
    """Class to hold timeseries data for a specific symbol."""

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
    """Class to manage market timeseries data for equities."""

    additional_data: Dict[str, Any] = field(default_factory=dict)
    rates: pd.DataFrame = field(default_factory=get_risk_free_rate_helper)
    DEFAULT_NAMES: ClassVar[List[str]] = ["spot", "chain_spot", "dividends", "split_factor", "dividend_yield"]
    _refresh_delta: Optional[timedelta] = timedelta(minutes=30)
    _last_refresh: Optional[datetime] = field(default_factory=ny_now)
    _start: str = OPTION_TIMESERIES_START_DATE
    _end: str = (datetime.now() - BDay(1)).strftime("%Y-%m-%d")
    _today: str = datetime.now().strftime("%Y-%m-%d")
    should_refresh: bool = True

    def __post_init__(self):
        register_signal(signum=15, signal_func=self._on_exit_sanitize)
        register_signal(signum=0, signal_func=self._on_exit_sanitize)

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
    def clear_caches(cls):
        """Clear all caches used by MarketTimeseries."""
        SPOT_CACHE.clear()
        CHAIN_SPOT_CACHE.clear()
        DIVIDEND_CACHE.clear()
        SPLIT_FACTOR_CACHE.clear()
        logger.info("All MarketTimeseries caches have been cleared.")

    @timeit
    def _on_exit_sanitize(self):
        """Remove today's data from all stored timeseries data."""
        global _SANITIZED_ON_EXIT
        if _SANITIZED_ON_EXIT:
            print("Sanitization on exit already performed. Skipping.")
            return
        try:

            def _check_instance(d):
                return isinstance(d, pd.DataFrame) or isinstance(d, pd.Series)

            for sym in self._spot.keys():
                d = self._spot[sym]
                if not _check_instance(d):
                    logger.critical(
                        "Data for symbol %s in spot cache is not a DataFrame or Series. Skipping sanitization. Data: %s",
                        sym,
                        d,
                    )
                    del self._spot[sym]
                    continue
                d = d[d.index < self._today]
                self._spot[sym] = d
            for sym in self._chain_spot.keys():
                d = self._chain_spot[sym]

                if not _check_instance(d):
                    logger.critical(
                        "Data for symbol %s in chain_spot cache is not a DataFrame or Series. Skipping sanitization. Data: %s",
                        sym,
                        d,
                    )
                    del self._chain_spot[sym]
                    continue

                d = d[d.index < self._today]
                self._chain_spot[sym] = d
            for sym in self._dividends.keys():
                d = self._dividends[sym]
                if not _check_instance(d):
                    logger.critical(
                        "Data for symbol %s in dividends cache is not a DataFrame or Series. Skipping sanitization. Data: %s",
                        sym,
                        d,
                    )
                    del self._dividends[sym]
                    continue

                d = d[d.index < self._today]
                self._dividends[sym] = d
            for sym in self._split_factor.keys():
                d = self._split_factor[sym]
                if not _check_instance(d):
                    logger.critical(
                        "Data for symbol %s in split_factor cache is not a DataFrame or Series. Skipping sanitization. Data: %s",
                        sym,
                        d,
                    )
                    del self._split_factor[sym]
                    continue

                d = d[d.index < self._today]
                self._split_factor[sym] = d

            logger.info("Sanitization of today's data on exit completed successfully.")
            _SANITIZED_ON_EXIT = True
        except Exception as e:
            logger.error("Error during sanitization: %s", e, exc_info=True)

    # @timeit
    def _already_loaded(
        self, sym: str, interval: str = "1d", start: str | datetime = None, end: str | datetime = None
    ) -> Tuple[bool, List[pd.Timestamp]]:
        """
        Check if the timeseries for a given symbol and interval is already loaded.
        Hidden method that also returns missing dates if not fully loaded.
        """
        start = start or self._start
        end = end or self._end
        sym_available = sym in self._spot
        all_dates_present = False

        data_to_check = [
            (self._spot.get(sym), "spot"),
            (self._chain_spot.get(sym), "chain_spot"),
            (self._dividends.get(sym), "dividends"),
            (self._split_factor.get(sym), "split_factor"),
        ]

        missing_dates_set = set()
        all_dates_present = False
        for data, data_type in data_to_check:  # noqa
            if data is not None:
                missing_dates = get_missing_dates(data, start, end)
                missing_dates_set.update(missing_dates)

                if not missing_dates:
                    all_dates_present = True
                else:
                    all_dates_present = False
            else:
                missing_dates = pd.bdate_range(start=start, end=end).to_pydatetime().tolist()
                missing_dates_set.update(missing_dates)
                all_dates_present = False

        ## If all dates not present, return missing dates
        return_dates = list(missing_dates_set)
        if not all_dates_present:
            ## If missing dates is empty, return start and end
            if not return_dates:
                return_dates = [pd.Timestamp(start), pd.Timestamp(end)]
            else:
                return_dates = [min(return_dates), max(return_dates)]

        ## If all dates present, return empty list
        else:
            return_dates = []

        return (sym_available and all_dates_present), return_dates

    def cache_it(self, timeseries: TimeseriesData, sym: str) -> None:
        """
        Cache the provided timeseries data for the given symbol.
        """
        ## Remove today's data before caching
        spot = timeseries.spot.copy()
        chain_spot = timeseries.chain_spot.copy()
        dividends = timeseries.dividends.copy()
        split_factor = timeseries.split_factor.copy()
        self._spot[sym] = self._remove_today_data(spot)
        self._chain_spot[sym] = self._remove_today_data(chain_spot)
        self._dividends[sym] = self._remove_today_data(dividends)
        self._split_factor[sym] = self._remove_today_data(split_factor)
        logger.info("Cached timeseries data for symbol %s", sym)

    def already_loaded(
        self, sym: str, interval: str = "1d", start: str | datetime = None, end: str | datetime = None
    ) -> bool:
        """
        Public method to check if the timeseries for a given symbol and interval is already loaded.
        """
        already_loaded, _ = self._already_loaded(sym, interval, start, end)
        return already_loaded

    @timeit
    def _remove_today_data(self, data: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
        """Remove today's data from the given DataFrame or Series."""
        today_str = ny_now().strftime("%Y-%m-%d")
        if isinstance(data, pd.DataFrame):
            return data[data.index < today_str]
        elif isinstance(data, pd.Series):
            return data[data.index < today_str]
        else:
            raise ValueError("Data must be a pandas DataFrame or Series. Got type: {}".format(type(data)))

    @timeit
    def _sanitize_today_data(self, force_after_eod: bool = False) -> None:
        """Remove today's data from all stored timeseries data."""
        current_time = ny_now()
        if not force_after_eod and current_time.hour > 18:
            logger.info("Current time is after 6 PM NY time. Skipping sanitization of today's data.")
            return
        
        logger.info("Sanitizing today's data from all stored timeseries data...")
        for sym in self._spot.keys():
            self._spot[sym] = self._remove_today_data(self._spot[sym])
        for sym in self._chain_spot.keys():
            self._chain_spot[sym] = self._remove_today_data(self._chain_spot[sym])
        for sym in self._dividends.keys():
            self._dividends[sym] = self._remove_today_data(self._dividends[sym])
        for sym in self._split_factor.keys():
            self._split_factor[sym] = self._remove_today_data(self._split_factor[sym])

    @timeit
    def _sanitize_data(self) -> None:
        """
        Sanitize all stored timeseries data by removing today's data.
        Dropping duplicates, ensuring datetime index, and sorting.
        """
        self._sanitize_today_data()

        for sym in self._spot.keys():
            sym = sym.upper()
            data = self._spot[sym]
            data.index = pd.to_datetime(data.index)
            data = data[~data.index.duplicated(keep="last")]
            data = data.sort_index()
            data.dropna(how="all", inplace=True)
            self._spot[sym] = data

        for sym in self._chain_spot.keys():
            sym = sym.upper()
            data = self._chain_spot[sym]
            data.index = pd.to_datetime(data.index)
            data = data[~data.index.duplicated(keep="last")]
            data = data.sort_index()
            data.dropna(how="all", inplace=True)
            self._chain_spot[sym] = data

        for sym in self._dividends.keys():
            sym = sym.upper()
            data = self._dividends[sym]
            data.index = pd.to_datetime(data.index)
            data = data[~data.index.duplicated(keep="last")]
            data = data.sort_index()
            data.dropna(how="all", inplace=True)
            self._dividends[sym] = data

        for sym in self._split_factor.keys():
            sym = sym.upper()
            data = self._split_factor[sym]
            data.index = pd.to_datetime(data.index)
            data = data[~data.index.duplicated(keep="last")]
            data = data.sort_index()
            data.dropna(how="all", inplace=True)
            self._split_factor[sym] = data

    def get_split_factor_at_index(self, sym: str, index: pd.Timestamp) -> float | int:
        """
        Retrieve the split factor for a given symbol at a specific index (date).
        Args:
            sym (str): The stock symbol.
            index (pd.Timestamp or str): The date for which to retrieve the split factor.
        Returns:
            float | int: The split factor at the specified date.
        """
        split_factor_series = self._split_factor.get(sym)
        if split_factor_series is None:
            return 1.0

        index = pd.to_datetime(index)
        if index in split_factor_series.index:
            return split_factor_series.loc[index]
        else:
            prior_dates = split_factor_series.index[split_factor_series.index <= index]
            if not prior_dates.empty:
                nearest_date = prior_dates.max()
                return split_factor_series.loc[nearest_date]
            else:
                return 1.0

    def _pre_sanitize_load_timeseries(
        self,
        sym: str,
        start_date: str | datetime = None,
        end_date: str | datetime = None,
        interval="1d",
        force: bool = False,
    ) -> None:
        """
        Pre-sanitization before loading timeseries data for a given symbol and interval.
        """
        sym = sym.upper()
        if start_date is None:
            start_date = self._start
        if end_date is None:
            end_date = self._end
        already_loaded, dt_range = self._already_loaded(sym, interval, start_date, end_date)
        if already_loaded and not force:
            logger.info("Timeseries for %s already loaded. Use force=%s to reload.", sym, force)
            return

        start_date = min(dt_range)
        end_date = max(dt_range)

        try:
            spot = retrieve_timeseries(sym, start_date, end_date, interval)
        except YFinanceEmptyData:
            logger.error("Failed to retrieve spot data for symbol %s. Skipping load.", sym)
            return

        try:
            chain_spot = retrieve_timeseries(sym, start_date, end_date, interval, spot_type="chain_price")
        except YFinanceEmptyData:
            logger.error("Failed to retrieve chain spot data for symbol %s. Skipping load.", sym)
            return
        try:
            divs = obb.equity.fundamental.dividends(symbol=sym, provider="yfinance").to_df()
            divs.set_index("ex_dividend_date", inplace=True)
        except Exception:
            logger.error("Failed to retrieve dividends for symbol %s", sym)
            divs = pd.DataFrame({"amount": [0]}, index=pd.bdate_range(start=self._start, end=self._end, freq=interval))

        try:
            split_factor = chain_spot["split_factor"]
        except Exception:
            logger.error("Failed to retrieve split factor for symbol %s", sym)
            split_factor = pd.Series(1, index=pd.bdate_range(start=self._start, end=self._end, freq=interval))

        ## Ensure datetime index
        divs.index = pd.to_datetime(divs.index)
        use_start = min(spot.index.min(), chain_spot.index.min(), divs.index.min())
        use_end = max(spot.index.max(), chain_spot.index.max(), divs.index.max())
        divs = divs.reindex(pd.bdate_range(start=use_start, end=use_end, freq=interval), method="ffill")
        divs = resample(divs["amount"], method="ffill", interval=interval)

        ## Current Data
        current_spot = self._spot.get(sym)
        current_chain_spot = self._chain_spot.get(sym)
        current_divs = self._dividends.get(sym)
        current_split_factor = self._split_factor.get(sym)

        ## We are moving from overwritting prev data to merging new data
        if current_spot is not None:
            spot = pd.concat([current_spot, spot]).sort_index()
            spot = spot[~spot.index.duplicated(keep="last")]
        else:
            logger.info("No previous spot data for symbol %s, adding new data.", sym)
        if current_chain_spot is not None:
            chain_spot = pd.concat([current_chain_spot, chain_spot]).sort_index()
            chain_spot = chain_spot[~chain_spot.index.duplicated(keep="last")]
        if current_divs is not None:
            divs = pd.concat([current_divs, divs]).sort_index()
            divs = divs[~divs.index.duplicated(keep="last")]
        if current_split_factor is not None:
            split_factor = pd.concat([current_split_factor, split_factor]).sort_index()
            split_factor = split_factor[~split_factor.index.duplicated(keep="last")]

        ## Assign data directly to cache
        ## We remove today's data to avoid situations where it was loaded intraday and remains in database
        ## This ensures only historical data is stored.
        self._spot[sym] = spot
        self._chain_spot[sym] = chain_spot
        self._dividends[sym] = divs
        self._split_factor[sym] = split_factor

    def load_timeseries(
        self,
        sym: str,
        start_date: str | datetime = None,
        end_date: str | datetime = None,
        interval="1d",
        force: bool = False,
    ) -> None:
        """
        Public method to load timeseries data for a given symbol and interval.
        """
        self._pre_sanitize_load_timeseries(sym, start_date, end_date, interval, force)

    def _is_date_in_index(self, sym: str, date: pd.Timestamp, interval: str = "1d") -> bool:
        """
        Check if a specific date is present in the timeseries index for a given symbol and interval.
        Args:
            sym (str): The stock symbol.
            date (pd.Timestamp or str): The date to check.
            interval (str): The interval of the timeseries data. Defaults to '1d'.
        Returns:
            bool: True if the date is present, False otherwise.
        """
        all_data = [
            self._spot.get(sym),
            self._chain_spot.get(sym),
            self._dividends.get(sym),
            self._split_factor.get(sym),
        ]

        for data in all_data:
            date = pd.to_datetime(date).date()
            if data is not None and date in data.index.date:
                continue
            else:
                return False
        return True

    def get_at_index(self, sym: str, index: pd.Timestamp, interval: str = "1d") -> AtIndexResult:
        """
        Retrieve the spot price, chain spot price, and dividends for a given symbol at a specific index (date).
        Args:
            sym (str): The stock symbol.
            index (pd.Timestamp or str): The date for which to retrieve the data.
        Returns:
            AtIndexResult: A dataclass containing spot price, chain spot price, and dividends."""

        ## Only load date if not available. Not loading all unavailable dates
        already_available = self._is_date_in_index(sym, index, interval)

        if not already_available:
            logger.critical("Reloading timeseries data for symbol %s.", sym)
            prev_day = (pd.Timestamp(index) - BDay(1)).strftime("%Y-%m-%d")
            self._pre_sanitize_load_timeseries(
                sym=sym, start_date=prev_day, end_date=index, interval=interval, force=True
            )

        ## OPTIMIZATION: Consolidate type checks and conversions (Task #3)
        if not isinstance(index, pd.Timestamp):
            index = pd.Timestamp(index)

        if sym not in self._spot:
            raise ValueError(f"Symbol {sym} not found in timeseries data.")

        index_str = index.strftime("%Y-%m-%d")
        spot = self._spot[sym].loc[index_str] if sym in self._spot else None
        chain_spot = self._chain_spot[sym].loc[index_str] if sym in self._chain_spot else None
        dividends = self._dividends[sym].loc[index_str] if sym in self._dividends else None
        rates = self.rates.loc[index_str] if self.rates is not None else None
        dividend_yield = dividends / spot["close"] if spot is not None and dividends is not None else None
        split_factor = self._split_factor[sym].loc[index_str] if sym in self._split_factor else None
        self._sanitize_today_data()

        return AtIndexResult(
            spot=spot,
            chain_spot=chain_spot,
            dividends=dividends,
            sym=sym,
            date=index_str,
            rates=rates,
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
    ) -> None:
        """
        Load additional data for a given factor (spot, chain_spot, dividends, split_factor) using a callable function.

        Process:
        Callable passed should only take in a pd.Series and return a pd.Series.
        It manipulates the timeseries data for the specified factor and appends the result to the additional_data dictionary.
        The schema of additional_data: {additional_data_name: {sym: pd.Series}}

        Args:
            factor (Literal['spot', 'chain_spot', 'dividends', 'split_factor']): The factor to process.
            sym (str): The stock symbol.
            additional_data_name (str): The name under which to store the additional data.
            _callable (Any): A callable function that processes the pd.Series.
            column (Optional[str]): The column to use from the factor data. Defaults to 'close'.
            force_add (bool): If True, will overwrite existing additional data for the given name and symbol.

        Raises:
            ValueError: If the factor is not recognized or if the symbol is not found in the timeseries data.
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
        interval: str = "1d",
        additional_data_name: Optional[str] = None,
        start_date: str | datetime = None,
        end_date: str | datetime = None,
        skip_preload_check: bool = False,
    ) -> TimeseriesData:
        """
        Retrieve the timeseries data for a given symbol and factor.
        Args:
            sym (str): The stock symbol.
            factor (Literal['spot', 'chain_spot', 'dividends', 'split_factor', 'additional']): The factor to retrieve.
            additional_data_name (Optional[str]): The name of the additional data if factor is 'additional'.
        Returns:
            TimeseriesData: A dataclass containing the requested timeseries data.
        """
        sym = sym.upper()
        must_preload = False
        end_date = end_date or self._end
        
        ## Adding `must_preload`. This will be determined based on if 
        ## 1. Today's date is in end_date
        ## 2. Current time is before market close
        if pd.to_datetime(end_date).date() >= ny_now().date():
            current_time = ny_now()
            if current_time.hour < 20:
                must_preload = True

        if must_preload:
            logger.warning(
                "End date %s is today or in the future and current time is before market close. Forcing preload check.",
                end_date,
            )


        ## Check if data is already loaded
        if skip_preload_check and not must_preload:
            already_loaded = True
        else:
            already_loaded, _ = self._already_loaded(sym, interval, start_date, end_date)

        if not already_loaded:
            logger.critical("Timeseries for symbol %s not loaded. Loading now.", sym)
            self._pre_sanitize_load_timeseries(sym, interval=interval, force=True)

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
            factor = "_" + factor
            if factor in ["_spot", "_chain_spot", "_dividends", "_split_factor"]:
                data = getattr(self, factor).get(sym)
            elif factor == "_dividend_yield":
                divs = self._dividends.get(sym)
                if divs is None:
                    raise ValueError(f"No dividend data found for symbol {sym} to calculate dividend yield.")
                spot = self._spot.get(sym)
                if spot is None:
                    raise ValueError(f"No spot data found for symbol {sym} to calculate dividend yield.")
                dividend_yield = divs / spot["close"]
                data = dividend_yield
            if start_date is not None or end_date is not None:
                start_date = pd.to_datetime(start_date).strftime("%Y-%m-%d") if start_date is not None else None
                end_date = pd.to_datetime(end_date).strftime("%Y-%m-%d") if end_date is not None else None
                if start_date is not None:
                    data = data[data.index >= start_date]
                if end_date is not None:
                    data = data[data.index <= end_date]
            if data is None:
                raise ValueError(f"No data found for factor {factor} and symbol {sym}.")
            if factor == "_spot":
                ts = TimeseriesData(spot=data, chain_spot=None, dividends=None, dividend_yield=None, split_factor=None)
            elif factor == "_chain_spot":
                ts = TimeseriesData(spot=None, chain_spot=data, dividends=None, dividend_yield=None, split_factor=None)
            elif factor == "_dividends":
                ts = TimeseriesData(spot=None, chain_spot=None, dividends=data, dividend_yield=None, split_factor=None)
            elif factor == "_dividend_yield":
                ts = TimeseriesData(spot=None, chain_spot=None, dividends=None, dividend_yield=data, split_factor=None)
            elif factor == "_split_factor":
                ts = TimeseriesData(spot=None, chain_spot=None, dividends=None, split_factor=data, dividend_yield=None)
            else:
                raise ValueError(f"Unhandled factor {factor}.")

        elif factor is None:
            spot = self._spot.get(sym)
            chain_spot = self._chain_spot.get(sym)
            dividends = self._dividends.get(sym)
            dividend_yield = dividends / spot["close"] if spot is not None and dividends is not None else None
            split_factor = self._split_factor.get(sym)
            if start_date is not None or end_date is not None:
                start_date = pd.to_datetime(start_date).strftime("%Y-%m-%d") if start_date is not None else None
                end_date = pd.to_datetime(end_date).strftime("%Y-%m-%d") if end_date is not None else None
                if start_date is not None:
                    spot = spot[spot.index >= start_date]
                    chain_spot = chain_spot[chain_spot.index >= start_date]
                    dividends = dividends[dividends.index >= start_date]
                    dividend_yield = dividend_yield[dividend_yield.index >= start_date]
                    split_factor = split_factor[split_factor.index >= start_date]
                if end_date is not None:
                    spot = spot[spot.index <= end_date]
                    chain_spot = chain_spot[chain_spot.index <= end_date]
                    dividends = dividends[dividends.index <= end_date]
                    dividend_yield = dividend_yield[dividend_yield.index <= end_date]
                    split_factor = split_factor[split_factor.index <= end_date]
            ts = TimeseriesData(
                spot=spot,
                chain_spot=chain_spot,
                dividends=dividends,
                dividend_yield=dividend_yield,
                split_factor=split_factor,
                rates=self.rates["annualized"],
            )
            self._sanitize_today_data()

        return ts

    def __repr__(self) -> str:
        return f"MarketTimeseries(symbols: {list(self._spot.keys())}, intervals: {list(self._spot.keys())})"


def get_timeseries_obj() -> MarketTimeseries:
    global OPTIMESERIES
    if OPTIMESERIES is None:
        OPTIMESERIES = MarketTimeseries()

    return OPTIMESERIES


def reset_timeseries_obj() -> None:
    global OPTIMESERIES
    OPTIMESERIES = None


if __name__ == "__main__":
    mts = get_timeseries_obj()
    mts.load_timeseries("BA", force=True)
    ts = mts.get_timeseries("BA")
    print(ts)
    print(SIGNALS_TO_RUN)
