"""Backtest Market Timeseries and Option Data Management.

This module manages market data retrieval, caching, and transformation for options backtesting.
It provides efficient access to historical spot prices, option chains, Greeks, corporate actions,
and position-level analytics across the entire backtest window.

Core Class:
    BacktestTimeseries: Centralized timeseries manager for backtest market data

Key Features:
    - Lazy loading of option data with intelligent caching
    - Position-specific data retrieval with corporate action adjustments
    - Skip column calculation for data quality management
    - Special dividend and split handling
    - Adjusted strike calculation for corporate actions
    - Thread-safe data access with locking mechanisms
    - Integration with DataManager for pricing and Greeks

Data Types Managed:
    Spot Data:
        - Underlying equity prices (open, high, low, close)
        - Volume and liquidity metrics
        - Chain-specific spot prices

    Option Data:
        - Full option chains by expiration
        - Bid/ask/mid prices
        - Black-Scholes and binomial Greeks
        - Open interest and volume
        - Implied volatility surfaces

    Corporate Actions:
        - Regular dividend timeseries
        - Special dividends with ex-dates
        - Stock splits (forward and reverse)
        - Strike adjustments from splits

    Risk-Free Rates:
        - Treasury yield curve
        - Interpolated rates for option pricing
        - Annualized rate timeseries

Caching Strategy:
    - Position data cached by position_id (trade_id)
    - Option data cached by option ticker
    - Special dividends and splits cached globally
    - Adjusted strikes cached to avoid recalculation
    - Cache expiration managed automatically

Skip Column Logic:
    Identifies unreliable data points to skip during analysis:
        - Missing or stale prices (no updates)
        - Zero volume/open interest
        - Wide bid-ask spreads
        - Missing Greeks or volatility
        - Configured per column (Midpoint, Delta, Vega, etc.)

Key Methods:
    get_option_data(opttick):
        Retrieve full timeseries for option ticker

    get_position_data(position_id):
        Get position-specific data with adjustments

    get_at_time_option_data(opttick, date):
        Point-in-time option data snapshot

    get_at_time_position_data(position_id, date):
        Position data at specific date with skip info

    skip(position_id, date, column):
        Check if position should be skipped on date

Configuration:
    SkipCalcConfig:
        - skip_columns: List of columns to monitor
        - threshold: Staleness threshold
        - window: Lookback window for checks

    UndlTimeseriesConfig:
        - price_source: Which price to use (close, mid)
        - data_provider: Source for spot data

    OptionPriceConfig:
        - pricing_model: BS vs Binomial
        - greeks_source: Analytical vs numerical

Corporate Action Handling:
    Splits:
        - Automatic strike adjustment (strike / split_ratio)
        - Position quantity adjustment
        - Historical price normalization

    Dividends:
        - Regular dividend forecasting
        - Special dividend adjustments
        - Ex-date identification
        - Impact on pricing models

Usage:
    bt_ts = BacktestTimeseries(
        _start='2024-01-01',
        _end='2024-12-31'
    )

    # Get option data at specific time
    option_data = bt_ts.get_at_time_option_data(
        opttick='AAPL250117P175000',
        date='2024-10-15'
    )

    # Check if should skip
    should_skip = bt_ts.skip(
        position_id='AAPL_20241015_175P_long',
        date='2024-11-01',
        column='Midpoint'
    )

    # Get position data with adjustments
    pos_data = bt_ts.get_at_time_position_data(
        position_id='MSFT_20240301_350C_short',
        date='2024-03-15'
    )

Performance:
    - All data cached in memory after first load
    - Custom cache with configurable expiration
    - Thread-safe concurrent access
    - Minimal disk I/O after initial population
    - Efficient pandas operations throughout

Integration:
    - Used by RiskManager for all market data needs
    - Feeds PositionAnalyzer for position evaluation
    - Provides data to OrderPicker for chain filtering
    - Supplies Greeks to position sizers for limit calculations

Notes:
    - Dates automatically converted to pandas Timestamps
    - Missing data returns None rather than raising exceptions
    - Critical errors logged but don't halt execution
    - Pool logging level changed to WARNING by default
    - MySQL queries can be disabled via set_skip_mysql_query()
"""

## Options Timeseries class for handling data retrieval
from datetime import datetime
from dateutil.relativedelta import relativedelta
from trade.datamanager.market_data import MarketTimeseries
from EventDriven._vars import load_riskmanager_cache, ADD_COLUMNS_FACTORY
from EventDriven.riskmanager.utils import parse_position_id, swap_ticker, add_skip_columns, load_position_data_new
from EventDriven.types import BacktestRunMixin
from trade.helpers.decorators import timeit
from trade.helpers.threads import runThreads  # noqa
from trade.helpers.pools import runProcesses  # noqa
from trade.helpers.helper import compare_dates, parse_option_tick, generate_option_tick_new, to_datetime
from EventDriven.configs.core import SkipCalcConfig, UndlTimeseriesConfig, OptionPriceConfig
from trade.assets.rates import get_risk_free_rate_helper_v2
from threading import Lock
from typing import Dict, Any, Union, List
import pandas as pd
import numpy as np
import yaml
from importlib.resources import files
from trade.helpers.Logging import setup_logger
from trade.helpers.pools import _change_global_stream_level
from EventDriven.dataclasses.timeseries import AtTimeOptionData, AtTimePositionData
from EventDriven.types import TradeID

logger = setup_logger("EventDriven.riskmanager.market_timeseries", stream_log_level="WARNING")
logger.info("Changing pools log level to WARNING for market_timeseries module")
_change_global_stream_level("WARNING")
SPECIAL_DIVIDENDS: Dict[str, List[Dict[str, Any]]] = {}
NO_SPECIAL_DIVIDENDS: List[str] = []


def load_special_divs():
    global SPECIAL_DIVIDENDS, NO_SPECIAL_DIVIDENDS
    loc = files("EventDriven.riskmanager").joinpath("special_dividends.yaml")
    with open(loc, "r") as f:
        info = yaml.safe_load(f)

    SPECIAL_DIVIDENDS = info["special_dividends"]
    NO_SPECIAL_DIVIDENDS = info["no_special_dividends"]


class BacktestTimeseries(BacktestRunMixin):
    """
    Class for managing and retrieving market timeseries data for options and positions during backtesting.
    """

    def __init__(self, _start: Union[datetime, str], _end: Union[datetime, str]):
        self.start_date = _start
        self.end_date = _end
        self._skip_calc_config = SkipCalcConfig(skip_columns=["Midpoint"])
        self.market_timeseries = MarketTimeseries(_start=self.start_date, _end=self.end_date)
        self.options_cache = load_riskmanager_cache(
            target="processed_option_data", clear_on_exit=True
        )  ## Cache to store processed option data, cleared on exit to avoid polluting the cache with potentially large data that might not be needed in future sessions.
        self.position_data_cache = load_riskmanager_cache(target="position_data")

        self.special_dividends = load_riskmanager_cache(target="special_dividend")
        self.splits = load_riskmanager_cache(target="splits_raw")
        self.adjusted_strike_cache = load_riskmanager_cache(target="adjusted_strike_cache")
        self.session_loaded_option_cache = load_riskmanager_cache(
            target="session_loaded_option_cache", create_on_missing=True, clear_on_exit=True
        )  ## Cache to store options that have been loaded during the session, to avoid repeated loading of the same option data during the session. Cleared on exit to avoid polluting the persistent cache.
        self.rf_timeseries = get_risk_free_rate_helper_v2()["annualized"]
        self.undl_timeseries_config = UndlTimeseriesConfig()
        self.option_price_config = OptionPriceConfig()
        self.lock = Lock()

        ## Private attrs
        self._backup_position_data_cache = load_riskmanager_cache(
            target="position_data_backup", clear_on_exit=True, create_on_missing=True
        )  ## Backup cache to store original position data before any adjustments, cleared on exit to avoid polluting the cache.
        self._loaded_special_dividends = {}

        ## Load special dividends info from yaml file into cache.
        ## Reloads on initialization to ensure we have the most up-to-date information, and to avoid issues with mutable global state if the module is reloaded during the session.
        load_special_divs()

    def skip(self, position_id: str, date: Union[datetime, str], column: str = "Midpoint") -> bool:
        """
        Check if a specific position should be skipped on a given date based on skip calculation configuration.
        """
        at_time = self.get_at_time_position_data(position_id, date)
        if at_time is None:
            return False
        skips_meta = at_time.skips
        skip = skips_meta.get(column.capitalize())

        return skip.get("skip_day", False)

    def pre_run_setup(self):
        """
        Pre-run setup for BacktestTimeseries. Currently, all necessary setup is done in __init__, so this method is a placeholder for any future setup steps that might be needed before the backtest run starts.
        """
        ## Clear session related caches
        self.session_loaded_option_cache.clear()
        self.position_data_cache.clear()
        self._backup_position_data_cache.clear()
        self.adjusted_strike_cache.clear()

    def _get_trade_id(self, trade_id: Union[str, TradeID]) -> TradeID:
        """Normalize a trade identifier to a TradeID instance.

        Args:
            trade_id: Trade identifier as a raw string or TradeID.

        Returns:
            Normalized TradeID instance.
        """
        return trade_id if isinstance(trade_id, TradeID) else TradeID(str(trade_id))

    def _get_associated_option_ticks(self, opttick: str) -> set[str]:
        """Get base and split/dividend-adjusted option ticks for a leg.

        Builds the associated option tick universe by replaying all valid
        split/dividend event boundaries across the backtest window.
        """
        associated_ticks: set[str] = {opttick}
        meta = parse_option_tick(opttick)

        start_dt = to_datetime(self.start_date)
        effective_end = min(to_datetime(self.end_date), to_datetime(meta["exp_date"]))

        try:
            splits = self.get_splits(meta["ticker"], bkt_end_date=effective_end)
        except Exception as exc:
            logger.warning(f"Could not load splits for {opttick}: {exc}")
            splits = []

        try:
            dividends = self.get_special_dividends(meta["ticker"])
        except Exception as exc:
            logger.warning(f"Could not load special dividends for {opttick}: {exc}")
            dividends = {}

        events: List[tuple[pd.Timestamp, float, str]] = []

        for split_date, split_factor in splits:
            split_date = to_datetime(split_date)
            if compare_dates.inbetween(split_date, start_dt, effective_end):
                events.append((split_date, float(split_factor), "SPLIT"))

        for div_date, div_amount in dividends.items():
            div_date = to_datetime(div_date)
            if compare_dates.inbetween(div_date, start_dt, effective_end):
                events.append((div_date, float(div_amount), "DIVIDEND"))

        events.sort(key=lambda item: item[0])
        if not events:
            return associated_ticks

        # Boundary model mirrors generate_option_data_for_trade behavior where
        # check_date partitions events into post-event and pre-event regimes.
        for boundary in range(len(events) + 1):
            adj_strike = float(meta["strike"])
            for idx, (_, factor, event_type) in enumerate(events):
                pre_event_regime = idx >= boundary
                if pre_event_regime:
                    if event_type == "SPLIT":
                        adj_strike /= factor
                    elif event_type == "DIVIDEND":
                        adj_strike -= factor
                else:
                    if event_type == "SPLIT":
                        adj_strike *= factor
                    elif event_type == "DIVIDEND":
                        adj_strike += factor

                associated_ticks.add(
                    generate_option_tick_new(
                        symbol=meta["ticker"],
                        strike=adj_strike,
                        right=meta["put_call"],
                        exp=meta["exp_date"],
                    )
                )

        return associated_ticks

    def delete_position_data(self, trade_id: Union[str, TradeID]) -> None:
        """Delete cached data for a single trade and its legs.

        Removes position-level data from the main and backup caches, then clears
        any session-loaded option data and option cache entries for the trade's
        individual legs.

        Args:
            trade_id: Trade identifier to delete.
        """
        trade_id_obj = self._get_trade_id(trade_id)

        self.position_data_cache.pop(str(trade_id_obj), None)
        self._backup_position_data_cache.pop(str(trade_id_obj), None)

        leg_option_ticks = {
            leg_meta[1] for leg_meta in trade_id_obj.legs if isinstance(leg_meta, (list, tuple)) and len(leg_meta) > 1
        }
        associated_leg_option_ticks: set[str] = set()
        for leg_tick in leg_option_ticks:
            associated_leg_option_ticks.update(self._get_associated_option_ticks(leg_tick))
        logger.info(
            f"Deleting position data for trade {trade_id_obj}. "
            f"Removing option data for legs and associated adjusted ticks: {associated_leg_option_ticks}"
        )

        for cache_key in list(self.session_loaded_option_cache.keys()):
            if isinstance(cache_key, tuple) and cache_key and cache_key[0] in associated_leg_option_ticks:
                self.session_loaded_option_cache.pop(cache_key, None)

        for option_tick in associated_leg_option_ticks:
            self.options_cache.pop(option_tick, None)

    def delete_all_positions_data(self) -> None:
        """Clear all cached position-level data.

        Removes both the active position cache and the backup cache.
        """
        self.position_data_cache.clear()
        self._backup_position_data_cache.clear()

    def delete_all_timeseries_data(self) -> None:
        """Clear all cached position and option timeseries data.

        Removes all position-level data plus session-loaded option data and the
        global option cache.
        """
        self.delete_all_positions_data()
        self.session_loaded_option_cache.clear()
        self.options_cache.clear()

    def set_splits(self, d):
        """
        Setter for splits
        """
        splits_dict = {}
        for k, v in d.items():
            splits_dict[k] = []
            for d in v:
                if compare_dates.inbetween(d[0], self.start_date, self.end_date):
                    splits_dict[k].append(d)
        return splits_dict

    def get_special_dividends(self, ticker: str) -> List[Dict[str, float]]:
        """
        Retrieve special dividend information for a given ticker.
         - If the ticker is in the no_special_dividends list, returns an empty list.
         - If the ticker is in the special_dividends list, returns the corresponding dividend information.
         - If the ticker is not found in either list, raises a ValueError.
        """
        load_special_divs()
        if ticker in self._loaded_special_dividends:
            return self._loaded_special_dividends[ticker]

        if ticker in NO_SPECIAL_DIVIDENDS:
            return {}
        elif ticker in SPECIAL_DIVIDENDS:
            divs = SPECIAL_DIVIDENDS[ticker]
            new_fmt = {}
            for d in divs:
                new_fmt[to_datetime(d["adjusted_ex_date"])] = d["amount"]
            return new_fmt
        else:
            raise ValueError(
                f"Ticker {ticker} not found in either special or no special dividends list. Please update the yaml file accordingly."
            )

    def get_list_of_splits(self, ticker: str) -> List[Dict[str, float]]:
        """
        Retrieve list of splits for a given ticker.
        """
        t = self.market_timeseries._get_chain_spot_timeseries(ticker, "1990-01-01")
        return list((t[t["split_ratio"] != 1]["split_ratio"]).items())

    def get_splits(self, ticker: str, bkt_end_date: pd.Timestamp):
        """
        Retrieve splits for a given ticker, updating the cache if necessary.
        """
        split = self.splits.get(ticker, None)
        if split is None:
            split = self.get_list_of_splits(ticker)
            self.splits[ticker] = {"split": split, "last_updated": pd.Timestamp.now()}
        else:
            last_updated = self.splits[ticker].get("last_updated", pd.Timestamp(0))
            if pd.Timestamp(bkt_end_date) > last_updated:
                split = self.get_list_of_splits(ticker)
                self.splits[ticker] = {"split": split, "last_updated": pd.Timestamp.now()}
        return self.splits[ticker]["split"]

    def get_option_data(self, opttick: str) -> pd.DataFrame:
        """
        Retrieve option data for a given option ticker.
        """
        if opttick in self.options_cache:
            logger.info(f"Option data for {opttick} found in options cache, returning cached data")
            return self.options_cache[opttick]

        raise KeyError(
            f"Option data for {opttick} not found in options cache. Please ensure option data is loaded before access."
        )

    def get_position_data(self, position_id: str) -> pd.DataFrame:
        """
        Retrieve position data for a given position ID.
        """
        d = self.position_data_cache.get(position_id, pd.DataFrame())
        if not d.empty:
            logger.info(f"Position data for {position_id} found in position data cache, returning cached data")
            d = self._backup_position_data_cache.get(
                position_id, d
            )  ## Return the backup data if it exists, otherwise return the original data. This allows us to retain the original position data before any adjustments, while still allowing for adjustments to be made to the position data in the main cache.
        return d

    def get_at_time_option_data(self, opttick: str, date: Union[datetime, str]) -> AtTimeOptionData:
        """
        Retrieve option data for a given option ticker at a specific date.
        """
        option_data = self.get_option_data(opttick)
        if option_data.empty:
            logger.critical(f"Option data for {opttick} not found in cache.")
            return None
        date = pd.to_datetime(date)
        if date not in option_data.index:
            logger.critical(f"Date {date} not found in option data for {opttick}.")
            return None
        row = option_data.loc[date]

        return AtTimeOptionData(
            opttick=opttick,
            date=date,
            close=row["Midpoint"],
            bid=row["Closebid"],
            ask=row["Closeask"],
            midpoint=row["Midpoint"],
            use_price=self.option_price_config.use_price,
            delta=row.get("Delta", np.nan),
            gamma=row.get("Gamma", np.nan),
            vega=row.get("Vega", np.nan),
            theta=row.get("Theta", np.nan),
        )

    def get_at_time_position_data(self, position_id: str, date: Union[datetime, str]) -> AtTimePositionData:
        """
        Retrieve position data for a given position ID at a specific date.
        """
        position_data = self.get_position_data(position_id)
        if position_data.empty:
            logger.critical(f"Position data for {position_id} not found in cache.")
            return None
        ## OPTIMIZATION: Accept pre-formatted date string to avoid repeated conversion
        if not isinstance(date, str):
            date = pd.to_datetime(date).strftime("%Y-%m-%d")
        if date not in position_data.index:
            logger.critical(f"Date {date} not found in position data for {position_id}.")
            return None
        row = position_data.loc[date]
        ## OPTIMIZATION: Dict comprehension for skip dict building (Task #4)
        skips = {
            col: {
                "skip_day": row.get(f"{col}_skip_day", False),
                "skip_day_count": row.get(f"{col}_skip_day_count", 0),
            }
            for col in self._skip_calc_config.skip_columns
        }
        return AtTimePositionData(
            position_id=position_id,
            date=date,
            close=row["Midpoint"],
            bid=row["Closebid"],
            skips=skips,
            ask=row["Closeask"],
            midpoint=row["Midpoint"],
            delta=row.get("Delta", np.nan),
            gamma=row.get("Gamma", np.nan),
            vega=row.get("Vega", np.nan),
            theta=row.get("Theta", np.nan),
        )

    def calculate_option_data(self, position_id: str, date: Union[datetime, str]) -> Dict[str, Any]:
        """
        Calculate Greeks for a given position at a specific date.

        If position data is already cached, returns cached data immediately.
        Otherwise, calculates greeks, applies skip column adjustments, caches, and returns.
        """
        import time

        logger.info(
            f"Calculate Greeks Dates Start: {self.start_date}, End: {self.end_date}, Position ID: {position_id}, Date: {date}"
        )

        ## Check cache first - early return if data exists
        d = self.get_position_data(position_id)
        if not d.empty and pd.to_datetime(date) in d.index:
            logger.info(f"Position Data for {position_id} already available in cache, returning cached data")
            return d

        ## Data not in cache - perform full calculation
        logger.info(f"Position Data for {position_id} not available, calculating greeks. Load time ~5 minutes")

        ## Initialize the Long and Short Lists
        long = []
        short = []
        thread_input_list = [[], []]

        date = pd.to_datetime(date)  ## Ensure date is in datetime format

        ## First get position info
        position_dict, positon_meta = parse_position_id(position_id)

        ## Now ensure that the spot and dividend data is available
        for p in position_dict.values():
            for s in p:
                ticker = swap_ticker(s["ticker"])
        start_time = time.time()

        ## TODO: Take this out, instead, use the info from loaded per leg option timeseries to get spot, dividend, etc. It's redundant to load the entire timeseries just to get the spot and dividend data for the position, when we can just get it from the option timeseries that we will be loading for the position anyway. This is especially important if we are calculating the greeks for a single point in time, as we don't need the entire timeseries for that.
        timeseries_data = self.market_timeseries.get_timeseries(sym=ticker)
        logger.info(f"Timeseries loading for {ticker} took {time.time() - start_time:.2f} seconds")

        @timeit
        def get_timeseries(_id, direction):
            logger.info("Calculate Greeks dates")
            logger.info(f"Start Date: {self.start_date}")
            logger.info(f"End Date: {self.end_date}")

            logger.info(f"Calculating Greeks for {_id} on {date} in {direction} direction")
            with self.lock:
                data = self.generate_option_data_for_trade(_id, date)  ## Generate the option data for the trade

            if direction == "L":
                long.append(data)
            elif direction == "S":
                ask = data["Closeask"]
                bid = data["Closebid"]

                ## Swap bid and ask for short positions to reflect the perspective of the position holder
                data["Closeask"] = bid
                data["Closebid"] = ask
                short.append(data)
            else:
                raise ValueError(f"Position Type {_set[0]} not recognized")

            return data

        ## Calculating IVs & Greeks for the options
        for _set in positon_meta:
            thread_input_list[0].append(_set[1])  ## Append the option id to the thread input list
            thread_input_list[1].append(_set[0])  ## Append the direction to the thread input list

        start_time = time.time()
        runThreads(
            get_timeseries, thread_input_list, block=True
        )  ## Run the threads to get the timeseries data for the options
        print(f"Threads execution took {time.time() - start_time:.2f} seconds")
        position_data = sum(long) - sum(short)
        position_data = position_data[~position_data.index.duplicated(keep="first")]
        position_data.columns = [x.capitalize() for x in position_data.columns]

        ## Retain the spot, risk free rate, and dividend yield for the position, after the greeks have been calculated & spread values subtracted
        position_data["s0_close"] = timeseries_data.spot["close"]
        position_data["s"] = timeseries_data.chain_spot["close"]
        position_data["r"] = self.rf_timeseries
        position_data["y"] = timeseries_data.dividends
        position_data["spread"] = position_data["Closeask"] - position_data["Closebid"]

        ## Apply skip columns adjustment (only on newly calculated data)
        position_data = self._skip_columns_adjustment(position_data=position_data, position_id=position_id)
        logger.info(f"Completed calculation of Greeks for Position ID: {position_id}")

        ## Cache the position data
        self.position_data_cache[position_id] = position_data
        self._backup_position_data_cache[position_id] = (
            position_data.copy()
        )  ## Store a copy of the original position data in the backup cache before any adjustments are made, to ensure we have the original data available for future reference if needed.

        return position_data

    def _skip_columns_adjustment(
        self,
        position_data: pd.DataFrame,
        position_id: str,
    ) -> pd.DataFrame:
        """
        Apply skip columns adjustment to the position data based on the skip calculation configuration.
        """

        ## PRICE_ON_TO_DO: No need to change
        ## Add the additional columns to the position data
        skip_columns = self._skip_calc_config.skip_columns
        remove_from_skip = [x for x in skip_columns if x not in position_data.columns]

        ## Remove columns not in position_data
        for col in remove_from_skip:
            logger.critical(
                f"Col: `{col}` cannot be processed in skip_columns adjustment. It is not a member column in position_data"
            )
            skip_columns.remove(col)

        position_data["spread_ratio"] = (
            (position_data["spread"] / position_data["Midpoint"]).abs().replace(np.inf, np.nan).fillna(0)
        )  ## Spread ratio is the spread divided by the midpoint price
        position_data = add_skip_columns(
            df=position_data,
            id=position_id,
            skip_columns=skip_columns,
            window=self._skip_calc_config.window,
            skip_threshold=self._skip_calc_config.skip_threshold,
        )

        ## Loop through the skip columns and apply the skip logic
        for col in skip_columns:
            combined = pd.Series([False] * len(position_data), index=position_data.index)

            ## Check if individual skip is enabled and apply the skip logic
            if self._skip_calc_config.skip_enabled:
                if self._skip_calc_config.abs_zscore_threshold:
                    combined = combined | position_data[f"{col}_abs_zscore"]

                if self._skip_calc_config.pct_zscore_threshold:
                    combined = combined | position_data[f"{col}_pct_zscore"]

                if self._skip_calc_config.spike_flag:
                    combined = combined | position_data[f"{col}_spike_flag"]

                if self._skip_calc_config.std_window_bool:
                    combined = combined | position_data[f"{col}_window"]

                if self._skip_calc_config.zero_filter:
                    combined = combined | position_data[f"{col}_zero"]

                position_data[f"{col}_skip_day"] = combined
            else:
                position_data[f"{col}_skip_day"] = False

            position_data[f"{col}_skip_day_count"] = position_data[f"{col}_skip_day"].rolling(60).sum()

        if self._skip_calc_config.add_columns:
            for col in self._skip_calc_config.add_columns:
                func = ADD_COLUMNS_FACTORY.get(col[1], None)
                if func is None:
                    logger.critical(
                        f"Function {col[1]} not found in ADD_COLUMNS_FACTORY. Skipping addition of column {col[0]}_{col[1]}"
                    )
                    continue
                position_data[f"{col[0]}_{col[1]}".capitalize()] = func(position_data[col[0]])

        ## Final clean up to ensure no infinite or NaN values in Midpoint after adjustments, and forward fill any missing values to maintain continuity for skip calculations
        ## NOTE: RETURN TO THIS. IDK IF THIS IS A GOOD IDEA
        position_data["Midpoint"] = position_data["Midpoint"].replace(0, np.nan).ffill()

        return position_data

    def load_position_data(self, opttick) -> pd.DataFrame:  # noqa
        """
        Load position data for a given option tick.

        This function ONLY retrives the data for the option tick, it does not apply any splits or adjustments.
        This function will NOT check for splits or special dividends. It will only retrieve the data for the given option tick.
        """
        ## Get Meta
        meta = parse_option_tick(opttick)
        self.market_timeseries.load_timeseries(sym=meta["ticker"])

        data = load_position_data_new(
            opttick=opttick, processed_option_data=self.options_cache, start=self.start_date, end=self.end_date
        )
        return data

    def _ffill_adj_strike_business_days(
        self,
        strike_series: pd.Series,
        start_date: Union[datetime, str],
        end_date: Union[datetime, str],
    ) -> pd.Series:
        """Forward-fill adjusted strike values across business days only."""
        if strike_series is None or strike_series.empty:
            return strike_series

        series = strike_series.copy()
        series.index = pd.DatetimeIndex(to_datetime(series.index)).normalize()
        series = series[~series.index.duplicated(keep="last")].sort_index()

        bday_index = pd.bdate_range(start=to_datetime(start_date), end=to_datetime(end_date))
        return series.reindex(bday_index).ffill()

    def generate_option_data_for_trade(self, opttick, check_date) -> pd.DataFrame:
        """
        Generate option data for a given trade.
        This function retrieve the option data to backtest on. Data will not be saved, as it will be applying splits and adjustments.
        This function is written with the assumption that there is no cummulative splits. Expectation is only one split per option tick.
            Obviously, this might not be the case if the option was alive for ~5 years or more. But most options are not alive for that long.
        """
        key = (opttick, to_datetime(check_date).strftime("%Y-%m-%d"))
        all_optticks = []
        if key in self.session_loaded_option_cache:
            logger.info(
                f"Option data for {opttick} on {check_date} already generated in session, returning cached data"
            )
            return self.session_loaded_option_cache[key]

        meta = parse_option_tick(opttick)
        exp = to_datetime(meta["exp_date"])
        effective_end = min(
            to_datetime(self.end_date), exp
        )  ## Ensure that we load data at least until the expiration date of the option, even if it is after the backtest end date, to account for any splits or dividends that might happen until expiration.

        ## Check if there's any split/special dividend
        splits = self.get_splits(meta["ticker"], bkt_end_date=effective_end)
        dividends = self.get_special_dividends(meta["ticker"])
        to_adjust_split = []

        ## To avoid loading multiple data to account for splits everytime, we check if the PM_date range includes the split date
        for pack in splits:
            if compare_dates.inbetween(
                pack[0],
                # self.start_date,
                check_date,
                effective_end,
            ):
                pack = list(pack)  ## Convert to list to append later
                pack.append("SPLIT")
                to_adjust_split.append(pack)

        for pack in dividends.items():
            if compare_dates.inbetween(
                pack[0],
                # self.start_date,
                check_date,
                effective_end,
            ):
                pack = list(pack)
                pack.append("DIVIDEND")
                to_adjust_split.append(pack)

        ## Sort the splits by date
        to_adjust_split.sort(key=lambda x: x[0])  ## Sort by date
        logger.info(
            f"Splits and Dividends to adjust for {opttick}: {to_adjust_split} range: {self.start_date} to {self.end_date}"
        )

        ## If there are no splits, we can just load the data
        if not to_adjust_split:
            logger.info(f"No splits or dividends to adjust for {opttick}, loading data directly")
            data = self.load_position_data(opttick).copy()  ## Copy to avoid modifying the original data
            data["adj_strike"] = meta["strike"]
            data["factor"] = 1.0
            window_start = to_datetime(self.start_date) - relativedelta(months=3)
            window_end = to_datetime(effective_end) + relativedelta(months=3)
            data = data[(data.index >= window_start) & (data.index <= window_end)]
            self.adjusted_strike_cache[opttick] = self._ffill_adj_strike_business_days(
                data["adj_strike"],
                start_date=window_start,
                end_date=window_end,
            )
            self.session_loaded_option_cache[key] = (
                data  ## Cache the loaded data for the session to avoid re-loading it if the same option and date is requested again during the session
            )
            self._option_data_sanity_check(
                data=data, associated_optticks={opttick}, check_date=check_date
            )
            return data

        # If there are splits, we need to load the data for each tick after adjusting strikes
        else:
            logger.info(f"Generating data for {opttick} with splits: {to_adjust_split}")
            adj_meta = meta.copy()
            adj_strike = meta["strike"]
            logger.info(f"Generating data for {opttick} with splits: {to_adjust_split}")
            segments = []

            for event_date, factor, event_type in to_adjust_split:
                if compare_dates.is_before(check_date, event_date):
                    # You're in the PRE-event regime
                    if event_type == "SPLIT":
                        adj_strike /= factor
                    elif event_type == "DIVIDEND":
                        adj_strike -= factor
                else:
                    # You're in the POST-event regime
                    if event_type == "SPLIT":
                        adj_strike *= factor
                    elif event_type == "DIVIDEND":
                        adj_strike += factor

                adj_opttick = generate_option_tick_new(
                    symbol=adj_meta["ticker"], strike=adj_strike, right=adj_meta["put_call"], exp=adj_meta["exp_date"]
                )
                logger.info(
                    f"Adjusted option tick: {adj_opttick} for event {event_type} on {event_date} with factor {factor}"
                )

                # Load adjusted data
                if adj_opttick not in self.options_cache:
                    adj_data = self.load_position_data(adj_opttick).copy()
                else:
                    adj_data = self.options_cache[adj_opttick]
                logger.info(f"Loaded data for adjusted option tick: {adj_opttick}")

                # Slice around the event
                if compare_dates.is_before(check_date, event_date):
                    adj_data = adj_data[adj_data.index >= event_date]
                else:
                    adj_data = adj_data[adj_data.index < event_date]

                adj_data["adj_strike"] = adj_strike
                adj_data["factor"] = factor

                # Apply price transformation if SPLIT
                ## PRICE_ON_TO_DO: No need to change this. These are necessary columns
                if event_type == "SPLIT":
                    cols = ["Midpoint", "Closeask", "Closebid"]
                    if compare_dates.is_before(check_date, event_date):
                        adj_data[cols] *= factor
                    else:
                        adj_data[cols] /= factor

                segments.append(adj_data)
                all_optticks.append(adj_opttick)

        base_data = self.load_position_data(opttick).copy()
        base_data["adj_strike"] = meta["strike"]  ## Original strike
        base_data["factor"] = 1.0
        all_optticks.append(opttick)

        first_event_date = to_adjust_split[0][0] if to_adjust_split else self.start_date
        if compare_dates.is_before(check_date, first_event_date):
            base_data = base_data[base_data.index < first_event_date]

        else:
            base_data = base_data[base_data.index >= first_event_date]

        segments.insert(0, base_data)
        final_data = pd.concat(segments).sort_index()
        final_data = final_data[~final_data.index.duplicated(keep="last")]

        ## Leave residual data outside the PM date range
        window_start = to_datetime(self.start_date) - relativedelta(months=3)
        window_end = to_datetime(effective_end) + relativedelta(months=3)
        final_data = final_data[(final_data.index >= window_start) & (final_data.index <= window_end)]
        self.adjusted_strike_cache[opttick] = self._ffill_adj_strike_business_days(
            final_data["adj_strike"],
            start_date=window_start,
            end_date=window_end,
        )
        

        self.session_loaded_option_cache[key] = (
            final_data  ## Cache the generated data for the session to avoid re-generating it if the same option and date is requested again during the session
        )
        self._option_data_sanity_check(
            data=final_data, associated_optticks=set(all_optticks), check_date=check_date
        )  ## Perform sanity check to ensure the generated option data includes the check date for all associated option ticks, and if not, clear the relevant cache entries to maintain cache integrity.
        return final_data
    
    def _option_data_sanity_check(
            self,
            data: pd.DataFrame,
            associated_optticks: set[str],
            check_date: Union[datetime, str],
    ) -> None:
        """Perform sanity checks on the option data for the associated option ticks."""
        if check_date not in data.index:
            logger.warning(f"Check date {check_date} not found in option data index for associated ticks: {associated_optticks}")
            
            ## Delete the cached data for the associated option ticks.
            for opttick in associated_optticks:
                if opttick in self.options_cache:
                    logger.warning(f"Deleting cached option data for {opttick} due to missing check date {check_date}")
                    self.options_cache.pop(opttick, None)
                session_key = (opttick, to_datetime(check_date).strftime("%Y-%m-%d"))
                if session_key in self.session_loaded_option_cache:
                    logger.warning(f"Deleting cached session option data for {opttick} on {check_date} due to missing check date in options cache")
                    self.session_loaded_option_cache.pop(session_key, None)
            
            ## Finally, raise an error to indicate the issue with the option data for the associated ticks.
            raise ValueError(f"Check date {check_date} not found in option data index for associated ticks: {associated_optticks}. Cached data for these ticks has been deleted to maintain cache integrity.")
