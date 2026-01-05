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
from EventDriven.riskmanager.market_data import MarketTimeseries
from EventDriven._vars import load_riskmanager_cache, ADD_COLUMNS_FACTORY
from EventDriven.riskmanager.utils import (
    parse_position_id,
    swap_ticker,
    load_position_data,
    add_skip_columns,
)
from trade.helpers.decorators import timeit
from trade.helpers.threads import runThreads
from trade.helpers.helper import compare_dates, parse_option_tick, generate_option_tick_new
from EventDriven.configs.core import SkipCalcConfig, UndlTimeseriesConfig, OptionPriceConfig
from trade.assets.rates import get_risk_free_rate_helper
from threading import Lock
from typing import Dict, Any, Union
import pandas as pd
import numpy as np
from trade.helpers.Logging import setup_logger
from trade.helpers.pools import _change_global_stream_level
from EventDriven.dataclasses.timeseries import AtTimeOptionData, AtTimePositionData
from module_test.raw_code.DataManagers.DataManagers import set_skip_mysql_query

logger = setup_logger("EventDriven.riskmanager.market_timeseries", stream_log_level="WARNING")
logger.info("Changing pools log level to WARNING for market_timeseries module")
_change_global_stream_level("WARNING")


class BacktestTimeseries:
    """
    Class for managing and retrieving market timeseries data for options and positions during backtesting.
    """

    def __init__(self, _start: Union[datetime, str], _end: Union[datetime, str]):
        self.start_date = _start
        self.end_date = _end
        self._skip_calc_config = SkipCalcConfig(skip_columns=["Midpoint"])
        self.market_timeseries = MarketTimeseries(_start=self.start_date, _end=self.end_date)
        self.options_cache = load_riskmanager_cache(target="processed_option_data")
        self.position_data_cache = load_riskmanager_cache(target="position_data")
        self.special_dividends = load_riskmanager_cache(target="special_dividend")
        self.splits = load_riskmanager_cache(target="splits_raw")
        self.adjusted_strike_cache = load_riskmanager_cache(target="adjusted_strike_cache")
        self.rf_timeseries = get_risk_free_rate_helper()["annualized"]
        self.undl_timeseries_config = UndlTimeseriesConfig()
        self.option_price_config = OptionPriceConfig()
        self.lock = Lock()

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

    def get_option_data(self, opttick: str) -> pd.DataFrame:
        """
        Retrieve option data for a given option ticker.
        """
        return self.options_cache.get(opttick, pd.DataFrame())

    def get_position_data(self, position_id: str) -> pd.DataFrame:
        """
        Retrieve position data for a given position ID.
        """
        return self.position_data_cache.get(position_id, pd.DataFrame())

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
        """
        set_skip_mysql_query(True)

        logger.info(
            f"Calculate Greeks Dates Start: {self.start_date}, End: {self.end_date}, Position ID: {position_id}, Date: {date}"
        )
        if position_id in self.position_data_cache:
            logger.info(f"Position Data for {position_id} already available, skipping calculation")
            position_data = self.position_data_cache[position_id]
        else:
            logger.critical(f"Position Data for {position_id} not available, calculating greeks. Load time ~5 minutes")

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
            self.market_timeseries.load_timeseries(sym=ticker, interval=self.undl_timeseries_config.interval)
            timeseries_data = self.market_timeseries.get_timeseries(sym=ticker)

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
                    short.append(data)
                else:
                    raise ValueError(f"Position Type {_set[0]} not recognized")

                return data

            ## Calculating IVs & Greeks for the options
            for _set in positon_meta:
                thread_input_list[0].append(_set[1])  ## Append the option id to the thread input list
                thread_input_list[1].append(_set[0])  ## Append the direction to the thread input list

            runThreads(
                get_timeseries, thread_input_list, block=True
            )  ## Run the threads to get the timeseries data for the options
            position_data = sum(long) - sum(short)
            position_data = position_data[~position_data.index.duplicated(keep="first")]
            position_data.columns = [x.capitalize() for x in position_data.columns]

            ## Retain the spot, risk free rate, and dividend yield for the position, after the greeks have been calculated & spread values subtracted
            position_data["s0_close"] = timeseries_data.spot["close"]
            position_data["s"] = timeseries_data.chain_spot["close"]
            position_data["r"] = self.rf_timeseries
            position_data["y"] = timeseries_data.dividends
            position_data["spread"] = position_data["Closeask"] - position_data["Closebid"]

        ## Apply skip columns adjustment
        position_data = self._skip_columns_adjustment(position_data=position_data, position_id=position_id)
        logger.info(f"Completed calculation of Greeks for Position ID: {position_id}")

        ## Cache the position data
        self.position_data_cache[position_id] = position_data

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
        return position_data
        # self.position_data[position_id] = position_data

    def load_position_data(self, opttick) -> pd.DataFrame:
        """
        Load position data for a given option tick.

        This function ONLY retrives the data for the option tick, it does not apply any splits or adjustments.
        This function will NOT check for splits or special dividends. It will only retrieve the data for the given option tick.
        """
        ## Get Meta
        meta = parse_option_tick(opttick)
        self.market_timeseries.load_timeseries(sym=meta["ticker"], interval=self.undl_timeseries_config.interval)
        timeseries_data = self.market_timeseries.get_timeseries(sym=meta["ticker"])
        return load_position_data(
            opttick,
            self.options_cache,
            self.start_date,
            self.end_date,
            s=timeseries_data.chain_spot["close"],
            r=self.rf_timeseries,
            y=timeseries_data.dividends,
            s0_close=timeseries_data.spot["close"],
        )

    def generate_option_data_for_trade(self, opttick, check_date) -> pd.DataFrame:
        """
        Generate option data for a given trade.
        This function retrieve the option data to backtest on. Data will not be saved, as it will be applying splits and adjustments.
        This function is written with the assumption that there is no cummulative splits. Expectation is only one split per option tick.
            Obviously, this might not be the case if the option was alive for ~5 years or more. But most options are not alive for that long.
        """

        meta = parse_option_tick(opttick)

        ## Check if there's any split/special dividend
        splits = self.splits.get(meta["ticker"], [])
        dividends = self.special_dividends.get(meta["ticker"], {})
        to_adjust_split = []

        ## To avoid loading multiple data to account for splits everytime, we check if the PM_date range includes the split date
        for pack in splits:
            if compare_dates.inbetween(
                pack[0],
                # self.start_date,
                check_date,
                self.end_date,
            ):
                pack = list(pack)  ## Convert to list to append later
                pack.append("SPLIT")
                to_adjust_split.append(pack)

        for pack in dividends.items():
            if compare_dates.inbetween(
                pack[0],
                # self.start_date,
                check_date,
                self.end_date,
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
            self.adjusted_strike_cache[opttick] = data["adj_strike"]
            return data[
                (data.index >= pd.to_datetime(self.start_date) - relativedelta(months=3))
                & (data.index <= pd.to_datetime(self.end_date) + relativedelta(months=3))
            ]

        # If there are splits, we need to load the data for each tick after adjusting strikes
        else:
            logger.info(f"Generating data for {opttick} with splits: {to_adjust_split}")
            adj_meta = meta.copy()
            adj_strike = meta["strike"]
            logger.info(f"Generating data for {opttick} with splits: {to_adjust_split}")

            # ## Load the data for picked option first
            # first_set_data = self.load_position_data(opttick).copy()

            # ## If check_date is before the first split date, first_set_data is only up to the first split date
            # if compare_dates.is_before(check_date, to_adjust_split[0][0]):
            #     first_set_data = first_set_data[first_set_data.index < to_adjust_split[0][0]]

            # ## If check_date is after the first split date, first_set_data is only from the first split date onwards
            # else:
            #     first_set_data = first_set_data[first_set_data.index >= to_adjust_split[0][0]]

            # ## Add Strike to keep track of adjustments
            # print(f"Initial adj_strike for {opttick}: {adj_strike}")
            # first_set_data['adj_strike'] = adj_strike
            # first_set_data['factor'] = 1.0
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

        base_data = self.load_position_data(opttick).copy()
        base_data["adj_strike"] = meta["strike"]  ## Original strike
        base_data["factor"] = 1.0

        first_event_date = to_adjust_split[0][0] if to_adjust_split else self.start_date
        if compare_dates.is_before(check_date, first_event_date):
            base_data = base_data[base_data.index < first_event_date]

        else:
            base_data = base_data[base_data.index >= first_event_date]

        segments.insert(0, base_data)
        final_data = pd.concat(segments).sort_index()
        final_data = final_data[~final_data.index.duplicated(keep="last")]

        ## Leave residual data outside the PM date range
        final_data = final_data[
            (final_data.index >= pd.to_datetime(self.start_date) - relativedelta(months=3))
            & (final_data.index <= pd.to_datetime(self.end_date) + relativedelta(months=3))
        ]
        self.adjusted_strike_cache[opttick] = final_data["adj_strike"]
        return final_data
