from queue import Empty as emptyEventQueue
from typing import Dict, Optional, cast
import pandas as pd
from EventDriven.data import HistoricTradeDataHandler
from EventDriven.event import Event
from EventDriven.strategy import OptionSignalStrategy
from EventDriven.new_portfolio import OptionSignalPortfolio
from EventDriven.execution import SimulatedExecutionHandler
from EventDriven.riskmanager.new_base import RiskManager
from EventDriven.eventScheduler import EventScheduler
from trade.backtester_._multi_asset_strategy import MultiAssetStrategy
from trade.helpers.Logging import setup_logger
from trade.helpers.helper import change_to_last_busday, is_USholiday
from EventDriven.helpers import generate_signal_id
from copy import deepcopy # noqa
import traceback
from pandas.tseries.offsets import BDay
from EventDriven.types import EventTypes, SignalTypes
from EventDriven.configs.core import BacktesterConfig

LOGGER = setup_logger("EventDriven.backtest", stream_log_level="WARNING")


class OptionSignalBacktest:
    """
    Event-driven backtesting engine for option trading strategies.

    This class orchestrates the complete backtesting workflow by coordinating multiple components:
    - Data handling (HistoricTradeDataHandler)
    - Strategy signal generation (OptionSignalStrategy)
    - Risk management (RiskManager)
    - Portfolio management (OptionSignalPortfolio)
    - Order execution with slippage simulation (SimulatedExecutionHandler)
    - Event scheduling and processing (EventScheduler)

    The backtester processes trades chronologically through an event queue, simulating realistic
    market conditions including entry/exit timing, slippage, position sizing, and risk limits.

    Key Features:
        - T+N settlement adjustments for realistic entry/exit timing
        - Configurable slippage range for execution realism
        - Position-level P&L tracking and attribution
        - Risk management with position limits and exposure controls
        - Support for both options and equity positions
        - Detailed logging and error handling

    Workflow:
        1. Initialize with trade data and configuration
        2. Call run() to execute the backtest
        3. Access results via portfolio.ledger, portfolio.holdings, etc.
        4. Use clean_run() to backtest with different parameters

    Attributes:
        config (BacktesterConfig): Configuration settings for the backtest
        bars (HistoricTradeDataHandler): Market data handler
        strategy (OptionSignalStrategy): Strategy signal generator
        portfolio (OptionSignalPortfolio): Portfolio state manager
        risk_manager (RiskManager): Risk controls and validation
        executor (SimulatedExecutionHandler): Order execution simulator
        eventScheduler (EventScheduler): Event queue coordinator
        start_date (date): Backtest start date
        end_date (date): Backtest end date
        initial_capital (float): Starting portfolio value
        unadjusted_trades (pd.DataFrame): Original trades before T+N adjustment

    Example:
        >>> # Basic usage
        >>> trades_df = pd.DataFrame({
        ...     'Ticker': ['AAPL', 'AAPL'],
        ...     'EntryTime': ['2024-01-02', '2024-01-03'],
        ...     'ExitTime': ['2024-01-05', '2024-01-08'],
        ...     'Size': [100, -100],
        ...     'EntryPrice': [150.0, 155.0],
        ...     'ExitPrice': [155.0, 150.0]
        ... })
        >>>
        >>> config = BacktesterConfig(t_plus_n=1, max_slippage_pct=0.001)
        >>> backtest = OptionSignalBacktest(
        ...     trades=trades_df,
        ...     initial_capital=100000,
        ...     config=config
        ... )
        >>> backtest.run()
        >>>
        >>> # Access results
        >>> ledger = backtest.portfolio.ledger
        >>> final_value = backtest.portfolio.current_holdings['total']

    See Also:
        BacktesterConfig: Configuration dataclass for backtest settings
        OptionSignalPortfolio: Portfolio management and P&L tracking
        RiskManager: Risk controls and position validation
    """

    def __init__(
        self,
        trades: pd.DataFrame = None,
        initial_capital: int | float = 100000,
        symbol_list=None,
        *,
        eq_strategy: Optional[MultiAssetStrategy] = None,
        config: Optional[BacktesterConfig] = None,
        end_date: Optional[pd.Timestamp] = None,
    ) -> None:
        """
        Initialize the backtesting engine with trade data and configuration.

        This constructor sets up all necessary components for the backtest including data handlers,
        portfolio manager, risk controls, and event scheduler. It preprocesses trades to handle
        T+N settlement adjustments and generates unique signal IDs for tracking.

        Args:
            trades (pd.DataFrame):
                DataFrame containing trade signals to backtest. Must include the following columns:
                - 'Ticker' or 'Symbol': Stock/option ticker symbol
                - 'EntryTime': Trade entry timestamp (str or datetime)
                - 'ExitTime': Trade exit timestamp (str or datetime, can be NaT for open positions)
                - 'Size': Position size (positive for long, negative for short)
                - 'EntryPrice': Entry execution price
                - 'ExitPrice': Exit execution price (can be NaN for open positions)

                Optional columns:
                - 'signal_id': Unique identifier for each signal (auto-generated if missing)
                - 'EntryType': Order type at entry (e.g., 'MKT', 'LMT')
                - 'ExitType': Order type at exit

            initial_capital (int | float, optional):
                Starting portfolio value in dollars. Defaults to 100000.
                Used for position sizing and P&L calculations.

            symbol_list (list, optional):
                List of symbols to track. If None, extracted from trades DataFrame.
                Useful for including symbols that may appear later in the backtest.

            eq_strategy (MultiAssetStrategy, optional):
                Equity strategy instance. When provided, trades DataFrame is ignored and
                signals are generated from the strategy.

            config (BacktesterConfig, optional):
                Configuration object controlling backtest behavior. If None, uses defaults.
                Key configuration options:
                - t_plus_n (int): Settlement delay in business days (0 or 1)
                - max_slippage_pct (float): Maximum slippage as % of price (e.g., 0.001 = 0.1%)
                - min_slippage_pct (float): Minimum slippage as % of price
                - finalize_trades (bool): Whether to finalize incomplete trades
                - raise_errors (bool): Whether to raise exceptions or log them

            end_date (pd.Timestamp, optional):
                Override backtest end date. If None, uses the latest ExitTime in trades.
                Useful for extending backtests beyond the last trade exit.

        Raises:
            TypeError: If config is not a BacktesterConfig instance or None
            ValueError: If trades DataFrame is empty
            ValueError: If t_plus_n is not 0 or 1
            ValueError: If both trades and eq_strategy are provided

        Notes:
            - Trades are automatically adjusted for T+N settlement if configured
            - Signal IDs are auto-generated using ticker, entry time, and direction
            - Start date is set to 1 business day before earliest entry
            - All timestamps are converted to business days (skipping weekends/holidays)
            - Original unadjusted trades are preserved in self.unadjusted_trades

        Example:
            >>> # With custom configuration
            >>> config = BacktesterConfig(
            ...     t_plus_n=1,  # Next-day settlement
            ...     max_slippage_pct=0.002,  # 0.2% max slippage
            ...     finalize_trades=True
            ... )
            >>> backtest = OptionSignalBacktest(
            ...     trades=my_trades_df,
            ...     initial_capital=500000,
            ...     symbol_list=['AAPL', 'MSFT', 'GOOGL'],
            ...     config=config,
            ...     end_date=pd.Timestamp('2024-12-31')
            ... )
            >>>
            >>> # Equity strategy mode
            >>> backtest_eq = OptionSignalBacktest(
            ...     eq_strategy=my_multi_asset_strategy,
            ...     initial_capital=500000,
            ...     end_date=pd.Timestamp('2024-12-31')
            ... )

        See Also:
            BacktesterConfig: For detailed configuration options
            clean_run(): To re-run backtest with different parameters
        """

        if eq_strategy is not None and trades is not None:
            raise ValueError("Cannot provide both trades DataFrame and eq_strategy. Please choose one.")
        self.is_eq_strategy = eq_strategy is not None
        self.eq_strategy: Optional[MultiAssetStrategy] = eq_strategy
        self.strategy: Optional[OptionSignalStrategy] = None
        self.initial_capital = initial_capital
        self.trades = trades
        self.symbol_list = symbol_list

        ## Tracker for eq_strategy runs, to ensure we only run the strategy once per date
        self.run_dates: Dict[pd.Timestamp, bool] = {}
        if config is not None and not isinstance(config, BacktesterConfig):
            raise TypeError("config must be an instance of BacktesterConfig or None")

        self.config: BacktesterConfig = cast(
            BacktesterConfig,
            config if config is not None else BacktesterConfig(),
        )
        self.end_date = end_date
        if self.is_eq_strategy:
            self.logger.info("Initializing backtest with equity strategy. Trades DataFrame will be ignored.")
            self.__init__with_equity_strategy(eq_strategy, cash=initial_capital)
        else:
            self.logger.info("Initializing backtest with trades DataFrame. Equity strategy will not be used.")
            if trades is None or trades.empty:
                raise ValueError("Trades DataFrame cannot be None or empty when not using an equity strategy.")
            self.__init__with_trades(trades, initial_capital, symbol_list, end_date=end_date)

    def __init__with_equity_strategy(
        self,
        eq_strategy: MultiAssetStrategy,
        cash: int | float = 100000,
    ) -> None:
        """
        Initializes the backtest using an equity strategy. This method sets up the backtest components based on the provided MultiAssetStrategy instance.
        Args:
            eq_strategy (MultiAssetStrategy): An instance of MultiAssetStrategy containing the strategy logic and data.
            cash (int | float, optional): Initial capital for the backtest. Defaults to 100000.
        Raises:
            AssertionError: If eq_strategy is not provided, not an instance of MultiAssetStrategy, or if end_date is not provided.
            EVBacktestError: If tplusn value of the equity strategy does not match the backtest configuration.
        """
        assert eq_strategy is not None, "Equity strategy must be provided for this initialization method"
        assert isinstance(eq_strategy, MultiAssetStrategy), (
            f"eq_strategy must be an instance of MultiAssetStrategy, got {type(eq_strategy)}"
        )
        assert self.end_date is not None, "end_date must be provided for this initialization method"

        ## We will not use trades dataframe in this process.
        self.start_date = pd.to_datetime(eq_strategy.start_date).date()
        if is_USholiday(self.start_date):
            self.logger.warning(f"Start date {self.start_date} is a US holiday. Adjusting to previous business day.")
            self.start_date = change_to_last_busday(self.start_date, -1).date()

        start_date, end_date = self.start_date, self.end_date
        self.eq_strategy.reset_strategies()

        ## Initialize critical components
        self.eventScheduler = EventScheduler(start_date, end_date)
        self.bars = HistoricTradeDataHandler(
            self.eventScheduler,
            trades_df=pd.DataFrame(),  # No initial trades, will be generated by strategy
            symbol_list=list(eq_strategy.data.keys()),
            finalize_trades=self.config.finalize_trades,
            start_date=start_date,
            end_date=end_date,
        )
        self.executor = SimulatedExecutionHandler(self.eventScheduler)
        self.risk_manager = RiskManager(
            symbol_list=self.bars.symbol_list,
            bkt_start=start_date,
            bkt_end=end_date,
            initial_capital=cash,
        )
        self.portfolio = OptionSignalPortfolio(
            self.bars,
            self.eventScheduler,
            risk_manager=self.risk_manager,
            initial_capital=float(cash),
            eq_strategy=eq_strategy,
        )
        self.events = []

    def __init__with_trades(
        self,
        trades: pd.DataFrame,
        initial_capital: int | float = 100000,
        symbol_list=None,
        *,
        end_date: Optional[pd.Timestamp] = None,
    ) -> None:
        ## Initialize trades dataframe. Trades to be preprocessed to handle t_plus_n logic and unadjusted trades
        ## to be stored for reference
        trades = trades.copy()
        unadjusted = trades.copy()
        if trades.empty:
            raise ValueError("Trades DataFrame cannot be empty. Please provide valid trade data.")
        trades = self.__handle_t_plus_n(trades)
        if "signal_id" not in trades.columns:
            self.logger.info("Generating 'signal_id' for trades DataFrame")
            unadjusted["signal_id"] = trades.apply(
                lambda row: generate_signal_id(
                    row["Ticker"],
                    row["EntryTime"],
                    SignalTypes.LONG.value if row["Size"] > 0 else SignalTypes.SHORT.value,
                ),
                axis=1,
            )
        else:
            self.logger.critical(
                "Trades DataFrame already contains 'signal_id' column. If this is unintended, please remove it to allow automatic generation."
            )
            unadjusted["signal_id"] = trades["signal_id"]
        unadjusted["unadjusted_signal_id"] = unadjusted.apply(
            lambda row: generate_signal_id(
                row["Ticker"], row["EntryTime"], SignalTypes.LONG.value if row["Size"] > 0 else SignalTypes.SHORT.value
            ),
            axis=1,
        )
        ## Store unadjusted trades for reference
        self.unadjusted_trades = unadjusted.copy()
        self.end_date = end_date
        self.__construct_data(trades, initial_capital, symbol_list)

    @property
    def logger(self):
        return LOGGER

    def __construct_data(self, trades: pd.DataFrame, initial_capital: int, symbol_list: list) -> None:
        ## Date range setup
        ## Move back a day if not business day
        self.start_date = change_to_last_busday(pd.to_datetime(trades["EntryTime"]).min() - BDay(1), 1).date()

        ## Move forward a day if not business day
        self.end_date = self.end_date or change_to_last_busday(pd.to_datetime(trades["ExitTime"]).max(), -1).date()

        ## Store trades and initial capital for clean runs
        self.bars_trades = trades
        self.initial_capital = initial_capital

        # initialize critical components
        self.eventScheduler = EventScheduler(self.start_date, self.end_date)
        self.bars = HistoricTradeDataHandler(
            self.eventScheduler,
            trades,
            symbol_list,
            finalize_trades=self.config.finalize_trades,
            end_date=self.end_date,
        )
        self.strategy = OptionSignalStrategy(self.bars, self.eventScheduler)
        self.executor = SimulatedExecutionHandler(self.eventScheduler)
        self.risk_manager = RiskManager(
            symbol_list=self.bars.symbol_list,
            bkt_start=self.start_date,
            bkt_end=self.end_date,
            initial_capital=initial_capital,
        )

        self.portfolio = OptionSignalPortfolio(
            self.bars, self.eventScheduler, risk_manager=self.risk_manager, initial_capital=float(initial_capital)
        )
        self.events = []

    def __handle_t_plus_n(self, trades: pd.DataFrame) -> pd.DataFrame:
        """
        Handles the t_plus_n logic for trades, adjusting entry and exit times based on the t_plus_n value.
        """
        if self.config.t_plus_n > 1:
            raise ValueError("t_plus_n must be either 0 or 1.")
        if self.config.t_plus_n > 0:
            self.logger.info(f"Adjusting EntryTime and ExitTime by {self.config.t_plus_n} business days")
            ## Adjust EntryTime and ExitTime by t_plus_n business days
            trades["EntryTime"] = trades["EntryTime"].apply(
                lambda x: change_to_last_busday(pd.to_datetime(x) + BDay(self.config.t_plus_n), -1).replace(hour=0)
            )  ## Adjust EntryTime by t_plus_n business days, and offseting to next business day if holiday

            ## Only adjust ExitTime if it is not NaT
            trades["ExitTime"] = trades["ExitTime"].apply(
                lambda x: (
                    change_to_last_busday(pd.to_datetime(x) + BDay(self.config.t_plus_n), -1).replace(hour=0)
                    if pd.notna(x)
                    else x
                )
            )  ## Adjust ExitTime by t_plus_n business days, and offseting to next business day if holiday
        return trades
    
    def reset(self):
        """
        Resets the backtest to its initial state, allowing for a fresh run with the original trades and configuration.
        This method reinitializes all components and clears any generated events or portfolio state.
        """
        self.logger.info("Resetting backtest to initial state.")
        self.__init__(
            trades=self.trades,
            initial_capital=self.initial_capital,
            eq_strategy=self.eq_strategy,
            config=self.config,
            end_date=self.end_date,
            symbol_list=self.symbol_list,
        )
    def _pre_signal_analysis(self):
        """
        Placeholder for any analysis or operations that need to be performed before signal processing in each loop iteration.
        This can include things like checking for roll conditions, updating market data, or any other pre-signal logic.
        """

        has_run_strategy = self.run_dates.get(self.eventScheduler.current_date, False)
        if self.is_eq_strategy and self.eq_strategy.tplusn == 0 and not has_run_strategy:
            ## For equity strategy, we want to run the strategy at the beginning of the loop before processing any events, 
            ## to ensure that we capture signals generated for the current date in the same loop iteration. If we put it after get_nowait, 
            ## we might miss signals generated for the current date until the next loop iteration, which could lead to delayed signal processing and execution
            self.logger.info(f"Running equity strategy with T+0 settlement on {self.eventScheduler.current_date}")
            self.portfolio.analyze_multiasset_strategy()
            self.run_dates[self.eventScheduler.current_date] = True
            self.logger.info(f"Completed running equity strategy for {self.eventScheduler.current_date}")
    
    def _post_signal_analysis(self):
        """
        Placeholder for any analysis or operations that need to be performed after signal processing in each loop iteration.
        This can include things like analyzing positions, updating portfolio metrics, or any other post-signal logic.
        """
        meta = self.portfolio.analyze_positions()  # noqa
        self.logger.info(f"Position Analysis Meta: {meta}")

        ## For equity strategy with T+n (n>=1), we want to run the strategy after analyzing positions, 
        ## to ensure that we are using the most up-to-date position information for the strategy analysis. 
        ## This is especially important for T+1 strategies, where the signals generated on the current date will only be actionable on the next business day. 
        ## By running the strategy after position analysis, we can ensure that any new signals generated based on the current positions and market data are captured and processed in a timely manner in the next loop iteration.
        if self.is_eq_strategy and self.eq_strategy.tplusn >= 1:
            self.logger.info(f"Running equity strategy with T+{self.eq_strategy.tplusn} settlement on {self.eventScheduler.current_date}")
            self.portfolio.analyze_multiasset_strategy()

    def run(self):
        ## Runtime configurations changes
        self.portfolio.t_plus_n = self.config.t_plus_n
        self.executor.max_slippage_pct = self.config.max_slippage_pct
        self.executor.min_slippage_pct = self.config.min_slippage_pct
        self.executor.commission_rate = self.config.commission_per_contract_in_units

        ## Begin backtest by looping through event scheduler dates
        while True:
            # Get current event queue
            if self.eventScheduler.current_date is None:
                self.logger.info("No more dates left.")
                print("No more dates left.")
                break

            self.logger.info(f"Processing events for {self.eventScheduler.current_date}")
            current_event_queue = self.eventScheduler.get_current_queue()
            event_count = 0
            _post_signal_ran = False

            # Process events for the current bar
            # Avoid blocking. Loops through the event queue
            while True:
                self._pre_signal_analysis()  # Placeholder for any pre-signal processing logic  

                try:
                    # ## Placing before get_nowait because I want to check for roll, and if there is no roll, I want to break out of the loop
                    # if len(list(deepcopy(current_event_queue.queue))) == 0:
                    #     meta = self.portfolio.analyze_positions()  # noqa
                    #     # print(f"Position Analysis Meta: {meta}")

                    ## Placing before get_nowait because I want to check for roll, and if there is no roll, I want to break out of the loop
                    if current_event_queue.empty() and not _post_signal_ran:
                        self.logger.info(f"Event queue is empty, processed {event_count} event(s)")

                        self._post_signal_analysis() 
                        _post_signal_ran = True

                    event = current_event_queue.get_nowait()

                except emptyEventQueue:
                    self.logger.info(f"Event queue is empty, processed {event_count} event(s)")

                    # Update portfolio time index after processing all events
                    self.portfolio.update_timeindex()

                    # advance scheduler queue to next date
                    self.eventScheduler.advance_date()
                    break
                except Exception as e:
                    if self.config.raise_errors:
                        raise e
                    self.logger.error(f"Error fetching event: {e}\n{traceback.format_exc()}")
                    break

                if event:
                    self.store_event(event)  # Store the event in the events list
                    event_count += 1
                    try:
                        self.logger.info(f"Processing event: {event}")

                        if event.type == EventTypes.SIGNAL.value:
                            self.portfolio.analyze_signal(event)
                        elif event.type == EventTypes.ORDER.value:
                            self.executor.execute_order_randomized_slippage(event)
                        elif event.type == EventTypes.FILL.value:
                            self.portfolio.update_fill(event)
                            self.portfolio.update_timeindex()
                        elif event.type == EventTypes.EXERCISE.value:
                            self.executor.execute_exercise(event)
                        elif event.type == EventTypes.ROLL.value:
                            self.logger.info("\nPerforming Roll Operation\n")
                            self.portfolio.execute_roll(event)

                        else:
                            self.logger.warning(f"Unrecognized event type: {event.type}")
                    except Exception as e:
                        if self.config.raise_errors:
                            raise e

                        self.logger.error(f"Error processing event: {e}\n{traceback.format_exc()}")
                        self.logger.error(f"Error processing event: {e}")

    def clean_run(self, trades: pd.DataFrame = None, initial_capital: int = None):
        """
        Rerun the backtest with fresh set of data, the only set of data that persists are the last set of trades and capital data passed to the backtest, unless new data is passed in this function
        """
        if trades is None:
            trades = pd.DataFrame()
        clean_trades = self.bars_trades if trades.empty else trades
        clean_capital = self.initial_capital if initial_capital is None else initial_capital
        self.__construct_data(clean_trades, clean_capital)
        self.run()

    def get_all_holdings(self) -> pd.DataFrame:
        """
        return timeseries of portfolio holdings
        """
        df = pd.DataFrame(self.portfolio.all_holdings)
        df.set_index("datetime", inplace=True)
        return df

    def get_all_positions(self) -> pd.DataFrame:
        """
        return timeseries of portfolio positions
        """
        pos_arr = []
        for position in self.portfolio.all_positions:
            pos_obj = {}
            pos_obj["AMD"] = position["AMD"]["option"]
            pos_obj["AAPL"] = position["AAPL"]["option"]
            pos_obj["MSFT"] = position["MSFT"]["option"]
            pos_obj["GOOGL"] = position["GOOGL"]["option"]
            pos_obj["datetime"] = position["datetime"]
            pos_arr.append(pos_obj)
        pos_df = pd.DataFrame(pos_arr)
        pos_df.set_index("datetime", inplace=True)
        return pos_df

    def store_event(self, event: Event):
        """
        Store an event in the events list
        """
        self.events.append(event.__dict__)

    def get_events(self) -> pd.DataFrame:
        """
        Returns a DataFrame of all events that have been processed during the backtest.
        """
        if not self.events:
            return pd.DataFrame()

        events_df = pd.DataFrame(self.events)
        events_df["datetime"] = pd.to_datetime(events_df["datetime"])
        events_df.set_index("datetime", inplace=True)
        return events_df
