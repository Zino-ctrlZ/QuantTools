from queue import Empty as emptyEventQueue
from dbase.DataAPI.ThetaData import * #type: ignore
from dbase.database.SQLHelpers import * #type: ignore
import pandas as pd
from EventDriven.data import  HistoricTradeDataHandler
from EventDriven.event import *
from EventDriven.strategy import OptionSignalStrategy
from EventDriven.portfolio import OptionSignalPortfolio
from EventDriven.execution import SimulatedExecutionHandler
from EventDriven.riskmanager import RiskManager
from EventDriven.eventScheduler import EventScheduler
from trade.helpers.Logging import setup_logger
from trade.helpers.helper import change_to_last_busday
from EventDriven.helpers import generate_signal_id
from trade.backtester_.utils.utils import *
from copy import deepcopy
import traceback
from pandas.tseries.offsets import BDay


class OptionSignalBacktest():
    """
    Encapsulates the settings and components for carrying out an event-driven backtest
    """
    
    def __init__(self, trades: pd.DataFrame, 
                 initial_capital: int | float =100000, 
                 t_plus_n: float = 0,
                 symbol_list = None,
                 finalize_trades: bool = True
                    ) -> None:
        """
            trades: pd.DataFrame
                Dataframe of trades to be used for backtesting, necessary columns are EntryTime, ExitTime, EntryPrice, ExitPrice, EntryType, ExitType, Symbol
            initial_capital: int
                Initial capital to be used for backtesting
            t_plus_n: int
                Number of business days to add to the entry and exit times of trades, defaults to 0, meaning no adjustment is made
        """
        self.logger = setup_logger('OptionSignalBacktest')
        self.t_plus_n = t_plus_n
        trades = trades.copy()
        unadjusted = trades.copy() ## Store unadjusted trades for reference
        if trades.empty:
            raise ValueError("Trades DataFrame cannot be empty. Please provide valid trade data.")
        trades = self.__handle_t_plus_n(trades)
        unadjusted['signal_id'] = trades.apply(lambda row: generate_signal_id(row['Ticker'], 
                                                                              row['EntryTime'], 
                                                                              SignalTypes.LONG.value if row['Size'] > 0 else SignalTypes.SHORT.value), 
                                                                              axis=1)
        unadjusted['unadjusted_signal_id'] = unadjusted.apply(lambda row: generate_signal_id(row['Ticker'], 
                                                                              row['EntryTime'], 
                                                                              SignalTypes.LONG.value if row['Size'] > 0 else SignalTypes.SHORT.value), 
                                                                              axis=1)
        self.unadjusted_trades = unadjusted.copy() ## Store unadjusted trades for reference
        self.finalize_trades = finalize_trades
        self.__construct_data(trades, initial_capital, symbol_list)
        
    def __construct_data(self, trades: pd.DataFrame, initial_capital: int, symbol_list: list) -> None: 
        self.start_date = pd.to_datetime(trades['EntryTime']).min() - BDay(1)
        self.end_date = pd.to_datetime(trades['ExitTime']).max()
        self.bars_trades = trades
        self.initial_capital = initial_capital
        
        #initialize critical components 
        self.eventScheduler = EventScheduler(self.start_date, self.end_date); 
        self.bars = HistoricTradeDataHandler(self.eventScheduler, trades, symbol_list)
        self.strategy = OptionSignalStrategy(self.bars, self.eventScheduler)
        self.executor =  SimulatedExecutionHandler(self.eventScheduler)
        self.risk_manager = RiskManager(self.bars, self.eventScheduler, initial_capital, self.start_date, self.end_date, self.executor, self.unadjusted_trades,t_plus_n= self.t_plus_n)
        self.portfolio = OptionSignalPortfolio(self.bars, self.eventScheduler, risk_manager=self.risk_manager, initial_capital= float(initial_capital))
        self.final_date = pd.to_datetime(list(self.eventScheduler.events_map)[-1]).strftime('%Y%m%d')
        self.risk_free_rate = 0.055
        self.events = []
        self.order_cache = {}
    
    def __handle_t_plus_n(self, trades: pd.DataFrame) -> pd.DataFrame:
        """
        Handles the t_plus_n logic for trades, adjusting entry and exit times based on the t_plus_n value.
        """
        if self.t_plus_n > 0:
            self.logger.info(f"Adjusting EntryTime and ExitTime by {self.t_plus_n} business days")
            trades['EntryTime'] = trades['EntryTime'].apply(lambda x: change_to_last_busday(pd.to_datetime(x) + BDay(self.t_plus_n), -1).replace(hour = 0)) ## Adjust EntryTime by t_plus_n business days, and offseting to next business day if holiday
            trades['ExitTime'] = trades['ExitTime'].apply(lambda x: change_to_last_busday(pd.to_datetime(x) + BDay(self.t_plus_n), -1).replace(hour = 0)) ## Adjust ExitTime by t_plus_n business days, and offseting to next business day if holiday
        elif self.t_plus_n > 1:
            raise ValueError("t_plus_n must be either 0 or 1.")
        return trades
        
    def run(self):
        while True: ##Loop through the dates
            # Get current event queue
            if self.eventScheduler.current_date is None: 
                self.logger.info("No more dates left.")
                print("No more dates left.")
                break
            
            self.logger.info(f"Processing events for {self.eventScheduler.current_date}")
            current_event_queue = self.eventScheduler.get_current_queue()
            event_count = 0

            # Process events for the current bar
            while True:  # Avoid blocking. Loops through the event queue
                try:
                    if len(list(deepcopy(current_event_queue.queue))) == 0: ## Placing before get_nowait because I want to check for roll, and if there is no roll, I want to break out of the loop
                        ## Analyze positions if theres no events in the queue, this happens before getting from the queue cause the process can add a roll event to the queue
                        actions = self.risk_manager.analyze_position() 
                        print("Risk Manager Actions: ", actions)

                    event = current_event_queue.get_nowait()

                except emptyEventQueue:
                    self.logger.info(f"Event queue is empty, processed {event_count} event(s)")
                    print(f"Event queue is empty, processed {event_count} event(s)")
                   
                    # Update portfolio time index after processing all events
                    
                    self.portfolio.update_timeindex()
                    
                    #advance scheduler queue to next date 
                    self.eventScheduler.advance_date()
                    break
                except Exception as e:
                    self.logger.error(f"Error fetching event: {e}\n{traceback.format_exc()}")
                    print(f"Error fetching event: {e}")
                    break

                if event:
                    self.store_event(event)  # Store the event in the events list
                    event_count += 1
                    try:
                        self.logger.info(f"Processing event: {event}")
                        print(f"Processing event: {event.type} {event.datetime}")

                        if event.type == EventTypes.SIGNAL.value:
                            if not self.finalize_trades and self.eventScheduler.current_date == self.final_date:
                                self.logger.info("Finalizing trades is disabled, skipping signal processing")
                                continue
                            self.portfolio.analyze_signal(event)
                        elif event.type == EventTypes.ORDER.value:
                            self.order_cache.setdefault(event.direction, {})\
                                .setdefault(event.datetime, []).append(event)
                            self.executor.execute_order_randomized_slippage(event)
                        elif event.type == EventTypes.FILL.value:
                            self.portfolio.update_fill(event)
                            self.portfolio.update_timeindex()
                        elif event.type == EventTypes.EXERCISE.value:
                            self.executor.execute_exercise(event)
                        elif event.type == EventTypes.ROLL.value:
                            print("\nPerforming Roll Operation\n")
                            self.portfolio.execute_roll(event)
                            # self.__roll(event, current_event_queue)
                        else:
                            self.logger.warning(f"Unrecognized event type: {event.type}")
                    except Exception as e:
                        self.logger.error(f"Error processing event: {e}\n{traceback.format_exc()}")
                        print(f"Error processing event: {e}")   

        
    def clean_run(self, trades: pd.DataFrame = pd.DataFrame(), initial_capital: int = None):
        """
            Rerun the backtest with fresh set of data, the only set of data that persists are the last set of trades and capital data passed to the backtest, unless new data is passed in this function
        """
        clean_trades = self.bars_trades if trades.empty else trades
        clean_capital = self.initial_capital if initial_capital == None else initial_capital
        self.__construct_data(clean_trades, clean_capital)
        self.run()

    def __roll(self, roll_event, current_event_queue) -> None:
        """
        Performs a roll in the same day by closing the current position and opening a new one.
        Closing Operation first executes a close order and fills, then opening operation executes an open order and fills
        """
        print("Using roll function")
        roll_action = ['CLOSE', 'OPEN']
        event_count = 0
        for action in roll_action: ## For each action, we want to carry out all processes
            # print(f"Processing {action} action")
            self.portfolio.execute_roll(roll_event, action) ## Execute the roll event
            event_count += 1
            
            while True: ##Starts event queue processing
                try: ## Gets current event from the queue for that date
                    event = current_event_queue.get_nowait()
                    
                except emptyEventQueue: 
                    ## If the queue is empty, we break out of the loop, and return to outer loop
                    ## If there is no actions in outta loop, we return control to the main loop
                    break
                ## Processes the event
                if event.type == EventTypes.MARKET.value:
                    self.portfolio.analyze_positions(event)
                elif event.type == EventTypes.SIGNAL.value:
                    self.portfolio.analyze_signal(event)
                elif event.type == EventTypes.ORDER.value:
                    self.executor.execute_order_randomized_slippage(event)
                elif event.type == EventTypes.FILL.value:
                    self.portfolio.update_fill(event)
        print(f"Roll processed {event_count} event(s)")
        self.logger.info(f"Roll Function processed {event_count} roll event(s)")


        
    def get_all_holdings(self) -> pd.DataFrame:
        """
            return timeseries of portfolio holdings
        """
        df = pd.DataFrame(self.portfolio.all_holdings)
        df.set_index('datetime', inplace=True)
        return df

    
    def get_all_positions(self) -> pd.DataFrame:
        """
            return timeseries of portfolio positions
        """
        pos_arr = []
        for position in self.portfolio.all_positions:
            pos_obj = {}
            pos_obj['AMD'] = position['AMD']['option']
            pos_obj['AAPL'] = position['AAPL']['option']
            pos_obj['MSFT'] = position['MSFT']['option']
            pos_obj['GOOGL'] = position['GOOGL']['option']
            pos_obj['datetime'] = position['datetime']
            pos_arr.append(pos_obj)
        pos_df = pd.DataFrame(pos_arr)
        pos_df.set_index('datetime', inplace=True)
        return pos_df


    def store_event(self,event: Event):
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
        events_df['datetime'] = pd.to_datetime(events_df['datetime'])
        events_df.set_index('datetime', inplace=True)
        return events_df