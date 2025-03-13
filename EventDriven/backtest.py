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
from trade.backtester_.utils.utils import *
import traceback


class OptionSignalBacktest():
    """
    Encapsulates the settings and components for carrying out an event-driven backtest
    """
    
    def __init__(self, trades: pd.DataFrame, initial_capital: int | float =100000 ) -> None:
        """
            trades: pd.DataFrame
                Dataframe of trades to be used for backtesting, necessary columns are EntryTime, ExitTime, EntryPrice, ExitPrice, EntryType, ExitType, Symbol
            initial_capital: int
                Initial capital to be used for backtesting
        """
        self.__construct_data(trades, initial_capital)
        
    def __construct_data(self, trades: pd.DataFrame, initial_capital: int) -> None: 
        self.start_date = pd.to_datetime(trades['EntryTime']).min()
        self.end_date = pd.to_datetime(trades['ExitTime']).max()
        self.bars_trades = trades
        self.initial_capital = initial_capital
        
        #initialize critical components 
        self.events = EventScheduler(self.start_date, self.end_date); 
        self.bars = HistoricTradeDataHandler(self.events, trades)
        self.strategy = OptionSignalStrategy(self.bars, self.events)
        self.risk_manager = RiskManager(self.bars, self.events, initial_capital)
        self.portfolio = OptionSignalPortfolio(self.bars, self.events, risk_manager=self.risk_manager, initial_capital= float(initial_capital))
        self.executor =  SimulatedExecutionHandler(self.events)
        self.logger = setup_logger('OptionSignalBacktest')
        self.risk_free_rate = 0.055
        
    def run(self):
        while True: ##Loop through the dates
            # Get current event queue
            if self.events.current_date is None: 
                self.logger.info("No more dates left.")
                print("No more dates left.")
                break
            
            self.logger.info(f"Processing events for {self.events.current_date}")
            current_event_queue = self.events.get_current_queue()
            event_count = 0

            # Process events for the current bar
            while True:  # Avoid blocking. Loops through the event queue
                try:
                    event = current_event_queue.get_nowait()
                    roll_action = ['CLOSE', 'OPEN'] ## Roll order to enforce full execution of close before opening a new position
                    action_idx = 0
                    roll_on_this_date = False ## Flag to determine if we should roll on this date, necessary to ensure that we close and open a position on the same date
                except emptyEventQueue:
                    self.logger.info(f"Event queue is empty, processed {event_count} event(s)")
                    print(f"Event queue is empty, processed {event_count} event(s)")
                   
                    # Update portfolio time index after processing all events
                    self.portfolio.update_timeindex()
                    
                    #advance scheduler queue to next date 
                    self.events.advance_date()
                    break
                except Exception as e:
                    self.logger.error(f"Error fetching event: {e}")
                    print(f"Error fetching event: {e}")
                    break

                if event:
                    event_count += 1
                    try:
                        self.logger.info(f"Processing event: {event}")
                        print(f"Processing event: {event.type}")

                        if event.type == EventTypes.MARKET.value:
                            self.portfolio.analyze_positions(event)
                        elif event.type == EventTypes.SIGNAL.value:
                            self.portfolio.analyze_signal(event)
                        elif event.type == EventTypes.ORDER.value:
                            self.executor.execute_order_randomized_slippage(event)
                        elif event.type == EventTypes.FILL.value:
                            self.portfolio.update_fill(event)
                        elif event.type == EventTypes.EXERCISE.value:
                            self.executor.execute_exercise(event)
                        elif event.type == EventTypes.ROLL.value:
                            if action_idx < len(roll_action):
                                print("\nPerforming Roll Operation\n")
                                self.__roll(event, current_event_queue)
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


