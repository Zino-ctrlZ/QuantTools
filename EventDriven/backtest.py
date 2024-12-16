
from dotenv import load_dotenv
load_dotenv()
import os
import sys
sys.path.append(
    os.environ.get('WORK_DIR')) #type: ignore
sys.path.append(
    os.environ.get('DBASE_DIR')) #type: ignore
from dbase.DataAPI.ThetaData import * #type: ignore
from dbase.database.SQLHelpers import * #type: ignore
import pandas as pd
from EventDriven.data import  HistoricTradeDataHandler
from EventDriven.event import *
from EventDriven.strategy import OptionSignalStrategy
from EventDriven.portfolio import OptionSignalPortfolio
from EventDriven.execution import SimulatedExecutionHandler
from queue import Queue
from trade.helpers.Logging import setup_logger
from trade.backtester_.utils.utils import *

##NOTE:
## - Create an `Assistant Portfolio Manager` that allows custom functionality
## - Create a `Risk Manager` that allows custom functionality
## - Include an option to trade on next days open/close or current days close
## - 
    ## - Eg: Picking the best option to trade/Structure/Cash position

## - Strategy Abstract class should have self.open.buy, self.open.sell, self.close.buy, self.close.sell
    ## - This will be to simplify SignalEvent generation
## - Strategy class currently only checks for signal True/False to generate SignalEvent. There needs to be a functionality to check if there is no active position as well
##   This is to avoid generating a SignalEvent when there is already an active position.

## - For Backtest, we can find a way to flip btwn a signal backtest and a backtest that uses the current price to generate a signal.

class OptionSignalBacktest():
    """
    Encapsulates the settings and components for carrying out an event-driven backtest
    """
    
    def __init__(self, trades: pd.DataFrame, initial_capital: int =100000 ) -> None:
        self.events = Queue(maxsize=0)  
        self.bars = HistoricTradeDataHandler(self.events, trades)
        self.strategy = OptionSignalStrategy(self.bars, self.events)
        self.portfolio = OptionSignalPortfolio(self.bars, self.events, initial_capital)
        self.executor =  SimulatedExecutionHandler(self.events)
        self.logger = setup_logger('OptionSignalBacktest')
        self.risk_free_rate = 0.055
        
    def run(self):
       while True: ## loops bars
        if self.bars.continue_backtest == True: 
            self.bars.update_bars()
            print(self.bars.get_latest_bars(''))
        else:
            self.trades  = self.portfolio.get_trades()
            self.logger.info('no more data to feed backtest')
            print('no more data to feed backtest')
            break
         while True: ## Loops Events in a bar
            try: 
                event = self.events.get(False)
            except Exception as e:
                self.logger.error(f'exception occured: {e}')
                print('exception occured: ', e)
                break
            else: 
                try: 
                    self.logger.info(f'event: {event.type}')
                    print('event ', event.type)
                    if event is not None: 
                        if event.type == 'MARKET':
                            self.strategy.calculate_signals()
                        elif event.type == 'SIGNAL':
                            self.portfolio.generate_naive_option_order(event)
                        elif event.type == 'ORDER':
                            self.logger.info(f'order: {event.option}')
                            self.executor.execute_order(event)
                        elif event.type == 'FILL':
                            self.logger.info(f'fill: {event.option}')
                            self.portfolio.update_fill(event)
                        else:
                            self.logger.error('event type not recognized')

                except Exception as e:  
                    self.logger.error(f'exception occured: {e}')
                    print('exception occured: ', e)
                    break                    
        
        self.portfolio.update_timeindex()
        
    
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


