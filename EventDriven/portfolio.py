
from dotenv import load_dotenv
load_dotenv()
import os
import sys
sys.path.append(
    os.environ.get('WORK_DIR')) #type: ignore
sys.path.append(
    os.environ.get('DBASE_DIR')) #type: ignore

import pandas as pd

from dbase.DataAPI.ThetaData import list_contracts, retrieve_option_ohlc, is_theta_data_retrieval_successful #type: ignore
from trade.assets.Stock import Stock

from abc import ABCMeta, abstractmethod

from EventDriven.event import  FillEvent, OrderEvent, SignalEvent
from EventDriven.data import HistoricTradeDataHandler
from trade.helpers.Logging import setup_logger


class Portfolio(object):
    """
    The Portfolio class handles the positions and market
    value of all instruments at a resolution of a "bar",
    i.e. secondly, minutely, 5-min, 30-min, 60 min or EOD.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def update_signal(self, event):
        """
        Acts on a SignalEvent to generate new orders 
        based on the portfolio logic.
        """
        raise NotImplementedError("Should implement update_signal()")

    @abstractmethod
    def update_fill(self, event):
        """
        Updates the portfolio current positions and holdings 
        from a FillEvent.
        """
        raise NotImplementedError("Should implement update_fill()")
    
       
    
    
class OptionSignalPortfolio(Portfolio): 
    """
    The OptionSignalPortfolio object is designed to send orders to a brokerage.
    This class offers no risk management or position sizing as yet.
    The class also stores basic information about the portfolio.
    
    start_date - The start date of the portfolio. format: 'YYYY-MM-DD'
    """
    
    def __init__(self, bars : HistoricTradeDataHandler, events, initial_capital = 10000): 
        self.bars = bars
        self.events = events
        self.symbol_list = self.bars.symbol_list
        self.start_date = bars.start_date.strftime("%Y%m%d")
        self.initial_capital = initial_capital
        self.all_positions = self.construct_all_positions()
        self.current_positions = self.construct_current_positions()
        self.all_holdings = self.construct_all_holdings()
        self.current_holdings = self.construct_current_holdings()
        self._generate_underlier_data()
        self.max_option_budget = 0.1 * self.initial_capital
        self.logger = setup_logger('OptionSignalPortfolio')
        self.options_data = {}
            
    def construct_current_positions(self):
        d = dict((k, v) for k, v in [(s, {'quantity': 0.0, 'option': None}) for s in self.symbol_list])
        return d
    
    def construct_all_holdings(self):
        """
        Constructs the holdings list using the start_date
        to determine when the time index will begin.
        """
        d = dict( (k,v) for k, v in [(s, 0.0) for s in self.symbol_list] )
        d['datetime'] = self.bars.start_date
        d['cash'] = self.initial_capital
        d['commission'] = 0.0
        d['total'] = self.initial_capital
        return [d]
    
    def construct_all_positions(self): 
        d = dict((k, v) for k, v in [(s, {'quantity' : 0.0, 'option': None}) for s in self.symbol_list]) #key is underlier, value is list of option contracts
        d['datetime'] = self.bars.start_date
        return [d]
    
    def construct_current_holdings(self): 
        d = dict((k, v) for k, v in [(s, 0.0) for s in self.symbol_list]) #key is underlier, value is total value of all postitions held
        d['cash'] = self.initial_capital
        d['commission'] = 0.0
        d['total'] = self.initial_capital
        return d
    
    def _generate_underlier_data(self):
        self.underlier_list_data = {}
        for underlier in self.symbol_list:
            self.underlier_list_data[underlier] = Stock(underlier, run_chain = False)
    
    def generate_naive_option_order(self, signal : SignalEvent):
        """
        Takes a signal event and creates an order object
        The order has a default quantity of contracts at 1
        """
        self.__latest_signals = self.bars.get_latest_bars().iloc[0]
        order = None 
        symbol = signal.symbol
        direction = signal.signal_type
        quantity = 2; 
        order_type = 'MKT'
        contract = None
        
        if direction == 'LONG':
            underlier_stock = self.underlier_list_data[symbol]
            contract = self.generate_option_to_buy(underlier_stock)
            if contract is None:  
                self.logger.warning(f'No contracts found for {symbol} at {signal.datetime}')
                return None
            self.logger.info(f'Buying contract for {symbol} at {signal.datetime}')
            
            order = OrderEvent(symbol, signal.datetime ,order_type, quantity, 'BUY', option = self.get_option_id(contract)) #pass id instead of the option dataframe here
            self.events.put(order)
        if direction == 'SHORT':
            sell_contract_id = self.current_positions[symbol]['option']
            if sell_contract_id is None: 
                self.logger.warning(f'No contracts held for {symbol} to sell')
                return None
            self.logger.info(f'Selling contract for {symbol} at {signal.datetime}')
            
            order = OrderEvent(symbol, signal.datetime, order_type, quantity, 'SELL', option = sell_contract_id) #pass id instead of the option dataframe here
            self.events.put(order)
        #TODO: atm short means sell, but this should mean to buy a put option in other implementations 
        return order
            
            
    def update_signal(self, event):
        """
        Acts on a SignalEvent to generate new orders 
        based on the portfolio logic.
        """
        if event.type == 'SIGNAL':
            order_event = self.generate_naive_option_order(event)
            if order_event is not None:
                self.events.put(order_event)
              
            
    def generate_option_to_buy(self, underlier: Stock, contract_fetch_attempts = 0, invalid_contracts = []) -> pd.Series | None:
            """
            Buy an option based on the underlier.
            """
            time = self.__latest_signals['Date']
            next_day_time = time + pd.DateOffset(days=1)
            option_spot = underlier.spot(ts=True, ts_start = time, ts_end = next_day_time)
            option_spot = option_spot.iloc[0] #type: ignore
            stock_price = option_spot['open']#use open price as spot price on the assumption of making trades at start of day
            expiry_benchmark_lower = time + pd.DateOffset(months=5)
            expiry_benchmark_upper = expiry_benchmark_lower + pd.DateOffset(months=12)
            oom_price_lower = stock_price * (1 +0.1) #10% out of the money
            oom_price_upper = stock_price * (1 +0.4) #40% out of the money
            time_str = time.strftime("%Y%m%d")
            self.logger.info(f'getting contract for {underlier.ticker} at {time_str}')
            contracts = list_contracts(underlier.ticker, time_str)
            if (contracts is None) or (len(contracts.columns) == 1):
                self.logger.info(f'No contracts found for {underlier.ticker} at {time_str}')
                return None
            contracts = contracts[contracts['right'] == 'C'] #filter out puts        
    
            #Filter out contracts that are out of the money
            contracts = contracts[(contracts['strike'] >= oom_price_lower) & (contracts['strike'] <= oom_price_upper)]
            
            #filter out contracts that are  below the expiry benchmark
            contracts = contracts[(pd.to_datetime(contracts['expiration'], format="%Y%m%d") >= expiry_benchmark_lower) & (pd.to_datetime(contracts['expiration'], format="%Y%m%d") <= expiry_benchmark_upper)]
            
            #select a random contract to buy
            contract = contracts.sample(n=1); 
            contract = contract.iloc[0]
            
            #TODO: make this its own function that gets called in the generate_naive_option_order function
            #TODO: add volume and open interest as criteria for selecting the contract
            #check if the contract is valid
            if contract_fetch_attempts < 30:
                option_data = self.get_options_data_on_contract(contract)
                if option_data is not None and self.__is_valid_dataframe(option_data) and self.get_option_id(contract) not in invalid_contracts:
                    #store option data
                    self.options_data[self.get_option_id(contract)] = option_data
                    return contract
                else:
                    invalid_contracts.append(self.get_option_id(contract))
                    # self.logger.warning(f'Contract {contract} is not valid')
                    return self.generate_option_to_buy(underlier, contract_fetch_attempts + 1, invalid_contracts)
            else: 
                print('Failed to fetch contract for {underlier.ticker} at {time_str}')
                # self.logger.warning(f'Failed to fetch contract for {underlier.ticker} {contract["strike"]}{contract["strike"]}  at {time_str}')
                return None
                
                
                
    def update_positions_from_fill(self, fill_event: FillEvent):
        """
        Takes a FilltEvent object and updates the position matrix
        to reflect the new position.

        Parameters:
        fill - The FillEvent object to update the positions with.
        """
        # Check whether the fill is a buy or sell
        new_quantities = {}
        if fill_event.direction == 'BUY':
            if fill_event.option is not None: 
                new_quantities['quantity'] = fill_event.quantity
                new_quantities['option'] = fill_event.option
                option_data = self.__get_latest_option_data(fill_event.option)
                fill_cost = option_data['close'] * 100
                new_quantities['entry_price'] = fill_cost
        if fill_event.direction == 'SELL':
            if fill_event.option is not None: 
                new_quantities['quantity'] = fill_event.quantity
                new_quantities['option'] = fill_event.option
                option_data = self.__get_latest_option_data(fill_event.option)
                fill_cost = option_data['close'] * 100
                new_quantities['exit_price'] = fill_cost
            
            #fill_option should be None cause you now hold no options 
            

        # Update positions list with new quantities
        self.current_positions[fill_event.symbol] = new_quantities
        
        
    def update_holdings_from_fill(self, fill_event):
        """
        Takes a FillEvent object and updates the holdings matrix
        to reflect the holdings value.

        Parameters:
        fill - The FillEvent object to update the holdings with.
        """
        # Check whether the fill is a buy or sell
        #TODO: if no price is found at that date, return the close price of the nearest date before the date that couldnt be found
        fill_dir = 0
        fill_cost = 0.0
        if fill_event.direction == 'BUY':
            fill_dir = 1
            option_data = self.__get_latest_option_data(fill_event.option)
            fill_cost = option_data['close'] * 100 
        if fill_event.direction == 'SELL':
            fill_dir = -1
            option_data = self.__get_latest_option_data(fill_event.option)
            fill_cost = option_data['close'] * 100 

        
        # Update holdings list with new quantities
        cost = fill_dir * fill_cost * fill_event.quantity
        self.current_holdings[fill_event.symbol] += cost
        self.current_holdings['commission'] += fill_event.commission
        self.current_holdings['cash'] -= (cost + fill_event.commission)
        self.current_holdings['total'] -= (cost + fill_event.commission)
        

    def get_trades(self) -> pd.DataFrame:
        """
        return timeseries of portfolio trades
        """
        trades_data = []
        visited_option_id =[]
        for position in self.all_positions:
            for ticker, data in position.items():
                if ticker == 'datetime':
                    continue
                if data['option'] is not None and  'exit_price' in data and data['option'] not in visited_option_id:
                    entry_price_obj = next((pos for pos in self.all_positions if pos[ticker]['option'] == data['option'] and 'entry_price' in pos[ticker]), {})
                    entry_price = entry_price_obj[ticker]['entry_price']
                    exit_price = data['exit_price']
                    pnl = exit_price - entry_price
                    return_pct = (pnl / entry_price) * 100
                    exit_time = position['datetime']
                    entry_time = next((pos['datetime'] for pos in self.all_positions if pos[ticker]['option'] == data['option'] and 'entry_price' in pos[ticker]), None)
                    duration = (exit_time - entry_time).days if exit_time and entry_time else None #type: ignore
                    if duration is not None:
                        visited_option_id.append(data['option'])
                        trades_data.append({
                        'EntryPrice': entry_price,
                        'ExitPrice': exit_price,
                        'PnL': pnl,
                        'ReturnPct': return_pct,
                        'EntryTime': entry_time,
                        'ExitTime': exit_time,
                        'Duration': duration,
                        'Ticker': ticker
                        }) 

        trades = pd.DataFrame(trades_data)
        return trades
        
    def update_fill(self, event):
        """
        Updates the portfolio current positions and holdings 
        from a FillEvent.
        """
        if event.type == 'FILL': 
            self.update_positions_from_fill(event)
            self.update_holdings_from_fill(event)
        
    def update_timeindex(self):
        """
        Adds a new record to the positions matrix for the current 
        market data bar. This reflects the PREVIOUS bar, i.e. all
        current market data at this stage is known (OLHCVI).

        Makes use of a MarketEvent from the events queue.
        """
        
        latest_bars = self.bars.get_latest_bars() 
        current_date = latest_bars.iloc[0]['Date']

        # Check if the current date is a weekend (Saturday or Sunday)
        if current_date.weekday() >= 5:
            return
        
        bars = {} # a dict that holds the market value of each symbol
        for sym in self.symbol_list:
            if self.current_positions[sym]['option'] is not None:
                option_data = self.get_options_data(self.current_positions[sym]['option'])
                option_data = self.__get_latest_option_data(self.current_positions[sym]['option'])
                if not option_data.empty:
                    bars[sym]  = option_data['close'] * 100 
                else: 
                    bars[sym] = 0.0
            else:
                bars[sym] = 0.0

        # Update positions
        dp = dict( (k,v) for k, v in [(s, {}) for s in self.symbol_list] )
        dp['datetime'] = current_date

        for sym in self.symbol_list:
            dp[sym] = self.current_positions[sym]

        # Append the current positions
        self.all_positions.append(dp)

        # Update holdings
        dh = dict( (k,v) for k, v in [(s, 0.0) for s in self.symbol_list] )
        dh['datetime'] = current_date
        dh['cash'] = self.current_holdings['cash']
        dh['commission'] = self.current_holdings['commission']
        dh['total'] = self.current_holdings['cash']

        for sym in self.symbol_list:
            # Approximation to the real value
            market_value = self.current_positions[sym]['quantity'] * bars[sym] # TODO: query close price or avg of bid and ask from sql ? or theta data
            dh[sym] = market_value
            dh['total'] += market_value

        # Append the current holdings
        self.all_holdings.append(dh)
        
        
    def create_equity_curve(self) :
        """
            create equity curve
        """
        curve = pd.DataFrame(self.all_holdings)
        curve.set_index('datetime', inplace=True)
        curve['returns'] = curve['total'].pct_change()
        curve['equity_curve'] = (1.0 + curve['returns']).cumprod()
        self.equity_curve = curve
        
    def __get_latest_option_data(self, option_id: str) -> pd.Series:
        """
        Get the latest option data for a symbol
        return a series with columns: ms_of_day,open,high,low,close,volume,count,date
        """
        latest_bars = self.bars.get_latest_bars()
        current_date = latest_bars.iloc[0]['Date']
        option_data_df = self.get_options_data(option_id)
        closest_date_index = (option_data_df.index - current_date).to_series().abs().argsort()[:1] #index of nearest date to the current date
        option_data = option_data_df.iloc[closest_date_index] #get the nearest date to the current date
        return option_data.iloc[0]

        
        
    def get_options_data_on_contract(self, contract : pd.Series) -> pd.DataFrame | None: 
        """
        Updates the option data based on the fill contract
        """
        if contract is not None:
            option_id = self.get_option_id(contract)
            if option_id not in self.options_data: 
                start_date = self.bars.start_date.strftime('%Y%m%d')
                end_date = self.bars.end_date.strftime('%Y%m%d')
                exp = f'{contract["expiration"]}'
                strike = contract['strike']
                options = retrieve_option_ohlc(symbol = contract['root'], exp = exp, strike= strike, right=contract["right"], start_date=start_date, end_date=end_date)
                options.set_index('date', inplace=True)
                if is_theta_data_retrieval_successful(options):
                    return options
                    # self.options_data[option_id] = options # a dataframe with columns: ms_of_day,open,high,low,close,volume,count,date
                else: 
                    self.logger.warning(f"Option data not available for {option_id} {options}")
                    return None
            
    def get_options_data(self, option_id: str) -> pd.DataFrame:
        """
         returns a dataframe with columns: ms_of_day,open,high,low,close,volume,count,date
        """
        return self.options_data[option_id]
    
    
    def get_option_id(self, contract: pd.Series) -> str: 
        """
            returns a string format of underlier-expiration-strike-type from the dataframe of the columns root, expiration, strike, right
        """
        return f"{contract['root']}-{contract['expiration']}-{contract['strike']}-{contract['right']}"
        
        
    def __is_valid_dataframe(self, df: pd.DataFrame) -> bool:
        """
        Check if a dataframe has more rows with empty/null values than filled values
        """
        is_valid = True
        # Count rows with any empty (NaN) values
        empty_rows = df.isnull().any(axis=1).sum()

        # Count rows where all values are 0.0
        zero_rows = (df == 0.0).any(axis=1).sum()

        # Count valid rows (not empty and not all zeros)
        valid_rows = len(df) - (empty_rows + zero_rows)

        # Compare counts
        if (empty_rows + zero_rows) > valid_rows:
            is_valid = False
        
        return is_valid
        