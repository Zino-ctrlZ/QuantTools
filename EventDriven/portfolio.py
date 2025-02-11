## Portfolio Expected Responsiblities:
# - Portfolio Construction
# - Trade management: Selection handled by risk manager, Execution handled by broker class.
# - Performance Monitoring: PnL, Reports, Sharpe Ratio
# - Position Management: Rolling Options, Hedging, Position sizing
# - 



from dotenv import load_dotenv
load_dotenv()

import os
import sys
sys.path.append(
    os.environ.get('WORK_DIR')) #type: ignore
sys.path.append(
    os.environ.get('DBASE_DIR')) #type: ignore
from trade.helpers.helper import parse_option_tick
import pandas as pd

from dbase.DataAPI.ThetaData import list_contracts, retrieve_option_ohlc, is_theta_data_retrieval_successful, retrieve_eod_ohlc #type: ignore
from trade.assets.Stock import Stock

from abc import ABCMeta, abstractmethod

from EventDriven.event import  FillEvent, OrderEvent, SignalEvent
from EventDriven.data import HistoricTradeDataHandler
if eval(os.environ.get('ASYNC', True)):
    from EventDriven.riskmanager_async import RiskManager
else:
    from EventDriven.riskmanager import RiskManager

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
    
    def __init__(self, bars : HistoricTradeDataHandler, events, risk_manager : RiskManager, weight_map = None, initial_capital = 10000): 
        self.bars = bars
        self.events = events
        self.symbol_list = self.bars.symbol_list
        self.start_date = bars.start_date.strftime("%Y%m%d")
        self.initial_capital = initial_capital
        self.all_positions = self.__construct_all_positions_new()
        self.current_positions = self.__construct_current_positions_new()
        self.all_holdings = self.__construct_all_holdings()
        self.current_holdings = self.__construct_current_holdings()
        self._generate_underlier_data()
        self.max_option_budget = 0.1 * self.initial_capital
        self.logger = setup_logger('OptionSignalPortfolio')
        self.risk_manager = risk_manager
        self.options_data = {}
        self.weight_map = self.__construct_weight(weight_map = None)
        self._order_settings =  {
            'type': 'spread',
            'specifics': [
                {'direction': 'long', 'rel_strike': 1.0, 'dte': 365, 'moneyness_width': 0.1},
                {'direction': 'short', 'rel_strike': 0.85, 'dte': 365, 'moneyness_width': 0.1} 
            ],
            'name': 'vertical_spread'
        }
        self.trades = {}

    @property
    def order_settings(self):
        return self._order_settings
    
    @order_settings.setter
    def order_settings(self, settings, *args, **kwargs):

        if isinstance(settings, dict):
            _setting = settings
            self.__enfore_order_settings(_setting)
            
        elif isinstance(settings, callable):
            _settings = settings(*args, **kwargs)
            self.__enfore_order_settings(_settings)

        else:
            raise ValueError('Order Settings can either be a callable or a dicitonary')
        self._order_settings = _setting
        
        
    def __enfore_order_settings(self, settings):
        self.logger.warn('Each index in specifics list should have: `direction`: str, `rel_strike`: float, `dte`: int, `moneyness_width`: float')
        available_types = ['spread', 'naked', 'stock']
        assert 'type' in settings.keys() and 'specifics' in settings.keys() and 'name' in settings.keys(), f'Expected both of `type`, `name` and `specifics` in settings keys'
        assert settings['type'] in available_types, f'`type` must be one of {available_types}'
        assert isinstance(settings['specifics'], list), f'Order Specifics should be a list'

        if settings['type'] == 'spread' and len(settings['specific']) < 2:
                raise ValueError(f'Expected 2 legs for spreads')
    
    def __construct_weight(self, weight_map): 
        if weight_map is not None:
            weight_total = sum(weight_map.values())
            assert weight_total <= 1.0, f"Sum of weights must be less than or equal to 1.0, got {weight_total}"
            return weight_map
        
        weight = round(1/len(self.symbol_list), 2) #spread capital between all symbols 
        return {sym: weight for sym in self.symbol_list}
    
    def construct_current_positions(self):
        d = dict((k, v) for k, v in [(s, {}) for s in self.symbol_list])
        return d
    
    def __construct_current_positions_new(self):
        d = {s: {} for s in self.symbol_list}
        return d
    
    def __construct_all_holdings(self):
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
    
    ## Change
    def construct_all_positions(self): 
        d = dict((k, v) for k, v in [(s, {'quantity' : 0.0, 'option': None}) for s in self.symbol_list]) #key is underlier, value is list of option contracts
        d['datetime'] = self.bars.start_date
        return [d]
    
    def __construct_all_positions_new(self): 
        d = {s: {} for s in self.symbol_list} #key is underlier, value is list of option contracts
        d['datetime'] = self.bars.start_date
        return [d]
    
    def __construct_current_holdings(self): 
        d = dict((k, v) for k, v in [(s, 0.0) for s in self.symbol_list]) #key is underlier, value is total value of all postitions held
        d['cash'] = self.initial_capital
        d['commission'] = 0.0
        d['total'] = self.initial_capital
        return d
    
    def _generate_underlier_data(self):
        self.underlier_list_data = {}
        for underlier in self.symbol_list:
            self.underlier_list_data[underlier] = Stock(underlier, run_chain = False)
             
    def generate_order(self, signal : SignalEvent):
        """ 
        Takes a signal event and creates an order event based on the signal parameters
        
        This interacts with RiskManager to determine what positions to open
        """
        self.__latest_signals = self.bars.get_latest_bars().iloc[0]
        order = None 
        symbol = signal.symbol
        direction = signal.signal_type
        quantity = 2; 
        order_type = 'MKT'
        contract = None
        date_str = self.__latest_signals['Date'].strftime('%Y-%m-%d')
        print(f'Signal: {(signal.symbol, date_str) }')
        
        if direction == 'LONG':
            underlier_stock = self.underlier_list_data[symbol]
            
            contract = self.generate_option_to_buy(underlier_stock,  'C')
            if contract is None:  
                self.logger.warning(f'No contracts found for {symbol} at {signal.datetime}')
                return None
            self.logger.info(f'Buying LONG contract for {symbol} at {signal.datetime}')
            
            order = OrderEvent(symbol, signal.datetime ,order_type, quantity, 'BUY', option = self.get_option_id(contract))
        elif direction == 'SHORT':
            underlier_stock = self.underlier_list_data[symbol]
            
            contract = self.generate_option_to_buy(underlier_stock, 'P')
            if contract is None:  
                self.logger.warning(f'No contracts found for {symbol} at {signal.datetime}')
                return None
            self.logger.info(f'Buying SHORT contract for {symbol} at {signal.datetime}')
            
            order = OrderEvent(symbol, signal.datetime ,order_type, quantity, 'BUY', option = self.get_option_id(contract))
        elif direction == 'CLOSE':
            sell_contract_id = self.current_positions[symbol]['option']
            if sell_contract_id is None: 
                self.logger.warning(f'No contracts held for {symbol} to sell')
                return None
            self.logger.info(f'Selling contract for {symbol} at {signal.datetime}')
            
            order = OrderEvent(symbol, signal.datetime, order_type, quantity, 'SELL', option = sell_contract_id)
        return order

    async def generate_order_new(self, signal : SignalEvent):
        """
        Takes a signal event and creates an order event based on the signal parameters
        Interacts with RiskManager to get order based on settings and signal
        returns: OrderEvent
        """
        self.__latest_signals = self.bars.get_latest_bars().iloc[0]
        date_str = self.__latest_signals['Date'].strftime('%Y-%m-%d')
        symbol = signal.symbol
        signal_type = signal.signal_type
        order_type = 'MKT'
        max_price = int(5) # max price is multiplied by 100 , so 5 is 500
        cash_at_hand = self.current_holdings['cash']
        
        if signal_type == 'LONG': #buy calls
            position_result = await self.risk_manager.OrderPicker.get_order(symbol, date_str, 'C', max_price, self.order_settings)
            
            position = position_result['data'] if position_result['data'] is not None else None
            if position is None:  
                self.logger.warning(f'No contracts found for {symbol} at {signal.datetime}, Inputs {locals()}')
                return None
            self.logger.info(f'Buying LONG contract for {symbol} at {signal.datetime}')
            order_quantity = (cash_at_hand * self.weight_map[symbol] )/ (position['close'] * 100)
            order = OrderEvent(symbol, signal.datetime, order_type, quantity=order_quantity, direction= 'BUY', position = position)
            return order
        elif signal_type == 'SHORT': #buy puts
            position_result = await self.risk_manager.OrderPicker.get_order(symbol, date_str, 'P', max_price, self.order_settings)
            position = position_result['data'] if position_result['data'] is not None else None
            if position is None:  
                self.logger.warning(f'No contracts found for {symbol} at {signal.datetime}, Inputs {locals()}')
                return None
            self.logger.info(f'Buying LONG contract for {symbol} at {signal.datetime}')
            order_quantity = (cash_at_hand * self.weight_map[symbol] )/ (position['close'] * 100)
            order = OrderEvent(symbol, signal.datetime, order_type, quantity=order_quantity,direction= 'BUY', position = position)
            return order
        elif signal_type == 'CLOSE':
            current_position = self.current_positions[symbol]
            if 'position' not in current_position:
                self.logger.warning(f'No contracts held for {symbol} to sell at {signal.datetime}, Inputs {locals()}')
                return None
            self.logger.info(f'Selling contract for {symbol} at {signal.datetime}')
            order = OrderEvent(symbol, signal.datetime, order_type, quantity=current_position['quantity'],direction= 'SELL', position = current_position['position'])
            return order
        return None
        
            
    async def update_signal(self, event : SignalEvent):
        """
        Acts on a SignalEvent to generate new orders 
        based on the portfolio logic.
        throws: AssertionError if event type is not 'SIGNAL'
        """
        assert event.type == 'SIGNAL', f"Expected 'SIGNAL' event type, got {event.type}"
        # order_event = self.generate_order(event)
        order_event = await self.generate_order_new(event)
        if order_event is not None:
            self.events.put(order_event)
              
    ## Change
    def generate_option_to_buy(self, underlier: Stock, right: str,  contract_fetch_attempts = 0, invalid_contracts = [] ) -> pd.Series | None:
            """
            Buy an option based on the underlier.
            """

            ## Generate should by risk manager
            ## Thoughts when generating order:
            """
            - How to trade the position.
                - Trade statically: Just defining what to buy and sell at all times
                - Trade Dynamically: Allowing for strategies to pick order type.

                Order Settings:
                    Keys:
                        Type: Naked, Vertical Spread
                        Option Specifics:
                            Single Legs: Relative Strike, DTE
                            Spreads: 
                                Both Legs Legs: Relative Strike, DTE

                {
                    'Type': Spread
                    'Specifics': {
                            [{'direction': 'short', 'details': {'rel_strike': 0.95, 'dte': 365}}, 
                            {'direction': 'long', 'details': {'rel_strike': 1.00, 'dte': 365}}]
                    }
                }

                Dynamically: Function that creates that.
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
            contracts = contracts[contracts['right'] == right] #filter out right type 'C' or 'P'        
    
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
                    return self.generate_option_to_buy(underlier, right, contract_fetch_attempts + 1, invalid_contracts)
            else: 
                print(f'Failed to fetch contract for {underlier.ticker} at {time_str}')
                # self.logger.warning(f'Failed to fetch contract for {underlier.ticker} {contract["strike"]}{contract["strike"]}  at {time_str}')
                return None
                
                
    #TODO: this needs to be modified to match the new order settings, change fill event types as well 
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
        
    def update_positions_from_fill_new(self, fill_event: FillEvent):
        """
        Takes a FilltEvent object and updates the current positions in the portfolio 
        When a buy is filled, the options data related to the contract is stored in the options_data dictionary. 
        This is so it can be fetched easily when needed 

        Parameters:
        fill - The FillEvent object to update the positions with.
        """
        # Check whether the fill is a buy or sell
        new_position_data = {}
        if fill_event.direction == 'BUY': 
            if fill_event.position is not None: 
                new_position_data['position'] = fill_event.position
                new_position_data['quantity'] = fill_event.quantity
                new_position_data['entry_price'] = fill_event.fill_cost * 100
                new_position_data['market_value'] = fill_event.market_value * 100
                
                
                #update trade data on successful buy
                trade_id = fill_event.position['trade_id']
                self.trades[trade_id] = {}
                self.trades[trade_id]['entry_price'] = fill_event.fill_cost * 100
                self.trades[trade_id]['entry_date'] = fill_event.datetime
                self.trades[trade_id]['quantity'] = fill_event.quantity
                self.trades[trade_id]['symbol'] = fill_event.symbol
                
                #TODO: the get_options_data_on_contract_new function can be called more efficiently, possible bottleneck case here makin many calls to the API
                #retain long legs options_data dictionary for future use 
                if 'long' in fill_event.position: 
                    for option_id in fill_event.position['long']: 
                        option_meta = parse_option_tick(option_id)
                        option_data = self.get_options_data_on_contract_new(symbol = option_meta['ticker'], right=option_meta['put_call'], exp=option_meta['exp_date'], strike=option_meta['strike'])
                        if option_data is not None: 
                            self.options_data[option_id] = option_data
                
                #retain short legs options_data dictionary for future use 
                if 'short' in fill_event.position: 
                    for option_id in fill_event.position['short']: 
                        option_meta = parse_option_tick(option_id)
                        option_data = self.get_options_data_on_contract_new(symbol = option_meta['ticker'], right=option_meta['put_call'], exp=option_meta['exp_date'], strike=option_meta['strike'])
                        if option_data is not None: 
                            self.options_data[option_id] = option_data
                
                        
        if fill_event.direction == 'SELL':
            if fill_event.position is not None: 
                new_position_data['position'] = fill_event.position
                new_position_data['quantity'] = fill_event.quantity
                new_position_data['exit_price'] = fill_event.fill_cost * 100
                new_position_data['market_value'] = fill_event.fill_cost * 100
                
                # Update positions list with new quantities
        self.current_positions[fill_event.symbol] = new_position_data
        
    def calculate_close_on_position(self, position) -> float: 
        """
        Calculate the close price on a position
        the close price is the difference between the long and short legs of the position 
        """
        long_legs_cost = 0.0
        short_legs_cost = 0.0
        if 'long' in position:
            for option_id in position['long']: 
                option_data = self.__get_latest_option_data(option_id)
                if option_data is not None: 
                    long_legs_cost += option_data['Midpoint'] ## Find a way to make this dynamic

        if 'short'in position:
            for option_id in position['short']: 
                option_data = self.__get_latest_option_data(option_id)
                if option_data is not None: 
                    short_legs_cost += option_data['Midpoint']

        return long_legs_cost - short_legs_cost
    
    def update_holdings_from_fill_new(self, fill_event: FillEvent):
        """
        Takes a FillEvent object and updates the holdings matrix
        to reflect the holdings value.

        Parameters:
        fill - The FillEvent object to update the holdings with.
        """
        # Check whether the fill is buy or sell
        fill_dir = 1 if fill_event.direction == 'BUY' else -1   
        cost = fill_dir * fill_event.fill_cost
        self.current_holdings[fill_event.symbol] += cost
        self.current_holdings['commission'] += fill_event.commission
        self.current_holdings['cash'] -= cost
        self.current_holdings['total'] -= cost
        
    def update_timeindex_new(self): 
        """
        Adds a new record to the holdings and positions matrix based on the current market data bar.
        """
        
        latest_bars = self.bars.get_latest_bars() 
        current_date = latest_bars.iloc[0]['Date']

        # Check if the current date is a weekend (Saturday or Sunday)
        if current_date.weekday() >= 5:
            return
        
        
        new_holdings_entry = dict( (k,v) for k, v in [(s, 0.0) for s in self.symbol_list] ) #new holdings dictionary
        new_holdings_entry['datetime'] = current_date 
        new_holdings_entry['cash'] = self.current_holdings['cash']
        new_holdings_entry['commission'] = self.current_holdings['commission']
        new_holdings_entry['total'] = self.current_holdings['cash']
        
        new_positions_entry = {s: {} for s in self.symbol_list} #new positions dictionary
        new_positions_entry['datetime'] = current_date
        for sym in self.symbol_list:
            if 'position' in self.current_positions[sym]:
                current_close = self.calculate_close_on_position(self.current_positions[sym]['position'])
                market_value = self.current_positions[sym]['quantity'] * current_close * 100
                
                
                #update holdings
                if 'exit_price' in self.current_positions[sym]: 
                    #use the exit price to update the total holdings value
                    new_holdings_entry['total'] += self.current_positions[sym]['exit_price'] 
                    new_holdings_entry[sym] = self.current_positions[sym]['exit_price'] 
                else:
                    new_holdings_entry['total'] += market_value #update  value of total holdings with the market value of the position
                    new_holdings_entry[sym] = market_value #update the value of the symbol in the holdings dictionary
         

                #update positions
                if 'exit_price' in self.current_positions[sym]: #if position is closed, set current_positions to empty dict
                    #update trade data on successful sell
                    self.trades[self.current_positions[sym]['position']['trade_id']]['exit_price'] = self.current_positions[sym]['exit_price']
                    self.trades[self.current_positions[sym]['position']['trade_id']]['exit_date'] = current_date

                    self.current_positions[sym] = {}
                    new_positions_entry[sym] = {}
                else: 
                    current_position_data = {}
                    current_position_data['position'] = self.current_positions[sym]['position']
                    current_position_data['position']['close'] = current_close 
                    current_position_data['quantity'] = self.current_positions[sym]['quantity']
                    current_position_data['market_value'] = market_value
                    new_positions_entry[sym] = current_position_data
            else :
                new_positions_entry[sym] = {}
                new_holdings_entry[sym] = 0.0
                
        self.all_positions.append(new_positions_entry)
        self.all_holdings.append(new_holdings_entry)
        
        
    def cleanup_closed_positions(self):
        """
        Remove closed positions from the current_positions dictionary
        """
        for sym in self.symbol_list:
            if self.current_positions[sym]['exit_price'] is not None :
                self.current_positions[sym] = {}
                
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
        cost = fill_dir * fill_cost
        self.current_holdings[fill_event.symbol] += cost
        self.current_holdings['commission'] += fill_event.commission
        self.current_holdings['cash'] -= (cost + fill_event.commission)
        self.current_holdings['total'] -= (cost + fill_event.commission)
        
    def get_trades_new(self) -> pd.DataFrame:
        """
        return timeseries of portfolio trades
        """
        trades_data = []
        for trade_id, data in self.trades.items():
            pnl = data['exit_price'] - data['entry_price']
            return_pct = (pnl / data['entry_price']) * 100
            trades_data.append({
                'Ticker': data['symbol'],
                'PnL': pnl,
                'EntryPrice': data['entry_price']/data['quantity'],
                'ExitPrice': data['exit_price']/data['quantity'],
                'ReturnPct': return_pct,
                'Quantity': data['quantity'],
                'EntryTime': data['entry_date'],
                'ExitTime': data['exit_date'],
                'Duration': (data['exit_date'] - data['entry_date']).days,
                'Positions': trade_id
            }) 

        trades = pd.DataFrame(trades_data)
        return trades
    
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
                        'Ticker': ticker,
                        'Option': data['option']
                        }) 

        trades = pd.DataFrame(trades_data)
        return trades
        
    def update_fill(self, event):
        """
        Updates the portfolio current positions and holdings 
        from a FillEvent.
        """
        if event.type == 'FILL': 
            self.update_positions_from_fill_new(event)
            self.update_holdings_from_fill_new(event)
        
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
                option_data = self.get_option_data(self.current_positions[sym]['option'])
                option_data = self.__get_latest_option_data(self.current_positions[sym]['option'])
                if not option_data.empty and option_data is not None:
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
        params: option_id: str The option_id the contract was saved with during the fill process 
        returns: a series with columns: ms_of_day,open,high,low,close,volume,count,date
        """
        latest_bars = self.bars.get_latest_bars()
        current_date = latest_bars.iloc[0]['Date']
        option_data_df = self.get_option_data(option_id)
        if option_data_df is None:
            return None
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
                # options = retrieve_option_ohlc(symbol = contract['root'], exp = exp, strike= strike, right=contract["right"], start_date=start_date, end_date=end_date)
                options = retrieve_eod_ohlc(symbol = contract['root'], exp = exp, strike= float(strike), right=contract["right"], start_date=start_date, end_date=end_date)
                if isinstance(options, pd.DataFrame) and is_theta_data_retrieval_successful(options):
                    return options
                    # self.options_data[option_id] = options # a dataframe with columns: ms_of_day,open,high,low,close,volume,count,date
                else: 
                    self.logger.warning(f"Option data not available for {option_id} {options}")
                    return None
       
    def get_options_data_on_contract_new(self, symbol: str, exp: str, strike: float, right: str) -> pd.DataFrame | None: 
        """
        Updates the option data based on the fill contract
        """ 
        start_date = self.bars.start_date.strftime('%Y%m%d')
        end_date = self.bars.end_date.strftime('%Y%m%d')
        exp = pd.to_datetime(exp).strftime('%Y%m%d')
        # options = retrieve_option_ohlc(symbol = symbol, exp = exp, strike= strike, right=right, start_date=start_date, end_date=end_date)
        options = retrieve_eod_ohlc(symbol = symbol, exp = exp, strike= float(strike), right=right, start_date=start_date, end_date=end_date)
        if isinstance(options, pd.DataFrame) and is_theta_data_retrieval_successful(options):
            return options # a dataframe with columns: ms_of_day,open,high,low,close,volume,count,date
        else: 
            return None
        
    def get_option_data(self, option_id: str) -> pd.DataFrame:
        """
         returns a dataframe with columns: ms_of_day,open,high,low,close,volume,count,date
        """
        if option_id in self.options_data:
            return self.options_data[option_id]
        else :
            return None
    
    
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
        