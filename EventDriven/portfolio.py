## Portfolio Expected Responsiblities:
# - Portfolio Construction
# - Trade management: Selection handled by risk manager, Execution handled by broker class.
# - Performance Monitoring: PnL, Reports, Sharpe Ratio
# - Position Management: Rolling Options, Hedging, Position sizing
# - 

from copy import deepcopy
import math
from abc import ABCMeta, abstractmethod
import pandas as pd

from EventDriven.eventScheduler import EventScheduler
from trade.helpers.helper import parse_option_tick
from EventDriven.types import ResultsEnum
from trade.helpers.Logging import setup_logger
from trade.assets.Stock import Stock
from dbase.DataAPI.ThetaData import  is_theta_data_retrieval_successful, retrieve_eod_ohlc #type: ignore

from EventDriven.event import  ExerciseEvent, FillEvent, MarketEvent, OrderEvent, SignalEvent
from EventDriven.data import HistoricTradeDataHandler
from EventDriven.riskmanager import RiskManager, is_USholiday
from trade.backtester_.utils.aggregators import AggregatorParent
from trade.backtester_.utils.utils import plot_portfolio
from typing import Optional
import plotly


class Portfolio(AggregatorParent):
    """
    The Portfolio class handles the positions and market
    value of all instruments at a resolution of a "bar",
    i.e. secondly, minutely, 5-min, 30-min, 60 min or EOD.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def analyze_signal(self, event):
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
    The OptionSignalPortfolio object is designed to handle the tracking of portfolio positions, create new orders and update holdings & positions based on FillEvents.    
    
    bars: HistoricTradeDataHandler
    events: EventScheduler
    risk_manager: RiskManager
    weight_map: dict
    initial_capital: int
    """
    
    def __init__(self, bars : HistoricTradeDataHandler, events: EventScheduler, risk_manager : RiskManager, weight_map = None, initial_capital = 10000): 
        self.bars = bars
        self.events = events
        self.symbol_list = self.bars.symbol_list
        self.start_date = bars.start_date.strftime("%Y%m%d")
        self.initial_capital = initial_capital
        self.logger = setup_logger('OptionSignalPortfolio')
        self.risk_manager = risk_manager
        self.moneyness_width_factor = 0.5 
        self.min_moneyness_threshold = 5 #minimum number of times to adjust moneyness width before moving to next trading day
        self.max_contract_price_factor = 1.2 #increase max price by 20%
        self.options_data = {}
        self.underlier_list_data = {}
        self.moneyness_tracker = {}
        self._order_settings =  {
            'type': 'spread',
            'specifics': [
                {'direction': 'long', 'rel_strike': 1.0, 'dte': 365, 'moneyness_width': 0.1},
                {'direction': 'short', 'rel_strike': 0.85, 'dte': 365, 'moneyness_width': 0.1} 
            ],
            'name': 'vertical_spread'
        }
        self.__trades = {}
        self.__equity = None
        # call internal functions to construct key portfolio data
        self.__construct_all_positions()
        self.__construct_current_positions()
        self.__construct_weight_map(weight_map = weight_map)
        self.__construct_current_weighted_holdings()
        self.__construct_weighted_holdings()
        self.__construct_roll_map()

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
        self.logger.warning('Each index in specifics list should have: `direction`: str, `rel_strike`: float, `dte`: int, `moneyness_width`: float')
        available_types = ['spread', 'naked', 'stock']
        assert 'type' in settings.keys() and 'specifics' in settings.keys() and 'name' in settings.keys(), f'Expected both of `type`, `name` and `specifics` in settings keys'
        assert settings['type'] in available_types, f'`type` must be one of {available_types}'
        assert isinstance(settings['specifics'], list), f'Order Specifics should be a list'

        if settings['type'] == 'spread' and len(settings['specific']) < 2:
                raise ValueError(f'Expected 2 legs for spreads')
            
    
    # if weight map is set externally, recalculate the allocated cash map
    @property
    def weight_map(self): 
        return self.__weight_map
    
    @weight_map.setter
    def weight_map(self, weight_map):
        self.__construct_weight_map(weight_map)
        self.__construct_current_weighted_holdings()
        self.__construct_weighted_holdings()
        
    @property
    def max_contract_price(self):
        return self.__max_contract_price
    
    @max_contract_price.setter
    def max_contract_price(self, max_contract_price):
        assert isinstance(max_contract_price, dict), f'max_contract_price must be a dictionary'
        assert all(x <= self.__normalize_dollar_amount_to_decimal(self.allocated_cash_map[s]) for s, x in max_contract_price.items()), f'max_contract_price must be less than or equal to allocated cash'
        self.__max_contract_price = deepcopy(max_contract_price) ## Was editing the original dict
        

        
    @property
    def roll_map(self):
        return self.__roll_map
    
    @roll_map.setter
    def roll_map(self, roll_map: int | dict):
        self.__construct_roll_map(roll_map)
    
    # internal functions to construct key portfolio data
    def __construct_roll_map(self, roll: int | dict = 30): 
        if isinstance(roll, int): 
            roll_map = {s: roll for s in self.symbol_list}
        else: 
            assert isinstance(roll, dict), f'Roll must be an integer or a dictionary'
            roll_map = deepcopy(roll)
        
        self.__roll_map = roll_map
        
    def __construct_weight_map(self, weight_map): 
        unprocessed_symbols = []
        if weight_map is not None:
            for s in weight_map.keys():
                if s not in self.symbol_list:
                    unprocessed_symbols.append(s)
            if len(unprocessed_symbols) > 0:
                print(f"The following symbols: {unprocessed_symbols} are not being processed but present in weight_map" )
                self.logger.warning(f"The following symbols: {unprocessed_symbols} are not being processed but present in weight_map")
            weight_map = {x : weight_map[x] for x in self.symbol_list}
            weight_total = sum(weight_map.values())
            assert weight_total <= 1.0, f"Sum of weights must be less than or equal to 1.0, got {weight_total}"
            
        else: 
            weight_map = {x: 1/len(self.symbol_list) for x in self.symbol_list} #spread capital between all symbols 
        
        self.__weight_map = weight_map
        self.allocated_cash_map = {s: self.__weight_map[s] * self.initial_capital for s in self.symbol_list}
        self.__max_contract_price = {s: self.__normalize_dollar_amount_to_decimal(self.allocated_cash_map[s] * .5) for s in self.symbol_list} # default max contract price is 50% of allocated cash divided by 100
    
    def __construct_current_positions(self):
        d = {s: {} for s in self.symbol_list}
        self.current_positions = d
        
    def __construct_all_positions(self): 
        d = {s: {} for s in self.symbol_list} #key is underlier, value is list of option contracts
        d['datetime'] = self.bars.start_date
        self.all_positions = [d]
        
    
    def __construct_current_weighted_holdings(self): 
        self.current_weighted_holdings = {'commission': 0.0}   
    
    def __construct_weighted_holdings(self): 
        """
        improved version of current_holdings, this attributes each symbols holdings to the market value of the position + left over allocated cash for the symbol
        """
        left_over_capital = (1.0 - sum(self.__weight_map.values())) * self.initial_capital
        d = {s: self.allocated_cash_map[s] for s in self.symbol_list}
        d['datetime'] = self.bars.start_date
        d['cash'] = left_over_capital
        d['commission'] = 0.0
        d['total'] = self.initial_capital
        self.weighted_holdings = [d]
        
    #lazy intialize Stock objects
    def __get_underlier_data(self, symbol: str):
        if symbol not in self.underlier_list_data:
            self.underlier_list_data[symbol] = Stock(symbol, run_chain = False)
        
        return self.underlier_list_data[symbol]
        

    @property
    def _equity(self):
        holdings = self.weighted_holdings
        equity_curve = pd.DataFrame(holdings).set_index('datetime')
        equity_curve = equity_curve[~equity_curve.index.duplicated(keep='last')]
        equity_curve['total'] = equity_curve.iloc[:, :len(self.symbol_list)+1].sum(axis = 1) ##NOTE: Temp fix till calcs work
        equity_curve.rename(columns = {'total': 'Total'}, inplace=True)
        self.__equity = equity_curve
        return self.__equity
    
    @property
    def trades(self):
        #NOTE: In the event of exercising, exit price will not be set, pnl will be calculated differently see todo  
        trades_data = []
        for trade_id, data in self.__trades.items():
            trades_data.append({
                'Ticker': data['symbol'],
                'PnL': data['pnl'],
                'ReturnPct': data['return_pct'],
                'EntryPrice': data['entry_price'],
                'EntryCommission': data['entry_commission'],
                'EntrySlippage': data['entry_slippage'],
                'EntryMarketValue': data['entry_market_value'],
                'TotalEntryCost': data['total_entry_cost'],
                'AuxilaryEntryCost': data['auxilary_entry_cost'],
                'ExitPrice': data['exit_price'],
                'ExitCommission': data['exit_commission'], 
                'ExitSlippage': data['exit_slippage'],
                'ExitMarketValue': data['exit_market_value'],
                'TotalExitCost': data['total_exit_cost'],
                'AuxilaryExitCost': data['auxilary_exit_cost'],
                'Quantity': data['quantity'],
                'EntryTime': data['entry_date'],
                'ExitTime': data['exit_date'],
                'Duration': (data['exit_date'] - data['entry_date']).days,
                'Positions': data['trade_id'],
                'SignalID': data['signal_id']
            }) 

        trades = pd.DataFrame(trades_data)
        return trades

    @property
    def _trades(self):
        ## AggregatorParent uses _trades in some methods. See Expectancy in aggregator
        return self.trades
    
    def get_port_stats(self):
        ## NOTE: I want to pass false if backtest is not run. How?
        ## if the latest date on the bars is equal to the start date, backtest has yet to run
        latest_bars = self.bars.get_latest_bars()
        current_date = latest_bars.iloc[0]['Date']
        if pd.to_datetime(self.start_date) == pd.to_datetime(current_date):
            return False
        return True
    
    ##NOTE: Should move to performance.py?
    def dates_(self, start: bool = True):
        if start:
            return self._equity.index.min()
        else:
            return self._equity.index.max()
        
    def buyNhold(self):
        stock_ts = pd.DataFrame()
        for stock in self.symbol_list:
            stock_ts[stock] = self.underlier_list_data.get(stock, self.__get_underlier_data(stock)).spot(ts = True, ts_start = self.dates_(), ts_end = self.dates_(start = False))['close'] * self.__weight_map[stock]
            
        stock_ts['Total'] = stock_ts.sum(axis = 1)
        self.stock_equity = stock_ts
        return self.__normalize_dollar_amount(((stock_ts['Total'].iloc[-1] / stock_ts['Total'].iloc[0]) -1))
    
             

    def generate_order(self, signal : SignalEvent):
        """
        Takes a signal event and creates an order event based on the signal parameters
        Interacts with RiskManager to get order based on settings and signal
        returns: OrderEvent
        """
        symbol = signal.symbol
        signal_type = signal.signal_type
        order_type = 'MKT'
        
        if signal_type == 'LONG': #buy calls
            return self.create_order(symbol, signal, 'C', order_type)
        elif signal_type == 'SHORT': #buy puts
            return self.create_order(symbol, signal, 'P', order_type)
        elif signal_type == 'CLOSE':
            current_position = self.current_positions[symbol]
            if is_USholiday(signal.datetime): # check if trading day is holdiay before selling
                self.analyze_order_result({'result': ResultsEnum.IS_HOLIDAY.value}, signal)
                return None
            
            if 'position' not in current_position:
                self.logger.warning(f'No contracts held for {symbol} to sell at {signal.datetime}, Inputs {locals()}')
                return None
            
            self.logger.info(f'Selling contract for {symbol} at {signal.datetime}')
            order = OrderEvent(symbol, signal.datetime, order_type, quantity=current_position['quantity'],direction= 'SELL', position = current_position['position'], signal_id=current_position['signal_id'])
            return order
        return None
    
    def create_order(self, symbol: str, signal : SignalEvent, position_type: str, order_type: str = 'MKT'):
        """
        Takes a signal event and creates an order event based on the signal parameters
        position_type: C|P
        """
        date_str = signal.datetime.strftime('%Y-%m-%d')
        cash_at_hand = self.allocated_cash_map[symbol] * .9 #use 90% of cash to buy contracts, leaving room for slippage and commission
        position_result = self.risk_manager.OrderPicker.get_order(signal.symbol, date_str, position_type, self.__max_contract_price[signal.symbol], signal.order_settings if signal.order_settings is not None else self._order_settings)  
        position = position_result['data'] if position_result['data'] is not None else None
        if position is None:
            self.analyze_order_result(position_result, signal)
            self.logger.warning(f'position_type: {position_type}')
            return None
        self.logger.info(f'Buying LONG contract for {symbol} at {signal.datetime}')
        order_quantity = math.floor(cash_at_hand / (position['close'] * 100))
        return OrderEvent(symbol, signal.datetime, order_type, quantity=order_quantity, direction= 'BUY', position = position, signal_id = signal.signal_id)
        
    
    def analyze_order_result(self, position_result, signal: SignalEvent): 
        """
        Analyze the results of the order and update the portfolio or event scheduler accordingly
        
        MONEYNESS_TOO_TIGHT: adjust moneyness width by adding moneyness_width factor (default at 0.5) and add to queue
        MAX_PRICE_TOO_LOW: adjust max_price by multiplying max_contract_price factor (default at 20%) on max_price dict and add to queue
        IS_HOLIDAY: move signal to next trading day
        NO_TRADED_CLOSE: move signal to next trading day
        NO_ORDERS: log warning
        UNSUCCESSFUL: log warning
        UNAVAILABLE_CONTRACT: log warning
        """
        result = position_result['result'] #TODO: pass just the result and not the whole dict
        if result == ResultsEnum.MONEYNESS_TOO_TIGHT.value: # adjust moneyness width if moneyness_tracket_index has not exceeded threshold and add to queue   
            moneyness_tracker_index = self.moneyness_tracker.get(signal.signal_id, 0)
            # if moneyness width has been adjusted more than threshold, do not generate order
            if moneyness_tracker_index > self.min_moneyness_threshold: 
                self.logger.warning(f'Not generating order because:{result} {signal}, moneyness width has been adjusted {moneyness_tracker_index} times, greater than threshold of {self.min_moneyness_threshold}')
                print(f'Not generating order because:{result} {signal}, moneyness width has been adjusted {moneyness_tracker_index} times, greater than threshold of {self.min_moneyness_threshold}')
                return None
            
            if moneyness_tracker_index == 0:
                self.moneyness_tracker[signal.signal_id] = moneyness_tracker_index + 1
            else:
                self.moneyness_tracker[signal.signal_id] += 1
            
            
                
            order_settings = deepcopy(signal.order_settings if signal.order_settings is not None else self.order_settings) # use default order settings if signal order settings not set 
            order_settings['specifics'] = [{**x, 'moneyness_width': x['moneyness_width'] + self.moneyness_width_factor} for x in order_settings['specifics']] #increase moneyness width by 20%
            new_signal = deepcopy(signal)
            new_signal.order_settings = order_settings
            self.logger.warning(f'Not generating order because:{result} {signal}, adding new signal with adjusted moneyness. specifics: {order_settings["specifics"]}')
            print(f'Not generating order because:{result} {signal}, adding new signal with adjusted moneyness. specifics: {order_settings["specifics"]}')
            self.events.put(new_signal)
            
                
        elif result == ResultsEnum.IS_HOLIDAY.value or result == ResultsEnum.NO_TRADED_CLOSE.value: #move signal to next trading day
            next_trading_day = self.events.get_next_trading_day()
            new_signal = deepcopy(signal)
            new_signal.datetime = next_trading_day
            self.logger.warning(f'Not generating order because:{result} {signal}, moving event to {next_trading_day}')
            print(f'Not generating order because:{result} {signal}, moving event to {next_trading_day}')
            self.events.schedule_event(next_trading_day, new_signal)
            
                
        elif result == ResultsEnum.MAX_PRICE_TOO_LOW.value: #adjust max_price by 20% on max_price dict and add to queue
            initial_contract_max_price = self.__max_contract_price[signal.symbol]
            new_max_price = self.__max_contract_price[signal.symbol] * self.max_contract_price_factor
            allocated_cash =  self.__normalize_dollar_amount_to_decimal(self.allocated_cash_map[signal.symbol]) ## Max price should not exceed allocated cash
            
            if new_max_price > allocated_cash:
                self.logger.warning(f'Not generating order because:{result} {signal}, new price {new_max_price} exceeds allocated cash {allocated_cash}')
                print(f'Not generating order because:{result} {signal}, new price {new_max_price} exceeds allocated cash {allocated_cash}')
                return None
            
            self.__max_contract_price[signal.symbol] = min(new_max_price, self.__normalize_dollar_amount_to_decimal(self.allocated_cash_map[signal.symbol])) #increase max price by 20%
            new_signal = deepcopy(signal)
            self.logger.warning(f'Not generating order because:{result} at {initial_contract_max_price}, adjusted to {self.__max_contract_price[signal.symbol]} {signal} ')
            print(f'Not generating order because:{result} at {initial_contract_max_price}, adjusted to {self.__max_contract_price[signal.symbol]} {signal} ')
            self.events.put(new_signal)
            
            
        elif result == ResultsEnum.NO_ORDERS.value or result == ResultsEnum.UNSUCCESSFUL.value or result == ResultsEnum.UNAVAILABLE_CONTRACT.value :
            self.logger.warning(f'Not generating order because:{result} {signal}')  
            print(f'Not generating order because:{result} {signal}')     
        
        else:
            self.logger.warning(f'Not generating order because:{result} {signal}')
            print(f'Not generating order because:{result} {signal}')
        
            
    def analyze_signal(self, event : SignalEvent):
        """
        Acts on a SignalEvent to generate new orders 
        based on the portfolio logic.
        throws: AssertionError if event type is not 'SIGNAL'
        """
        assert event.type == 'SIGNAL', f"Expected 'SIGNAL' event type, got {event.type}"
        order_event = self.generate_order(event)
        if order_event is not None:
            self.events.put(order_event)
                
    def analyze_roll(self, event: MarketEvent): 
        """
        Analyze the current positions and determine if any need to be rolled
        """
        for symbol in self.symbol_list:
            current_position = self.current_positions[symbol]
            if 'position' not in current_position:
                continue
            
            expiry_date = ''
            
            if 'long' in current_position['position']:
                for option_id in current_position['position']['long']:
                    option_meta = parse_option_tick(option_id)
                    expiry_date = option_meta['exp_date']
                    break
            elif 'short' in current_position['position']:
                for option_id in current_position['position']['long']:
                    option_meta = parse_option_tick(option_id)
                    expiry_date = option_meta['exp_date']
                    break
            
            dte = (pd.to_datetime(expiry_date) - pd.to_datetime(event.datetime)).days
            #exercise contract if symbol not in roll map
            if symbol in self.roll_map and dte <= self.roll_map[symbol]:
                self.logger.info(f'Rolling contract for {symbol} at {event.datetime}')
                print(f'Rolling contract for {symbol} at {event.datetime}')
                sell_signal_event = SignalEvent( symbol, event.datetime,'CLOSE', signal_id=current_position['signal_id'])
                self.events.put(sell_signal_event)
                next_trading_day = self.events.get_next_trading_day()
                direction = 'LONG' if option_meta['put_call'] == 'C' else 'SHORT'
                buy_signal_event = SignalEvent( symbol, next_trading_day, direction , signal_id=current_position['signal_id'])
                self.events.schedule_event(next_trading_day, buy_signal_event)
                
            elif symbol not in self.roll_map and dte == 0:
                position = self.current_positions[symbol]['position']
                trade_data = self.__trades[position['trade_id']]
                quantity = position['quantity']
                entry_date = trade_data['entry_date']
                underlier = self.__get_underlier_data(symbol)
                spot = underlier.spot(ts = True, ts_start = entry_date, ts_end = entry_date)['close']
                long_premiums, short_premiums = self.get_premiums_on_position(current_position['position'], entry_date)
                self.logger.warning(f'Exercising contract for {symbol} at {event.datetime}')
                print(f'Exercising contract for {symbol} at {event.datetime}')
                self.events.put(ExerciseEvent(event.datetime, symbol, 'EXERCISE', quantity, entry_date, spot, long_premiums, short_premiums, position, trade_data['signal_id']))
                continue
            
            
            
                
                
    def update_positions_from_fill(self, fill_event: FillEvent):
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
                new_position_data['entry_price'] = self.__normalize_dollar_amount(fill_event.fill_cost)
                new_position_data['market_value'] = self.__normalize_dollar_amount(fill_event.market_value)
                new_position_data['signal_id'] = fill_event.signal_id
                
                self.update_trades_on_open(fill_event)
                
                #retain long legs options_data dictionary for future use 
                if 'long' in fill_event.position: 
                    for option_id in fill_event.position['long']: 
                        option_meta = parse_option_tick(option_id)
                        option_data = self.get_options_data_on_contract(symbol = option_meta['ticker'], right=option_meta['put_call'], exp=option_meta['exp_date'], strike=option_meta['strike'])
                        if option_data is not None: 
                            self.options_data[option_id] = option_data[~option_data.index.duplicated(keep='last')]
                        else:
                            self.logger.warning(f'No data found for {option_id}')
                
                #retain short legs options_data dictionary for future use 
                if 'short' in fill_event.position: 
                    for option_id in fill_event.position['short']: 
                        option_meta = parse_option_tick(option_id)
                        option_data = self.get_options_data_on_contract(symbol = option_meta['ticker'], right=option_meta['put_call'], exp=option_meta['exp_date'], strike=option_meta['strike'])
                        if option_data is not None: 
                            self.options_data[option_id] = option_data[~option_data.index.duplicated(keep='last')]
                        else:
                            self.logger.warning(f'No data found for {option_id}')
                
                        
        if fill_event.direction == 'SELL':
            if fill_event.position is not None: 
                new_position_data['position'] = fill_event.position
                new_position_data['quantity'] = fill_event.quantity
                new_position_data['exit_price'] = self.__normalize_dollar_amount(fill_event.fill_cost)
                new_position_data['market_value'] = self.__normalize_dollar_amount(fill_event.market_value)
                
                self.update_trades_on_sell(fill_event)

        if fill_event.direction == 'EXERCISE':
            if fill_event.position is not None:
                new_position_data['position'] = fill_event.position
                new_position_data['quantity'] = fill_event.quantity
                new_position_data['exit_price'] = self.__normalize_dollar_amount(fill_event.fill_cost)
                self.update_trades_on_exercise(fill_event)
            
            
        # update current_Positions with new position data
        self.current_positions[fill_event.symbol] = new_position_data

    def update_trades_on_open(self, fill_event: FillEvent):
        """
        Update trade data on open
        """
         #update trade data on successful buy
        trade_id = fill_event.position['trade_id']
        self.__trades[trade_id] = {}
        self.__trades[trade_id]['entry_price'] = self.__normalize_dollar_amount(fill_event.fill_cost/fill_event.quantity)
        self.__trades[trade_id]['entry_date'] = fill_event.datetime
        self.__trades[trade_id]['quantity'] = fill_event.quantity
        self.__trades[trade_id]['symbol'] = fill_event.symbol
        self.__trades[trade_id]['entry_commission'] = self.__normalize_dollar_amount(fill_event.commission)
        self.__trades[trade_id]['entry_market_value'] = self.__normalize_dollar_amount(fill_event.market_value)
        self.__trades[trade_id]['entry_slippage'] = self.__normalize_dollar_amount(fill_event.slippage)
        self.__trades[trade_id]['total_entry_cost'] = fill_event.fill_cost
        self.__trades[trade_id]['auxilary_entry_cost'] = abs(self.__trades[trade_id]['entry_commission']) + abs(self.__trades[trade_id]['entry_slippage'])
        self.__trades[trade_id]['signal_id'] = fill_event.signal_id
        self.__trades[trade_id]['trade_id'] = trade_id

    def update_trades_on_sell(self, fill_event: FillEvent):
        """
        Update trade data on successful sell
        """
        trade_data = self.__trades[fill_event.position['trade_id']]
        trade_data['exit_price'] = self.__normalize_dollar_amount(fill_event.fill_cost/fill_event.quantity)
        trade_data['exit_date'] = fill_event.datetime
        trade_data['exit_commission'] = self.__normalize_dollar_amount(fill_event.commission)
        trade_data['exit_slippage'] = self.__normalize_dollar_amount(fill_event.slippage)
        trade_data['exit_market_value'] = self.__normalize_dollar_amount(fill_event.market_value)
        trade_data['total_exit_cost'] = fill_event.fill_cost
        trade_data['auxilary_exit_cost'] = abs(trade_data['exit_commission']) + abs(trade_data['exit_slippage'])
        trade_data['pnl'] =  (trade_data['exit_price'] - trade_data['entry_price']) * trade_data['quantity']
        trade_data['return_pct'] = (trade_data['pnl'] / trade_data['entry_price'])
        trade_data['duration_days'] = (fill_event.datetime - trade_data['entry_date']).days
        trade_data['exit_method'] = 'sell' 
        
    def update_trades_on_exercise(self, fill_event: FillEvent):
        """
        Update trade data on successful exercise
        """
        trade_data = self.__trades[fill_event.position['trade_id']]
        trade_data['exit_price'] = 0.0
        trade_data['exit_date'] = fill_event.datetime
        trade_data['exit_commission'] = 0.0
        trade_data['exit_slippage'] = 0.0 
        trade_data['exit_market_value'] = 0.0
        trade_data['total_exit_cost'] = 0.0
        trade_data['auxilary_exit_cost'] = 0.0
        trade_data['pnl'] = fill_event.fill_cost * fill_event.quantity
        trade_data['return_pct'] = (trade_data['pnl'] / trade_data['entry_price'])
        trade_data['duration_days'] = (fill_event.datetime - trade_data['entry_date']).days
        trade_data['exit_method'] = 'exercise'
        
    def get_premiums_on_position(self, position: dict, entry_date: str) -> tuple[dict, dict] | tuple[None, None]:
        """
        get the premium of each contract in a position
        return [long_premiums | None, short_premiums | None]
        """
        long_premiums = {}
        short_premiums = {}
        if 'long' in position:
            for option_id in position['long']: 
                option_data = self.get_option_data(option_id)
                if option_data is not None: 
                    option_data_series = option_data.loc[pd.to_datetime(entry_date)]
                    if isinstance(option_data_series, pd.Series):
                        premium = option_data_series['Midpoint']
                    elif isinstance(option_data_series, pd.DataFrame):
                        premium = option_data_series.iloc[0]['Midpoint']

                    long_premiums[option_id] = premium
                    
        if 'short' in position:
            for option_id in position['short']: 
                option_data = self.get_option_data(option_id)
                if option_data is not None: 
                    option_data_series = option_data.loc[pd.to_datetime(entry_date)]
                    if isinstance(option_data_series, pd.Series):
                        premium = option_data_series['Midpoint']
                    elif isinstance(option_data_series, pd.DataFrame):
                        premium = option_data_series.iloc[0]['Midpoint']

                    short_premiums[option_id] = premium
                    
        if len(long_premiums) == 0:
            long_premiums = None
        if len(short_premiums) == 0:
            short_premiums = None
        return (long_premiums, short_premiums)
    
        
    def __normalize_dollar_amount_to_decimal(self, price: float) -> float:
        """
        divide by 100
        """
        return price / 100
    
    def __normalize_dollar_amount(self, price: float) -> float:
        """
        multiply by 100
        """
        return price * 100
    
    def update_holdings_from_fill(self, fill_event: FillEvent):
        """
        Takes a FillEvent object and updates the holdings matrix
        to reflect the holdings value.

        Parameters:
        fill - The FillEvent object to update the holdings with.
        """
        
        
        if fill_event.direction == 'BUY': 
            # available cash for the symbol is the left over cash after buying the contract
            self.allocated_cash_map[fill_event.symbol] -= self.__normalize_dollar_amount(fill_event.fill_cost)
            
        #track the total cash spent on commission
        self.current_weighted_holdings['commission'] += fill_event.commission
         
    def update_timeindex(self): 
        """
        Adds a new record to the holdings and positions matrix based on the current market data bar.
        """
        
        latest_bars = self.bars.get_latest_bars()
        current_date = latest_bars.iloc[0]['Date']
        # Check if the current date is a weekend (Saturday or Sunday)
        if current_date.weekday() >= 5:
            return
                
        #new positions dictionary
        new_positions_entry = {s: {} for s in self.symbol_list} 
        new_positions_entry['datetime'] = current_date
        
        #new weighted holdings dictionary
        new_weighted_holdings_entry = {s: self.allocated_cash_map[s] for s in self.symbol_list}
        new_weighted_holdings_entry['datetime'] = current_date
        new_weighted_holdings_entry['cash'] = (1.0 - sum(self.__weight_map.values())) * self.initial_capital
        new_weighted_holdings_entry['commission'] = self.current_weighted_holdings['commission']
        new_weighted_holdings_entry['total'] = new_weighted_holdings_entry['cash']
        
        for sym in self.symbol_list:
            if 'position' in self.current_positions[sym]:
                current_close = self.calculate_close_on_position(self.current_positions[sym]['position'])
                market_value = self.__normalize_dollar_amount(self.current_positions[sym]['quantity'] * current_close)
                
                
                #update holdings
                if 'exit_price' in self.current_positions[sym]: 
                    #updated the available cash to include pnl from the closed position
                    self.allocated_cash_map[sym] += self.current_positions[sym]['exit_price']
                    new_weighted_holdings_entry[sym] = self.allocated_cash_map[sym] #update the holdings value to the market value of position + left over allocated cash
                else:
                    new_weighted_holdings_entry[sym] = market_value + self.allocated_cash_map[sym] #update the holdings value to the market value of position + left over allocated cash
                    

                #update positions
                if 'exit_price' in self.current_positions[sym]: #if position is closed, set current_positions to empty dict
                    self.current_positions[sym] = {}
                    new_positions_entry[sym] = {}
                else: 
                    current_position_data = deepcopy(self.current_positions[sym])
                    current_position_data['position']['close'] = current_close 
                    current_position_data['market_value'] = market_value
                    new_positions_entry[sym] = current_position_data
                    
                #update total weighted holdings
                new_weighted_holdings_entry['total'] += new_weighted_holdings_entry[sym]
            else: 
                # if no position held for symbol, add the available cash to the symbol to the total equity 
                new_weighted_holdings_entry['total'] += self.allocated_cash_map[sym]
                
        #append the new holdings and positions to the list of all holdings and positions
        self.all_positions.append(new_positions_entry)
        self.weighted_holdings.append(new_weighted_holdings_entry)
        
    def update_fill(self, event):
        """
        Updates the portfolio current positions and holdings 
        from a FillEvent.
        """
        if event.type == 'FILL': 
            self.update_positions_from_fill(event)
            self.update_holdings_from_fill(event)
            
    def calculate_close_on_position(self, position) -> float: 
        """
        Calculate the close price on a position
        the close price is the difference between the long and short legs of the position 
        """
        long_legs_cost = 0.0
        short_legs_cost = 0.0
        if 'long' in position:
            for option_id in position['long']: 
                option_data = self.get_latest_option_data(option_id)
                if option_data is not None: 
                    long_legs_cost += option_data['Midpoint']

        if 'short'in position:
            for option_id in position['short']: 
                option_data = self.get_latest_option_data(option_id)
                if option_data is not None: 
                    short_legs_cost += option_data['Midpoint']

        return long_legs_cost - short_legs_cost
    
    
    
    # Getters 
    def get_weighted_holdings(self) -> pd.DataFrame:
        """
        Converts `weighted_holdings` from a list of dictionaries to a Pandas DataFrame with datetime index.
        Returns:
            pd.DataFrame: A time-series DataFrame of weighted holdings.
        """
        df = pd.DataFrame(self.weighted_holdings)
        df['datetime'] = pd.to_datetime(df['datetime'])  # Ensure datetime format
        df.set_index('datetime', inplace=True)  # Set datetime as index
        return df 
    
    
    def get_all_positions(self) -> pd.DataFrame:
        """
        Converts `all_positions` from a list of dictionaries to a Pandas MultiIndex DataFrame.
        - Index Level 1: datetime
        - Index Level 2: symbol
        Returns:
            pd.DataFrame: A MultiIndex DataFrame of all positions.
        """
        records = []  # Temporary storage for DataFrame conversion
        all_positions_copy = deepcopy(self.all_positions)  # Avoid modifying original list
        for position_dict in all_positions_copy:
            dt = position_dict.pop('datetime', pd.to_datetime(0))  # Extract timestamp
            for symbol, position in position_dict.items():
                if position:  # If there is an active position
                    records.append([
                        dt, symbol, 
                        position.get('position', {}).get('long', []),
                        position.get('position', {}).get('short', []),
                        position.get('position', {}).get('trade_id', None),
                        position.get('position', {}).get('close', None),
                        position.get('quantity', 0),
                        position.get('market_value', 0.0)
                    ])

        # Create DataFrame
        df = pd.DataFrame(records, columns=['datetime', 'symbol', 'long', 'short', 'trade_id', 'close', 'quantity', 'market_value'])

        # Set MultiIndex (datetime → symbol)
        df.set_index(['datetime', 'symbol'], inplace=True)
        df.index = df.index.set_levels(pd.to_datetime(df.index.levels[0]), level=0)  # Ensure datetime index
        return df
    
        
    
    def get_equity_curve(self) :
        """
            create equity curve
        """
        curve = pd.DataFrame(self.weighted_holdings)
        curve.set_index('datetime', inplace=True)
        curve['returns'] = curve['total'].pct_change()
        curve['equity_curve'] = (1.0 + curve['returns']).cumprod()
        return curve
        
    def get_latest_option_data(self, option_id: str) -> pd.Series:
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

       
    def get_options_data_on_contract(self, symbol: str, exp: str, strike: float, right: str) -> pd.DataFrame | None: 
        """
        Updates the option data based on the fill contract
        """ 
        start_date = self.bars.start_date.strftime('%Y%m%d')
        end_date = self.bars.end_date.strftime('%Y%m%d')
        exp = pd.to_datetime(exp).strftime('%Y%m%d')
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
    
    

    def plot_portfolio(self,
                    benchmark: Optional[str] = 'SPY',
                    plot_bnchmk: Optional[bool] = True,
                    return_plot: Optional[bool] = False,
                    start_plot: Optional[str] = None,
                    **kwargs) -> Optional[plotly.graph_objects.Figure]:
        """
        Plots a graph of current porfolio metrics. These graphs are Equity Curve, Portfolio Drawdown, Trades, Periodic returns
        Plotting function is plotly. Through **kwargs, you can edit the subplot
        
        Parameters:
        benchmark (Optional[str]): Benchmark you would like to compare portfolio equity. Defaults to SPY
        plot_bnchmk (Optional[bool]): Optionality to plot a benchmark or not
        return_plot Optional[bool]: Returns the plot object. User may opt for this if they plan to make further editing beyond **kwargs functionality. 
                                    Note, best to designate this to a variable to avoid being displayed twice

        Returns: 
        Plot: For further editing by the user
        """
        
        stock = Stock(benchmark, run_chain = False)
        data = stock.spot(ts = True, ts_start = self._equity.index[0], ts_end = self._equity.index[-1])
        data.rename(columns = {x:x.capitalize() for x in data.columns}, inplace= True)
        data = data.asfreq('B', method = 'ffill')
        _bnch = data.fillna(0)
        eq = self._equity
        dd = self.dd(True)
        tr = self.trades.copy()
        tr['Size'] = tr['Quantity']

        return plot_portfolio(tr, eq, dd, _bnch,plot_bnchmk=plot_bnchmk, return_plot=return_plot, **kwargs)

    
        