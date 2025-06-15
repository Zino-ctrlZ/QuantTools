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
import numpy as np
from EventDriven.eventScheduler import EventScheduler
from EventDriven.trade import Trade
from trade.helpers.helper import parse_option_tick
from EventDriven.types import ResultsEnum, SignalTypes
from EventDriven.riskmanager import RiskManager
from trade.helpers.Logging import setup_logger
from trade.assets.Stock import Stock
from dbase.DataAPI.ThetaData import  is_theta_data_retrieval_successful, retrieve_eod_ohlc #type: ignore

from EventDriven.event import  ExerciseEvent, FillEvent, MarketEvent, OrderEvent, RollEvent, SignalEvent
from EventDriven.data import HistoricTradeDataHandler
from EventDriven.riskmanager import RiskManager
from trade.helpers.helper import is_USholiday
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
        raise NotImplementedError("Should implement analyze_signal()")

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
    
    def __init__(self, bars : HistoricTradeDataHandler, eventScheduler: EventScheduler, risk_manager : RiskManager, weight_map = None, initial_capital = 10000): 
        """
        Portfolio class for managing option trading strategies based on signals.
        Handles position tracking, order generation, portfolio valuation, and trade management.

        Attributes:
        bars (HistoricTradeDataHandler): Data handler containing historical price data for all symbols.
        events (EventScheduler): Event queue for processing market events, signals, orders, and fills.
        symbol_list (list): List of symbols being tracked in the portfolio.
        start_date (str): Start date of the backtest in YYYYMMDD format.
        initial_capital (float): Initial capital for the portfolio, default 10000.
        logger (Logger): Logger for portfolio activities.
        risk_manager (RiskManager): Manages trade selection and risk constraints.
        dte_reduction_factor (int): Days to reduce DTE by when a contract is illiquid (default: 60).
        min_acceptable_dte_threshold (int): Minimum acceptable DTE for a contract (default: 90).
        moneyness_width_factor (float): Factor to adjust moneyness width (default: 0.05).
        min_moneyness_threshold (int): Minimum threshold for adjusting moneyness before moving to next trading day (default: 5).
        max_contract_price_factor (float): Factor to increase max price (default: 1.2, 20% increase).
        options_data (dict): Dictionary mapping option IDs to their historical data.
        underlier_list_data (dict): Dictionary mapping symbols to their Stock objects.
        moneyness_tracker (dict): Tracks moneyness adjustments for signals.
        unprocessed_signals (list): List of signals that couldn't be processed.
        resolve_orders (bool): Whether to resolve orders if they are not processed (default: True).
        _order_settings (dict): Dictionary containing default order settings for trade strategy.
        __trades (dict): Dictionary of all trades executed.
        __equity (pd.DataFrame): DataFrame containing equity curve data.
        __transactions (list): List of all transactions made.
        __weight_map (dict): Dictionary mapping symbols to their portfolio weights.
        allocated_cash_map (dict): Dictionary mapping symbols to allocated capital.
        __max_contract_price (dict): Dictionary mapping symbols to maximum contract prices.
        __roll_map (dict): Dictionary mapping symbols to days before expiration to roll.
        all_positions (list): List of dictionaries containing position data snapshots.
        current_positions (dict): Dictionary of current positions by symbol.
        weighted_holdings (list): List of dictionaries containing portfolio valuation snapshots.
        current_weighted_holdings (dict): Dictionary of current portfolio values.
        trades_df (pd.DataFrame): DataFrame containing processed trade data.
        new_trades (dict): Dictionary of Trade objects to track trade performance.

        Methods:
        analyze_signal(event): Processes signal events and generates orders.
        update_fill(event): Updates portfolio positions and holdings from fill events.
        [Additional methods documented in their definitions]
        """
        self.bars = bars
        self.eventScheduler = eventScheduler
        self.symbol_list = self.bars.symbol_list
        self.start_date = bars.start_date.strftime("%Y%m%d")
        self.initial_capital = initial_capital
        self.logger = setup_logger('OptionSignalPortfolio')
        self.risk_manager = risk_manager
        self.dte_reduction_factor = 60 
        self.min_acceptable_dte_threshold = 90 
        self.moneyness_width_factor = 0.05 
        self.min_moneyness_threshold = 5 
        self.max_contract_price_factor = 1.2 
        self.options_data = {}
        self.underlier_list_data = {}
        self.moneyness_tracker = {}
        self.unprocessed_signals = []
        self.resolve_orders = True 
        self.risk_manager.pm = self #
        self._order_settings =  {
            'type': 'spread',
            'specifics': [
                {'direction': 'long', 'rel_strike': 1.0, 'dte': 365, 'moneyness_width': 0.1},
                {'direction': 'short', 'rel_strike': 0.85, 'dte': 365, 'moneyness_width': 0.1} 
            ],
            'name': 'vertical_spread',
            'strategy': 'vertical',
            'target_dte': 365,
            'structure_direction': 'long',
            'spread_ticks': 1,
            'dte_tolerance': 60,
            'min_moneyness': 0.75,
            'max_moneyness': 1.25,
            'min_total_price': 0.5
        }
        self.__equity = None
        self.__transactions = []
        # call internal functions to construct key portfolio data
        self.__construct_all_positions()
        self.__construct_current_positions()
        self.__construct_weight_map(weight_map = weight_map)
        self.__construct_current_weighted_holdings()
        self.__construct_weighted_holdings()
        self.__construct_roll_map()
        self.trades_df = None
        self.trades_map = {}

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
        
        necessary_keys = {
            'strategy': str,
            'target_dte': int,
            'structure_direction': str,
        }

        optional_keys = {
            'spread_ticks': int,
            'dte_tolerance': int,
            'min_moneyness': float,
            'max_moneyness': float,
            'min_total_price': float,
        }
        if settings['type'] == 'spread' and len(settings['specifics']) < 2:
                raise ValueError(f'Expected 2 legs for spreads')

        for key, value_type in necessary_keys.items():
            assert key in settings.keys(), f'Expected `{key}` in order settings'
            assert isinstance(settings[key], value_type), f'Expected `{key}` to be of type {value_type}, got {type(settings[key])}'
        
        for key, value_type in optional_keys.items():
            if key in settings.keys():
                assert isinstance(settings[key], value_type), f'Expected `{key}` to be of type {value_type}, got {type(settings[key])}'
            
    
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
        if isinstance(max_contract_price, int):
            max_contract_price = {s: max_contract_price for s in self.symbol_list}
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
    def get_underlier_data(self):
        return self.__get_underlier_data

    @property
    def transactions(self):
        return pd.DataFrame(self.__transactions)

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
        """
        Returns a DataFrame of trades executed in the portfolio.
        """
        if self.trades_df is not None:
            return self.trades_df
        
        self.trades_df = self.aggregate_trades()
        return self.trades_df
        
        
    def aggregate_trades(self): 
        trades_data = [self.trades_map[trade_id].stats for trade_id in self.trades_map.keys()]
        return pd.concat(trades_data, ignore_index=True) if trades_data else None
    
    
    @property
    def _trades(self):
        ## AggregatorParent uses _trades in some methods. See Expectancy in aggregator
        return self.trades
    
    def get_port_stats(self):
        ## NOTE: I want to pass false if backtest is not run. How?
        ## if the latest date on the bars is equal to the start date, backtest has yet to run
        current_date = pd.to_datetime(self.eventScheduler.current_date)
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
        
        if signal_type != 'CLOSE': #generate order for LONG or SHORT
            return self.create_order( signal, order_type)
        elif signal_type == 'CLOSE':
            if signal.signal_id not in self.current_positions[symbol]:
                self.logger.warning(f'No contracts held for {symbol} to sell at {signal.datetime}, Inputs {locals()}')
                unprocess_dict = signal.__dict__
                unprocess_dict['reason'] = (f'Signal not held in current positions at that time')
                self.unprocessed_signals.append(unprocess_dict)
                return None
            
            current_position = self.current_positions[symbol][signal.signal_id]
            if is_USholiday(signal.datetime): # check if trading day is holdiay before selling
                self.resolve_order_result({'result': ResultsEnum.IS_HOLIDAY.value}, signal)
                return None
            
            if 'position' not in current_position:
                self.logger.warning(f'No contracts held for {symbol} to sell at {signal.datetime}, Inputs {locals()}')
                return None
            
            
            position =  deepcopy(current_position['position'])
            self.logger.info(f'Selling contract for {symbol} at {signal.datetime} Position: {current_position}')
            position['close'] = self.calculate_close_on_position(position) #calculate close price on position
            skip = self.risk_manager.position_data[position['trade_id']].Midpoint_skip_day[signal.datetime]
            #on the off case where close price is negative, move sell to next trading day
            if position['close'] < 0 or skip == True:
                # move signal to next day 
                new_signal = deepcopy(signal)
                next_trading_day = new_signal.datetime + pd.offsets.BusinessDay(1)
                new_signal.datetime = next_trading_day
                self.logger.warning(f'Not generating order because: CLOSE price is negative {signal}, moving event to {next_trading_day}')
                print(f'Not generating order because: CLOSE price is negative {signal}, moving event to {next_trading_day}')
                self.eventScheduler.schedule_event(next_trading_day, new_signal)
                return None
            order = OrderEvent(symbol, signal.datetime, order_type, quantity=current_position['quantity'],direction= 'SELL', position = position, signal_id=signal.signal_id)
            return order
        return None
    
    def create_order(self, signal : SignalEvent, position_type: str, order_type: str = 'MKT'):
        """
        Takes a signal event and creates an order event based on the signal parameters
        position_type: C|P
        """
        date_str = signal.datetime.strftime('%Y-%m-%d')
        position_type = 'C' if signal.signal_type == 'LONG' else 'P'
        cash_at_hand = self.__normalize_dollar_amount_to_decimal(self.allocated_cash_map[signal.symbol] * 1) #use 90% of cash to buy contracts, leaving room for slippage and commission
        max_contract_price = self.__max_contract_price[signal.symbol] if signal.max_contract_price is None else signal.max_contract_price
        max_contract_price = max_contract_price if max_contract_price <= cash_at_hand else cash_at_hand 
        position_result = self.risk_manager.get_order(tick = signal.symbol, 
                                                                  date = date_str, 
                                                                  right = position_type, 
                                                                  option_type = position_type, 
                                                                  max_close = max_contract_price, 
                                                                  order_settings= signal.order_settings if signal.order_settings is not None else self._order_settings,
                                                                  signal_id = signal.signal_id,
                                                                  **self.order_settings)  
        
        position = None if position_result['result'] == ResultsEnum.NO_CONTRACTS_FOUND.value else position_result['data'] #if no contracts found, position is None
        if position is None :
            if self.resolve_orders == True :
                self.resolve_order_result(position_result['result'], signal)
            else:
                self.logger.warning(f'resolve_orders is {self.resolve_orders} hence not generating order because:{position_result["result"]} {signal}')
            return None
        
        self.moneyness_tracker[signal.signal_id] = 0 #reset moneyness tracker for signal after successful order generation
        self.logger.info(f'Buying LONG contract for {signal.symbol} at {signal.datetime} Position: {position}')
        print("===========================")
        print("Buy Details")
        print(f"Position: {position}, Date: {date_str}, Signal: {signal}")
        print(f"Max Contract Price: {max_contract_price}, Cash at Hand: {cash_at_hand}")
        print("Cash at Hand", cash_at_hand, "Close", position['close'])
        print("===========================")
        return OrderEvent(signal.symbol, signal.datetime, order_type, cash=cash_at_hand, direction= 'BUY', position = position, signal_id = signal.signal_id, quantity=position['quantity'])
        
    def __reduce_order_settings_dte_by_factor(self, order_settings):
        new_order_settings = deepcopy(order_settings)
        initial_dte = new_order_settings['specifics'][0]['dte']
        initial_dte = initial_dte - self.dte_reduction_factor
        new_order_settings['specifics'] = [{**x, 'dte': initial_dte} for x in new_order_settings['specifics']] #reduce dte by 1 day
        return new_order_settings
    
    def resolve_order_result(self, position_result: ResultsEnum, signal: SignalEvent): 
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
        if position_result == ResultsEnum.MONEYNESS_TOO_TIGHT.value: # adjust moneyness width if moneyness_tracket_index has not exceeded threshold and add to queue   
            order_settings = deepcopy(signal.order_settings if signal.order_settings is not None else self.order_settings) # use default order settings if signal order settings not set 
            order_settings['specifics'] = [{**x, 'moneyness_width': x['moneyness_width'] + self.moneyness_width_factor} for x in order_settings['specifics']] #increase moneyness width by 20%
            new_signal = deepcopy(signal)
            new_signal.order_settings = order_settings
            
            moneyness_tracker_index = self.moneyness_tracker.get(signal.signal_id, 0)

            if moneyness_tracker_index == 0:
                self.moneyness_tracker[signal.signal_id] = moneyness_tracker_index + 1
            else:
                self.moneyness_tracker[signal.signal_id] += 1
            
            # if moneyness width has been adjusted more than threshold, do not generate order
            if moneyness_tracker_index > self.min_moneyness_threshold: 
                new_max_price = self.__max_contract_price[signal.symbol]
                new_signal_on_dte = deepcopy(signal) 
                new_signal_on_dte.order_settings = deepcopy(self.order_settings) if moneyness_tracker_index == self.min_moneyness_threshold + 1 else signal.order_settings #first run threshold is exceeded, use default order settings, on next runs use signal order settings
                self.logger.warning(f'Not generating order because:{position_result} {signal}, performing resolve on reduced dte with intial moneyness width {self.__max_contract_price[signal.symbol]}')
                print(f'Not generating order because:{position_result} {signal}, performing resolve on reduced dte with intial moneyness width cash {self.__max_contract_price[signal.symbol]}')
                self.resolve_order_result(ResultsEnum.TOO_ILLIQUID.value, new_signal_on_dte)
                return None
            

            
                

            self.logger.warning(f'Not generating order because:{position_result} {signal}, adding new signal with adjusted moneyness. specifics: {order_settings["specifics"]}')
            print(f'Not generating order because:{position_result} {signal}, adding new signal with adjusted moneyness. specifics: {order_settings["specifics"]}')
            self.eventScheduler.put(new_signal)
           
            
                
        elif position_result == ResultsEnum.IS_HOLIDAY.value or position_result == ResultsEnum.NO_TRADED_CLOSE.value: #move signal to next trading day
            next_trading_day = signal.datetime + pd.offsets.BusinessDay(1)
            new_signal = deepcopy(signal)
            new_signal.datetime = next_trading_day
            self.logger.warning(f'Not generating order because:{position_result} {signal}, moving event to {next_trading_day}')
            print(f'Not generating order because:{position_result} {signal}, moving event to {next_trading_day}')
            self.eventScheduler.schedule_event(next_trading_day, new_signal)
            
                
        elif position_result == ResultsEnum.MAX_PRICE_TOO_LOW.value: #adjust max_price by 20% on max_price dict and add to queue
            initial_contract_max_price = self.__max_contract_price[signal.symbol] if signal.max_contract_price is None else signal.max_contract_price
            new_max_price = initial_contract_max_price * self.max_contract_price_factor
            allocated_cash =  self.__normalize_dollar_amount_to_decimal(self.allocated_cash_map[signal.symbol]) ## Max price should not exceed allocated cash
            
            if new_max_price > allocated_cash:
                new_max_price = self.__max_contract_price[signal.symbol]
                new_signal_on_dte = deepcopy(signal)
                new_signal_on_dte.max_contract_price = None
                self.logger.warning(f'Not generating order because:{position_result} {signal}, performing resolve on reduced dte with intial max cash {self.__max_contract_price[signal.symbol]}')
                print(f'Not generating order because:{position_result} {signal}, performing resolve on reduced dte with intial max cash {self.__max_contract_price[signal.symbol]}')
                self.resolve_order_result(ResultsEnum.TOO_ILLIQUID.value, new_signal_on_dte)
                return None
            
            new_max_price = min(new_max_price, self.__normalize_dollar_amount_to_decimal(self.allocated_cash_map[signal.symbol]))
            new_signal = deepcopy(signal)
            new_signal.max_contract_price = new_max_price
            self.logger.warning(f'Not generating order because:{position_result} at {initial_contract_max_price}, adjusted to {new_max_price} {signal} ')
            print(f'Not generating order because:{position_result} at {initial_contract_max_price}, adjusted to {new_max_price} {signal} ')
            self.eventScheduler.put(new_signal)
    
            
        elif position_result == ResultsEnum.TOO_ILLIQUID.value or position_result == ResultsEnum.NO_ORDERS.value:
            order_settings = deepcopy(signal.order_settings if signal.order_settings is not None else self.order_settings) # use default order settings if signal order settings not set
            order_settings = self.__reduce_order_settings_dte_by_factor(order_settings)
            dte = order_settings['specifics'][0]['dte']
            
            if dte < self.min_acceptable_dte_threshold or dte <= 0:
                self.logger.warning(f'Not generating order because:{position_result} {signal}')
                print(f'Not generating order because:{position_result} {signal}')
                unprocess_dict = signal.__dict__
                unprocess_dict['reason'] = position_result
                self.unprocessed_signals.append(unprocess_dict)
                return None
                
            new_signal = deepcopy(signal)
            new_signal.order_settings = order_settings
            self.logger.warning(f'Not generating order because:{position_result} {signal}, adding new signal with adjusted dte. specifics: {order_settings["specifics"]}')
            print(f'Not generating order because:{position_result} {signal}, adding new signal with adjusted dte. specifics: {order_settings["specifics"]}')
            self.eventScheduler.put(new_signal)
        else:
            self.logger.warning(f'Not generating order because:{position_result} {signal}')
            print(f'Not generating order because:{position_result} {signal}')
            unprocess_dict = signal.__dict__
            unprocess_dict['reason'] = position_result
            self.unprocessed_signals.append(unprocess_dict)
        
            
    def analyze_signal(self, event : SignalEvent):
        """
        Acts on a SignalEvent to generate new orders 
        based on the portfolio logic.
        throws: AssertionError if event type is not 'SIGNAL'
        """
        assert event.type == 'SIGNAL', f"Expected 'SIGNAL' event type, got {event.type}"
        order_event = self.generate_order(event)
        if order_event is not None:
            self.eventScheduler.put(order_event)
                
    def analyze_positions(self, market_event: MarketEvent): 
        """
        Analyze the current positions and determine if any need to be rolled
        """
        if is_USholiday(market_event.datetime):
            self.logger.warning(f"Market is closed on {market_event.datetime}, skipping")
            return
        
        for symbol in self.symbol_list:
            for signal_id in self.current_positions[symbol]: 
                current_position = self.current_positions[symbol][signal_id]
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
                
                dte = (pd.to_datetime(expiry_date) - pd.to_datetime(market_event.datetime)).days

                
                if symbol in self.roll_map and dte <= self.roll_map[symbol]:
                    self.logger.warning(f"On {market_event.datetime}, DTE for {symbol} is {dte}")
                    direction = SignalTypes.LONG.value if option_meta['put_call'] == 'C' else SignalTypes.SHORT.value
                    rollEvent = RollEvent(symbol=symbol, datetime=market_event.datetime, signal_type=direction, position=current_position, signal_id=signal_id)
                    self.eventScheduler.put(rollEvent)

                elif symbol not in self.roll_map and dte == 0:  # exercise contract if symbol not in roll map
                    position = current_position['position']
                    trade_data = self.trades_map[position['trade_id']]
                    quantity = position['quantity']
                    entry_date = trade_data['entry_date']
                    underlier = self.__get_underlier_data(symbol)
                    spot = underlier.spot(ts = True, ts_start = entry_date, ts_end = entry_date)['close']
                    long_premiums, short_premiums = self.get_premiums_on_position(current_position['position'], entry_date)
                    self.logger.warning(f'Exercising contract for {symbol} at {market_event.datetime}')
                    print(f'Exercising contract for {symbol} at {market_event.datetime}')
                    self.eventScheduler.put(ExerciseEvent(market_event.datetime, symbol, 'EXERCISE', quantity, entry_date, spot, long_premiums, short_premiums, position, trade_data['signal_id']))
                    ## if exercising, open new position if trade not closed yet.
                    continue
            
    def execute_roll(self, roll_event: RollEvent):
        """
        Execute the roll event by closing the current position and opening a new one
        rollEvent: RollEvent
        """
        self.logger.info(f'Rolling contract for {roll_event}')
        print(f'Rolling contract for {roll_event.symbol} at {roll_event.datetime}')
        sell_signal_event = SignalEvent( roll_event.symbol, roll_event.datetime, SignalTypes.CLOSE.value, signal_id=roll_event.signal_id)
        self.eventScheduler.put(sell_signal_event)
        buy_signal_event = SignalEvent( roll_event.symbol, roll_event.datetime, roll_event.signal_type , signal_id=roll_event.signal_id)
        self.eventScheduler.put(buy_signal_event)
                
        
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
    
    def update_positions_on_fill(self, fill_event: FillEvent):
        """
        Takes a FilltEvent object and updates the current positions in the portfolio 
        When a buy is filled, the options data related to the contract is stored in the options_data dictionary. 
        This is so it can be fetched easily when needed 
        Parameters:
        fill - The FillEvent object to update the positions with.
        """
        # Check whether the fill is a buy or sell
        new_position_data = {}
        
        if fill_event.position['trade_id'] not in self.trades_map:
            self.trades_map[fill_event.position['trade_id']] = Trade(fill_event.position['trade_id'], fill_event.symbol, fill_event.signal_id)
            self.trades_map[fill_event.position['trade_id']].update(fill_event)
        else:
            self.trades_map[fill_event.position['trade_id']].update(fill_event)
        
        if fill_event.direction == 'BUY': 
            if fill_event.position is not None: 
                new_position_data['position'] = fill_event.position
                if self.current_positions[fill_event.symbol] is not None and fill_event.signal_id in self.current_positions[fill_event.symbol]:
                    new_position_data['quantity'] = self.current_positions[fill_event.symbol][fill_event.signal_id]['quantity'] + fill_event.quantity
                else:
                    new_position_data['quantity'] = fill_event.quantity
                new_position_data['entry_price'] = self.__normalize_dollar_amount(fill_event.fill_cost)
                new_position_data['market_value'] = self.__normalize_dollar_amount(fill_event.market_value)
                new_position_data['signal_id'] = fill_event.signal_id
                
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
                if self.current_positions[fill_event.symbol] is not None and fill_event.signal_id in self.current_positions[fill_event.symbol]:
                    new_position_data['quantity'] = self.current_positions[fill_event.symbol][fill_event.signal_id]['quantity'] - fill_event.quantity
                else:
                    return ValueError(f'No position found for {fill_event.symbol} with signal_id {fill_event.signal_id}')
                new_position_data['market_value'] = self.__normalize_dollar_amount(fill_event.market_value)
                if (new_position_data['quantity']) == 0: 
                   new_position_data['exit_price'] = self.__normalize_dollar_amount(fill_event.fill_cost) 

        if fill_event.direction == 'EXERCISE':
            if fill_event.position is not None:
                new_position_data['position'] = fill_event.position
                new_position_data['quantity'] = self.current_positions[fill_event.symbol][fill_event.signal_id]['quantity'] - fill_event.quantity
                new_position_data['market_value'] = self.__normalize_dollar_amount(fill_event.market_value)   
                if (new_position_data['quantity']) == 0: 
                   new_position_data['exit_price'] = self.__normalize_dollar_amount(fill_event.fill_cost) 
                
                # open a new position after exercise
                new_signal = SignalEvent(fill_event.symbol, fill_event.datetime, SignalTypes.OPEN.value, signal_id=fill_event.signal_id)
                self.eventScheduler.put(new_signal)
            
            
        # update current_Positions with new position data
        # self.current_positions[fill_event.symbol]= new_position_data
        self.current_positions[fill_event.symbol][fill_event.signal_id] = new_position_data

    def update_holdings_on_fill(self, fill_event: FillEvent):
        """
        Takes a FillEvent object and updates the holdings matrix
        to reflect the holdings value.

        Parameters:
        fill - The FillEvent object to update the holdings with.
        """
        
        transaction = {}
        transaction['signal_id'] = fill_event.signal_id
        transaction['datetime'] = fill_event.datetime
        transaction['symbol'] = fill_event.symbol
        transaction['direction'] = fill_event.direction
        if fill_event.direction == 'BUY': 
            # available cash for the symbol is the left over cash after buying the contract
            transaction['cash_before'] = self.allocated_cash_map[fill_event.symbol]
            self.allocated_cash_map[fill_event.symbol] -= self.__normalize_dollar_amount(fill_event.fill_cost)
            transaction['cash_after'] = self.allocated_cash_map[fill_event.symbol]
        
        elif fill_event.direction == 'SELL':
            transaction['cash_before'] = self.allocated_cash_map[fill_event.symbol]
            self.allocated_cash_map[fill_event.symbol] += self.__normalize_dollar_amount(fill_event.fill_cost)
            transaction['cash_after'] = self.allocated_cash_map[fill_event.symbol] 

        self.__transactions.append(transaction)
        self.current_weighted_holdings['commission'] += fill_event.commission
         
    def update_timeindex(self): 
        """
        Adds a new record to the holdings and positions matrix based on the current market data bar. Runs at the end of the trading day (i.e all events for the day have been processed)
        """
        
        current_date = pd.to_datetime(self.eventScheduler.current_date)
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
            new_weighted_holdings_entry[sym] = self.allocated_cash_map[sym] 
            remove_signals = []
            for signal_id in self.current_positions[sym]:
                current_close = self.calculate_close_on_position(self.current_positions[sym][signal_id]['position'])
                market_value = self.__normalize_dollar_amount(self.current_positions[sym][signal_id]['quantity'] * current_close)
                
                self.trades_map[self.current_positions[sym][signal_id]['position']['trade_id']].update_current_price(self.__normalize_dollar_amount(current_close)) #update current price on trade
                
                self.current_positions[sym][signal_id]['position']['close'] = current_close ##Update close price for every iteration
                self.current_positions[sym][signal_id]['market_value'] = market_value
                
                #update holdings
                if 'exit_price' not in self.current_positions[sym][signal_id]: 
                     new_weighted_holdings_entry[sym] += market_value #update the holdings value to the market value of position + left over allocated cash
                    

                #update positions
                if 'exit_price' in self.current_positions[sym][signal_id]: #if position is closed, remove the signal from current_positions
                    #remove signal
                    remove_signals.append(signal_id)
                else: 
                    new_positions_entry[sym][signal_id] = deepcopy(self.current_positions[sym][signal_id])
                    
            
            #cleanup current_positions
            for signal_id in remove_signals:
                del self.current_positions[sym][signal_id]
            #update total weighted holdings
            new_weighted_holdings_entry['total'] += new_weighted_holdings_entry[sym]
                
        #append the new holdings and positions to the list of all holdings and positions
        self.all_positions.append(new_positions_entry)
        self.weighted_holdings.append(new_weighted_holdings_entry)
        
    def update_fill(self, event):
        """
        Updates the portfolio current positions and holdings 
        from a FillEvent.
        """
        if event.type == 'FILL': 
            self.update_positions_on_fill(event)
            self.update_holdings_on_fill(event)
            
    def calculate_close_on_position(self, position) -> float: 
        """
        Calculate the close price on a position
        the close price is the difference between the long and short legs of the position 
        """
        return self.risk_manager.position_data[position['trade_id']]['Midpoint'][pd.to_datetime(self.eventScheduler.current_date)]

    
    
    
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
            for symbol, positions in position_dict.items(): #TODO:all positions structure now by signal, get symbol from position 
                for signal_id, position in positions.items():
                    records.append([
                        dt, symbol, 
                        position.get('position', {}).get('long', []),
                        position.get('position', {}).get('short', []),
                        position.get('position', {}).get('trade_id', None),
                        position.get('position', {}).get('close', None),
                        position.get('quantity', 0),
                        position.get('market_value', 0.0),
                        signal_id
                    ])

        # Create DataFrame
        df = pd.DataFrame(records, columns=['datetime', 'symbol', 'long', 'short', 'trade_id', 'close', 'quantity', 'market_value', signal_id])

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
        
        current_date = pd.to_datetime(self.eventScheduler.current_date)
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

    
        