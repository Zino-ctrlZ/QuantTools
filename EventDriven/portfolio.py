## Portfolio Expected Responsiblities:
# - Portfolio Construction
# - Trade management: Selection handled by risk manager, Execution handled by broker class.
# - Performance Monitoring: PnL, Reports, Sharpe Ratio
# - Position Management: Rolling Options, Hedging, Position sizing
# - 

import math
from abc import ABCMeta, abstractmethod
import pandas as pd

from trade.helpers.helper import parse_option_tick
from trade.helpers.Logging import setup_logger
from trade.assets.Stock import Stock
from dbase.DataAPI.ThetaData import  is_theta_data_retrieval_successful, retrieve_eod_ohlc #type: ignore

from EventDriven.event import  FillEvent, OrderEvent, SignalEvent
from EventDriven.data import HistoricTradeDataHandler
from EventDriven.riskmanager import RiskManager
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
    The OptionSignalPortfolio object is designed to handle the tracking of portfolio positions, create new orders and update holdings & positions based on FillEvents.    
    
    bars: HistoricTradeDataHandler
    events: EventScheduler
    risk_manager: RiskManager
    weight_map: dict
    initial_capital: int
    """
    
    def __init__(self, bars : HistoricTradeDataHandler, events, risk_manager : RiskManager, weight_map = None, initial_capital = 10000, max_contract_price = 500.0): 
        self.bars = bars
        self.events = events
        self.symbol_list = self.bars.symbol_list
        self.start_date = bars.start_date.strftime("%Y%m%d")
        self.initial_capital = initial_capital
        self.logger = setup_logger('OptionSignalPortfolio')
        self.risk_manager = risk_manager
        self.options_data = {}
        self._order_settings =  {
            'type': 'spread',
            'specifics': [
                {'direction': 'long', 'rel_strike': 1.0, 'dte': 365, 'moneyness_width': 0.1},
                {'direction': 'short', 'rel_strike': 0.85, 'dte': 365, 'moneyness_width': 0.1} 
            ],
            'name': 'vertical_spread'
        }
        self.trades = {}
        self.__trades = None ## Temporarily store trades data. Will change once Zino fixes the format
        self.max_contract_price = max_contract_price / 100.0
        self.__equity = None
        # call internal functions to construct key portfolio data
        self.__construct_all_positions()
        self.__construct_current_positions()
        self.__construct_all_holdings()
        self.__construct_weight_map(weight_map = weight_map)
        self.__construct_current_holdings()
        self.__construct_current_weighted_Holdings()
        self.__construct_weighted_holdings()
        self.__generate_underlier_data()

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
        self.__construct_current_weighted_Holdings()
        self.__construct_weighted_holdings()
        
        
    # internal functions to construct key portfolio data 
    
    def __construct_weight_map(self, weight_map): 
        if weight_map is not None:
            for s in weight_map.keys():
                if s not in self.symbol_list:
                    print(f"Symbol {s} not being processed but present in weight_map" )
                    self.logger.warning(f"Symbol {s} not being processed but present in weight_map")
            weight_map = {x : weight_map[x] for x in self.symbol_list}
            weight_total = sum(weight_map.values())
            assert weight_total <= 1.0, f"Sum of weights must be less than or equal to 1.0, got {weight_total}"
            
        else: 
            weight_map = {x: 1/len(self.symbol_list) for x in self.symbol_list} #spread capital between all symbols 
        
        self.__weight_map = weight_map
        self.allocated_cash_map = {s: self.__weight_map[s] * self.initial_capital for s in self.symbol_list}
    
    def __construct_current_positions(self):
        d = {s: {} for s in self.symbol_list}
        self.current_positions = d
        
    def __construct_all_positions(self): 
        d = {s: {} for s in self.symbol_list} #key is underlier, value is list of option contracts
        d['datetime'] = self.bars.start_date
        self.all_positions = [d]
    
    def __construct_current_holdings(self): 
        d = {s: self.allocated_cash_map[s] for s in self.symbol_list} #key is underlier, value is total value of all postitions held
        d['cash'] = self.initial_capital
        d['commission'] = 0.0
        d['total'] = self.initial_capital
        self.current_holdings = d
        
    def __construct_all_holdings(self):
        """
        Constructs the holdings list using the start_date
        to determine when the time index will begin.
        """
        d = {s: 0.0 for s in self.symbol_list}
        d['datetime'] = self.bars.start_date
        d['cash'] = self.initial_capital
        d['commission'] = 0.0
        d['total'] = self.initial_capital
        self.all_holdings =  [d]
    
    def __construct_current_weighted_Holdings(self): 
        d = {s: self.allocated_cash_map[s] for s in self.symbol_list}
        d['commission'] = 0.0
        d['total'] = self.initial_capital
        d['cash'] = self.initial_capital - sum(self.allocated_cash_map.values()) 
        self.current_weighted_holdings = d   
    
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
        
    def __generate_underlier_data(self):
        self.underlier_list_data = {}
        for underlier in self.symbol_list:
            self.underlier_list_data[underlier] = Stock(underlier, run_chain = False)

    @property
    def _equity(self):
        holdings = getattr(self, 'weighted_holdings')
        equity_curve = pd.DataFrame(holdings).set_index('datetime')
        equity_curve['total'] = equity_curve.iloc[:, :len(self.symbol_list)+1].sum(axis = 1) ##NOTE: Temp fix till calcs work
        equity_curve.rename(columns = {'total': 'Total'}, inplace=True)
        self.__equity = equity_curve
        return self.__equity
    
    @property
    def _trades(self):
        trades = self.get_trades()
        trades['ReturnPct'] = trades['ReturnPct']/100
        self.__trades = trades
        return self.__trades

    def get_port_stats(self):
        ## NOTE: I want to pass false if backtest is not run. How?
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
            stock_ts[stock] = self.underlier_list_data[stock].spot(ts = True, ts_start = self.dates_(), ts_end = self.dates_(start = False))['close'] * self.__weight_map[stock]
            
        stock_ts['Total'] = stock_ts.sum(axis = 1)
        self.stock_equity = stock_ts
        return ((stock_ts['Total'].iloc[-1] / stock_ts['Total'].iloc[0]) -1) * 100
    
             

    def generate_order(self, signal : SignalEvent):
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
        cash_at_hand = self.allocated_cash_map[symbol] * .9 #use 90% of cash to buy contracts
        
        if signal_type == 'LONG': #buy calls
            position_result = self.risk_manager.OrderPicker.get_order(symbol, date_str, 'C', self.max_contract_price, self.order_settings)  
            position = position_result['data'] if position_result['data'] is not None else None
            if position is None:  
                self.logger.warning(f'No contracts found for {symbol} at {signal.datetime}, Inputs {locals()}')
                return None
            self.logger.info(f'Buying LONG contract for {symbol} at {signal.datetime}')
            order_quantity = math.floor(cash_at_hand / (position['close'] * 100))
            order = OrderEvent(symbol, signal.datetime, order_type, quantity=order_quantity, direction= 'BUY', position = position)
            return order
        elif signal_type == 'SHORT': #buy puts
            position_result = self.risk_manager.OrderPicker.get_order(symbol, date_str, 'P', self.max_contract_price, self.order_settings)
            position = position_result['data'] if position_result['data'] is not None else None
            if position is None:  
                self.logger.warning(f'No contracts found for {symbol} at {signal.datetime}, Inputs {locals()}')
                return None
            self.logger.info(f'Buying LONG contract for {symbol} at {signal.datetime}')
            order_quantity = math.floor(cash_at_hand / (position['close'] * 100))
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
        
            
    def update_signal(self, event : SignalEvent):
        """
        Acts on a SignalEvent to generate new orders 
        based on the portfolio logic.
        throws: AssertionError if event type is not 'SIGNAL'
        """
        assert event.type == 'SIGNAL', f"Expected 'SIGNAL' event type, got {event.type}"
        order_event = self.generate_order(event)
        if order_event is not None:
            self.events.put(order_event)
                
        
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
                new_position_data['entry_price'] = fill_event.fill_cost * 100
                new_position_data['market_value'] = fill_event.market_value * 100
                
                
                #update trade data on successful buy
                trade_id = fill_event.position['trade_id']
                self.trades[trade_id] = {}
                self.trades[trade_id]['entry_price'] = fill_event.fill_cost * 100
                self.trades[trade_id]['entry_date'] = fill_event.datetime
                self.trades[trade_id]['quantity'] = fill_event.quantity
                self.trades[trade_id]['symbol'] = fill_event.symbol
                self.trades[trade_id]['commission'] = fill_event.commission
                self.trades[trade_id]['market_value'] = fill_event.market_value * 100
                
                #retain long legs options_data dictionary for future use 
                if 'long' in fill_event.position: 
                    for option_id in fill_event.position['long']: 
                        option_meta = parse_option_tick(option_id)
                        option_data = self.get_options_data_on_contract(symbol = option_meta['ticker'], right=option_meta['put_call'], exp=option_meta['exp_date'], strike=option_meta['strike'])
                        if option_data is not None: 
                            self.options_data[option_id] = option_data
                
                #retain short legs options_data dictionary for future use 
                if 'short' in fill_event.position: 
                    for option_id in fill_event.position['short']: 
                        option_meta = parse_option_tick(option_id)
                        option_data = self.get_options_data_on_contract(symbol = option_meta['ticker'], right=option_meta['put_call'], exp=option_meta['exp_date'], strike=option_meta['strike'])
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

    
    def update_holdings_from_fill(self, fill_event: FillEvent):
        """
        Takes a FillEvent object and updates the holdings matrix
        to reflect the holdings value.

        Parameters:
        fill - The FillEvent object to update the holdings with.
        """
        
        
        if fill_event.direction == 'BUY': 
            # available cash for the symbol is the left over cash after buying the contract
            self.allocated_cash_map[fill_event.symbol] -= fill_event.fill_cost * 100
            
            
            
        # Check whether the fill is buy or sell
        fill_dir = 1 if fill_event.direction == 'BUY' else -1   
        cost = fill_dir * fill_event.fill_cost
        self.current_holdings[fill_event.symbol] += cost
        self.current_holdings['commission'] += fill_event.commission
        self.current_holdings['cash'] -= cost
        self.current_holdings['total'] -= cost
        
        
        
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
        
        #new holdings dictionary
        new_holdings_entry = {s: 0.0 for s in self.symbol_list} 
        new_holdings_entry['datetime'] = current_date 
        new_holdings_entry['cash'] = self.current_holdings['cash']
        new_holdings_entry['commission'] = self.current_holdings['commission']
        new_holdings_entry['total'] = self.current_holdings['cash']
        
        #new positions dictionary
        new_positions_entry = {s: {} for s in self.symbol_list} 
        new_positions_entry['datetime'] = current_date
        
        #new weighted holdings dictionary
        new_weighted_holdings_entry = {s: self.allocated_cash_map[s] for s in self.symbol_list}
        new_weighted_holdings_entry['datetime'] = current_date
        new_weighted_holdings_entry['cash'] = (1.0 - sum(self.__weight_map.values())) * self.initial_capital
        new_weighted_holdings_entry['commission'] = self.current_weighted_holdings['commission']
        new_weighted_holdings_entry['total'] = 0.0
        
        for sym in self.symbol_list:
            if 'position' in self.current_positions[sym]:
                current_close = self.calculate_close_on_position(self.current_positions[sym]['position'])
                market_value = self.current_positions[sym]['quantity'] * current_close * 100
                
                
                #update holdings
                if 'exit_price' in self.current_positions[sym]: 
                    #use holdings to include pnl from closed position and set holdings to price at exit
                    new_holdings_entry['total'] += self.current_positions[sym]['exit_price'] 
                    new_holdings_entry[sym] = self.current_positions[sym]['exit_price'] 
                
                    #updated the available cash to include pnl from the closed position
                    self.allocated_cash_map[sym] += self.current_positions[sym]['exit_price']
                    new_weighted_holdings_entry[sym] = self.allocated_cash_map[sym] #update the holdings value to the market value of position + left over allocated cash
                else:
                    new_holdings_entry['total'] += market_value #update  value of total holdings with the market value of the position
                    new_holdings_entry[sym] = market_value #update the value of the symbol in the holdings dictionary
         
                    new_weighted_holdings_entry[sym] = market_value + self.allocated_cash_map[sym] #update the holdings value to the market value of position + left over allocated cash
                    

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
                    
                #update total weighted holdings
                new_weighted_holdings_entry['total'] += new_weighted_holdings_entry[sym]
            else: 
                # if no position held for symbol, add the available cash to the symbol to the total equity 
                new_weighted_holdings_entry['total'] += self.allocated_cash_map[sym]
                
        #append the new holdings and positions to the list of all holdings and positions
        self.all_positions.append(new_positions_entry)
        self.all_holdings.append(new_holdings_entry)
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
                option_data = self.__get_latest_option_data(option_id)
                if option_data is not None: 
                    long_legs_cost += option_data['Midpoint']

        if 'short'in position:
            for option_id in position['short']: 
                option_data = self.__get_latest_option_data(option_id)
                if option_data is not None: 
                    short_legs_cost += option_data['Midpoint']

        return long_legs_cost - short_legs_cost
    
    def get_trades(self) -> pd.DataFrame:
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
    
    
    def get_option_id(self, contract: pd.Series) -> str: 
        """
            returns a string format of underlier-expiration-strike-type from the dataframe of the columns root, expiration, strike, right
        """
        return f"{contract['root']}-{contract['expiration']}-{contract['strike']}-{contract['right']}"
    

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
        tr = self._trades.copy()
        tr['Size'] = tr['Quantity']

        return plot_portfolio(tr, eq, dd, _bnch,plot_bnchmk=plot_bnchmk, return_plot=return_plot, **kwargs)

    
        