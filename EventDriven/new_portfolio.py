## Portfolio Expected Responsiblities:
# - Portfolio Construction
# - Trade management: Selection handled by risk manager, Execution handled by broker class.
# - Performance Monitoring: PnL, Reports, Sharpe Ratio
# - Position Management: Rolling Options, Hedging, Position sizing
# - 

from copy import deepcopy
from abc import ABCMeta, abstractmethod
import pandas as pd
from EventDriven.dataclasses.orders import OrderRequest
from EventDriven.eventScheduler import EventScheduler
from EventDriven.trade import Trade
from EventDriven.types import EventTypes, FillDirection, ResultsEnum, SignalTypes
from EventDriven.riskmanager.new_base import RiskManager
from trade.helpers.Logging import setup_logger
from trade.assets.Stock import Stock
from EventDriven.event import  (
    ExerciseEvent, #noqa
    FillEvent, 
    MarketEvent, # noqa 
    OrderEvent, 
    RollEvent, 
    SignalEvent, 
    get_event_ancestor, 
    Event
)
from EventDriven.data import HistoricTradeDataHandler
from trade.helpers.helper import is_USholiday
from trade.backtester_.utils.aggregators import AggregatorParent
from trade.backtester_.utils.utils import plot_portfolio
from typing import Optional
import plotly
from EventDriven.dataclasses.states import (
    PositionState,
    PortfolioMetaInfo,
    PortfolioState,
    PositionAnalysisContext
)
from EventDriven.dataclasses.states import StrategyChangeMeta
from EventDriven.configs.core import PortfolioManagerConfig, CashAllocatorConfig
from EventDriven.portfolio_utils import extract_events
from EventDriven.exceptions import BacktestNotImplementedError
LOGGER = setup_logger("OptionSignalPortfolio")

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
    
    def __init__(self, bars : HistoricTradeDataHandler, 
                 eventScheduler: EventScheduler, 
                 risk_manager : RiskManager, 
                 weight_map = None, 
                 initial_capital = 10000,
                 *,
                 cash_allocator_config: CashAllocatorConfig | None = None,
                 t_plus_n: int = 1): 
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
        self.t_plus_n = t_plus_n
        self.bars = bars
        self.eventScheduler = eventScheduler
        self.symbol_list = self.bars.symbol_list
        self.start_date = bars.start_date.strftime("%Y%m%d")
        self.initial_capital = initial_capital
        self.risk_manager = risk_manager
        self.cash_allocator_config = cash_allocator_config or CashAllocatorConfig()
        self.underlier_list_data = {}
        self.unprocessed_signals = []
        self.allow_multiple_trades = True # allow multiple trades for the same signal_id
        self.__equity = None
        self.__transactions = []
        # call internal functions to construct key portfolio data
        self.__construct_all_positions()
        self.__construct_current_positions()
        self.__construct_weight_map(weight_map = weight_map)
        self.__construct_current_weighted_holdings()
        self.__construct_weighted_holdings()
        self.trades_df = None
        self.trades_map = {}
        self.current_cash = {}
        self.order_cache = {
            'CLOSE': {},
            'OPEN': {},
        }
        self.position_cache = {}
        self.config = PortfolioManagerConfig()

    @property
    def logger(self):
        return LOGGER
    
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
        
        for s in max_contract_price.keys():
            if max_contract_price[s] > self.allocated_cash_map[s]:
                raise ValueError(f'max_contract_price for {s} cannot be greater than allocated cash of {self.allocated_cash_map[s]}')
        self.__max_contract_price = deepcopy(max_contract_price)

    def __get_underlier_data(self, symbol: str):
        if symbol not in self.underlier_list_data:
            self.underlier_list_data[symbol] = Stock(symbol, run_chain=False)
        return self.underlier_list_data[symbol]

    @property
    def get_underlier_data(self):
        return self.__get_underlier_data

        
    def __construct_weight_map(self, weight_map): 
        unprocessed_symbols = []
        if weight_map is not None:
            for s in weight_map.keys():
                if s not in self.symbol_list:
                    unprocessed_symbols.append(s)
            if len(unprocessed_symbols) > 0:
                self.logger.critical(f"The following symbols: {unprocessed_symbols} are not being processed but present in weight_map")
            weight_map = {x : weight_map[x] * (1 - self.config.weights_haircut) for x in self.symbol_list}
            weight_total = round(sum(weight_map.values()), 4)
            assert weight_total <= 1.0, f"Sum of weights must be less than or equal to 1.0, got {weight_total}"
            
        else: 
            weight_map = {x: 1/len(self.symbol_list) for x in self.symbol_list} #spread capital between all symbols 
        
        self.__weight_map = weight_map
        self.allocated_cash_map = {s: self.__weight_map[s] * self.initial_capital for s in self.symbol_list}
        self.__max_contract_price = self.__construct_max_contract_price()

    def __construct_max_contract_price(self):
        # Try config-driven buckets first; fallback to legacy behavior
        if self.cash_allocator_config is not None:
            try:
                max_cash_map = self.cash_allocator_config.build_max_cash_map(
                    weights=self.__weight_map,
                    cash=self.initial_capital,
                )
                return {
                    s: max_cash_map.get(
                        s,
                        self.__normalize_dollar_amount_to_decimal(self.allocated_cash_map[s] * 0.5),
                    )
                    for s in self.symbol_list
                }
            except Exception as exc:  # pragma: no cover - defensive
                self.logger.warning(
                    f"Falling back to default max_contract_price due to allocator error: {exc}"
                )
        return {
            s: self.__normalize_dollar_amount_to_decimal(self.allocated_cash_map[s] * 0.5)
            for s in self.symbol_list
        }  # default max contract price is 50% of allocated cash divided by 100
    
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
    

    def generate_order(self, signal_event : SignalEvent):
        """
        Takes a signal event and creates an order event based on the signal parameters
        Interacts with RiskManager to get order based on settings and signal
        returns: OrderEvent
        """
        symbol = signal_event.symbol
        signal_type = signal_event.signal_type
        order_type = 'MKT'
        
        if signal_type != 'CLOSE': #generate order for LONG or SHORT
            order =  self.create_order( signal_event, order_type)
            self.order_cache['OPEN'].setdefault(signal_event.datetime, {})[signal_event.symbol] = order
            return order
        elif signal_type == 'CLOSE':

            ## Check if we have signal_id in current positions. If not, log warning and skip. 
            if signal_event.signal_id not in self.current_positions[symbol]:
                self.logger.warning(f'No contracts held for {symbol} to sell at {signal_event.datetime}, Inputs {locals()}')
                unprocess_dict = signal_event.__dict__
                unprocess_dict['reason'] = ('Signal not held in current positions at that time')
                self.unprocessed_signals.append(unprocess_dict)
                return None
            
            current_position = self.current_positions[symbol][signal_event.signal_id]
            ## Check if market is holiday
            if is_USholiday(signal_event.datetime):
                self.resolve_order_result({'result': ResultsEnum.IS_HOLIDAY.value}, signal_event)
                return None
            
            ## Check if we have position to close
            if 'position' not in current_position:
                self.logger.warning(f'No contracts held for {symbol} to sell at {signal_event.datetime}, Inputs {locals()}')
                return None
            
            ## Prepare order details
            position =  deepcopy(current_position['position'])
            self.logger.info(f'Selling contract for {symbol} at {signal_event.datetime} Position: {current_position}')
            position['close'] = self.calculate_close_on_position(position)

            ## Access skip from risk_manager market data
            skip = self.risk_manager.market_data.skip(position_id=position['trade_id'],
                                                      date=signal_event.datetime)
            
            ## If skip is true, either move to next trading day or skip if rolling
            if skip:

                ## If rolling, do not move to next trading day. Let it fall through
                if isinstance(signal_event.parent_event, RollEvent): 
                    self.logger.warning(f'Not generating order because: CLOSE price is negative {signal_event}, skipping sell for roll event')
                    return None
                
                # Move signal to next day 
                new_signal = deepcopy(signal_event)
                next_trading_day = new_signal.datetime + pd.offsets.BusinessDay(1)
                new_signal.datetime = next_trading_day
                self.logger.warning(f'Not generating order because: CLOSE price is negative {signal_event}, moving event to {next_trading_day}')
                self.eventScheduler.schedule_event(next_trading_day, new_signal)
                return None
            
            ## Create sell order if not skipping and return.
            order = OrderEvent(symbol, signal_event.datetime, order_type, quantity=current_position['quantity'],direction= 'SELL', position = position, signal_id=signal_event.signal_id, parent_event=signal_event)
            self.order_cache['CLOSE'].setdefault(signal_event.datetime, {})[signal_event.symbol] = order
            return order
        
        return None

    def resolve_order_result(self, position_result, signal):
        """
        Placeholder for legacy resolve_order_result logic.
        """
        self.logger.warning(f"resolve_order_result not implemented for {position_result}, {signal}")
    
    def create_order(self, signal_event : SignalEvent, position_type: str, order_type: str = 'MKT'):
        """
        Takes a signal event and creates an order event based on the signal parameters
        position_type: C|P
        """
        date_str = signal_event.datetime.strftime('%Y-%m-%d')
        position_type = 'c' if signal_event.signal_type == 'LONG' else 'p'
        cash_at_hand = self.__normalize_dollar_amount_to_decimal(self.allocated_cash_map[signal_event.symbol] * 1)
        max_contract_price = self.__max_contract_price[signal_event.symbol] if signal_event.max_contract_price is None else signal_event.max_contract_price
        max_contract_price = max_contract_price if max_contract_price <= cash_at_hand else cash_at_hand  
        print(f"Cash at Hand: {cash_at_hand}, Max Contract Price: {max_contract_price} for Signal: {signal_event.signal_id}")
        position_state = self.risk_manager.get_order(OrderRequest(date=date_str, symbol=signal_event.symbol, option_type=position_type, max_close=max_contract_price, tick_cash=cash_at_hand, direction=signal_event.signal_type, signal_id=signal_event.signal_id))
        self.position_cache[signal_event.signal_id] = position_state
        position = position_state.order.data

        # print("===========================")
        # print("Buy Details")
        # print(f"Position: {position}, Date: {date_str}, Signal: {signal_event}")
        # print(f"Max Contract Price: {max_contract_price}, Cash at Hand: {cash_at_hand}")
        # print("Cash at Hand", cash_at_hand, "Close", position['close'])
        # print("===========================")
        return OrderEvent(signal_event.symbol, signal_event.datetime, order_type, cash=cash_at_hand, direction= 'BUY', position = position, signal_id = signal_event.signal_id, quantity=position['quantity'], parent_event=signal_event)
            
    def analyze_signal(self, event : SignalEvent):
        """
        Acts on a SignalEvent to generate new orders 
        based on the portfolio logic.
        throws: AssertionError if event type is not 'SIGNAL'
        """
        assert event.type == 'SIGNAL', f"Expected 'SIGNAL' event type, got {event.type}"
        
        
        if not self.allow_multiple_trades and event.signal_type != SignalTypes.CLOSE.value:
            if len(self.current_positions[event.symbol].keys()) > 0: 
                for signal_id in self.current_positions[event.symbol]:
                   if 'exit_price' not in self.current_positions[event.symbol][signal_id]: 
                        self.logger.warning(f'Pushing signal {event} to next trading day because a position already exists for {event.symbol} with signal_id {signal_id}')
                        print(f'Pushing signal {event.signal_id} to next trading day because a position already exists for {event.symbol} with signal_id {signal_id}')
                        next_trading_day = event.datetime + pd.offsets.BusinessDay(1)
                        new_signal = deepcopy(event)
                        new_signal.datetime = next_trading_day
                        self.eventScheduler.schedule_event(next_trading_day, new_signal)
                        return None
                    
        order_event = self.generate_order(event)
        if order_event is not None:
            self.eventScheduler.put(order_event)
                
    def analyze_positions(self) -> StrategyChangeMeta : 
        """
        Analyze the current positions and determine if any need to be rolled
        """
        if not self.risk_manager.position_analyzer.config.enabled:
            self.logger.info('Position analysis is disabled in RiskManager, skipping')
            return StrategyChangeMeta(date=pd.to_datetime(self.eventScheduler.current_date), actionables=[])
        
         ## Check if current date is a holiday
         ## If holiday, skip position analysis
         ## Market is closed on holidays
         ## Use pandas to_datetime for date conversion
         ## Use is_USholiday function to check for holidays
         ## Log a warning message if market is closed
         ## Return None if market is closed
         ## Else, proceed with position analysis
         ## Create Context for current positions
         ## Analyze positions using RiskManager
         ## Extract events from meta changes and schedule them
        dt = pd.to_datetime(self.eventScheduler.current_date)
        if is_USholiday(dt):
            self.logger.warning(f"Market is closed on {dt}, skipping")
            return
        
        ## Create Context for current positions
        ctx = self._create_ctx(dt)

        ## Analyze positions using RiskManager
        meta_changes = self.risk_manager.analyze_position(ctx)

        ## Extract events from meta changes and schedule them
        events = self.extract_events(meta_changes)
        if not events:
            self.logger.info(f'No events to schedule for position analysis on {dt}')
            return meta_changes

        ## Loop through events and schedule them
        for event in events:
            self.eventScheduler.schedule_event(event.datetime, event)
        return meta_changes
        
    def _create_ctx(self, date: pd.Timestamp) -> PositionAnalysisContext:
        """
        Create a Context object for the given date
        """

        ## Create PositionState objects for all current positions
        positions = self.current_positions
        positions_states = []
        for tick, pos_pack in positions.items():
            for signal_id, position in pos_pack.items():
                trade_id = position["position"]["trade_id"]
                qty = position["position"]["quantity"]
                entry_price = position["entry_price"] / qty
                current_position_data = self.risk_manager.market_data.get_at_time_position_data(position_id=trade_id, date=date)
                current_underlier_data = self.risk_manager.market_data.market_timeseries.get_at_index(sym=tick, index=date)
                current_price = position["market_value"] / qty
                pnl = (current_price - entry_price) * qty

                pos_state = PositionState(
                    trade_id=trade_id,
                    underlier_tick=tick,
                    signal_id=signal_id,
                    quantity=qty,
                    entry_price=entry_price,
                    current_position_data=current_position_data,
                    current_underlier_data=current_underlier_data,
                    pnl=pnl,
                    last_updated=date,
                )
                positions_states.append(pos_state)
        
        ## Create Portfolio State
        cash = sum(self.allocated_cash_map.values())
        positions = positions_states
        pnl = sum([x.pnl for x in positions_states])
        total_value = cash + pnl
        last_updated = date

        portfolio_state = PortfolioState(
            total_value=total_value,
            cash=cash,
            positions=positions,
            pnl=pnl,
            last_updated=last_updated,
        )
        
        ## Create PortfolioMetaInfo
        meta = PortfolioMetaInfo(
            portfolio_name="bkt_test_11",
            initial_cash=self.initial_capital,
            start_date=self.risk_manager.start_date,
            end_date=self.risk_manager.end_date,
            t_plus_n=self.t_plus_n,
            is_backtest=True,
        )

        ## Create AnalysisContext
        ctx = PositionAnalysisContext(
            date=date,
            portfolio=portfolio_state,
            portfolio_meta=meta,
        )
        
        return ctx
    
    def extract_events(self, meta_changes: StrategyChangeMeta) -> list[Event]:
        """
        Extract events from the strategy meta changes
        """
        events = extract_events(actionables=meta_changes.actionables,
                                current_positions=self.current_positions)
        return events
            
    def execute_roll(self, roll_event: RollEvent):
        """
        Execute the roll event by closing the current position and opening a new one
        rollEvent: RollEvent
        """
        self.logger.info(f'Rolling contract for {roll_event}')
        print(f'Rolling contract (sell side) for {roll_event.symbol} at {roll_event.datetime}')
        sell_signal_event = SignalEvent( roll_event.symbol, roll_event.datetime, SignalTypes.CLOSE.value, signal_id=roll_event.signal_id, parent_event=roll_event)
        self.eventScheduler.put(sell_signal_event)
        
    def execute_roll_buy(self, roll_event: RollEvent):
        """
        Run after a successful fill on the sell side of the roll event 
        rollEvent: RollEvent
        """
        self.logger.info(f'Rolling contract for {roll_event}')
        print(f'Rolling contract (buy side) for {roll_event.symbol} at {roll_event.datetime}')
        buy_signal_event = SignalEvent( roll_event.symbol, roll_event.datetime, roll_event.signal_type , signal_id=roll_event.signal_id)
        self.eventScheduler.put(buy_signal_event)    
        
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
            raise BacktestNotImplementedError('Exercise fill handling not implemented yet')
            

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
        current_cash = {'datetime': current_date}
        
        #new weighted holdings dictionary
        new_weighted_holdings_entry = {s: self.allocated_cash_map[s] for s in self.symbol_list}
        new_weighted_holdings_entry['datetime'] = current_date
        new_weighted_holdings_entry['cash'] = (1.0 - sum(self.__weight_map.values())) * self.initial_capital
        new_weighted_holdings_entry['commission'] = self.current_weighted_holdings['commission']
        new_weighted_holdings_entry['total'] = new_weighted_holdings_entry['cash']
        
        for sym in self.symbol_list:
            new_weighted_holdings_entry[sym] = self.allocated_cash_map[sym] 
            current_cash[sym] = self.allocated_cash_map[sym] #update current cash for the symbol
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
        self.current_cash[current_date] = current_cash
        
    def update_fill(self, fill_event: FillEvent):
        """
        Updates the portfolio current positions and holdings 
        from a FillEvent.
        """
        if fill_event.type == 'FILL': 
            self.update_positions_on_fill(fill_event)
            self.update_holdings_on_fill(fill_event)
            # check if fill_event has roll event ancestor. if so execute roll buy side
            if fill_event.direction == FillDirection.SELL.value and fill_event.position is not None: 
                roll_event = get_event_ancestor(fill_event, EventTypes.ROLL.value)
                if roll_event is not None:
                    self.execute_roll_buy(roll_event)
            
    def calculate_close_on_position(self, position) -> float: 
        """
        Calculate the close price on a position
        the close price is the difference between the long and short legs of the position 
        """
        return self.risk_manager.market_data.get_at_time_position_data(position['trade_id'], self.eventScheduler.current_date).get_price()

    
    
    
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

        df = pd.DataFrame(
            records,
            columns=[
                'datetime',
                'symbol',
                'long',
                'short',
                'trade_id',
                'close',
                'quantity',
                'market_value',
                'signal_id',
            ],
        )
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

    
        
