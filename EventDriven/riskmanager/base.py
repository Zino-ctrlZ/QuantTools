## NOTE:
## 1) If a split happens during a backtest window, the trade id won't be updated. The dataframe will simply be uploaded with a the split adjusted strike.
## 2) All Greeks &  Midpoint with Zero values will be FFWD'ed

from pprint import pprint
from .utils import *
from .utils import (logger, 
                    get_timeseries_start_end,
                    set_deleted_keys,
                    date_in_cache_index,
                    add_skip_columns,
                    _clean_data,
                    PERSISTENT_CACHE)
from .actions import *
from .picker import *
from .sizer import BaseSizer, DefaultSizer, ZscoreRVolSizer
from .config import ffwd_data
from trade.helpers.helper import printmd, CustomCache, date_inbetween, compare_dates
from EventDriven.event import (
    RollEvent,
    ExerciseEvent,
    OrderEvent
)
import numpy as np
import os
from cachetools import cached, LRUCache
from EventDriven.execution import ExecutionHandler
from cachetools.keys import hashkey
import time
from cachetools import cachedmethod
from functools import lru_cache
from trade.assets.helpers.utils import (swap_ticker)
from dateutil.relativedelta import relativedelta


BASE = Path(os.environ["WORK_DIR"])/ ".riskmanager_cache"
HOME_BASE = Path(os.environ["WORK_DIR"])/".cache"
BASE.mkdir(exist_ok=True)
# logger = setup_logger('QuantTools.EventDriven.riskmanager.base')
order_cache = CustomCache(BASE, fname = "order")

special_dividend = CustomCache(HOME_BASE, fname = 'special_dividend', expire_days=1000) ## Special dividend cache for handling special dividends
special_dividend['COST'] = {
    '2020-12-01': 10,
    '2023-12-27': 15
}

def get_order_cache():
    """
    Returns the order cache
    """
    global order_cache
    return order_cache



def resolve_schema(schema: OrderSchema, 
                   tries: int, 
                   max_dte_tolerance: int, 
                   otm_moneyness_width: float, 
                   itm_moneyness_width: float,
                   max_close: float, max_tries: int =6) -> (OrderSchema, int):
    """
    Resolving schema by order of importance
    1. DTE Tolerance
    2. Min Moneyness width
    3. Max Moneyness width
    4. Max Close Price
    5. Max Schema Tries
    If no schema is found after max tries, return False and the number of tries.

    Args:
        schema (OrderSchema): The schema to resolve.
        tries (int): The number of tries already made.
        max_dte_tolerance (int): The maximum DTE tolerance to allow.
        moneyness_width (float): The moneyness width to allow.
        max_close (float): The maximum close price to allow.
        max_tries (int): The maximum number of tries allowed.

    Returns:
        tuple: A tuple containing the resolved schema or False if no schema was found, and the number of tries made.
    """

    ##0). Max schema tries
    if tries >= max_tries:
        return False, tries

    #1). DTE Resolve
    tries +=1
    if schema['dte_tolerance'] <= max_dte_tolerance:
        logger.info(f"Resolving Schema: {schema['dte_tolerance']} <= {max_dte_tolerance}, increasing DTE Tolerance by 10")
        schema['dte_tolerance'] += 20
        return schema, tries
    
    #2). Min Moneyness Resolve
    elif 1 - schema['min_moneyness'] <= otm_moneyness_width:
        logger.info(f"Resolving Schema: {1 - schema['min_moneyness']} <= {otm_moneyness_width}, decreasing Min Moneyness by 0.1")
        schema['min_moneyness'] -= 0.1
        return schema, tries    

    #3). Max Moneyness Resolve
    elif schema['max_moneyness'] - 1 <= itm_moneyness_width:
        logger.info(f"Resolving Schema: {schema['max_moneyness'] - 1} <= {itm_moneyness_width}, increasing Max Moneyness by 0.1")
        schema['max_moneyness'] += 0.1
        return schema, tries
    
    #4). Close Resolve
    elif schema['max_total_price'] <= max_close:
        logger.info(f"Resolving Schema: {schema['max_total_price']} <= {max_close}, increasing Max Close by 0.5")
        schema['max_total_price'] += 1
        return schema, tries
    
    return False, tries





class OrderPicker:
    def __init__(self, 
                 start_date: str|datetime,
                 end_date: str|datetime,
                 liquidity_threshold: int = 250, 
                 data_availability_threshold: float = 0.7, 
                 lookback: int = 30):
        """
        initializes the OrderPicker class
        
        params:
        liquidity_threshold: int: liquidity threshold. Default is 250
        data_availability_threshold: float: data availability threshold. Default is 0.7
        lookback: int: lookback. Default is 30
        """
        self.liquidity_threshold = liquidity_threshold
        self.data_availability_threshold = data_availability_threshold
        self.__lookback = lookback
        self.start_date = start_date
        self.end_date = end_date
        self.order_cache = {}
        
    @property
    def lookback(self):
        return self.__lookback
    
    @lookback.setter
    def lookback(self, value):
        global LOOKBACKS
        initial_lookback_key = list(LOOKBACKS.keys())[0]
        if value not in LOOKBACKS[initial_lookback_key].keys():
            precompute_lookbacks('2000-01-01', '2030-12-31', _range = [value])
        self.__lookback = value

    def get_order_new(self,
                      schema: OrderSchema, 
                      date: str|datetime,
                      spot,
                      print_url: bool = False):
        
        schema = tuple(schema.data.items())
        
        return self.__get_order(schema, date, spot, print_url=print_url)
    
    
    # @lru_cache(maxsize=128)
    @staticmethod
    @PERSISTENT_CACHE.memoize()
    def __get_order(schema:tuple,
                date: str|datetime,
                spot: float,
                print_url: bool = False) -> dict:
        """
        Get the order for the given schema, date, and spot price.
        """

        schema = OrderSchema(dict(schema))
        if schema['option_type'] =='C': ## This ensures that both call and put OTM are < 1.0 and ITM are > 1.0
            logger.info(f"Call Option Detected, Pre-Adjustment Moneyness: {schema['min_moneyness']} - {schema['max_moneyness']}")
            min_m, max_m = 2-schema['min_moneyness'], 2 - schema['max_moneyness']
            schema['min_moneyness'] = min(min_m, max_m) ## For Calls, we want the min moneyness to be 2 - min_moneyness
            schema['max_moneyness'] = max(min_m, max_m) ## For Calls, we want the max moneyness to be 2 - max_moneyness
            logger.info(f"Call Option Detected, Adjusting Moneyness: {schema['min_moneyness']} - {schema['max_moneyness']}")
        elif schema['option_type'] == 'P': ## This ensures that both call and put OTM are < 1.0 and ITM are > 1.0
            logger.info(f"Put Option Detected, Pre-Adjustment Moneyness: {schema['min_moneyness']} - {schema['max_moneyness']}")

        chain = populate_cache_with_chain(
            schema['tick'], 
            date, 
            print_url = print_url
        )
        raw_order = build_strategy(chain, schema, spot, dict(get_cache('spot')))
        return extract_order(raw_order)
    

    @log_error_with_stack(logger)
    def get_order(self, 
                  tick: str, 
                  date: str,
                  right: str, 
                  max_close: str,
                  order_settings: dict) -> dict:
        
        """
        returns the order for the given tick, date, right, max_close, and order_settings

        params:
        tick: str: ticker to get the order for
        date: str: date to get the order for
        right: str: right of the option contract (P or C)
        max_close: str: maximum close price
        order_settings: dict: settings for the order
            example: {'type': 'naked',
                        'specifics': [{'direction': 'long',
                        'rel_strike': .900,
                        'dte': 365,
                        'moneyness_width': 0.15},
                        {'direction': 'short',
                        'rel_strike': .80,
                        'dte': 365,
                        'moneyness_width': 0.15}],

                        'name': 'vertical_spread'}

        returns:
        dict: order
        """
        global spot_cache, close_cache, oi_cache, chain_cache
        order_cache = self.order_cache
        order_cache.setdefault(date, {})
        order_cache[date].setdefault(tick, {})

        ## Create necessary data structures
        direction_index = {}
    
        str_direction_index = {}
        for indx, v in enumerate(order_settings['specifics']):
            if v['direction'] == 'long':
                str_direction_index[indx] = 'long'
                direction_index[indx] = 1
            elif v['direction'] == 'short':
                str_direction_index[indx] = 'short'
                direction_index[indx] = -1

        order_candidates = produce_order_candidates(order_settings, tick, date, right)
        if any([x2 is None for x in order_candidates.values() for x2 in x]):
            return_item = {
                'result': "MONEYNESS_TOO_TIGHT",
                'data': None
            } 
            order_cache[date][tick] = return_item
            return return_item

        returned = populate_cache(order_candidates = order_candidates, 
                                  target_date=date, 
                                  start_date=self.start_date, 
                                  end_date=self.end_date) 
        refresh_cache()
    
        if returned == 'holiday':
            return_item = {
                'result': "IS_HOLIDAY",
                'data': None
            }
            order_cache[date][tick] = return_item
            return return_item
        
        elif returned == 'theta_data_error':

            return_item = {
                'result': "UNAVAILABLE_CONTRACT",
                'data': None
            }
            order_cache[date][tick] = return_item
            return return_item
        
        elif returned == 'weekend':
            return_item = {
                'result': "IS_WEEKEND",
                'data': None
            }
            order_cache[date][tick] = return_item
            return return_item
    

        SKIP_ORDER_CRITERIA = []
        for s in order_settings['specifics']:
            SKIP_ORDER_CRITERIA.append(s['moneyness_width'])
            
        SKIP_ORDER_CRITERIA = not all(SKIP_ORDER_CRITERIA)
        if SKIP_ORDER_CRITERIA: ## We will return the order as is.

            return_order = {
                'result': ResultsEnum.SUCCESSFUL.value,
                'data':{}
            }
            id = ''
            close = 0
            for direction in order_candidates.keys():
                return_order['data'][direction] = []
                for data in order_candidates[direction]:
                    optid = data["option_id"].unique()[0]
                    return_order['data'][direction].append(optid)
                    id+= f'&{direction.upper()[0]}:{optid}'
                    spot = get_cache('spot')[f'{optid}_{date}'] if direction == 'long' else -get_cache('spot')[f'{optid}_{date}']
                    close += spot
            return_order['data']['trade_id'] = id
            return_order['data']['close'] = close
            return return_order

        
        for direction in order_candidates: ## Fix this to use .items()
            for i,data in enumerate(order_candidates[direction]):
                data['date_available'] = data.apply(lambda x: date_in_cache_index( date, x.option_id), axis=1)
                data = data[data.date_available == True] ## Filter out contracts that are not available on the date.
                data['liquidity_check'] = data.option_id.apply(lambda x: liquidity_check(x, date, pass_threshold=self.liquidity_threshold, lookback=self.lookback))
                data = data[data.liquidity_check == True]
                if data.empty:
                    return_item = {
                        'result': "TOO_ILLIQUID",
                        'data': None
                    }
                    order_cache[date][tick] = return_item
                    return return_item
                
                data['available_close_check'] = data.option_id.apply(lambda x: available_close_check(x, date, threshold=self.data_availability_threshold))
                data = data[data.available_close_check == True] ## Filter out contracts that do not have close data.
                if data.empty:
                    return_item = {
                        'result': "NO_TRADED_CLOSE",
                        'data': None
                    }
                    order_cache[date][tick] = return_item
                    return return_item
            
                order_candidates[direction][i] = data


        ## Filter Unique Combinations per leg.
        unique_ids = {'long': [], 'short': []}
        for direction in order_candidates:
            for i,data in enumerate(order_candidates[direction]):
                unique_ids[direction].append(data[(data.liquidity_check == True) & (data.available_close_check == True)].option_id.unique().tolist())

        ## Produce Tradeable Combinations
        tradeable_ids = list(product(*unique_ids['long'], *unique_ids['short']))
        tradeable_ids, unique_ids 

        ## Keep only unique combinations. Not repeating a contract.
        filtered = [t for t in tradeable_ids if len(set(t)) == len(t)]


        ## Get the price of the structure
        ## Using List Comprehension to sum the prices of the structure per index
        results = [
            (*items, sum([direction_index[i] * spot_cache[f'{item}_{date}'] for i, item in enumerate(items)])) for items in filtered
        ]

        ## Convert to DataFrame, and sort by the price of the structure.
        return_dataframe = pd.DataFrame(results)
        if return_dataframe.empty:
            return_item = {
                'result': ResultsEnum.MONEYNESS_TOO_TIGHT.value,
                'data': None
            }
            order_cache[date][tick] = return_item

            return return_item
        cols = return_dataframe.columns.tolist()
        cols[-1] = 'close'
        return_dataframe.columns= cols
        return_dataframe = return_dataframe[(return_dataframe.close<= max_close) & (return_dataframe.close> 0)].sort_values('close', ascending = False).head(1) ## Implement for shorts. Filtering automatically removes shorts.


        if return_dataframe.empty:
            return_item = {
                'result': ResultsEnum.MAX_PRICE_TOO_LOW.value,
                'data': None
            }
            order_cache[date][tick] = return_item
            return return_item
            
        ## Rename the columns to the direction names
        return_dataframe.columns = list(str_direction_index.values()) + ['close']
        return_order = return_dataframe[list(str_direction_index.values())].to_dict(orient = 'list')
        return_order

        ## Create the trade_id with the direction and the id of the contract.
        id = ''
        for k, v in return_order.items():
            if len(v) > 0:
                id += f"&{k[0].upper()}:{v[0]}"
        return_order['trade_id'] = id
        return_order['close'] = return_dataframe.close.values[0]
        
        return_dict = {
            'result': ResultsEnum.SUCCESSFUL.value,
            'data': return_order
        }
        order_cache[date][tick] = return_dict

        return return_dict

class RiskManager:
    """
    RiskManager class for managing portfolio risk and executing strategies.

    Attributes:
    ----------
    Core Attributes:
    ----------------
    bars : Bars
        The Bars object containing historical price data for the symbols.
    events : Events
        The Events object used for event-driven backtesting.
    initial_capital : float
        The initial capital allocated for the portfolio.
    start_date : str | datetime
        The start date for the backtest, recommended to match the start date of the Bars object.
    end_date : str | datetime
        The end date for the backtest, recommended to match the end date of the Bars object.
    pm_start_date : str | datetime
        The start date for the portfolio manager.
    pm_end_date : str | datetime
        The end date for the portfolio manager.
    symbol_list : list[str]
        List of symbols available in the Bars object.
    OrderPicker : OrderPicker
        The OrderPicker object used for selecting orders based on criteria.

    Cache Attributes:
    -----------------
    spot_timeseries : CustomCache
        Cache for storing the spot price timeseries data.
    chain_spot_timeseries : CustomCache
        Cache for storing the chain spot price timeseries data, used for pricing and accounting for option strikes during splits.
    processed_option_data : CustomCache
        Cache for storing processed individual option data.
    position_data : CustomCache
        Cache for storing position data.
    dividend_timeseries : CustomCache
        Cache for storing dividend timeseries data.
    splits_raw : CustomCache
        Cache for storing raw split names and dates.
    splits : dict
        Processed split data derived from splits_raw.
    schema_cache : dict
        Cache for storing schema-related data.
    _order_cache : dict
        Cache for storing order-related data.
    id_meta : dict
        Metadata for tracking IDs.
    _analyzed_date_list : list
        List of dates that have been analyzed for actions.
        To re-analyze a date, it must be removed from this list.

    Risk Management Attributes:
    ---------------------------
    sizing_type : str
        Specifies the sizing type for calculating quantities (e.g., 'delta', 'vega', 'gamma', 'price').
    sizing_lev : float
        Multiplier for equity equivalent size (leverage). Default is 5.0.
    limits : dict[str, bool]
        Specifies which risk limits are enabled.
    greek_limits : dict[str, dict]
        Specifies the limits for individual Greeks (e.g., delta, gamma, vega, theta).
    max_moneyness : float
        Maximum moneyness before rolling positions. Default is 1.2.
    max_dte_tolerance : int
        Maximum days-to-expiration tolerance for options. Default is 90 days.
    moneyness_width : float
        Width of moneyness for filtering options. Default is 0.45.
    max_slippage : float
        Maximum allowable slippage for trades. Default is 0.25.
    re_update_on_roll : bool
        If True, the limits will be re-evaluated on roll events. Default is False.

    Pricing and Data Attributes:
    ----------------------------
    price_on : str
        Specifies the price type used for calculations (e.g., 'mkt_close'). Default is 'mkt_close'.
    option_price : str
        Specifies the option price used for pricing. Default is 'Midpoint'. Available options include:
        'Midpoint', 'Bid', 'Ask', 'Close', 'Weighted Midpoint'.
    rf_timeseries : pd.Series
        Risk-free rate timeseries data, retrieved using get_risk_free_rate_helper().
    unadjusted_signals : pd.DataFrame
        Unadjusted signals for the risk manager, used for analysis and actions.

    Miscellaneous Attributes:
    -------------------------
    data_managers : dict
        Dictionary for managing data-related objects.
    _actions : dict
        Internal dictionary for storing actions related to risk management.
    executor : ExecutionHandler
        The execution handler responsible for executing trades.
    t_plus_n : int
        Settlement delay for orders (T+N). Default is 0, meaning no settlement delay.
    """

    def __init__(self,
                 bars: DataHandler,
                 events: EventScheduler,
                 initial_capital: int|float,
                 start_date: str|datetime,
                 end_date: str|datetime,
                 executor: ExecutionHandler,
                 unadjusted_signals: pd.DataFrame,
                 portfolio_manager: 'Portfolio' = None,
                 price_on = 'close',
                 option_price = 'Midpoint',
                 sizing_type = 'delta',
                 leverage = 5.0,
                 max_moneyness = 1.2,
                 t_plus_n = 0,
                 **kwargs
                 ):
        """
        Methods:
            --------
            __init__(self, bars: Bars, events: Events, initial_capital: float, start_date: str | datetime, end_date: str | datetime,
                    executor: ExecutionHandler, unadjusted_signals: pd.DataFrame, portfolio_manager: PortfolioManager = None,
                    price_on: str = 'mkt_close', option_price: str = 'Midpoint', sizing_type: str = 'delta', leverage: float = 5.0,
                    max_moneyness: float = 1.2, t_plus_n: int = 0, **kwargs):
                Initializes the RiskManager class and sets up attributes for managing portfolio risk.

                Parameters:
                ----------
                bars : Bars
                    The Bars object containing historical price data for the symbols.
                events : Events
                    The Events object used for event-driven backtesting.
                initial_capital : float
                    The initial capital allocated for the portfolio.
                start_date : str | datetime
                    The start date for the backtest, recommended to match the start date of the Bars object.
                end_date : str | datetime
                    The end date for the backtest, recommended to match the end date of the Bars object.
                executor : ExecutionHandler
                    The execution handler responsible for executing trades.
                unadjusted_signals : pd.DataFrame
                    Unadjusted signals for the risk manager, used for analysis and actions.
                portfolio_manager : PortfolioManager, optional
                    The PortfolioManager object for managing portfolio positions and orders. Default is None.
                price_on : str, optional
                    Specifies the price type used for calculations (e.g., 'mkt_close'). Default is 'mkt_close'.
                option_price : str, optional
                    Specifies the option price used for pricing. Default is 'Midpoint'. Available options include:
                    'Midpoint', 'Bid', 'Ask', 'Close', 'Weighted Midpoint'.
                sizing_type : str, optional
                    Specifies the sizing type for calculating quantities (e.g., 'delta', 'vega', 'gamma', 'price'). Default is 'delta'.
                leverage : float, optional
                    Multiplier for equity equivalent size (leverage). Default is 5.0.
                    Example: (Cash Available / Spot Price) * Leverage = Equity Equivalent Size.
                max_moneyness : float, optional
                    Maximum moneyness before rolling positions. Default is 1.2.
                t_plus_n : int, optional
                    Settlement delay for orders (T+N). Default is 0, meaning no settlement delay.
                **kwargs : dict, optional
                    Additional keyword arguments for customization. Expected keys include:
                    - `max_dte_tolerance` (int): Maximum days-to-expiration tolerance for options. Default is 90 days.
                    - `moneyness_width` (float): Width of moneyness for filtering options. Default is 0.45.
                    - `max_tries` (int): Maximum number of tries to resolve schema. Default is 20.
        """
        
        assert sizing_type in ['delta', 'vega', 'gamma', 'price'], f"Sizing Type {sizing_type} not recognized, expected 'delta', 'vega', 'gamma', or 'price'"
        order_cache.clear()
        global DELETED_KEYS
        set_deleted_keys([]) ## Set the deletion keys for the cache
        start, end = get_timeseries_start_end()
        self.bars = bars
        self.events = events
        self.initial_capital = initial_capital
        self.__pm = portfolio_manager
        self.start_date = start
        self.pm_start_date = start_date
        self.pm_end_date = end_date
        self.end_date = end
        self.symbol_list = self.bars.symbol_list
        self.OrderPicker = OrderPicker(start, end)
        self.spot_timeseries = CustomCache(BASE, fname = "rm_spot_timeseries", expire_days=100)
        self.chain_spot_timeseries = CustomCache(BASE, fname = "rm_chain_spot_timeseries", expire_days=100) ## This is used for pricing, to account option strikes for splits
        self.processed_option_data = CustomCache(BASE, fname = "rm_processed_option_data", expire_days=100)
        self.position_data = CustomCache(BASE, fname = "rm_position_data", clear_on_exit=True)
        self.dividend_timeseries = CustomCache(BASE, fname = "rm_dividend_timeseries", expire_days=100)
        self.sizing_type = sizing_type
        self.sizing_lev = leverage
        self.limits = {
            'delta': True,
            'gamma': False,
            'vega': False,
            'theta': False,
            'dte': False,
            'moneyness': False
        }
        self.greek_limits = {
            'delta': {},
            'gamma': {},
            'vega': {},
            'theta': {}
        }
        self.data_managers = {}

        ## Might want to make this changeable in future
        self.rf_timeseries = get_risk_free_rate_helper()['annualized']
        self.price_on = price_on
        self.max_moneyness = max_moneyness
        self.option_price = option_price
        self._actions = {}
        self.splits_raw =CustomCache(HOME_BASE, fname = "split_names_dates", expire_days = 1000)
        self.splits = self.set_splits(self.splits_raw)
        self.schema_cache = {}
        self.max_dte_tolerance = kwargs.get('max_dte_tolerance', 90) ## Default is 90 days
        self.otm_moneyness_width = kwargs.get('moneyness_width', 0.45) ## Default is 0.45
        self.itm_moneyness_width = kwargs.get('itm_moneyness_width', 0.45) ## Default is 0.45
        self.max_tries = kwargs.get('max_tries', 20) ## Default is 20 tries to resolve schema
        self.__analyzed_date_list = [] ## List of dates that have been analyzed for actions
        self._order_cache = {}
        self.id_meta = {}
        self.t_plus_n = t_plus_n ## T+N settlement for the orders, default is 0, meaning no settlement delay. Orders will be placed on the same day.
        self.max_slippage = 0.25
        self.executor = executor
        self.re_update_on_roll = False ## If True, the limits will be re-evaluated on roll events. Default is False
        self.unadjusted_signals = unadjusted_signals ## Unadjusted signals for the risk manager, used for analysis and actions
        self.special_dividends = CustomCache(HOME_BASE, fname = 'special_dividend', expire_days=1000) ## Special dividend cache for handling special dividends
        self.__sizer = None

    @property 
    def option_data(self):
        global close_cache
        return close_cache  
    
    @property
    def order_cache(self):
        """
        Returns the order cache
        """
        return self._order_cache
    
    @property
    def sizer(self):
        """
        Getter for the sizer
        """
        if isinstance(self.__sizer, (BaseSizer, DefaultSizer, ZscoreRVolSizer)):
            return self.__sizer
        elif self.__sizer is None:
            self.__sizer = DefaultSizer(pm=self.__pm, rm=self, sizing_lev=self.sizing_lev)
            return self.__sizer
        else:
            raise TypeError("Sizer must be an instance of BaseSizer or its subclasses. Reset with None to use DefaultSizer.")

    @sizer.setter
    def sizer(self, value):
        """
        Setter for the sizer
        """
        if isinstance(value, (BaseSizer, DefaultSizer, ZscoreRVolSizer)) or value is None:
            self.__sizer = value
        else:
            raise TypeError("Sizer must be an instance of BaseSizer or its subclasses.")
    
    def clear_caches(self):
        """
        Clears the caches
        """
        self.spot_timeseries.clear()
        self.chain_spot_timeseries.clear()
        # self.position_data.clear()
        self.dividend_timeseries.clear()

    
    @property
    def pm(self):
        return self.__pm
    
    @pm.setter
    def pm(self, value):
        self.__pm = value

    @property
    def actions(self):
        return pd.DataFrame(self._actions).T


    def set_splits(self, d):
        """
        Setter for splits
        """
        splits_dict = {}
        for k, v in d.items():
            splits_dict[k] = []
            for d in v:
                if date_inbetween(d[0], self.start_date, self.end_date):
                    splits_dict[k].append(d)
        return splits_dict


    def print_settings(self):
        msg = f"""
Risk Manager Settings:
Start Date: {self.pm_start_date}
End Date: {self.pm_end_date}
Current Limits State (Position Adjusted when these thresholds are reached):
    Delta: {self.limits['delta']}
    Gamma: {self.limits['gamma']}
    Vega: {self.limits['vega']}
    Theta: {self.limits['theta']}
    Roll On DTE: {self.limits['dte']}
        Min DTE Threshold: {self.pm.min_acceptable_dte_threshold}
    Roll On Moneyness: {self.limits['moneyness']}
        Max Moneyness: {self.max_moneyness}
Quanitity Sizing Type: {self.sizing_type}
            """
        print(msg)
        

    def get_order(self, *args, **kwargs):
        """
        Compulsory variables for OrderSchema:
        signal_id: str: Unique identifier for the signal
        date: str|datetime: Date for which the order is to be placed
        tick: str: Ticker for the option contract
        max_close: float: Maximum close price for the order
        strategy: str: Strategy type
        option_type: str: Option type
        target_dte: int: Target days to expiration
        structure_direction: str: Direction of the structure
        

        Optional variables:
        spread_ticks: int: Number of ticks for the spread, default is 1
        dte_tolerance: int: Tolerance for days to expiration, default is 60
        min_moneyness: float: Minimum moneyness for the order, default is 0.75
        max_moneyness: float: Maximum moneyness for the order, default is 1.25
        min_total_price: float: Minimum total price for the order, default is max_close/2

        This function generates an order based on the provided parameters and returns it.
        """
        ## Initialize the order cache if it doesn't exist
        order_cache = self.order_cache
        signalID = kwargs.pop('signal_id')
        date = kwargs.get('date')
        tick = kwargs.get('tick')
        max_close = kwargs.get('max_close', 2.0) 
        option_strategy = kwargs.pop('strategy')
        option_type = kwargs.pop('option_type')
        structure_direction = kwargs.pop('structure_direction')
        spread_ticks = kwargs.pop('spread_ticks', 1)
        dte_tolerance = kwargs.pop('dte_tolerance', 60)
        min_moneyness = kwargs.pop('min_moneyness', 0.75)
        max_moneyness = kwargs.pop('max_moneyness', 1.25)
        target_dte = kwargs.pop('target_dte')
        min_total_price = kwargs.pop('min_total_price', max_close/2)
        
        if is_USholiday(date):
            logger.info(f"Date {date} is a US Holiday, skipping order generation")
            return {
                'result': ResultsEnum.IS_HOLIDAY.value,
                'data': None
            }


        self.generate_data(tick)
        spot = self.chain_spot_timeseries[tick][date] 
        logger.info(f"## ***Signal ID: {signalID}***")

        ## I cannot calculate greeks here. I need option_data to be available first.
        # order = self.OrderPicker.get_order(*args, **kwargs)    

        ## Testing new order picker
        schema = OrderSchema({
            "strategy": option_strategy, "option_type": option_type, "tick": tick,
            "target_dte": target_dte, "dte_tolerance": dte_tolerance,
            "structure_direction": structure_direction, "max_total_price": max_close,
            "spread_ticks":spread_ticks, "min_moneyness": min_moneyness, "max_moneyness": max_moneyness, "increment": 0.5,
            "min_total_price": min_total_price
        })
        order = self.OrderPicker.get_order_new(schema, date, spot, print_url = False)

        ## Resolve the schema if the order is not successful
        tries = 0
        while order['result'] != ResultsEnum.SUCCESSFUL.value:
            logger.info(f"Failed to produce order with schema: {schema}, trying to resolve schema, on try {tries}")
            pack = resolve_schema(schema,
                                           tries = tries,
                                            max_dte_tolerance = self.max_dte_tolerance,
                                            max_close = self.pm.allocated_cash_map[tick]/100,
                                            max_tries = self.max_tries,
                                            otm_moneyness_width = self.otm_moneyness_width,
                                            itm_moneyness_width = self.itm_moneyness_width)
            schema, tries = pack

            if schema is False:
                logger.info(f"Unable to resolve schema after {tries} tries, returning None")
                self.schema_cache.setdefault(date, {}).update({signalID: schema})
                return {
                    'result': ResultsEnum.NO_CONTRACTS_FOUND.value,
                    'data': None
                }
            logger.info(f"Resolved Schema: {schema}, tries: {tries}")
            order = self.OrderPicker.get_order_new(schema, date, spot, print_url = False) ## Get the order from the OrderPicker

            
        self.schema_cache.setdefault(date, {}).update({signalID: schema}) ## Update the schema cache with the date and signalID

        
        signal_meta = parse_signal_id(signalID)
        logger.info(f"Order Produced: {order}")


        if order['result'] == ResultsEnum.SUCCESSFUL.value:
            print(f"\nOrder Received: {order}\n")

            position_id = order['data']['trade_id']
            # self.register_option_meta_frame(date, position_id) ## Register the option meta frame for the position
            
        else:
            print(f"\nOrder Failed: {order}\n")
            logger.info(f"Signal ID: {signalID}, Unable to produce order, returning None")
            return order
        
        logger.info(f"Position ID: {position_id}")
        logger.info("Calculating Position Greeks")
        self.calculate_position_greeks(position_id, kwargs['date'])
        order = self.update_order_close(position_id, kwargs['date'], order) ## Update the order with the close price from the position data
        logger.info('Updating Signal Limits')
        # self.update_greek_limits(signalID, position_id)
        self.sizer.update_delta_limit(signalID, position_id, date)
        logger.info("Calculating Quantity")
        # quantity = self.calculate_quantity(position_id, signalID, kwargs['date'], order['data']['close'])
        quantity = self.sizer.calculate_position_size(signalID, position_id, order['data']['close'], kwargs['date'])
        logger.info(f"Quantity for Position ({position_id}) Date {kwargs['date']}, Signal ID {signalID} is {quantity}")
        order['data']['quantity'] = quantity
        order['data']['cash_equivalent_qty'] = self.pm.allocated_cash_map[tick] // (order['data']['close'] * 100)
        logger.info(order)

        ## save the order in the cache
        if date not in order_cache:
            cache_dict = {tick: order}
            order_cache[date] = cache_dict
        else:
            cache_dict = order_cache[date]
            cache_dict[tick] = order
            order_cache[date] = cache_dict
        
        self.adjust_slippage(position_id, date) ## Adjust the slippage for the position based on the position data

        return order
    
    def adjust_slippage(self, position_id, date):
        position_data = self.position_data.get(position_id, None)
        if position_data is None:
            logger.error(f"Position Data for {position_id} not available, cannot adjust slippage")
            return None
        
        if 'spread_ratio' in position_data:  
            spread_ratio = position_data['spread_ratio'][date] if position_data['spread_ratio'][date] else self.max_slippage
            decided_slippage = min(spread_ratio, self.max_slippage)
            logger.info(f"Position {position_id} on date {date} has spread ratio {spread_ratio}, adjusting slippage to {decided_slippage}")
            self.executor.max_slippage_pct = decided_slippage
        else:
            logger.warning(f"Spread Ratio not available for position {position_id}, using default max slippage of {self.max_slippage}")
            self.executor.max_slippage_pct = self.max_slippage
    
    def update_order_close(self, position_id:str, date:str|datetime, order:dict)-> dict:
        """
        Updates the close price of the order based on the position data.
        Parameters:
        position_id: str: ID of the position
        date: str|datetime: Date for which the order is to be updated
        order: dict: Order dictionary containing the order details
        Returns:
        dict: Updated order dictionary with the close price
        """

        close = self.position_data[position_id][self.option_price.capitalize()][date]
        order['data']['close'] = close
        return order
    

    def register_option_meta_frame(
            self,
            date: str|datetime,
            trade_id:str,
    ) -> None:
        
        ## Generate a DataFrame for each direction in the trade
        trade_meta = self.parse_position_id(trade_id)[0]
        direction_pair = self.id_meta.setdefault(trade_id, {'L': pd.DataFrame(
            index = bus_range(self.pm_start_date, self.pm_end_date, freq='1d'),
        ), 'S': pd.DataFrame(
            index = bus_range(self.pm_start_date, self.pm_end_date, freq='1d'),
        )})

        ## Get split info
        splits = self.splits

        ## Loop through each direction 
        for direction, details in trade_meta.items():
            direction_frame = direction_pair.get(direction, pd.DataFrame())

            ## Loop through each option in the direction
            for i, option in enumerate(details):

                ## First populate the Given Option Detail
                direction_frame[i] = generate_option_tick_new(*option.values())
                tick = option['ticker']
                split_info = splits.get(tick, None)
                if split_info is None:
                    continue

                ## If there is split info, we adjust
                for split_meta in split_info:
                    if not compare_dates.is_after(split_meta[0], date):
                        continue
                    split_start, split_ratio = split_meta
                    new_details = deepcopy(option)
                    new_details['strike']/=split_meta[1]
                    direction_frame.loc[split_start:, i] = generate_option_tick_new(*new_details.values())


    def calculate_position_greeks(self, positionID, date):
        """
        Calculate the greeks of a position

        date: Evaluation Date for the greeks (PS: This is not the pricing date)
        positionID: str: position string. (PS: This function assumes ticker for position is the same)
        """
        logger.info(f"Calculate Greeks Dates Start: {self.start_date}, End: {self.end_date}, Position ID: {positionID}, Date: {date}")
        if positionID in self.position_data:
            ## If the position data is already available, then we can skip this step
            logger.info(f"Position Data for {positionID} already available, skipping calculation")
            return self.position_data[positionID]
        else:
            logger.critical(f"Position Data for {positionID} not available, calculating greeks. Load time ~5 minutes")
        ## Initialize the Long and Short Lists
        long = []
        short = []
        threads = []
        thread_input_list = [
            [], []
        ]

        date = pd.to_datetime(date) ## Ensure date is in datetime format
        
        ## First get position info
        position_dict, positon_meta = self.parse_position_id(positionID)

        ## Now ensure that the spot and dividend data is available
        for p in position_dict.values():
            for s in p:
                self.generate_data(swap_ticker(s['ticker']))
        ticker = swap_ticker(s['ticker'])

        # @log_time(time_logger)
        def get_timeseries(_id, direction):
            logger.info("Calculate Greeks dates")
            logger.info(f"Start Date: {self.start_date}")
            logger.info(f"End Date: {self.end_date}")
            
            print(f"Calculating Greeks for {_id} on {date} in {direction} direction")
            data = self.generate_option_data_for_trade(_id, date) ## Generate the option data for the trade

            if direction == 'L':
                long.append(data)
            elif direction == 'S':
                short.append(data)
            else:
                raise ValueError(f"Position Type {_set[0]} not recognized")
            
            return data

        ## Calculating IVs & Greeks for the options
        for _set in positon_meta:
            thread_input_list[0].append(_set[1]) ## Append the option id to the thread input list
            thread_input_list[1].append(_set[0]) ## Append the direction to the thread input list
        runThreads(get_timeseries, thread_input_list, block=True) ## Run the threads to get the timeseries data for the options
            
        position_data = sum(long) - sum(short)
        position_data = position_data[~position_data.index.duplicated(keep = 'first')]
        position_data.columns = [x.capitalize() for x in position_data.columns]
        ## Retain the spot, risk free rate, and dividend yield for the position, after the greeks have been calculated & spread values subtracted
        position_data['s0_close'] = self.spot_timeseries[ticker] ## Spot price at the time of the position
        position_data['s'] = self.chain_spot_timeseries[ticker] ## Chain spot price at the time of the position
        position_data['r'] = self.rf_timeseries ## Risk free rate at the time of the position
        position_data['y'] = self.dividend_timeseries[ticker] ## Dividend yield at the time of the position
        position_data['spread'] = position_data['Closeask'] - position_data['Closebid'] ## Spread is the difference between the ask and bid prices
        position_data['spread_ratio'] = (position_data['spread'] / position_data['Midpoint'] ).abs().replace(np.inf, np.nan).fillna(0) ## Spread ratio is the spread divided by the midpoint price
        position_data = add_skip_columns(position_data, positionID, ['Delta', 'Gamma', 'Vega', 'Theta', 'Midpoint'], window = 20, skip_threshold=3)
        self.position_data[positionID] = position_data
        return position_data


    def load_position_data(self, opttick):
        """
        Load position data for a given option tick.

        This function ONLY retrives the data for the option tick, it does not apply any splits or adjustments.
        This function will NOT check for splits or special dividends. It will only retrieve the data for the given option tick.
        """
        ## Check if the option tick is already processed
        if opttick in self.processed_option_data:
            return self.processed_option_data[opttick]

        ## Get Meta
        meta = parse_option_tick(opttick)

        ## Generate data
        data = self.generate_spot_greeks( opttick)
        data = self.enrich_data(data, meta['ticker'])
        self.processed_option_data[opttick] = data
        return data

    def enrich_data(self, data, ticker):
        """
        Enrich the data with additional information.
        """
        data = _clean_data(data)
        data = data[~data.index.duplicated(keep = 'last')]
        data['s'] = self.chain_spot_timeseries[ticker]
        data['r'] = self.rf_timeseries
        data['y'] = self.dividend_timeseries[ticker]
        data['s0_close'] = self.spot_timeseries[ticker]
        data = ffwd_data(data, ticker)
        return data
        
    def generate_spot_greeks(self, opttick):
        """
        Generate spot greeks for a given option tick.
        """
        meta = parse_option_tick(opttick)
        data_manager = OptionDataManager(opttick=opttick)
        greeks = data_manager.get_timeseries(start = self.start_date,
                                                end = self.end_date,
                                                interval = '1d',
                                                type_ = 'greeks',).post_processed_data ## Multiply by the shift to account for splits
        greeks_cols = [x for x in greeks.columns if 'Midpoint' in x]
        greeks = greeks[greeks_cols]
        greeks[greeks_cols] = greeks[greeks_cols].replace(0, np.nan).fillna(method = 'ffill') ## FFill NaN values and 0 Values
        greeks.columns = [x.split('_')[1].capitalize() for x in greeks.columns]

        spot = data_manager.get_timeseries(start = self.start_date,
                                            end = self.end_date,
                                            interval = '1d',
                                            type_ = 'spot',
                                            extra_cols=['bid', 'ask']).post_processed_data ## Using chain spot data to account for splits
        spot = spot[[self.option_price.capitalize()] + ['Closeask', 'Closebid']]
        data = greeks.join(spot)
        return data


    def generate_option_data_for_trade(self, opttick, check_date):
        """
        Generate option data for a given trade.
        This function retrieve the option data to backtest on. Data will not be saved, as it will be applying splits and adjustments.
        This function is written with the assumption that there is no cummulative splits. Expectation is only one split per option tick.
            Obviously, this might not be the case if the option was alive for ~5 years or more. But most options are not alive for that long.
        """

        meta = parse_option_tick(opttick)

        ## Check if there's any split/special dividend
        splits = self.splits.get(meta['ticker'], [])
        dividends = self.special_dividends.get(meta['ticker'], {})
        to_adjust_split = []

        ## To avoid loading multiple data to account for splits everytime, we check if the PM_date range includes the split date  
        for pack in splits:
            if compare_dates.inbetween(
                pack[0], 
                self.pm_start_date, 
                self.pm_end_date,
            ):
                pack = list(pack)  ## Convert to list to append later
                pack.append('SPLIT')
                to_adjust_split.append(pack)

        for pack in dividends.items():
            if compare_dates.inbetween(
                pack[0], 
                self.pm_start_date, 
                self.pm_end_date,
            ):
                pack = list(pack)
                pack.append('DIVIDEND')
                to_adjust_split.append(pack)

        ## Sort the splits by date
        to_adjust_split.sort(key=lambda x: x[0])  ## Sort by date

        ## If there are no splits, we can just load the data
        if not to_adjust_split:
            data = self.load_position_data(opttick).copy()  ## Copy to avoid modifying the original data
            return data[(data.index >= pd.to_datetime(self.pm_start_date) - relativedelta(months = 3))\
                         & (data.index<= pd.to_datetime(self.pm_end_date) + relativedelta(months = 3))]

        # If there are splits, we need to load the data for each tick after adjusting strikes
        else:
            adj_meta = meta.copy()
            adj_strike = meta['strike']
            logger.info(f"Generating data for {opttick} with splits: {to_adjust_split}")
            ## Load the data for picked option first
            first_set_data = self.load_position_data(opttick).copy()  ## Copy to avoid modifying the original data
            if compare_dates.is_before(check_date, to_adjust_split[0][0]):
                first_set_data = first_set_data[first_set_data.index < to_adjust_split[0][0]]
            else:
                first_set_data = first_set_data[first_set_data.index >= to_adjust_split[0][0]]

            segments = []

            for event_date, factor, event_type in to_adjust_split:
                if compare_dates.is_before(check_date, event_date):
                    # You're in the PRE-event regime
                    if event_type == 'SPLIT':
                        adj_strike /= factor
                    elif event_type == 'DIVIDEND':
                        adj_strike -= factor
                else:
                    # You're in the POST-event regime
                    if event_type == 'SPLIT':
                        adj_strike *= factor
                    elif event_type == 'DIVIDEND':
                        adj_strike += factor

                adj_opttick = generate_option_tick_new(
                    symbol=adj_meta['ticker'],
                    strike=adj_strike,
                    right=adj_meta['put_call'],
                    exp=adj_meta['exp_date']
                )
                logger.info(f"Adjusted option tick: {adj_opttick} for event {event_type} on {event_date} with factor {factor}")

                # Load adjusted data
                if adj_opttick not in self.processed_option_data:
                    adj_data = self.load_position_data(adj_opttick).copy()
                else:
                    adj_data = self.processed_option_data[adj_opttick]

                # Slice around the event
                if compare_dates.is_before(check_date, event_date):
                    adj_data = adj_data[adj_data.index >= event_date]
                else:
                    adj_data = adj_data[adj_data.index < event_date]

                # Apply price transformation if SPLIT
                if event_type == 'SPLIT':
                    cols = ['Midpoint', 'Closeask', 'Closebid']
                    if compare_dates.is_before(check_date, event_date):
                        adj_data[cols] *= factor
                    else:
                        adj_data[cols] /= factor
                    
                segments.append(adj_data)

        
        base_data = self.load_position_data(opttick).copy()
        first_event_date = to_adjust_split[0][0] if to_adjust_split else self.pm_start_date
        if compare_dates.is_before(check_date, first_event_date):
            base_data = base_data[base_data.index < first_event_date]
            
        else:
            base_data = base_data[base_data.index >= first_event_date]
        
        segments.insert(0, base_data)
        final_data = pd.concat(segments).sort_index()
        final_data = final_data[~final_data.index.duplicated(keep='last')]
        
        ## Leave residual data outside the PM date range
        final_data = final_data[(final_data.index >= pd.to_datetime(self.pm_start_date) - relativedelta(months = 3)) & \
                                (final_data.index <= pd.to_datetime(self.pm_end_date) + relativedelta(months = 3))]
        return final_data

    @log_time(time_logger)
    def update_greek_limits(self, signal_id, position_id):
        """
        Updates the limits associated with a signal
        ps: This should only be updated on first purchase of the signal
            Limits are saved in absolute values to account for both long and short positions
        
        """
        
        ## We want to update delta limits for now.
        ## This should be based on the SignalID.
        ## I will use The date from Signal ID To create the limit
        ## Goal is to enfore the limit on the signal, not the position
        
        if signal_id in self.greek_limits['delta'] and not self.re_update_on_roll: ## May consider to maximize cash on roll
            logger.info(f"Greek Limits for Signal ID: {signal_id} already updated, skipping")
            return
        logger.info(f"Updating Greek Limits for Signal ID: {signal_id} and Position ID: {position_id}")
        id_details = parse_signal_id(signal_id)
        cash_available = self.pm.allocated_cash_map[swap_ticker(id_details['ticker'])]
        delta_at_purchase = self.position_data[position_id]['Delta'][id_details['date']] 
        s0_at_purchase = self.position_data[position_id]['s'][id_details['date']] ## As always, we use the chain spot data to account for splits
        # equivalent_delta_size = (math.floor(cash_available/s0_at_purchase)/100) * self.sizing_lev
        equivalent_delta_size = ((cash_available/s0_at_purchase)/100) * self.sizing_lev
        self.greek_limits['delta'][signal_id] = abs(equivalent_delta_size)
        logger.info(f"Spot Price at Purchase: {s0_at_purchase} at time {id_details['date']}")
        logger.info(f"Delta at Purchase: {delta_at_purchase}")
        logger.info(f"Equivalent Delta Size: {equivalent_delta_size}, with Cash Available: {cash_available}, and Leverage: {self.sizing_lev}")
        logger.info(f"Equivalent Delta Size: {equivalent_delta_size}")

    def calculate_quantity(self, positionID, signalID, date, opt_price) -> int:
        """
        Returns the quantity of the position that can be bought based on the sizing type
        """
        logger.info(f"Calculating Quantity for Position ID: {positionID} and Signal ID: {signalID} on Date: {date}")
        if positionID not in self.position_data: ## If the position data isn't available, calculate the greeks
            self.calculate_position_greeks(positionID, date)
        
        ## First get position info and ticker
        position_dict, _ = self.parse_position_id(positionID)
        key = list(position_dict.keys())[0]
        ticker = swap_ticker(position_dict[key][0]['ticker'])

        ## Now calculate the max size cash can buy
        cash_available = self.pm.allocated_cash_map[ticker]
        purchase_date = pd.to_datetime(date)
        s0_at_purchase = self.position_data[positionID]['s'][purchase_date]  ## s -> chain spot, s0_close -> adjusted close
        logger.info(f"Spot Price at Purchase: {s0_at_purchase} at time {purchase_date}")
        # opt_price = self.position_data[positionID]['Midpoint'][purchase_date]
        logger.info(f"Cash Available: {cash_available}, Option Price: {opt_price}, Cash_Available/OptPRice: {(cash_available/(opt_price*100))}")
        max_size_cash_can_buy = abs(math.floor(cash_available/(opt_price*100))) ## Assuming Allocated Cash map is already in 100s

        if self.sizing_type == 'price':
            return max_size_cash_can_buy
          
        elif self.sizing_type.capitalize() == 'Delta':
            delta = self.position_data[positionID]['Delta'][purchase_date]
            if signalID not in self.greek_limits['delta']:
                self.update_greek_limits(signalID,positionID )
            target_delta = self.greek_limits['delta'][signalID]
            logger.info(f"Target Delta: {target_delta}")
            delta_size = (math.floor(target_delta/abs(delta)))
            logger.info(f"Delta from Full Cash Spend: {max_size_cash_can_buy * delta}, Size: {max_size_cash_can_buy}")
            logger.info(f"Delta with Size Limit: {delta_size * delta}, Size: {delta_size}")
            return delta_size if abs(delta_size) <= abs(max_size_cash_can_buy) else max_size_cash_can_buy
        
        elif self.sizing_type.capitalize() in ['Gamma', 'Vega']:
            raise NotImplementedError(f"Sizing Type {self.sizing_type} not yet implemented, please use 'delta' or 'price'")
        
        else:
            raise ValueError(f"Sizing Type {self.sizing_type} not recognized")
        
    def analyze_position(self):
        """
        Analyze the current positions and determine if any need to be rolled, closed, or adjusted
        """
        position_action_dict = {} ## This will be used to store the actions for each position
        date = pd.to_datetime(self.pm.eventScheduler.current_date)
        if date in self.__analyzed_date_list: ## If the date has already been analyzed, return
            logger.info(f"Positions already analyzed on {date}, skipping")
            return "ALREADY_ANALYZED"
        
        self.__analyzed_date_list.append(date) ## Add the date to the analyzed list
        event_date = pd.to_datetime(date) + BDay(self.t_plus_n) ## Order date is the next business day after the current date
        logger.info(f"Analyzing Positions on {date}")
        is_holiday = is_USholiday(date)
        if is_holiday:
            self.pm.logger.warning(f"Market is closed on {date}, skipping")
            logger.info(f"Market is closed on {date}, skipping")
            return "IS_HOLIDAY"

        ## First check if the position needs to be rolled
        if self.limits['dte']:
            roll_dict = self.dte_check()
        else:
            logger.info("Roll Check Not Enabled")
            roll_dict = {}
            for sym in self.pm.symbol_list:
                current_position = self.pm.current_positions[sym]
                if 'position' not in current_position:
                    continue
                # roll_dict[current_position['position']['trade_id']] = OpenPositionAction.HOLD.value
                roll_dict[current_position['position']['trade_id']] = HOLD(current_position['position']['trade_id'])

        logger.info(f"Roll Dict {roll_dict}")

        ## Check if the position needs to be adjusted based on moneyness
        if self.limits['moneyness']:
            moneyness_dict = self.moneyness_check()
        else:
            logger.info("Moneyness Check Not Enabled")
            moneyness_dict = {}
            for sym in self.pm.symbol_list:
                current_position = self.pm.current_positions[sym]
                if 'position' not in current_position:
                    continue
                # moneyness_dict[current_position['position']['trade_id']] = OpenPositionAction.HOLD.value
                moneyness_dict[current_position['position']['trade_id']] = HOLD(current_position['position']['trade_id'])
        logger.info(f"Moneyness Dict: {moneyness_dict}")

        ## Check if the position needs to be adjusted based on greeks
        greek_dict = self.limits_check()
        logger.info(f"Greek Dict {greek_dict}")

        check_dicts = [roll_dict, moneyness_dict, greek_dict]
        all_empty = all([len(x)==0 for x in check_dicts])

        if all_empty: ## Return if all are empty
            self.pm.logger.info(f"No positions need to be adjusted on {date}")
            print(f"No positions need to be adjusted on {date}")
            return "NO_POSITIONS_TO_ADJUST"
        
        actions_dicts = {
            'dte': roll_dict,
            'moneyness': moneyness_dict,
            'greeks': greek_dict
        }
        ## Aggregate the results
        trades_df = self.unadjusted_signals
        bars_trade = self.bars.trades_df
        for sym in self.pm.symbol_list:
            position = self.pm.current_positions[sym]
            for signal_id, current_position in position.items():
                if 'position' not in current_position:
                    continue
                k = current_position['position']['trade_id']
                exit_signal_date = trades_df[trades_df['signal_id'] == signal_id].ExitTime.values[0] ## This is not look ahead because Signal is gotten on bars_df - t_plus_n
                entry_signal_date = trades_df[trades_df['signal_id'] == signal_id].EntryTime.values[0] ## This is not look ahead because Signal is gotten on bars_df - t_plus_n        
                exit_date, entry_date = bars_trade[bars_trade['signal_id'] == signal_id].ExitTime.values[0], bars_trade[bars_trade['signal_id'] == signal_id].EntryTime.values[0]
                if compare_dates.is_on_or_after(date, exit_signal_date) or compare_dates.is_on_or_before(date, entry_date):
                    logger.info(f"Position has exited on {exit_signal_date} or not yet entered on {entry_date}, skipping")
                    continue
                




                ## There are 4 possible actions: roll, Hold, Exercise, Adjust
                ## Roll happens on DTE & Moneyness. Exercise happens on DTE. Adjust happens on Greeks
                actions = []
                reasons = []
                for action in actions_dicts:
                    if k in actions_dicts[action]:
                        actions.append(actions_dicts[action][k])
                        reasons.append(action)
                    else:
                        actions.append(OpenPositionAction.HOLD.value)
                        reasons.append('hold')
                
                sub_action_dict = {'action': '', 'quantity_diff': 0}

                ## If the position needs to be rolled or exercised, do that first, no need to check other actions or adjust quantity
                if OpenPositionAction.ROLL.value in actions:
                    pos_action = ROLL(k, {})
                    pos_action.reason = reasons[actions.index(OpenPositionAction.ROLL.value)]
                    
                    event = RollEvent(
                        datetime = event_date,
                        symbol = sym,
                        signal_type = parse_signal_id(signal_id)['direction'],
                        position = current_position,
                        signal_id = signal_id

                    )
                    pos_action.event = event
                    position_action_dict[k] = pos_action
                    continue

                ## If exercise is needed, do that first, no need to check other actions or adjust quantity
                elif OpenPositionAction.EXERCISE.value in actions:
                    pos_action = EXERCISE(k, {})
                    pos_action.reason = reasons[actions.index(OpenPositionAction.EXERCISE.value)]
                    long_premiums, short_premiums = self.pm.get_premiums_on_position(current_position['position'], date)
                    
                    event = ExerciseEvent(
                        datetime = date, ## Exercise happens on the same day as the action.
                        symbol = sym,
                        quantity = current_position['quantity'],
                        entry_date = date,
                        spot = self.chain_spot_timeseries[sym][date], ## Using chain spot because strikes are unadjusted for splits
                        long_premiums = long_premiums,
                        short_premiums = short_premiums,
                        position = current_position,
                        signal_id = signal_id
                        
                    )
                    pos_action.event = event
                    sub_action_dict[k] = pos_action
                    
                    continue

            
                ## If the position is a hold, check if it needs to be adjusted based on greeks
                elif OpenPositionAction.HOLD.value in actions:
                    pos_action = HOLD(k)
                    pos_action.reason = None
                    position_action_dict[k] = pos_action

                quantity_change_list = [0] ## Initialize the quantity change list with 0
                value = greek_dict.get(k, {}) ## Get the greek dict for each position
                for greek, res in value.items(): ## Looping through each greek adjustments
                    quantity_change_list.append(res['quantity_diff'])
                sub_action_dict['quantity_diff'] = min(quantity_change_list) ## Ultimate adjustment would be the minimum reduction factor
                if sub_action_dict['quantity_diff'] < 0: ## If the quantity needs to be reduced, set the action to adjust
                    pos_action = ADJUST(k, sub_action_dict)
                    pos_action.reason = "greek_limit"

                    event = OrderEvent(
                        symbol = sym,
                        datetime = event_date,
                        order_type = 'MKT',
                        quantity= abs(sub_action_dict['quantity_diff']),
                        direction = 'SELL' if sub_action_dict['quantity_diff'] < 0 else 'BUY',
                        position = current_position['position'],
                        signal_id = signal_id
                    )
                    pos_action.event = event
                    position_action_dict[k] = pos_action ## If adjust position, override HOLD.
        self._actions[date] = position_action_dict
        logger.info(f"Position Action Dict: {position_action_dict}")
        for id, action in position_action_dict.items():
            logger.info(f"Position ID: {id}, Action: {action}, Reason: {action.reason}")
            if not isinstance(action, HOLD):
                logger.info(f"Event: {action.event}")
                logger.info((f"Risk Manager Scheduling Action: Position ID: {id}, Action: {action}, Reason: {action.reason}"))
                self.pm.eventScheduler.schedule_event(event_date, action.event)

        return position_action_dict

                

        

    def limits_check(self):
        limits = self.limits
        delta_limit = limits['delta']
        position_limit = {}

        date = pd.to_datetime(self.pm.eventScheduler.current_date)
        if is_USholiday(date):
            self.pm.logger.warning(f"Market is closed on {date}, skipping")
            return 

        for symbol in self.pm.symbol_list:
            position = self.pm.current_positions[symbol]
            for signal_id, current_position in position.items():
                if 'position' not in current_position:
                    continue
                logger.info(f"Checking Position {current_position['position']['trade_id']} for Greek Limits on {date}")
                trade_id = current_position['position']['trade_id']
                quantity = current_position['quantity']
                signal_id = signal_id
                max_delta = self.greek_limits['delta'][signal_id]
                pos_data = self.position_data[trade_id]

                status = {'status': False, 'quantity_diff': 0}
                greek_limit_bool = dict(delta=status, gamma=status, vega=status, theta=status)

                if delta_limit:
                    delta_val = abs(pos_data.at[date, 'Delta'])
                    skip = pos_data.at[date, 'Delta_skip_day'] if 'Delta_skip_day' in pos_data.columns else False

                    if skip or delta_val == 0:
                        continue

                    current_delta = delta_val * quantity
                    if current_delta > max_delta:
                        # Compute how many contracts to reduce
                        required_quantity = max(int(max_delta // delta_val), 1) ## Ensure at least 1 contract is required. If last contract exceeds delta limit, we will still hold it.
                        quantity_diff = required_quantity - quantity
                        logger.info(f"Position {trade_id} exceeds delta limit. Current Delta: {current_delta}, Max Delta: {max_delta}, Required Quantity: {required_quantity}, Current Quantity: {quantity}")
                        greek_limit_bool['delta'] = {'status': True, 'quantity_diff': quantity_diff}

                    position_limit[trade_id] = greek_limit_bool

        return position_limit




    def dte_check(self):
        """
        Analyze the current positions and determine if any need to be rolled
        """
        date = pd.to_datetime(self.pm.eventScheduler.current_date)
        logger.info(f"Checking DTE on {date}")
        if is_USholiday(date):
            self.pm.logger.warning(f"Market is closed on {date}, skipping")
            return
        
        roll_dict = {}
        for symbol in self.pm.symbol_list:
            position = self.pm.current_positions[symbol]
            for signal_id, current_position in position.items():
                if 'position' not in current_position:
                    continue
            
                logger.info(f"Checking Position {current_position['position']['trade_id']} for DTE on {date}")
                id = current_position['position']['trade_id']
                expiry_date = ''
                
                if 'long' in current_position['position']:
                    for option_id in current_position['position']['long']:
                        option_meta = parse_option_tick(option_id)
                        expiry_date = option_meta['exp_date']
                        break
                elif 'short' in current_position['position']:
                    for option_id in current_position['position']['short']:
                        option_meta = parse_option_tick(option_id)
                        expiry_date = option_meta['exp_date']
                        break


                dte = (pd.to_datetime(expiry_date) - pd.to_datetime(date)).days
                logger.info(f"ID: {id}, DTE: {dte}, Expiry: {expiry_date}, Date: {date}")

                if symbol in self.pm.roll_map and dte <= self.pm.roll_map[symbol]:
                    logger.info(f"{id} rolling because {dte} <= {self.pm.roll_map[symbol]}")
                    roll_dict[id] = OpenPositionAction.ROLL.value
                elif symbol not in self.pm.roll_map and dte == 0:  # exercise contract if symbol not in roll map
                    logger.info(f"{id} exercising because {dte} == 0")
                    roll_dict[id] = OpenPositionAction.EXERCISE.value
                else:
                    logger.info(f"{id} holding because {dte} > {self.pm.roll_map[symbol]}")
                    roll_dict[id] = OpenPositionAction.HOLD.value
        return roll_dict
    
    def moneyness_check(self):
        """
        Analyze the current positions and determine if any need to be rolled based on moneyness
        """
        date = pd.to_datetime(self.pm.eventScheduler.current_date)
        logger.info(f"Checking Moneyness on {date}")
        if is_USholiday(date):
            self.pm.logger.warning(f"Market is closed on {date}, skipping")
            return
        
        
        roll_dict = {}
        for symbol in self.pm.symbol_list:
            strike_list = []
            position = self.pm.current_positions[symbol]
            for signal_id, current_position in position.items():
                if 'position' not in current_position:
                    continue
                
                logger.info(f"Checking Position {current_position['position']['trade_id']} for Moneyness on {date}")
                id = current_position['position']['trade_id']
                try:
                    entry_date = self.pm.trades_map[id].entry_date
                except Exception as e:
                    print(f"Error getting entry date for position {id}: {e}")
                    entry_date = date
                spot = self.chain_spot_timeseries[symbol][date] ## Use the spot price on the date (from chain cause of splits)
                
                if 'long' in current_position['position']:
                    for option_id in current_position['position']['long']:
                        option_meta = self.adjust_for_events(entry_date, date, parse_option_tick(option_id))
                        # option_meta =  parse_option_tick(option_id)
                        strike_list.append(option_meta['strike']/spot if option_meta['put_call'] == 'P' else spot/option_meta['strike'])

                if 'short' in current_position['position']:
                    for option_id in current_position['position']['short']:
                        option_meta = self.adjust_for_events(entry_date, date, parse_option_tick(option_id))
                        # option_meta =  parse_option_tick(option_id)
                        strike_list.append(option_meta['strike']/spot if option_meta['put_call'] == 'P' else spot/option_meta['strike'])
                
                logger.info(f"{id} moneyness list {strike_list}, spot: {spot}, date: {date}, entry_date: {entry_date}")
                logger.info(f"{id} moneyness bool list {[x > self.max_moneyness for x in strike_list]}")
                
                roll_dict[id] = OpenPositionAction.ROLL.value if any([x > self.max_moneyness for x in strike_list]) else OpenPositionAction.HOLD.value
        return roll_dict

    def hedge_check(self,
                    hedge_func: callable,
                    hedge_args: list,
                    hedge_kwargs: dict,
                    ) -> dict:
        """
        Responsible for checking if the hedge is needed and if so, queueing in analyze_position
        Hedge function should allow 1st argument to be Risk Manager and 2nd argument to be Portfolio Manager
        Expected return type is: List[HEDGE]. Where HEDGE is a subclass of RMAction

        params:
        hedge_func: callable: function to be called for the hedge
        hedge_args: list: arguments to be passed to the hedge function
        hedge_kwargs: dict: keyword arguments to be passed to the hedge function

        returns:
        dict: dictionary of the hedge actions
        """
        pass 

    ## Lazy Loading Spot Data
    def generate_data(self, symbol):
        stk = self.pm.get_underlier_data(symbol)  ## Performance isn't affected because of singletons in stock class
        if symbol not in self.spot_timeseries:
            self.spot_timeseries[symbol] = stk.spot(
                ts = True,
                ts_start = pd.to_datetime(self.start_date) - BDay(30),
                ts_end = pd.to_datetime(self.end_date),
            )[self.price_on]

        if symbol not in self.chain_spot_timeseries:
            self.chain_spot_timeseries[symbol] = stk.spot(
                ts = True,
                spot_type = OptionModelAttributes.spot_type.value,
                ts_start = pd.to_datetime(self.start_date) - BDay(30),
                ts_end = pd.to_datetime(self.end_date),
            )[self.price_on]
        
        if symbol not in self.dividend_timeseries:
            divs = stk.div_yield_history(start = pd.to_datetime(self.start_date) - BDay(30))
            if not isinstance(divs, (pd.DataFrame, pd.Series)): ## When a ticker has no dividends, it returns None/0
                divs = pd.Series(divs, index = self.spot_timeseries[symbol].index)
            self.dividend_timeseries[symbol] = divs

    def parse_position_id(self, positionID):
        position_str = positionID
        position_list = position_str.split('&')
        position_list = [x.split(':') for x in position_list if x]
        position_list_parsed = [(x[0], parse_option_tick(x[1])) for x in position_list]
        position_dict = dict(L = [], S = [])
        for x in position_list_parsed:
            position_dict[x[0]].append(x[1])
        return position_dict, position_list

    def get_position_dict(self, positionID):
        return self.parse_position_id(positionID)[0]

    def get_position_list(self, positionID):
        return self.parse_position_id(positionID)[1]

    def get_option_price(self, optID, date):
        portfolio = self.pm
        return portfolio.options_data[optID][self.option_price][date]
    
    def adjust_for_events(
            self,
            start: str,
            date: str,
            option: str|dict,
    ):
        """
        Adjusts the option tick for events like splits or dividends.
        """
        if isinstance(option, str):
            meta = parse_option_tick(option)
        elif isinstance(option, dict):
            meta = option
        else:
            raise ValueError("Option must be a string or a dictionary.")
        split = self.splits.get(swap_ticker(meta['ticker']), None)
        if split is None:
            return meta
        for pack in split:
            if compare_dates.is_before(start, pack[0]) and compare_dates.is_after(date, pack[0]):
                meta['strike'] /= pack[1]
        return meta

    
    