## NOTE:
## 1) If a split happens during a backtest window, the trade id won't be updated. The dataframe will simply be uploaded with a the split adjusted strike.
## 2) All Greeks &  Midpoint with Zero values will be FFWD'ed


from .utils import *
from .utils import (logger, 
                    get_timeseries_start_end,
                    set_deleted_keys,
                    date_in_cache_index)
from .actions import *
from trade.helpers.helper import printmd, CustomCache, date_inbetween
from EventDriven.event import (
    RollEvent,
    ExerciseEvent,
    OrderEvent
)
import numpy as np
import os
BASE = Path(os.environ["WORK_DIR"])/ ".riskmanager_cache"
HOME_BASE = Path(os.environ["WORK_DIR"])/".cache"
BASE.mkdir(exist_ok=True)
# logger = setup_logger('QuantTools.EventDriven.riskmanager.base')
order_cache = CustomCache(BASE, fname = "order")



def get_order_cache():
    """
    Returns the order cache
    """
    global order_cache
    return order_cache
def refresh_cache():
    """
    Refreshes the cache for the order picker
    """
    global order_cache, spot_cache, close_cache, oi_cache, chain_cache
    spot_cache = get_cache('spot')
    close_cache = get_cache('close')
    oi_cache = get_cache('oi')
    chain_cache = get_cache('chain')


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
        global order_cache, spot_cache, close_cache, oi_cache, chain_cache
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
    def __init__(self,
                 bars: DataHandler,
                 events: EventScheduler,
                 initial_capital: int|float,
                 start_date: str|datetime,
                 end_date: str|datetime,
                 portfolio_manager: 'Portfolio' = None,
                 price_on = 'close',
                 option_price = 'Midpoint',
                 sizing_type = 'delta',
                 leverage = 5.0,
                 max_moneyness = 1.2,
                 ):
        
        """
        initializes the RiskManager class

        params:
        bars: Bars: bars
        events: Events: events
        initial_capital: float: initial capital
        start_date: str: start date, recommended to match with the start date of the bars
        end_date: str: end date, recommended to match with the end date of the bars
        portfolio_manager: PortfolioManager: portfolio manager. Default is None
        price_on: str: price on. Default is 'mkt_close'
        option_price: str: option price. The Option Price used for pricing. Default is 'Midpoint'. Available Options are 'Midpoint', 'Bid', 'Ask', 'Close', 'Weighted Midpoint'
        sizing_type: str: sizing type. This is what you want your quantity to be calculated on. Default is 'delta'. Available Options are 'delta', 'vega', 'gamma', 'price'
        leverage: float: Multiplier for Equity Equivalent Size. Default is 5.0. Eg (Cash Available/Spot Price) * Leverage = Equity Equivalent Size
        max_moneyness: float: Maximum Moneyness before rolling. Default is 1.2

        Other Attributes:


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
        self.spot_timeseries = CustomCache(BASE, fname = "rm_spot_timeseries")
        self.chain_spot_timeseries = CustomCache(BASE, fname = "rm_chain_spot_timeseries") ## This is used for pricing, to account option strikes for splits
        self.processed_option_data = CustomCache(BASE, fname = "rm_processed_option_data")
        self.position_data = CustomCache(BASE, fname = "rm_position_data")
        self.dividend_timeseries = CustomCache(BASE, fname = "rm_dividend_timeseries")
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
        self.splits_raw =CustomCache(HOME_BASE, fname = "split_names_dates", expiry = 1000)
        self.splits = self.set_splits(self.splits_raw)
        # self.clear_caches()
        


    @property 
    def option_data(self):
        global close_cache
        return close_cache  
    
    @property
    def order_cache(self):
        """
        Returns the order cache
        """
        global order_cache
        return order_cache
    
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
Start Date: {self.start_date}
End Date: {self.end_date}
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
        signalID = kwargs.pop('signal_id')
        date = kwargs.get('date')
        tick = kwargs.get('tick')
        logger.info(f"## ***Signal ID: {signalID}***")

        ## I cannot calculate greeks here. I need option_data to be available first.
        order = self.OrderPicker.get_order(*args, **kwargs)     
        logger.info(f"Order Produced: {order}")

        ## save the order in the cache
        if date not in order_cache:
            cache_dict = {tick: order}
            order_cache[date] = cache_dict
        else:
            cache_dict = order_cache[date]
            cache_dict[tick] = order
            order_cache[date] = cache_dict

        if order['result'] == ResultsEnum.SUCCESSFUL.value:
            print(f"\nOrder Received: {order}\n")

            position_id = order['data']['trade_id']
            
        else:
            logger.info(f"Signal ID: {signalID}, Unable to produce order, returning None")
            return order
        
        logger.info(f"Position ID: {position_id}")
        logger.info("Calculating Position Greeks")
        self.calculate_position_greeks(position_id, kwargs['date'])
        logger.info('Updating Signal Limits')
        self.update_greek_limits(signalID, position_id)
        logger.info("Calculating Quantity")
        quantity = self.calculate_quantity(position_id, signalID, kwargs['date'])
        logger.info(f"Quantity for Position ({position_id}): {quantity}")
        order['data']['quantity'] = quantity
        logger.info(order)
        return order


        

    @log_time(time_logger)
    def calculate_position_greeks(self, positionID, date):
        """
        Calculate the greeks of a position

        date: Evaluation Date for the greeks (PS: This is not the pricing date)
        positionID: str: position string. (PS: This function assumes ticker for position is the same)
        """
        print(f"Calculate Greeks Dates Start: {self.start_date}, End: {self.end_date}, Position ID: {positionID}, Date: {date}")
        if positionID in self.position_data:
            ## If the position data is already available, then we can skip this step
            print(f"Position Data for {positionID} already available, skipping calculation")
            logger.info(f"Position Data for {positionID} already available, skipping calculation")
            return self.position_data[positionID]
        else:
            logger.critical(f"Position Data for {positionID} not available, calculating greeks. Load time ~2 minutes")
        ## Initialize the Long and Short Lists
        long = []
        short = []
        threads = []
        thread_input_list = [
            [], [], [], [], [], []
        ]

        date = pd.to_datetime(date) ## Ensure date is in datetime format
        
        ## First get position info
        position_dict, positon_meta = self.parse_position_id(positionID)

        ## Now ensure that the spot and dividend data is available
        for p in position_dict.values():
            for s in p:
                self.generate_data(s['ticker'])
        ticker = s['ticker']

        ## Get the spot, risk free rate, and dividend yield for the date
        s = self.chain_spot_timeseries[ticker]
        s0_close = self.spot_timeseries[ticker]
        r = self.rf_timeseries
        y = self.dividend_timeseries[ticker]

        @log_time(time_logger)
        def get_timeseries(ids, s, r, y, s0_close, direction):
            logger.info("Calculate Greeks dates")
            logger.info(f"Start Date: {self.start_date}")
            logger.info(f"End Date: {self.end_date}")
            full_data = pd.DataFrame()
            for id in ids:
                data_manager = OptionDataManager(opttick = id)
                greeks = data_manager.get_timeseries(start = self.start_date,
                                                        end = self.end_date,
                                                        interval = '1d',
                                                        type_ = 'greeks',).post_processed_data
                greeks_cols = [x for x in greeks.columns if 'Midpoint' in x]
                greeks = greeks[greeks_cols]
                greeks[greeks_cols] = greeks[greeks_cols].replace(0, np.nan).fillna(method = 'ffill') ## FFill NaN values and 0 Values
                greeks.columns = [x.split('_')[1].capitalize() for x in greeks.columns]

                spot = data_manager.get_timeseries(start = self.start_date,
                                                    end = self.end_date,
                                                    interval = '1d',
                                                    type_ = 'spot',).post_processed_data ## Using chain spot data to account for splits
                spot = spot[[self.option_price.capitalize()]]
                data = greeks.join(spot)
                full_data = pd.concat([full_data, data], axis = 0)
            full_data = full_data[~full_data.index.duplicated(keep = 'last')]
            full_data['s'] = s
            full_data['r'] = r
            full_data['y'] = y
            full_data['s0_close'] = s0_close
            self.processed_option_data[data_manager.opttick] = full_data
            if direction == 'L':
                long.append(full_data)
            elif direction == 'S':
                short.append(full_data)
            else:
                raise ValueError(f"Position Type {_set[0]} not recognized")
            
            return data

    
        ## Check for splits
        split = self.splits.get(ticker, [])

        ## Calculating IVs & Greeks for the options
        for _set in positon_meta:
            # To-do: Thread thisto speed up the process
            ids = [_set[1]]
            if len(split) > 0:
                for i in split:
                    split_date = i[0]
                    if pd.to_datetime(split_date) < pd.to_datetime(date): ## Strike is already adjusted for the split
                        continue
                    shift = i[1]
                    id = _set[1]
                    meta = parse_option_tick(id)
                    meta['strike'] = meta['strike'] / shift
                    ids.append(generate_option_tick_new(*meta.values()))
            # data_manager = OptionDataManager(opttick = id)


            for input, list_ in zip([ids, s, r, y, s0_close, _set[0]], thread_input_list):
                list_.append(input)
        
        
        runThreads(get_timeseries, thread_input_list)
        # return long
            
        position_data = sum(long) - sum(short)
        position_data = position_data[~position_data.index.duplicated(keep = 'first')]
        position_data.columns = [x.capitalize() for x in position_data.columns]
        ## Retain the spot, risk free rate, and dividend yield for the position, after the greeks have been calculated & spread values subtracted
        position_data['s0_close'] = s0_close
        position_data['s'] = s
        position_data['r'] = r
        position_data['y'] = y
        self.position_data[positionID] = position_data

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
        
        if signal_id in self.greek_limits['delta']: ## May consider to maximize cash on roll
            logger.info(f"Greek Limits for Signal ID: {signal_id} already updated, skipping")
            return
        logger.info(f"Updating Greek Limits for Signal ID: {signal_id} and Position ID: {position_id}")
        id_details = parse_signal_id(signal_id)
        cash_available = self.pm.allocated_cash_map[id_details['ticker']]
        delta_at_purchase = self.position_data[position_id]['Delta'][id_details['date']] 
        s0_at_purchase = self.position_data[position_id]['s'][id_details['date']] ## As always, we use the chain spot data to account for splits
        equivalent_delta_size = (math.floor(cash_available/s0_at_purchase)/100) * self.sizing_lev
        self.greek_limits['delta'][signal_id] = abs(equivalent_delta_size)
        logger.info(f"Spot Price at Purchase: {s0_at_purchase} at time {id_details['date']}")
        logger.info(f"Delta at Purchase: {delta_at_purchase}")
        logger.info(f"Equivalent Delta Size: {equivalent_delta_size}, with Cash Available: {cash_available}, and Leverage: {self.sizing_lev}")
        logger.info(f"Equivalent Delta Size: {equivalent_delta_size}")

    def calculate_quantity(self, positionID, signalID, date) -> int:
        """
        Returns the quantity of the position that can be bought based on the sizing type
        """
        logger.info(f"Calculating Quantity for Position ID: {positionID} and Signal ID: {signalID} on Date: {date}")
        if positionID not in self.position_data: ## If the position data isn't available, calculate the greeks
            self.calculate_position_greeks(positionID, date)
        
        ## First get position info and ticker
        position_dict, _ = self.parse_position_id(positionID)
        key = list(position_dict.keys())[0]
        ticker = position_dict[key][0]['ticker']

        ## Now calculate the max size cash can buy
        cash_available = self.pm.allocated_cash_map[ticker]
        purchase_date = pd.to_datetime(date)
        s0_at_purchase = self.position_data[positionID]['s'][purchase_date]  ## s -> chain spot, s0_close -> adjusted close
        logger.info(f"Spot Price at Purchase: {s0_at_purchase} at time {purchase_date}")
        opt_price = self.position_data[positionID]['Midpoint'][purchase_date]
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
        date = pd.to_datetime(self.pm.events.current_date)
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
        for sym in self.pm.symbol_list:
            current_position = self.pm.current_positions[sym]
            if 'position' not in current_position:
                continue
            k = current_position['position']['trade_id']


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
                    datetime = date,
                    symbol = sym,
                    signal_type = parse_signal_id(current_position['signal_id'])['direction'],
                    position = current_position,
                    signal_id = current_position['signal_id']

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
                    datetime = date,
                    symbol = sym,
                    quantity = current_position['quantity'],
                    entry_data = date,
                    spot = self.chain_spot_timeseries[sym][date], ## Using chain spot because strikes are unadjusted for splits
                    long_premiums = long_premiums,
                    short_premiums = short_premiums,
                    position = current_position,
                    signal_id = current_position['signal_id']
                    
                )
                pos_action.event = event
                sub_action_dict[k] = pos_action
                continue

        
            ## If the position is a hold, check if it needs to be adjusted based on greeks
            elif OpenPositionAction.HOLD.value in actions:
                pos_action = HOLD(k)
                pos_action.reason = reasons[actions.index(OpenPositionAction.HOLD.value)]
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
                    datetime = date,
                    order_type = 'MKT',
                    quantity= sub_action_dict['quantity_diff'],
                    direction = 'SELL' if sub_action_dict['quantity_diff'] < 0 else 'BUY',
                    position = current_position['position'],
                    signal_id = current_position['signal_id']
                )
                position_action_dict[k] = pos_action ## If adjust position, override HOLD.
        self._actions[date] = position_action_dict

        return position_action_dict

                

        
    def limits_check(self):
        """
        Checks if the order is within the limits of the portfolio
        """
        limits = self.limits
        delta_limit = limits['delta']
        position_limit = {}

        date = pd.to_datetime(self.pm.events.current_date)
        logger.info(f"Checking Limits on {date}")
        if is_USholiday(date):
            self.pm.logger.warning(f"Market is closed on {date}, skipping")
            return 
        

        current_positions = self.pm.current_positions
        for symbol, position in current_positions.items():
            if 'position' not in position:
                continue

            ## Initialize the greeks limits to False and other essentials variables
            status = {'status': False, 'quantity_diff': 0} ## Status is False by default
            greek_limit_bool = dict(vega = status, gamma = status, delta = status, theta = status) ## Initialize the greek limits to False
            max_delta = self.greek_limits['delta'][position['signal_id']]
            quantity, q = position['quantity'], position['quantity']
            trade_id = position['position']['trade_id']
            date = pd.to_datetime(self.pm.events.current_date)
            current_delta = abs(self.position_data[trade_id]['Delta'][date] * quantity)

            if delta_limit:
                quantity_diff = 0 ## Quantity difference to be used in case of limit breach, I want to return negative values
                if current_delta < max_delta:
                    logger.info(f"Delta for Position {trade_id} is within limits")
                else:
                    logger.info(f"Delta for Position {trade_id} is above limits")
                    while current_delta > max_delta:
                        ## Reduce the quantity of the position until it is within limits
                        quantity_diff -= 1
                        q = q -1
                        current_delta = abs(self.position_data[trade_id]['Delta'][date]) * q
                        logger.info(f"Current Delta: {current_delta}, Max Delta: {max_delta}, Quantity: {q}")
                    greek_limit_bool['delta'] = {'status': True, 'quantity_diff': quantity_diff}
                position_limit[trade_id] = greek_limit_bool
        return position_limit
    
    def dte_check(self):
        """
        Analyze the current positions and determine if any need to be rolled
        """
        date = pd.to_datetime(self.pm.events.current_date)
        logger.info(f"Checking DTE on {date}")
        if is_USholiday(date):
            self.pm.logger.warning(f"Market is closed on {date}, skipping")
            return
        
        roll_dict = {}
        for symbol in self.pm.symbol_list:
            current_position = self.pm.current_positions[symbol] 
            
            if 'position' not in current_position:
                continue

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

            
            if symbol in self.pm.roll_map and dte <= self.pm.roll_map[symbol]:
                roll_dict[id] = OpenPositionAction.ROLL.value
            elif symbol not in self.pm.roll_map and dte == 0:  # exercise contract if symbol not in roll map
                roll_dict[id] = OpenPositionAction.EXERCISE.value
            else:
                roll_dict[id] = OpenPositionAction.HOLD.value
        return roll_dict
    
    def moneyness_check(self):
        """
        Analyze the current positions and determine if any need to be rolled based on moneyness
        """
        date = pd.to_datetime(self.pm.events.current_date)
        logger.info(f"Checking Moneyness on {date}")
        if is_USholiday(date):
            self.pm.logger.warning(f"Market is closed on {date}, skipping")
            return
        
        strike_list = []
        roll_dict = {}
        for symbol in self.pm.symbol_list:
            current_position = self.pm.current_positions[symbol] 
            if 'position' not in current_position:
                continue

            id = current_position['position']['trade_id']
            spot = self.chain_spot_timeseries[symbol][date] ## Use the spot price on the date (from chain cause of splits)
            
            if 'long' in current_position['position']:
                for option_id in current_position['position']['long']:
                    option_meta = parse_option_tick(option_id)
                    strike_list.append(option_meta['strike']/spot if option_meta['put_call'] == 'P' else spot/option_meta['strike'])

            if 'short' in current_position['position']:
                for option_id in current_position['position']['short']:
                    option_meta = parse_option_tick(option_id)
                    strike_list.append(option_meta['strike']/spot if option_meta['put_call'] == 'P' else spot/option_meta['strike'])
            
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
    
    