## NOTE:
## 1) If a split happens during a backtest window, the trade id won't be updated. The dataframe will simply be uploaded with a the split adjusted strike.
## 2) All Greeks &  Midpoint with Zero values will be FFWD'ed
## 3) Do something about all these caches locations. I don't like it. It's confusing

from .utils import (
                    get_timeseries_start_end,
                    get_persistent_cache,)
from EventDriven._vars import get_use_temp_cache
from trade.helpers.helper import  CustomCache, is_USholiday
import pandas as pd
from typing import List
from datetime import datetime
from trade.helpers.Logging import setup_logger
from EventDriven.riskmanager.picker.order_picker import OrderPicker, order_failed
from EventDriven.riskmanager.market_timeseries import BacktestTimeseries
from EventDriven.riskmanager.position.analyzer import PositionAnalyzer
from EventDriven.riskmanager.position.cogs.limits import LimitsAndSizingCog
from EventDriven.dataclasses.orders import OrderRequest
from EventDriven.dataclasses.states import NewPositionState, PositionAnalysisContext, StrategyChangeMeta
from EventDriven.types import ResultsEnum, Order
from EventDriven.configs.core import RiskManagerConfig
from EventDriven._vars import CONTRACT_MULTIPLIER, load_riskmanager_cache
logger = setup_logger('EventDriven.riskmanager.new_base', stream_log_level="WARNING")


class RiskManager:
    """
    Manages portfolio risk and executes options trading strategies with position sizing and Greek-based limits.

    The RiskManager orchestrates the entire risk management lifecycle: analyzing positions, generating orders
    based on strategy signals, enforcing risk limits, and managing position rolls. It integrates multiple
    specialized components to handle market data, position analytics, and order selection.

    Key Responsibilities:
    - Position analysis and order generation based on risk limits and signals
    - Greek-based risk management (delta, gamma, vega, theta)
    - Position rolling based on moneyness and DTE thresholds
    - Market data caching and option pricing across backtests
    - Corporate action handling (splits, dividends)

    Core Components:
        order_picker (OrderPicker): Selects optimal orders from available options based on moneyness,
            DTE, liquidity, and strategy-specific criteria. Filters option chains and ranks candidates.
        
        timeseries (BacktestTimeseries): Manages and caches all market data including spot prices,
            option chains, Greeks, dividends, and risk-free rates. Handles data loading and caching
            strategies for efficient backtesting.
        
        position_analyzer (PositionAnalyzer): Analyzes current positions against risk limits and signals
            to determine required actions (entry, exit, roll, adjust). Calculates portfolio Greeks and
            sizing requirements.
        
        limits_cog (LimitsAndSizingCog): Enforces risk limits and calculates position sizes based on
            available capital, leverage, and Greek constraints. Validates orders against risk parameters.

    Core Attributes:
        symbol_list (List[str]): List of tradable symbols in the universe
        start_date/end_date (str|datetime): Backtest window boundaries
        initial_capital (float): Starting portfolio capital for the backtest
        config (RiskManagerConfig): Configuration object containing all risk management parameters
        
    """

    def __init__(self,
                 symbol_list: List[str],
                 bkt_start: str|datetime,
                 bkt_end: str|datetime,
                 initial_capital: float,
                 *args,
                 **kwargs
                 ):
        """
        Initialize the RiskManager with core backtest parameters and component setup.

        Parameters
        ----------
        symbol_list : List[str]
            Trading symbols to manage in the portfolio
        bkt_start : str | datetime
            Backtest start date
        bkt_end : str | datetime
            Backtest end date
        initial_capital : float
            Starting portfolio capital
        **kwargs : dict, optional
            Additional configuration passed to RiskManagerConfig
        
        Notes
        -----
        - Initializes OrderPicker, BacktestTimeseries, PositionAnalyzer, and LimitsAndSizingCog
        - Loads configuration from RiskManagerConfig with overrides from kwargs
        - Sets up caching infrastructure for market data and position analytics
        """
        ## For testing override the passed args
        # bkt_start = '2025-01-01'
        # bkt_end = '2025-06-30'
        # symbol_list = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'JNJ', 'V', 'WMT']

        ## Backtest Window
        self.start_date = bkt_start
        self.end_date = bkt_end

        ## Configs
        self.config = RiskManagerConfig()

        ## Get start and end dates for timeseries loading
        start, end = get_timeseries_start_end()
        self.ts_start = start
        self.ts_end = end

        ## Other Core Attributes
        self.symbol_list = symbol_list

        ## Core Classes
        self.order_picker: OrderPicker = OrderPicker(start, end)
        self.market_data: BacktestTimeseries = BacktestTimeseries(_start=self.ts_start, _end=self.ts_end)
        self.position_analyzer: PositionAnalyzer = PositionAnalyzer()
        lmt_cog: LimitsAndSizingCog = LimitsAndSizingCog(underlier_list=self.symbol_list)
        self.position_analyzer.add_cog(lmt_cog)

        ## Misc Attributes
        self.initial_capital = initial_capital
        self.symbol_list = symbol_list

        ## Initialize on disk caches. These will be cleared on exit.

        ##Order Cache
        self.order_cache = load_riskmanager_cache(target="order_cache", create_on_missing=True, clear_on_exit=True)
        if len(self.order_cache.values()) > 0:
            logger.info(f"Order cache loaded with {len(self.order_cache.values())} orders")

        ##Position Analysis Cache
        self.analysis_cache = load_riskmanager_cache(target="position_analysis", create_on_missing=True, clear_on_exit=True)
        if len(self.analysis_cache.values()) > 0:
            logger.info(f"Position analysis cache loaded with {len(self.analysis_cache.values())} analyses")

        ##Order Request Cache
        self.order_request_cache = load_riskmanager_cache(target="order_request_cache", create_on_missing=True, clear_on_exit=True)
        if len(self.order_request_cache.values()) > 0:
            logger.info(f"Order request cache loaded with {len(self.order_request_cache.values())} requests")


    def clear_caches(self):
        """
        Clears all caches used by the RiskManager.
        """
        if get_use_temp_cache():
            self.market_data.options_cache.clear()
            self.market_data.position_data_cache.clear()
            get_persistent_cache().clear() ## Ensures any caching with `.memoize` is cleared as well.
        else:
            logger.critical("USE_TEMP_CACHE set to False. Cache will not be cleared")

    def get_order(self, req: OrderRequest) -> NewPositionState:
        """
        Generates an order based on the provided OrderRequest dataclass and returns
        a NewPositionState after processing through the PositionAnalyzer. The order will be
        generated based on the parameters specified in the OrderRequest as well as configs
        defined in the RiskManager & OrderPicker.

        To learn more about the OrderRequest dataclass, refer to EventDriven.dataclasses.orders.OrderRequest.
        To learn more about the NewPositionState dataclass, refer to EventDriven.dataclasses.states.NewPositionState.
        To learn more about the configs used in order generation, refer to EventDriven.configs.core

        Args:
            req (OrderRequest): The order request containing parameters for order generation.
        Returns:
            NewPositionState: The state of the new position after analysis.
        """
        if is_USholiday(req.date):
            logger.info(f"Date {req.date} is a US Holiday, skipping order generation")
            return {"result": ResultsEnum.IS_HOLIDAY.value, "data": None}
        
        ## Investigate if tick cash is scaled
        if not req.is_tick_cash_scaled:
            req.tick_cash = req.tick_cash * CONTRACT_MULTIPLIER  ## Scale tick cash to account for options contract multiplier
            req.is_tick_cash_scaled = True
        ## Generate Data
        self.market_data.market_timeseries.load_timeseries(sym=req.symbol)
        at_index = self.market_data.market_timeseries.get_at_index(sym=req.symbol, index=req.date)
        spot = at_index.spot["close"]
        chain_spot = at_index.chain_spot["close"]

        ## Update request with data
        req.spot = spot
        req.chain_spot = chain_spot

        ## Get order
        print(f"Generating order for request: {req}")
        if self.config.cache_order_requests:
            logger.info(f"Caching order request for signal ID: {req.signal_id}")
            self.order_request_cache[req.signal_id] = req
        order = self.order_picker.get_order(request=req).to_dict()
        logger.info(f"Order generated: {order}")

        ## Process order

        if not order_failed(order):
            print(f"\nOrder Received: {order}\n")
            position_id = order["data"]["trade_id"]
        else:
            print(f"\nOrder Failed: {order}\n")
            logger.info(f"Signal ID: {req.signal_id}, Unable to produce order, returning None")
            failed_order_state = NewPositionState(
                trade_id="None",
                symbol=req.symbol,
                order=Order.from_dict(order),
                request=req,
                at_time_data=None,
                undl_at_time_data=None,
                limits=None
            )
            return failed_order_state

        ## Get position data
        logger.info(f"Retrieving position data for position ID: {position_id}")
        self.market_data.calculate_option_data(position_id=position_id, date=req.date)

        ## Update Close price
        at_time_data = self.market_data.get_at_time_position_data(position_id=position_id, date=req.date)
        if at_time_data is not None:
            close_price = at_time_data.close
            logger.info(
                f"At time data found for position ID: {position_id} on date: {req.date}, updating close price to {close_price}"
            )
            order["data"]["close"] = close_price
        else:
            logger.critical(
                f"At time data NOT found for position ID: {position_id} on date: {req.date}, close price remains unchanged"
            )

        ## Create NewPositionState and run through position_analyzer
        logger.info(f"Quantity before analysis: {order['data']['quantity']}")
        undl_at_time_data = self.market_data.market_timeseries.get_at_index(sym=req.symbol, index=req.date)
        new_pos_state = NewPositionState(
            trade_id=position_id,
            order=Order.from_dict(order),
            request=req,
            symbol=req.symbol,
            at_time_data=at_time_data,
            undl_at_time_data=undl_at_time_data,
        )
        logger.info(f"Processing new position state through position_analyzer: {new_pos_state}")
        updated_pos_state = self.position_analyzer.on_new_position(new_position_state=new_pos_state)
        logger.info(f"Quantity after analysis: {updated_pos_state.order.data['quantity']}")


        if self.config.cache_orders:
            logger.info(f"Caching order for position ID: {position_id}")
            self.order_cache[position_id] = updated_pos_state

        q = updated_pos_state.order.data['quantity']
        if q == 0:
            logger.warning(f"Final calculated position size is 0 for order {position_id}. Order will not be placed.")
            updated_pos_state.order["result"] = ResultsEnum.POSITION_SIZE_ZERO.value
            
        order = updated_pos_state.order.to_dict()

        return updated_pos_state
    
    def analyze_position(self, context: PositionAnalysisContext) -> StrategyChangeMeta:
        """
        Analyze the given portfolio context using the PositionAnalyzer.

        Args:
            context (PositionAnalysisContext): The context containing portfolio information for analysis.

        Returns:
            CogActions: The actions determined by the analysis.
        """
        analysis = self.position_analyzer.analyze(context)
        if self.config.cache_position_analysis:
            logger.info(f"Caching position analysis for date: {context.date}")
            dt = pd.to_datetime(context.date).strftime('%Y-%m-%d')
            self.analysis_cache[dt] = analysis
        return analysis

    def append_option_data(
        self,
        option_id: str = None,
        position_data: pd.DataFrame = None,
        data_pack: dict | CustomCache = None,
    ):
        """
        Append option data to the processed_position_data cache.
        Parameters:
        position_id: str: ID of the position
        position_data: pd.DataFrame: DataFrame containing the position data
        data_pack: dict|CustomCache: Data pack containing the position data
        """
        if option_id:
            assert position_data is not None, "position_data must be provided if option_id is given"
            self.market_data.options_cache[option_id] = position_data

        elif isinstance(data_pack, (CustomCache, dict)):
            for k, v in data_pack.items():
                self.market_data.options_cache[k] = v

        else:
            raise ValueError("Either option_id or data_pack must be provided to append_option_data")

    def append_position_data(
        self,
        position_id: str = None,
        position_data: pd.DataFrame = None,
        data_pack: dict | CustomCache = None,
    ):
        """
        Append position data to the position_data cache.
        Parameters:
        position_id: str: ID of the position
        position_data: pd.DataFrame: DataFrame containing the position data
        data_pack: dict|CustomCache: Data pack containing the position data
        """
        if position_id:
            assert position_data is not None, "position_data must be provided if position_id is given"
            self.market_data.position_data_cache[position_id] = position_data

        elif data_pack:
            assert isinstance(data_pack, (dict, CustomCache)), "data_pack must be a dict or CustomCache"
            for k, v in data_pack.items():
                self.market_data.position_data_cache[k] = v

        else:
            raise ValueError("Either position_id or data_pack must be provided to append_position_data")

#     def analyze_position(self):
#         """
#         Analyze the current positions and determine if any need to be rolled, closed, or adjusted
#         """
#         return
#         position_action_dict = {}  ## This will be used to store the actions for each position
#         date = pd.to_datetime(self.pm.eventScheduler.current_date)
#         if date in self.__analyzed_date_list:  ## If the date has already been analyzed, return
#             logger.info(f"Positions already analyzed on {date}, skipping")
#             return "ALREADY_ANALYZED"

#         self.__analyzed_date_list.append(date)  ## Add the date to the analyzed list
#         event_date = pd.to_datetime(date) + BDay(
#             self.t_plus_n
#         )  ## Order date is the next business day after the current date
#         logger.info(f"Analyzing Positions on {date}")
#         is_holiday = is_USholiday(date)
#         if is_holiday:
#             self.pm.logger.warning(f"Market is closed on {date}, skipping")
#             logger.info(f"Market is closed on {date}, skipping")
#             return "IS_HOLIDAY"

#         ## First check if the position needs to be rolled
#         if self.limits["dte"]:
#             roll_dict = self.dte_check()
#         else:
#             logger.info("Roll Check Not Enabled")
#             roll_dict = {}
#             for sym in self.pm.symbol_list:
#                 current_position = self.pm.current_positions[sym]
#                 if "position" not in current_position:
#                     continue
#                 roll_dict[current_position["position"]["trade_id"]] = HOLD(current_position["position"]["trade_id"])

#         logger.info(f"Roll Dict {roll_dict}")

#         ## Check if the position needs to be adjusted based on moneyness
#         if self.limits["moneyness"]:
#             moneyness_dict = self.moneyness_check()
#         else:
#             logger.info("Moneyness Check Not Enabled")
#             moneyness_dict = {}
#             for sym in self.pm.symbol_list:
#                 current_position = self.pm.current_positions[sym]
#                 if "position" not in current_position:
#                     continue
#                 moneyness_dict[current_position["position"]["trade_id"]] = HOLD(
#                     current_position["position"]["trade_id"]
#                 )
#         logger.info(f"Moneyness Dict: {moneyness_dict}")

#         ## Check if the position needs to be adjusted based on greeks
#         greek_dict = self.limits_check()
#         logger.info(f"Greek Dict {greek_dict}")

#         check_dicts = [roll_dict, moneyness_dict, greek_dict]
#         all_empty = all([len(x) == 0 for x in check_dicts])

#         if all_empty:  ## Return if all are empty
#             self.pm.logger.info(f"No positions need to be adjusted on {date}")
#             print(f"No positions need to be adjusted on {date}")
#             return "NO_POSITIONS_TO_ADJUST"

#         actions_dicts = {"dte": roll_dict, "moneyness": moneyness_dict, "greeks": greek_dict}
#         ## Aggregate the results
#         trades_df = self.unadjusted_signals
#         bars_trade = self.bars.trades_df
#         for sym in self.pm.symbol_list:
#             position = self.pm.current_positions[sym]
#             for signal_id, current_position in position.items():
#                 if "position" not in current_position:
#                     continue
#                 k = current_position["position"]["trade_id"]
#                 exit_signal_date = trades_df[trades_df["signal_id"] == signal_id].ExitTime.values[
#                     0
#                 ]  ## This is not look ahead because Signal is gotten on bars_df - t_plus_n
#                 entry_signal_date = trades_df[trades_df["signal_id"] == signal_id].EntryTime.values[
#                     0
#                 ]  ## This is not look ahead because Signal is gotten on bars_df - t_plus_n
#                 exit_date, entry_date = (
#                     bars_trade[bars_trade["signal_id"] == signal_id].ExitTime.values[0],
#                     bars_trade[bars_trade["signal_id"] == signal_id].EntryTime.values[0],
#                 )
#                 if compare_dates.is_on_or_after(date, exit_signal_date) or compare_dates.is_on_or_before(
#                     date, entry_date
#                 ):
#                     logger.info(
#                         f"Position has exited on {exit_signal_date} or not yet entered on {entry_date}, skipping"
#                     )
#                     continue

#                 ## There are 4 possible actions: roll, Hold, Exercise, Adjust
#                 ## Roll happens on DTE & Moneyness. Exercise happens on DTE. Adjust happens on Greeks
#                 actions = []
#                 reasons = []
#                 for action in actions_dicts:
#                     if k in actions_dicts[action]:
#                         actions.append(actions_dicts[action][k])
#                         reasons.append(action)
#                     else:
#                         actions.append(EventTypes.HOLD.value)
#                         reasons.append("hold")

#                 sub_action_dict = {"action": "", "quantity_diff": 0}

#                 ## If the position needs to be rolled or exercised, do that first, no need to check other actions or adjust quantity
#                 if EventTypes.ROLL.value in actions:
#                     pos_action = ROLL(k, {})
#                     pos_action.reason = reasons[actions.index(EventTypes.ROLL.value)]

#                     event = RollEvent(
#                         datetime=event_date,
#                         symbol=sym,
#                         signal_type=parse_signal_id(signal_id)["direction"],
#                         position=current_position,
#                         signal_id=signal_id,
#                     )
#                     pos_action.event = event
#                     position_action_dict[k] = pos_action
#                     continue

#                 ## If exercise is needed, do that first, no need to check other actions or adjust quantity
#                 elif EventTypes.EXERCISE.value in actions:
#                     pos_action = EXERCISE(k, {})
#                     pos_action.reason = reasons[actions.index(EventTypes.EXERCISE.value)]
#                     long_premiums, short_premiums = self.pm.get_premiums_on_position(current_position["position"], date)

#                     event = ExerciseEvent(
#                         datetime=date,  ## Exercise happens on the same day as the action.
#                         symbol=sym,
#                         quantity=current_position["quantity"],
#                         entry_date=date,
#                         spot=self.chain_spot_timeseries[sym][
#                             date
#                         ],  ## Using chain spot because strikes are unadjusted for splits
#                         long_premiums=long_premiums,
#                         short_premiums=short_premiums,
#                         position=current_position,
#                         signal_id=signal_id,
#                     )
#                     pos_action.event = event
#                     sub_action_dict[k] = pos_action

#                     continue

#                 ## If the position is a hold, check if it needs to be adjusted based on greeks
#                 elif EventTypes.HOLD.value in actions:
#                     pos_action = HOLD(k)
#                     pos_action.reason = None
#                     position_action_dict[k] = pos_action

#                 quantity_change_list = [0]  ## Initialize the quantity change list with 0
#                 value = greek_dict.get(k, {})  ## Get the greek dict for each position
#                 for greek, res in value.items():  ## Looping through each greek adjustments
#                     quantity_change_list.append(res["quantity_diff"])
#                 sub_action_dict["quantity_diff"] = min(
#                     quantity_change_list
#                 )  ## Ultimate adjustment would be the minimum reduction factor because they're all negative values
#                 if (
#                     sub_action_dict["quantity_diff"] < 0
#                 ):  ## If the quantity needs to be reduced, set the action to adjust
#                     pos_action = ADJUST(k, sub_action_dict)
#                     pos_action.reason = "greek_limit"

#                     event = OrderEvent(
#                         symbol=sym,
#                         datetime=event_date,
#                         order_type="MKT",
#                         quantity=abs(sub_action_dict["quantity_diff"]),
#                         direction="SELL" if sub_action_dict["quantity_diff"] < 0 else "BUY",
#                         position=current_position["position"],
#                         signal_id=signal_id,
#                     )
#                     pos_action.event = event
#                     position_action_dict[k] = pos_action  ## If adjust position, override HOLD.
#         self._actions[date] = position_action_dict
#         logger.info(f"Position Action Dict: {position_action_dict}")
#         for id, action in position_action_dict.items():
#             logger.info(f"Position ID: {id}, Action: {action}, Reason: {action.reason}")
#             if not isinstance(action, HOLD):
#                 logger.info(f"Event: {action.event}")
#                 logger.info(
#                     (f"Risk Manager Scheduling Action: Position ID: {id}, Action: {action}, Reason: {action.reason}")
#                 )
#                 self.pm.eventScheduler.schedule_event(event_date, action.event)

#         return position_action_dict

#     def limits_check(self):
#         limits = self.limits
#         delta_limit = limits["delta"]
#         position_limit = {}

#         date = pd.to_datetime(self.pm.eventScheduler.current_date)
#         if is_USholiday(date):
#             self.pm.logger.warning(f"Market is closed on {date}, skipping")
#             return

#         for symbol in self.pm.symbol_list:
#             position = self.pm.current_positions[symbol]
#             for signal_id, current_position in position.items():
#                 if "position" not in current_position:
#                     continue
#                 logger.info(f"Checking Position {current_position['position']['trade_id']} for Greek Limits on {date}")
#                 trade_id = current_position["position"]["trade_id"]
#                 quantity = current_position["quantity"]
#                 signal_id = signal_id
#                 max_delta = self.greek_limits["delta"][signal_id]
#                 pos_data = self.position_data[trade_id]

#                 status = {"status": False, "quantity_diff": 0}
#                 greek_limit_bool = dict(delta=status, gamma=status, vega=status, theta=status)

#                 if delta_limit:
#                     delta_val = abs(pos_data.at[date, "Delta"])
#                     skip = pos_data.at[date, "Delta_skip_day"] if "Delta_skip_day" in pos_data.columns else False

#                     if skip or delta_val == 0:
#                         continue

#                     current_delta = delta_val * quantity
#                     if current_delta > max_delta:
#                         # Compute how many contracts to reduce
#                         required_quantity = max(
#                             int(max_delta // delta_val), 1
#                         )  ## Ensure at least 1 contract is required. If last contract exceeds delta limit, we will still hold it.
#                         quantity_diff = required_quantity - quantity
#                         logger.info(
#                             f"Position {trade_id} exceeds delta limit. Current Delta: {current_delta}, Max Delta: {max_delta}, Required Quantity: {required_quantity}, Current Quantity: {quantity}"
#                         )
#                         greek_limit_bool["delta"] = {"status": True, "quantity_diff": quantity_diff}

#                     position_limit[trade_id] = greek_limit_bool

#         return position_limit

#     def dte_check(self):
#         """
#         Analyze the current positions and determine if any need to be rolled
#         """
#         date = pd.to_datetime(self.pm.eventScheduler.current_date)
#         logger.info(f"Checking DTE on {date}")
#         if is_USholiday(date):
#             self.pm.logger.warning(f"Market is closed on {date}, skipping")
#             return

#         roll_dict = {}
#         for symbol in self.pm.symbol_list:
#             position = self.pm.current_positions[symbol]
#             for signal_id, current_position in position.items():
#                 if "position" not in current_position:
#                     continue

#                 logger.info(f"Checking Position {current_position['position']['trade_id']} for DTE on {date}")
#                 id = current_position["position"]["trade_id"]
#                 expiry_date = ""

#                 if "long" in current_position["position"]:
#                     for option_id in current_position["position"]["long"]:
#                         option_meta = parse_option_tick(option_id)
#                         expiry_date = option_meta["exp_date"]
#                         break
#                 elif "short" in current_position["position"]:
#                     for option_id in current_position["position"]["short"]:
#                         option_meta = parse_option_tick(option_id)
#                         expiry_date = option_meta["exp_date"]
#                         break

#                 dte = (pd.to_datetime(expiry_date) - pd.to_datetime(date)).days
#                 logger.info(f"ID: {id}, DTE: {dte}, Expiry: {expiry_date}, Date: {date}")

#                 if symbol in self.pm.roll_map and dte <= self.pm.roll_map[symbol]:
#                     logger.info(f"{id} rolling because {dte} <= {self.pm.roll_map[symbol]}")
#                     roll_dict[id] = EventTypes.ROLL.value
#                 elif symbol not in self.pm.roll_map and dte == 0:  # exercise contract if symbol not in roll map
#                     logger.info(f"{id} exercising because {dte} == 0")
#                     roll_dict[id] = EventTypes.EXERCISE.value
#                 else:
#                     logger.info(f"{id} holding because {dte} > {self.pm.roll_map[symbol]}")
#                     roll_dict[id] = EventTypes.HOLD.value
#         return roll_dict

#     def moneyness_check(self):
#         """
#         Analyze the current positions and determine if any need to be rolled based on moneyness
#         """
#         date = pd.to_datetime(self.pm.eventScheduler.current_date)
#         logger.info(f"Checking Moneyness on {date}")
#         if is_USholiday(date):
#             self.pm.logger.warning(f"Market is closed on {date}, skipping")
#             return

#         roll_dict = {}
#         for symbol in self.pm.symbol_list:
#             strike_list = []
#             position = self.pm.current_positions[symbol]
#             for signal_id, current_position in position.items():
#                 if "position" not in current_position:
#                     continue

#                 logger.info(f"Checking Position {current_position['position']['trade_id']} for Moneyness on {date}")
#                 id = current_position["position"]["trade_id"]
#                 try:
#                     entry_date = self.pm.trades_map[id].entry_date
#                 except Exception as e:
#                     logger.error(f"Error getting entry date for position {id}: {e}")
#                     entry_date = date
#                 spot = self.chain_spot_timeseries[symbol][
#                     date
#                 ]  ## Use the spot price on the date (from chain cause of splits)

#                 if "long" in current_position["position"]:
#                     for option_id in current_position["position"]["long"]:
#                         option_meta = self.adjust_for_events(entry_date, date, parse_option_tick(option_id))
#                         strike_list.append(
#                             option_meta["strike"] / spot
#                             if option_meta["put_call"] == "P"
#                             else spot / option_meta["strike"]
#                         )

#                 if "short" in current_position["position"]:
#                     for option_id in current_position["position"]["short"]:
#                         option_meta = self.adjust_for_events(entry_date, date, parse_option_tick(option_id))
#                         strike_list.append(
#                             option_meta["strike"] / spot
#                             if option_meta["put_call"] == "P"
#                             else spot / option_meta["strike"]
#                         )

#                 logger.info(f"{id} moneyness list {strike_list}, spot: {spot}, date: {date}, entry_date: {entry_date}")
#                 logger.info(f"{id} moneyness bool list {[x > self.max_moneyness for x in strike_list]}")

#                 roll_dict[id] = (
#                     EventTypes.ROLL.value
#                     if any([x > self.max_moneyness for x in strike_list])
#                     else EventTypes.HOLD.value
#                 )
#         return roll_dict

#     def submit_add_columns(self, columns: Tuple[str, str]):
#         """
#         Submits additional columns to be added to the risk manager's data.
#         Args:
#             columns (Tuple[str, str]): A tuple containing the column names to be added.
#         Raises:
#             AssertionError: If the first column is not one of the recognized columns or if the second column is not in the ADD_COLUMNS_FACTORY.
#         """
#         raise DeprecationWarning("submit_add_columns is deprecated, please use SkipCalcConfig.add_columns instead.")
#         assert isinstance(columns, tuple) and len(columns) == 2, "columns must be a tuple of two strings"
#         assert columns[0] in ['Midpoint', 'Delta', 'Open', 'Close', 'Closeask', 'Closebid'], f"Column {columns[0]} not recognized, expected 'Midpoint', 'Delta', 'Open', 'Close', 'Closeask', or 'Closebid'"
#         assert columns[1] in ADD_COLUMNS_FACTORY.keys(), f"Column {columns[1]} not recognized, expected {ADD_COLUMNS_FACTORY.keys()}"
#         ## Check if the columns already exist in the add_columns list
#         if columns in self.add_columns:
#             logger.info(f"Column {columns} already exists in add_columns, skipping addition")
#             return
#         self.add_columns.append(columns)




#     def print_settings(self):
#         raise DeprecationWarning("print_settings is deprecated, please use the RiskManager's __str__ method instead.")
#         msg = f"""
# Risk Manager Settings:
# Start Date: {self.pm_start_date}
# End Date: {self.pm_end_date}
# Current Limits State (Position Adjusted when these thresholds are reached):
#     Delta: {self.limits['delta']}
#     Gamma: {self.limits['gamma']}
#     Vega: {self.limits['vega']}
#     Theta: {self.limits['theta']}
#     Roll On DTE: {self.limits['dte']}
#         Min DTE Threshold: {self.pm.min_acceptable_dte_threshold}
#     Roll On Moneyness: {self.limits['moneyness']}
#         Max Moneyness: {self.max_moneyness}
# Quanitity Sizing Type: {self.sizing_type}
#             """
#         print(msg)


        
#     # @log_error_with_stack(logger)
#     # @log_time(time_logger)
#     def get_order_old(self, *args, **kwargs):
#         """
#         Compulsory variables for OrderSchema:
#         signal_id: str: Unique identifier for the signal
#         date: str|datetime: Date for which the order is to be placed
#         tick: str: Ticker for the option contract
#         max_close: float: Maximum close price for the order
#         strategy: str: Strategy type
#         option_type: str: Option type
#         target_dte: int: Target days to expiration
#         structure_direction: str: Direction of the structure
        

#         Optional variables:
#         spread_ticks: int: Number of ticks for the spread, default is 1
#         dte_tolerance: int: Tolerance for days to expiration, default is 60
#         min_moneyness: float: Minimum moneyness for the order, default is 0.75
#         max_moneyness: float: Maximum moneyness for the order, default is 1.25
#         min_total_price: float: Minimum total price for the order, default is max_close/2

#         This function generates an order based on the provided parameters and returns it.
#         """
#         ## Initialize the order cache if it doesn't exist
#         raise DeprecationWarning("get_order_old is deprecated, please use get_order with OrderRequest dataclass instead.")
#         order_cache = self.order_cache
#         signalID = kwargs.pop('signal_id')
#         date = kwargs.get('date')
#         tick = kwargs.get('tick')
#         max_close = kwargs.get('max_close', 2.0) 
#         option_strategy = kwargs.pop('strategy')
#         option_type = kwargs.pop('option_type')
#         structure_direction = kwargs.pop('structure_direction')
#         spread_ticks = kwargs.pop('spread_ticks', 1)
#         dte_tolerance = kwargs.pop('dte_tolerance', 60)
#         min_moneyness = kwargs.pop('min_moneyness', 0.75)
#         max_moneyness = kwargs.pop('max_moneyness', 1.25)
#         target_dte = kwargs.pop('target_dte')
#         min_total_price = kwargs.pop('min_total_price', max_close/2)
#         direction='LONG' if option_type == 'C' else 'SHORT' 
        
#         if is_USholiday(date):
#             logger.info(f"Date {date} is a US Holiday, skipping order generation")
#             return {
#                 'result': ResultsEnum.IS_HOLIDAY.value,
#                 'data': None
#             }


#         self.generate_data(tick)
#         spot = self.chain_spot_timeseries[tick][date] 

#         logger.info(f"## ***Signal ID: {signalID}***")

#         ## I cannot calculate greeks here. I need option_data to be available first.
#         # order = self.OrderPicker.get_order(*args, **kwargs)    

#         ## Testing new order picker
#         schema = OrderSchema({
#             "strategy": option_strategy, "option_type": option_type, "tick": tick,
#             "target_dte": target_dte, "dte_tolerance": dte_tolerance,
#             "structure_direction": structure_direction, "max_total_price": max_close,
#             "spread_ticks":spread_ticks, "min_moneyness": min_moneyness, "max_moneyness": max_moneyness, "increment": 0.5,
#             "min_total_price": min_total_price
#         })
#         logger.info(f"Initial Schema on {date}: {schema}")
#         order = self.OrderPicker.get_order_new(schema, date, spot, print_url = False)

#         ## Resolve the schema if the order is not successful
#         tries = 0
#         while order['result'] != ResultsEnum.SUCCESSFUL.value:
#             logger.info(f"Failed to produce order with schema: {schema}, trying to resolve schema, on try {tries}")
#             pack = resolve_schema(schema,
#                                            tries = tries,
#                                             max_dte_tolerance = self.max_dte_tolerance,
#                                             max_close = self.pm.allocated_cash_map[tick]/100,
#                                             max_tries = self.max_tries,
#                                             otm_moneyness_width = self.otm_moneyness_width,
#                                             itm_moneyness_width = self.itm_moneyness_width)
#             schema, tries = pack

#             if schema is False:
#                 logger.info(f"Unable to resolve schema after {tries} tries, returning None")
#                 self.schema_cache.setdefault(date, {}).update({signalID: schema})
#                 return {
#                     'result': ResultsEnum.NO_CONTRACTS_FOUND.value,
#                     'data': None
#                 }
#             logger.info(f"Resolved Schema: {schema}, tries: {tries}")
#             order = self.OrderPicker.get_order_new(schema, date, spot, print_url = False) ## Get the order from the OrderPicker

            
#         self.schema_cache.setdefault(date, {}).update({signalID: schema}) ## Update the schema cache with the date and signalID

        
#         signal_meta = parse_signal_id(signalID)
#         logger.info(f"Order Produced: {order}")


#         if order['result'] == ResultsEnum.SUCCESSFUL.value:
#             print(f"\nOrder Received: {order}\n")
#             position_id = order['data']['trade_id']
#             order['signal_id'] = signalID
#             order['direction'] = direction
            
#         else:
#             print(f"\nOrder Failed: {order}\n")
#             logger.info(f"Signal ID: {signalID}, Unable to produce order, returning None")
#             return order
        
#         logger.info(f"Position ID: {position_id}")
#         logger.info("Calculating Position Greeks")
#         self.calculate_position_greeks(position_id, kwargs['date'])
#         order = self.update_order_close(position_id, kwargs['date'], order) ## Update the order with the close price from the position data
#         logger.info('Updating Signal Limits')
#         self.sizer.update_delta_limit(signalID, position_id, date)

#         ## Hack to get limit meta
#         self.store_limits_meta(signalID, position_id, date)
#         logger.info("Calculating Quantity")
#         quantity = self.sizer.calculate_position_size(signalID, position_id, order['data']['close'], kwargs['date'])
#         logger.info(f"Quantity for Position ({position_id}) Date {kwargs['date']}, Signal ID {signalID} is {quantity}")
#         order['data']['quantity'] = quantity
#         order['data']['cash_equivalent_qty'] = self.pm.allocated_cash_map[tick] // (order['data']['close'] * 100)
#         logger.info(order)

#         ## save the order in the cache
#         if date not in order_cache:
#             cache_dict = {tick: order}
#             order_cache[date] = cache_dict
#         else:
#             cache_dict = order_cache[date]
#             cache_dict[tick] = order
#             order_cache[date] = cache_dict
        
#         self.adjust_slippage(position_id, date) ## Adjust the slippage for the position based on the position data

#         return order

#     def store_limits_meta(self, signal_id, position_id, date):
#         """
#         Stores the limits meta for the signal and position
#         """
#         raise DeprecationWarning("store_limits_meta is deprecated, please access LimitsCog directly for limits meta.")

#         delta = self.greek_limits['delta'].get(signal_id, None)
#         gamma = self.greek_limits['gamma'].get(signal_id, None)
#         vega = self.greek_limits['vega'].get(signal_id, None)
#         theta = self.greek_limits['theta'].get(signal_id, None)

#         self.limits_meta[(signal_id, position_id, date)] = {
#             'delta': delta,
#             'gamma': gamma,
#             'vega': vega,
#             'theta': theta,
#         }
    
#     def adjust_slippage(self, position_id, date):
#         raise DeprecationWarning("adjust_slippage is deprecated, It will no longer be supported in future versions.")
#         position_data = self.position_data.get(position_id, None)
#         if position_data is None:
#             logger.error(f"Position Data for {position_id} not available, cannot adjust slippage")
#             return None
        
#         if 'spread_ratio' in position_data:  
#             spread_ratio = position_data['spread_ratio'][date] if position_data['spread_ratio'][date] else self.max_slippage
#             decided_slippage = min(spread_ratio, self.max_slippage)
#             logger.info(f"Position {position_id} on date {date} has spread ratio {spread_ratio}, adjusting slippage to {decided_slippage}")
#             self.executor.max_slippage_pct = decided_slippage
#         else:
#             logger.warning(f"Spread Ratio not available for position {position_id}, using default max slippage of {self.max_slippage}")
#             self.executor.max_slippage_pct = self.max_slippage

#         ## Overriding to risk_manager set for now
#         self.executor.min_slippage_pct = self.min_slippage
#         self.executor.max_slippage_pct = self.max_slippage
    
#     def update_order_close(self, position_id:str, date:str|datetime, order:dict)-> dict:
#         """
#         Updates the close price of the order based on the position data.
#         Parameters:
#         position_id: str: ID of the position
#         date: str|datetime: Date for which the order is to be updated
#         order: dict: Order dictionary containing the order details
#         Returns:
#         dict: Updated order dictionary with the close price
#         """
#         raise DeprecationWarning("update_order_close is deprecated, please use get_at_time_position_data from BacktestTimeseries instead.")

#         skip = self.position_data[position_id]['Midpoint_skip_day'][date]
#         if skip:
#             self.skip_adj_count += 1
            
#             close = self.position_data[position_id]['Midpoint'].ewm(span = 3).mean()[date] if self.option_price == 'Midpoint' \
#                 else self.position_data[position_id][self.option_price.capitalize()][date]
#             logger.info(f"***TESTING WITH EWM PRICE*** Position ID: {position_id}, Date: {date} - Skipping Day, using EWM Price")
#             logger.info(f"OLD CLOSE: {order['data']['close']}, NEW CLOSE: {close}, PCT_CHANGE: {(close - order['data']['close'])/order['data']['close']*100:.2f}%")
#         else:
#             close = self.position_data[position_id][self.option_price.capitalize()][date]
#         order['data']['close'] = close
#         return order
    

#     def register_option_meta_frame(
#             self,
#             date: str|datetime,
#             trade_id:str,
#     ) -> None:
#         raise DeprecationWarning("register_option_meta_frame is deprecated, option data handling has been revamped.")
#         ## Generate a DataFrame for each direction in the trade
#         trade_meta = self.parse_position_id(trade_id)[0]
#         direction_pair = self.id_meta.setdefault(trade_id, {'L': pd.DataFrame(
#             index = bus_range(self.pm_start_date, self.pm_end_date, freq='1d'),
#         ), 'S': pd.DataFrame(
#             index = bus_range(self.pm_start_date, self.pm_end_date, freq='1d'),
#         )})

#         ## Get split info
#         splits = self.splits

#         ## Loop through each direction 
#         for direction, details in trade_meta.items():
#             direction_frame = direction_pair.get(direction, pd.DataFrame())

#             ## Loop through each option in the direction
#             for i, option in enumerate(details):

#                 ## First populate the Given Option Detail
#                 direction_frame[i] = generate_option_tick_new(*option.values())
#                 tick = option['ticker']
#                 split_info = splits.get(tick, None)
#                 if split_info is None:
#                     continue

#                 ## If there is split info, we adjust
#                 for split_meta in split_info:
#                     if not compare_dates.is_after(split_meta[0], date):
#                         continue
#                     split_start, split_ratio = split_meta
#                     new_details = deepcopy(option)
#                     new_details['strike']/=split_meta[1]
#                     direction_frame.loc[split_start:, i] = generate_option_tick_new(*new_details.values())

#     @log_time(time_logger)
#     def calculate_position_greeks(self, positionID, date):
#         """
#         Calculate the greeks of a position

#         date: Evaluation Date for the greeks (PS: This is not the pricing date)
#         positionID: str: position string. (PS: This function assumes ticker for position is the same)
#         """
#         raise DeprecationWarning("calculate_position_greeks is deprecated, please use BacktestTimeseries.calculate_option_data instead.")
#         logger.info(f"Calculate Greeks Dates Start: {self.start_date}, End: {self.end_date}, Position ID: {positionID}, Date: {date}")
#         if positionID in self.position_data:
#             ## If the position data is already available, then we can skip this step
#             logger.info(f"Position Data for {positionID} already available, skipping calculation")
#             return self.position_data[positionID]
#         else:
#             logger.critical(f"Position Data for {positionID} not available, calculating greeks. Load time ~5 minutes")
#         ## Initialize the Long and Short Lists
#         long = []
#         short = []
#         threads = []
#         thread_input_list = [
#             [], []
#         ]

#         date = pd.to_datetime(date) ## Ensure date is in datetime format
    
        
#         ## First get position info
#         position_dict, positon_meta = self.parse_position_id(positionID)

#         ## Now ensure that the spot and dividend data is available
#         for p in position_dict.values():
#             for s in p:
#                 self.generate_data(swap_ticker(s['ticker']))
#         ticker = swap_ticker(s['ticker'])
#         if ticker not in self.spot_timeseries:
#             self.spot_timeseries[ticker] = self.pm.get_underlier_data(ticker).spot(
#                 ts = True,
#                 ts_start = self.start_date,
#                 ts_end = self.end_date,
#             )

#         if ticker not in self.chain_spot_timeseries:
#             self.chain_spot_timeseries[ticker] = self.pm.get_underlier_data(ticker).spot(
#                 ts = True,
#                 ts_start = self.start_date,
#                 ts_end = self.end_date,
#                 spot_type = 'chain_price'
#             )

#         if ticker not in self.dividend_timeseries:
#             self.dividend_timeseries[ticker] = self.pm.get_underlier_data(ticker).div_yield_history(start = self.start_date)


#         @log_time(time_logger)
#         def get_timeseries(_id, direction):
#             logger.info("Calculate Greeks dates")
#             logger.info(f"Start Date: {self.start_date}")
#             logger.info(f"End Date: {self.end_date}")
            
#             logger.info(f"Calculating Greeks for {_id} on {date} in {direction} direction")
#             data = self.generate_option_data_for_trade(_id, date) ## Generate the option data for the trade

#             if direction == 'L':
#                 long.append(data)
#             elif direction == 'S':
#                 short.append(data)
#             else:
#                 raise ValueError(f"Position Type {_set[0]} not recognized")
            
#             return data

#         ## Calculating IVs & Greeks for the options
#         for _set in positon_meta:
#             thread_input_list[0].append(_set[1]) ## Append the option id to the thread input list
#             thread_input_list[1].append(_set[0]) ## Append the direction to the thread input list
#         runThreads(get_timeseries, thread_input_list, block=True) ## Run the threads to get the timeseries data for the options
            
#         position_data = sum(long) - sum(short)
#         position_data = position_data[~position_data.index.duplicated(keep = 'first')]
#         position_data.columns = [x.capitalize() for x in position_data.columns]
#         ## Retain the spot, risk free rate, and dividend yield for the position, after the greeks have been calculated & spread values subtracted
#         position_data['s0_close'] = self.spot_timeseries[ticker] ## Spot price at the time of the position
#         position_data['s'] = self.chain_spot_timeseries[ticker] ## Chain spot price at the time of the position
#         position_data['r'] = self.rf_timeseries ## Risk free rate at the time of the position
#         position_data['y'] = self.dividend_timeseries[ticker] ## Dividend yield at the time of the position
#         position_data['spread'] = position_data['Closeask'] - position_data['Closebid'] ## Spread is the difference between the ask and bid prices

#         ## PRICE_ON_TO_DO: No need to change
#         ## Add the additional columns to the position data
#         position_data['spread_ratio'] = (position_data['spread'] / position_data['Midpoint'] ).abs().replace(np.inf, np.nan).fillna(0) ## Spread ratio is the spread divided by the midpoint price
#         position_data = add_skip_columns(position_data, positionID, ['Delta', 'Gamma', 'Vega', 'Theta', 'Midpoint'], window = 20, skip_threshold=3)
#         if self.add_columns:
#             for col in self.add_columns:
#                 position_data[f"{col[0]}_{col[1]}".capitalize()] = ADD_COLUMNS_FACTORY[col[1]](position_data[col[0]])
#         self.position_data[positionID] = position_data

#         return position_data


#     def load_position_data(self, opttick) -> pd.DataFrame:
#         """
#         Load position data for a given option tick.

#         This function ONLY retrives the data for the option tick, it does not apply any splits or adjustments.
#         This function will NOT check for splits or special dividends. It will only retrieve the data for the given option tick.
#         """
#         raise DeprecationWarning("load_position_data is deprecated, please use load_position_data from BacktestTimeseries instead.")
#         ## Get Meta
#         meta = parse_option_tick(opttick)
#         return load_position_data(opttick, 
#                                   self.processed_option_data, 
#                                   self.start_date, 
#                                   self.end_date,
#                                   s=self.chain_spot_timeseries[meta['ticker']],
#                                   r=self.rf_timeseries,
#                                   y=self.dividend_timeseries[meta['ticker']],
#                                   s0_close=self.spot_timeseries[meta['ticker']],)

#     def enrich_data(self, data, ticker) -> pd.DataFrame:
#         """
#         Enrich the data with additional information.
#         """
#         raise DeprecationWarning("enrich_data is deprecated. This functionality has been moved to load_position_data function.")
#         return enrich_data(data, ticker, self.spot_timeseries, self.chain_spot_timeseries, self.rf_timeseries, self.dividend_timeseries)
        
#     def generate_spot_greeks(self, opttick) -> pd.DataFrame:
#         """
#         Generate spot greeks for a given option tick.
#         """
#         raise DeprecationWarning("generate_spot_greeks is deprecated, please use generate_spot_greeks from BacktestTimeseries instead.")
#         ## PRICE_ON_TO_DO: NO NEED TO CHANGE. This is necessary retrievals
#         return generate_spot_greeks(opttick, self.start_date, self.end_date)


#     def generate_option_data_for_trade(self, opttick, check_date) -> pd.DataFrame:
#         """
#         Generate option data for a given trade.
#         This function retrieve the option data to backtest on. Data will not be saved, as it will be applying splits and adjustments.
#         This function is written with the assumption that there is no cummulative splits. Expectation is only one split per option tick.
#             Obviously, this might not be the case if the option was alive for ~5 years or more. But most options are not alive for that long.
#         """
#         raise DeprecationWarning("generate_option_data_for_trade is deprecated, please use generate_option_data_for_trade from BacktestTimeseries instead.")
#         meta = parse_option_tick(opttick)

#         ## Check if there's any split/special dividend
#         splits = self.splits.get(meta['ticker'], [])
#         dividends = self.special_dividends.get(meta['ticker'], {})
#         to_adjust_split = []

#         ## To avoid loading multiple data to account for splits everytime, we check if the PM_date range includes the split date  
#         for pack in splits:
#             if compare_dates.inbetween(
#                 pack[0], 
#                 self.pm_start_date, 
#                 self.pm_end_date,
#             ):
#                 pack = list(pack)  ## Convert to list to append later
#                 pack.append('SPLIT')
#                 to_adjust_split.append(pack)

#         for pack in dividends.items():
#             if compare_dates.inbetween(
#                 pack[0], 
#                 self.pm_start_date, 
#                 self.pm_end_date,
#             ):
#                 pack = list(pack)
#                 pack.append('DIVIDEND')
#                 to_adjust_split.append(pack)

#         ## Sort the splits by date
#         to_adjust_split.sort(key=lambda x: x[0])  ## Sort by date
#         logger.info(f"Splits and Dividends to adjust for {opttick}: {to_adjust_split} range: {self.pm_start_date} to {self.pm_end_date}")

#         ## If there are no splits, we can just load the data
#         if not to_adjust_split:
#             data = self.load_position_data(opttick).copy()  ## Copy to avoid modifying the original data
#             return data[(data.index >= pd.to_datetime(self.pm_start_date) - relativedelta(months = 3))\
#                          & (data.index<= pd.to_datetime(self.pm_end_date) + relativedelta(months = 3))]

#         # If there are splits, we need to load the data for each tick after adjusting strikes
#         else:
#             adj_meta = meta.copy()
#             adj_strike = meta['strike']
#             logger.info(f"Generating data for {opttick} with splits: {to_adjust_split}")
#             ## Load the data for picked option first
#             first_set_data = self.load_position_data(opttick).copy()  ## Copy to avoid modifying the original data
#             if compare_dates.is_before(check_date, to_adjust_split[0][0]):
#                 first_set_data = first_set_data[first_set_data.index < to_adjust_split[0][0]]
#             else:
#                 first_set_data = first_set_data[first_set_data.index >= to_adjust_split[0][0]]

#             segments = []

#             for event_date, factor, event_type in to_adjust_split:
#                 if compare_dates.is_before(check_date, event_date):
#                     # You're in the PRE-event regime
#                     if event_type == 'SPLIT':
#                         adj_strike /= factor
#                     elif event_type == 'DIVIDEND':
#                         adj_strike -= factor
#                 else:
#                     # You're in the POST-event regime
#                     if event_type == 'SPLIT':
#                         adj_strike *= factor
#                     elif event_type == 'DIVIDEND':
#                         adj_strike += factor

#                 adj_opttick = generate_option_tick_new(
#                     symbol=adj_meta['ticker'],
#                     strike=adj_strike,
#                     right=adj_meta['put_call'],
#                     exp=adj_meta['exp_date']
#                 )
#                 logger.info(f"Adjusted option tick: {adj_opttick} for event {event_type} on {event_date} with factor {factor}")

#                 # Load adjusted data
#                 if adj_opttick not in self.processed_option_data:
#                     adj_data = self.load_position_data(adj_opttick).copy()
#                 else:
#                     adj_data = self.processed_option_data[adj_opttick]

#                 # Slice around the event
#                 if compare_dates.is_before(check_date, event_date):
#                     adj_data = adj_data[adj_data.index >= event_date]
#                 else:
#                     adj_data = adj_data[adj_data.index < event_date]

#                 # Apply price transformation if SPLIT
#                 ## PRICE_ON_TO_DO: No need to change this. These are necessary columns
#                 if event_type == 'SPLIT':
#                     cols = ['Midpoint', 'Closeask', 'Closebid']
#                     if compare_dates.is_before(check_date, event_date):
#                         adj_data[cols] *= factor
#                     else:
#                         adj_data[cols] /= factor
                    
#                 segments.append(adj_data)

        
#         base_data = self.load_position_data(opttick).copy()
#         first_event_date = to_adjust_split[0][0] if to_adjust_split else self.pm_start_date
#         if compare_dates.is_before(check_date, first_event_date):
#             base_data = base_data[base_data.index < first_event_date]
            
#         else:
#             base_data = base_data[base_data.index >= first_event_date]
        
#         segments.insert(0, base_data)
#         final_data = pd.concat(segments).sort_index()
#         final_data = final_data[~final_data.index.duplicated(keep='last')]
        
#         ## Leave residual data outside the PM date range
#         final_data = final_data[(final_data.index >= pd.to_datetime(self.pm_start_date) - relativedelta(months = 3)) & \
#                                 (final_data.index <= pd.to_datetime(self.pm_end_date) + relativedelta(months = 3))]
#         return final_data

#     @log_time(time_logger)
#     def update_greek_limits(self, signal_id, position_id) -> None:
#         """
#         Updates the limits associated with a signal
#         ps: This should only be updated on first purchase of the signal
#             Limits are saved in absolute values to account for both long and short positions
        
#         """
#         raise DeprecationWarning("update_greek_limits is deprecated, please use LimitsCog directly to manage greek limits.")
#         if signal_id in self.greek_limits['delta'] and not self.re_update_on_roll: ## May consider to maximize cash on roll
#             logger.info(f"Greek Limits for Signal ID: {signal_id} already updated, skipping")
#             return
#         logger.info(f"Updating Greek Limits for Signal ID: {signal_id} and Position ID: {position_id}")
#         id_details = parse_signal_id(signal_id)
#         cash_available = self.pm.allocated_cash_map[swap_ticker(id_details['ticker'])]
#         delta_at_purchase = self.position_data[position_id]['Delta'][id_details['date']] 
#         s0_at_purchase = self.position_data[position_id]['s'][id_details['date']] ## As always, we use the chain spot data to account for splits
#         equivalent_delta_size = ((cash_available/s0_at_purchase)/100) * self.sizing_lev
#         self.greek_limits['delta'][signal_id] = abs(equivalent_delta_size)
#         logger.info(f"Spot Price at Purchase: {s0_at_purchase} at time {id_details['date']}")
#         logger.info(f"Delta at Purchase: {delta_at_purchase}")
#         logger.info(f"Equivalent Delta Size: {equivalent_delta_size}, with Cash Available: {cash_available}, and Leverage: {self.sizing_lev}")
#         logger.info(f"Equivalent Delta Size: {equivalent_delta_size}")

#     def calculate_quantity(self, positionID, signalID, date, opt_price) -> int:
#         """
#         Returns the quantity of the position that can be bought based on the sizing type
#         """
#         raise DeprecationWarning("calculate_quantity is deprecated, please use LimitsCog and PositionSizer directly to manage position sizing.")
#         logger.info(f"Calculating Quantity for Position ID: {positionID} and Signal ID: {signalID} on Date: {date}")
#         if positionID not in self.position_data: ## If the position data isn't available, calculate the greeks
#             self.calculate_position_greeks(positionID, date)
        
#         ## First get position info and ticker
#         position_dict, _ = self.parse_position_id(positionID)
#         key = list(position_dict.keys())[0]
#         ticker = swap_ticker(position_dict[key][0]['ticker'])

#         ## Now calculate the max size cash can buy
#         cash_available = self.pm.allocated_cash_map[ticker]
#         purchase_date = pd.to_datetime(date)
#         s0_at_purchase = self.position_data[positionID]['s'][purchase_date]  ## s -> chain spot, s0_close -> adjusted close
#         logger.info(f"Spot Price at Purchase: {s0_at_purchase} at time {purchase_date}")
#         logger.info(f"Cash Available: {cash_available}, Option Price: {opt_price}, Cash_Available/OptPRice: {(cash_available/(opt_price*100))}")
#         max_size_cash_can_buy = abs(math.floor(cash_available/(opt_price*100))) ## Assuming Allocated Cash map is already in 100s

#         if self.sizing_type == 'price':
#             return max_size_cash_can_buy
          
#         elif self.sizing_type.capitalize() == 'Delta':
#             delta = self.position_data[positionID]['Delta'][purchase_date]
#             if signalID not in self.greek_limits['delta']:
#                 self.update_greek_limits(signalID,positionID )
#             target_delta = self.greek_limits['delta'][signalID]
#             logger.info(f"Target Delta: {target_delta}")
#             delta_size = (math.floor(target_delta/abs(delta)))
#             logger.info(f"Delta from Full Cash Spend: {max_size_cash_can_buy * delta}, Size: {max_size_cash_can_buy}")
#             logger.info(f"Delta with Size Limit: {delta_size * delta}, Size: {delta_size}")
#             return delta_size if abs(delta_size) <= abs(max_size_cash_can_buy) else max_size_cash_can_buy
        
#         elif self.sizing_type.capitalize() in ['Gamma', 'Vega']:
#             raise NotImplementedError(f"Sizing Type {self.sizing_type} not yet implemented, please use 'delta' or 'price'")
        
#         else:
#             raise ValueError(f"Sizing Type {self.sizing_type} not recognized")


#     def hedge_check(self,
#                     hedge_func: callable,
#                     hedge_args: list,
#                     hedge_kwargs: dict,
#                     ) -> dict:
#         """
#         Responsible for checking if the hedge is needed and if so, queueing in analyze_position
#         Hedge function should allow 1st argument to be Risk Manager and 2nd argument to be Portfolio Manager
#         Expected return type is: List[HEDGE]. Where HEDGE is a subclass of RMAction

#         params:
#         hedge_func: callable: function to be called for the hedge
#         hedge_args: list: arguments to be passed to the hedge function
#         hedge_kwargs: dict: keyword arguments to be passed to the hedge function

#         returns:
#         dict: dictionary of the hedge actions
#         """
#         pass 

#     ## Lazy Loading Spot Data
#     def generate_data(self, symbol):
#         raise DeprecationWarning("generate_data is deprecated, please use BacktestTimeseries instead.")
#         stk = self.pm.get_underlier_data(symbol)  ## Performance isn't affected because of singletons in stock class
#         if symbol not in self.spot_timeseries:
#             self.spot_timeseries[symbol] = stk.spot(
#                 ts = True,
#                 ts_start = pd.to_datetime(self.start_date) - BDay(30),
#                 ts_end = pd.to_datetime(self.end_date),
#             )[self.price_on]

#         if symbol not in self.chain_spot_timeseries:
#             self.chain_spot_timeseries[symbol] = stk.spot(
#                 ts = True,
#                 spot_type = OptionModelAttributes.spot_type.value,
#                 ts_start = pd.to_datetime(self.start_date) - BDay(30),
#                 ts_end = pd.to_datetime(self.end_date),
#             )[self.price_on]
        
#         if symbol not in self.dividend_timeseries:
#             divs = stk.div_yield_history(start = pd.to_datetime(self.start_date) - BDay(30))
#             if not isinstance(divs, (pd.DataFrame, pd.Series)): ## When a ticker has no dividends, it returns None/0
#                 divs = pd.Series(divs, index = self.spot_timeseries[symbol].index)
#             self.dividend_timeseries[symbol] = divs


#     def get_option_price(self, optID, date):
#         raise DeprecationWarning("get_option_price is deprecated, please use get_option_price from BacktestTimeseries directly.")
    
#     def adjust_for_events(
#             self,
#             start: str,
#             date: str,
#             option: str|dict,
#     ):
#         """
#         Adjusts the option tick for events like splits or dividends.
#         """
#         raise DeprecationWarning("adjust_for_events is deprecated, please use adjust_for_events from LimitsAndSizingCog directly.")
#         if isinstance(option, str):
#             meta = parse_option_tick(option)
#         elif isinstance(option, dict):
#             meta = option
#         else:
#             raise ValueError("Option must be a string or a dictionary.")
#         split = self.splits.get(swap_ticker(meta['ticker']), None)
#         if split is None:
#             return meta
#         for pack in split:
#             if compare_dates.is_before(start, pack[0]) and compare_dates.is_after(date, pack[0]):
#                 meta['strike'] /= pack[1]
#         return meta

    
    