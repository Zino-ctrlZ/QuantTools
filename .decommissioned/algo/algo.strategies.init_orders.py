"""
This module provides functions to initialize and manage orders for backtesting strategies.
It includes functions to set up the backtest environment, run the backtest, and manage orders and fills.
"""
raise DeprecationWarning("This module (strategies.init_orders) is deprecated and will be removed in future releases. Please use the new order management module.")
# from typing import List
# from datetime import datetime
# from copy import deepcopy
# import os
# import pandas as pd
# from pandas.tseries.offsets import BDay
# from trade.helpers.helper import (
#     change_to_last_busday, 
#     str_to_bool, 
#     ny_now
# )
# from trade.helpers.Logging import setup_logger
# from EventDriven.backtest import OptionSignalBacktest
# from EventDriven.riskmanager.sizer import ZscoreRVolSizer, DefaultSizer
# from EventDriven.riskmanager.utils import (
#     set_timeseries_start, 
#     set_timeseries_end,  
#     set_use_temp_cache
# )
# from EventDriven.riskmanager.utils import parse_position_id
# from EventDriven.helpers import parse_signal_id, generate_signal_id
# from EventDriven.types import SignalTypes
# from module_test.raw_code.DataManagers.DataManagers import set_skip_mysql_query, set_use_quotes
# from dbase.database.SQLHelpers import DatabaseAdapter
# from .init_strategies import get_fills
# from .enums import Action
# from .init_environ import (
#     get_custom_cache
# )
# from .utils import (create_max_cash_map,
#                     get_option_price_theta_data,
#                     get_option_price,
#                     get_option_realtime_quote,
#                     get_position_price,)
# from ..positions.loaders.limits._limits import save_limits_from_backtester
# logger = setup_logger('strategies.init_orders')
# bkt_logger = setup_logger('strategies.backtest')

# logger.critical("WARNING: This module (strategies.init_orders) is deprecated and will be removed in future releases. Please use the new order management module.")

# def get_use_csv() -> bool:
#     """
#     Get the value of USE_CSV.
#     Returns:
#         bool: The current value of USE_CSV.
#     """
#     use_csv = os.environ.get('USE_CSV', 'False').lower()
#     if use_csv not in ['true', '1', 'yes', 'false', '0', 'no']:
#         raise ValueError("USE_CSV must be a boolean value (True/False).")
#     return str_to_bool(use_csv)

# def set_use_csv(value: bool):
#     """
#     Set the value of USE_CSV.
#     Args:
#         value (bool): The value to set for USE_CSV.
#     """
#     if value not in ['true', 'false', 1, 0, True, False, '1', '0', 'yes', 'no']:
#         raise ValueError("USE_CSV must be a boolean value (True/False).")
#     os.environ['USE_CSV'] = str(value).lower()
#     logger.info(f"USE_CSV set to {value}")



# def delete_cached_chain(tick, date):
#     """
#     Delete cached chain data for a specific ticker and date.
#     Args:
#         tick (str): Ticker symbol.
#         date (str): Date in 'YYYY-MM-DD' format.
#     """
#     from EventDriven.riskmanager.utils import PERSISTENT_CACHE
#     func = 'EventDriven.riskmanager.utils.populate_cache_with_chain'
#     key = (func, tick, date, None, 'print_url',False)
#     if key in PERSISTENT_CACHE:
#         del PERSISTENT_CACHE[key]
#         print(f"Deleted Chain cache for {tick} on {date}")
#     else:
#         print(f"No cached chain found for {tick} on {date}")

# def delete_cached_get_order(tick, date):
#     """
#     Delete cached order data for a specific ticker and date.
#     Args:
#         tick (str): Ticker symbol.
#         date (str): Date in 'YYYY-MM-DD' format.
#     """
#     from EventDriven.riskmanager.utils import PERSISTENT_CACHE
#     f ='EventDriven.riskmanager.base.OrderPicker.__get_order'
#     deleted = False
#     for key in PERSISTENT_CACHE.keys():
#         if key[0] == f and key[1][2][1] == tick and key[2]== date:
#             del PERSISTENT_CACHE[key]
#             print(f"Deleted cache for {key}")
#             deleted = True
#     if not deleted:
#         print(f"No cached order found for {tick} on {date}")


# def generate_trades_data(strategy_folder_name: str, date:str|datetime=None) -> tuple[pd.DataFrame, list[str]]:
#     """
#     Generate trades data from the strategy folder.
#     Args:
#         strategy_folder_name (str): Name of the strategy folder.
#     Returns:
#         tuple: A tuple containing:
#             - pd.DataFrame: DataFrame with trade details.
#             - list[str]: List of unique signal IDs."""
#     db = DatabaseAdapter()
#     if get_use_csv():
#         logger.info(f"Using CSV for trades data in {strategy_folder_name}")
#         if date:
#             logger.critical("Date parameter is ignored when USE_CSV is True.")
#         trades = pd.read_csv(f"{os.environ['ALGO_DIR']}/algo/strategies/{strategy_folder_name}/trades.csv")
#     else:
#         logger.info(f"Using database for trades data in {strategy_folder_name}")
#         if date:
#             logger.info(f"Filtering trades data for date: {date}")
#             trades = db.query_database(db='strategy_trades_signals',
#                         table_name= 'historical_signals',
#                         query= f"SELECT * FROM strategy_trades_signals.historical_signals WHERE RUN_DATE = '{pd.to_datetime(date).strftime('%Y-%m-%d')}'")
#             if trades.empty:
#                 logger.warning(f"No trades found for date: {date}. Falling back to all trades.")
#                 print(f"No trades found for date: {date}. Falling back to all trades.")
#                 # trades = db.query_database(db='strategy_trades_signals',
#                 #             table_name= strategy_folder_name,
#                 #             query= f"SELECT * FROM {strategy_folder_name}")
#         else:
#             trades = db.query_database(db='strategy_trades_signals',
#                         table_name= strategy_folder_name,
#                         query= f"SELECT * FROM {strategy_folder_name}")
#     consumption_trades = trades[trades['ACTION'].isin([Action.OPEN.value, Action.CLOSE.value, Action.HOLD.value])].copy()
#     consumption_trades=consumption_trades[['Ticker', 'Size', 'NEW_ENTRY_TIME', 'NEW_EXIT_TIME', 'SIGNAL_ID', 'ACTION']]
#     consumption_trades.rename(columns={
#         'NEW_ENTRY_TIME': 'EntryTime',
#         'NEW_EXIT_TIME': 'ExitTime',
#         'SIGNAL_ID': 'PT_BKTEST_SIG_ID'}, inplace=True)
#     consumption_trades['ExitTime'] = consumption_trades['ExitTime'].fillna(ny_now().strftime('%Y-%m-%d'))
#     consumption_trades['EntryTime'] = consumption_trades['EntryTime'].fillna(ny_now().strftime('%Y-%m-%d'))
#     consumption_trades['MAP_ORDER_SIGNAL_ID'] = consumption_trades.apply(lambda x: generate_signal_id(
#         x['Ticker'],
#         x['EntryTime'],
#         SignalTypes.LONG.value if x['Size'] > 0 else SignalTypes.SHORT.value
#     ), axis=1)
#     signals = consumption_trades.PT_BKTEST_SIG_ID.unique().tolist()
#     return consumption_trades, signals


# def delete_cached_chain_and_order(trades: pd.DataFrame):
#     """
#     Delete cached chain and order data for each trade in the DataFrame.
#     Args:
#         trades (pd.DataFrame): DataFrame containing trade details.
#     """
#     signals = trades.to_dict(orient='records')
#     for signal in signals:
#         date = change_to_last_busday(pd.to_datetime(signal['EntryTime']) + BDay(1), offset=-1).strftime('%Y-%m-%d')
#         delete_cached_chain(signal['Ticker'], date)
#         delete_cached_get_order(signal['Ticker'], date)

# def add_attr(attr_config, obj, skip_keys=[]):
#     """
#     Add attributes to an object based on a configuration dictionary.
#     attr_config: dict where keys are attribute names and values are the values to set.
#     obj: the object to which attributes will be added.
#     skip_keys: list of keys to skip in the attr_config.
#     """
#     for attr, value in attr_config.items():
#         if attr in skip_keys:
#             continue
#         assert hasattr(obj, attr), f"{obj.__class__.__name__} does not have `{attr}`"
#         setattr(obj, attr, value)



# def transfer_processed_data_to_cache(bkt:OptionSignalBacktest):
#     """
#     Transfer processed option data to the custom cache.
#     """
#     for name, data in bkt.risk_manager.processed_option_data.items():
#         if name not in get_custom_cache().keys(): ## TODO: Should transfer all. But data is inconsistent at times. Fix this
#             get_custom_cache()[name] = data
#             print(f"Transferred processed data for {name} to custom cache.")


# def setup_backtest_env(
#         trades: pd.DataFrame,
#         portfolio_config: dict,
#         rm_config: dict,
#         sizer_settings: dict,
#         config: dict,
#         weights: dict = {},
#         cash: int|float=20_000,
#         skip_keys_map: dict = {},
#         bkt_config: dict = {}
# ):
#     """
#     Set up the backtest environment with the given configuration and trades.

#     Args:
#         trades (pd.DataFrame): DataFrame containing trades data.
#         portfolio_config (dict): Configuration for the portfolio.
#         rm_config (dict): Configuration for the risk manager.
#         sizer_settings (dict): Settings for the sizer.
#         config (dict): General configuration for the backtest.
#         weights (dict, optional): Weights for the portfolio. Defaults to {}.
#         cash (int|float, optional): Initial cash amount. Defaults to 20_000.
#         skip_keys_map (dict, optional): Keys to skip in the configuration. Defaults to {}.
#         bkt_config (dict, optional): Additional configuration for the backtest. Defaults to {}.
#     Returns:
#         OptionSignalBacktest: An instance of the OptionSignalBacktest class with the configured environment.
#     """

#     ## Set up backtest environment.
#     ## NOTE: Need to delete processed_option_data & position_data in other to allow updating the time stamps
#     portfolio_config, rm_config, bkt_config = deepcopy(portfolio_config), deepcopy(rm_config), deepcopy(bkt_config)
#     set_timeseries_start(config['rm_series_start'])
#     set_timeseries_end(config['rm_series_end'])
#     set_skip_mysql_query(True) ## To avoid time lag in getting option Timeseries
#     set_use_temp_cache(True) ## To ensure we use the temp cache for this backtest.
    
#     for key in skip_keys_map:
#         assert key in ['rm_config', 'portfolio_config', 'bkt_config'], f"Invalid key in skip_keys_map: {key}. Expected keys are ['rm_config', 'portfolio_config', 'bkt_config']"

#     for key in ['rm_config', 'portfolio_config', 'bkt_config']: ## Creating a default
#         if key not in skip_keys_map:
#             skip_keys_map[key] = []

#     ## Apply weight haircut to weights
#     weights = {x: 
#                v * portfolio_config.get('weights_haircut', 1) for x,v in weights.items()}
#     if 'weights_haircut' not in portfolio_config:
#         logger.warning("weights_haircut not found in portfolio_config, using default value of 1.0")
    
#     ## Produce max_cash_map
#     cash_map = portfolio_config.pop('max_cash_map', {})
#     portfolio_config['max_contract_price'] = create_max_cash_map(
#         weights=weights,
#         cash=cash,
#         threshold_map=cash_map
#     )
    
#     ## Set up the portfolio & Backtest
#     bkt = OptionSignalBacktest(
#         trades=trades,
#         initial_capital=cash,
#         t_plus_n=portfolio_config.pop('t_plus_n'),
#         symbol_list=config['traded_symbols'],
#         finalize_trades = False)

#     ## Clear any existing processed option data and position data then upload new one
#     bkt.risk_manager.clear_caches()
#     bkt.risk_manager.clear_core_data_caches() ## Clear core data caches to ensure fresh data is used
#     bkt.risk_manager.append_option_data(data_pack=get_custom_cache())

#     ## Set Enabled Limits:
#     if 'limits_enabled' in rm_config:
#         for lmt in rm_config['limits_enabled']:
#             if lmt not in bkt.risk_manager.limits:
#                 raise ValueError(f"Limit {lmt} not found in risk manager limits.")
#             bkt.risk_manager.limits[lmt]=True

#     ## Set up the Sizer in RM
#     if 'sizer_type' in rm_config:
#         _type = rm_config.pop('sizer_type')
#         if _type == 'ZscoreRVolSizer':
#             rm_config['sizer'] = ZscoreRVolSizer(pm=bkt.portfolio,
#                                                  rm=bkt.risk_manager,
#                                                  **sizer_settings)
#         elif _type == 'DefaultSizer':
#             rm_config['sizer'] = DefaultSizer(pm=bkt.portfolio,
#                                               rm=bkt.risk_manager,
#                                               **sizer_settings)
#         else:
#             raise ValueError(f"Invalid sizer type: {_type}. Expected 'ZscoreRVolSizer' or 'DefaultSizer'.")
#     else:
#         logger.info("No sizer type specified in rm_config, using DefaultSizer in RiskManager as default.")

#     ## Add Attributes to the objects
#     add_attr(rm_config, 
#              bkt.risk_manager, 
#              skip_keys=['rm_series_start', 'rm_series_end'] + skip_keys_map['rm_config'])
#     add_attr(portfolio_config, bkt.portfolio, skip_keys=['initial_cash'] + skip_keys_map['portfolio_config'])
#     add_attr(bkt_config, bkt.portfolio, skip_keys=['initial_cash', 'weights_haircut'])  
#     bkt.logger = bkt_logger  # Set the logger for the backtest
    
#     return bkt


# def run_backtest(
#         trades: pd.DataFrame,
#         portfolio_config: dict,
#         rm_config: dict,
#         sizer_settings: dict,
#         config: dict,
#         T_0: str,
#         strat_name: str,
#         strateg_slug: str,
#         weights: dict = {},
#         cash: int|float=20_000,
#         skip_keys_map: dict = {},
#         bkt_config: dict = {}
# ):
#     """
#     Run the backtest with the given configuration and trades.
#     Args:
#         trades (pd.DataFrame): DataFrame containing trades data.
#         portfolio_config (dict): Configuration for the portfolio.
#         rm_config (dict): Configuration for the risk manager.
#         sizer_settings (dict): Settings for the sizer.
#         config (dict): General configuration for the backtest.
#         weights (dict, optional): Weights for the portfolio. Defaults to {}.
#         cash (int|float, optional): Initial cash amount. Defaults to 20_000.
#         skip_keys_map (dict, optional): Keys to skip in the configuration. Defaults to {}.
#         bkt_config (dict, optional): Additional configuration for the backtest. Defaults to {}.
#     Returns:
#         OptionSignalBacktest: An instance of the OptionSignalBacktest class with the configured environment.

#     Sequence:
#     1. Prepare the data for backtest. Which includes:
#         - Loading the trades data.
#         - Adjusting latest date to match the current date.
#     2. Set up the backtest environment with the given configuration and trades.
#     3. Run the backtest.
#     4. Transfer processed data to the custom cache for future use.
#     5. Run post test tasks, which involves deleting today's data from the cache. It will be re-added EOD with eod_task
#     6. Return the backtest object for further analysis or inspection.

#     """
    
#     bkt = setup_backtest_env(
#         trades=trades,
#         portfolio_config=portfolio_config,
#         rm_config=rm_config,
#         sizer_settings=sizer_settings,
#         config=config,
#         weights=weights,
#         cash=cash,
#         skip_keys_map=skip_keys_map,
#         bkt_config=bkt_config
#     )
#     try:
#         set_use_quotes(True) 
#         bkt.run()  # Run the backtest
#         set_use_quotes(False)
#     except KeyError:
#         logger.error("Backtest complete, key error coming from T+1 not available")
#         print("Backtest complete, key error coming from T+1 not available")
#     except Exception as e:
#         raise e
    
#     ## Intra day needs to use quotes for backtest, not EOD.
#     ## T_0 is the run date of the order search
#     # t1 = change_to_last_busday(pd.to_datetime(T_0) + BDay(config['t_plus_n']), offset=-1).strftime('%Y-%m-%d') 
#     closed_orders = get_close_orders(trades, strateg_slug)
#     open_orders = get_open_orders(bkt, T_0)
#     actions=get_actions(bkt, T_0)

#     ## Save limits
#     save_limits_from_backtester(bkt, T_0)

#     # Transfer processed data to the custom cacheOy
#     print("Backtest completed. Transferring processed data to cache...")
#     transfer_processed_data_to_cache(bkt)  
#     print("Processed data transferred to cache. Running post test tasks...")
    
#     return {
#         'BACKTESTER': bkt,
#         'CLOSED_ORDERS': closed_orders,
#         'OPEN_ORDERS': open_orders,
#         'ACTIONS': actions,
#     }


# def get_close_orders_meta(consumption_trades:pd.DataFrame, strat_name:str) -> List[dict]:
#     """
#     Get the close orders meta data for the given consumption trades DataFrame.
#     Args:
#         consumption_trades (pd.DataFrame): DataFrame containing trades data with columns 'ACTION', 'PT_BKTEST_SIG_ID', etc.
#         strat_name (str): Name of the strategy to filter the trades.
#     Returns:
#         List[dict]: A list of dictionaries containing the close orders meta data.
#     """
#     ## Get closed trades signal_id
#     closed_signals = consumption_trades[consumption_trades['ACTION'] == Action.CLOSE.value]['PT_BKTEST_SIG_ID'].unique().tolist()

#     ## Get fills for the closed trades
#     fills = get_fills(closed_signals, strat_name)

#     ## Get only open fills.
#     open_fills = fills[fills['position_effect'] == Action.OPEN.value].copy()

#     ## Create order meta:
#     close_today_signals=[]
#     for i, row in open_fills.iterrows():
#         signal_id = row['signal_id']
#         signal_meta = parse_signal_id(signal_id)
#         meta= dict(
#         signal_id=signal_id,    
#         position_id = row['position_id'],
#         submitted_timestamp=ny_now(),
#         ticker=signal_meta['ticker'],
#         direction=SignalTypes.LONG.value if row['direction']>0 else SignalTypes.SHORT.value,
#         order_type='GTC',
#         quantity=row['quantity'],
#         limit_price=get_position_price(row['position_id'], 
#                                     ny_now().strftime('%Y-%m-%d')),
#         fill_ts=None,
#         fill_price=None,
#         filled_qty=None,
#         position_effect='CLOSE',
#         strategy_name='LongBBandsTrend_SL'
#         )
#         close_today_signals.append(meta)
#     return close_today_signals

# def get_close_orders(consumption_trades:pd.DataFrame, strat_name:str) -> dict:
#     """
#     Get the close orders for the given close today order meta data.
#     Args:
#         close_today_order_meta (List[dict]): List of dictionaries containing the close orders meta data.
#         Gotten from get_close_orders_meta function.
#     Returns:
#         dict: A dictionary where keys are tickers and values are dictionaries containing order details.
#     """
    
#     close_order_list={}
#     consumption_trades = consumption_trades.copy()
#     consumption_trades['signal_id'] = consumption_trades['PT_BKTEST_SIG_ID']

#     ## Get closed trades signal_id
#     closed_signals = consumption_trades[consumption_trades['ACTION'] == Action.CLOSE.value]['PT_BKTEST_SIG_ID'].unique().tolist()
#     if not closed_signals:
#         print("No closed trades found.")
#         return close_order_list

#     ## Get fills for the closed trades
#     fills = get_fills(closed_signals, strat_name)

#     ## Get only open fills.
#     open_fills = fills[fills['position_effect'] == Action.OPEN.value].copy()

#     for idx, sig in open_fills.iterrows():
#         map_signal_id = consumption_trades[consumption_trades['signal_id'] == sig['signal_id']]\
#             ['PT_BKTEST_SIG_ID'].unique()[0]
#         size=consumption_trades[consumption_trades['PT_BKTEST_SIG_ID'] == map_signal_id]['Size'].values[0]
#         meta= parse_signal_id(sig['signal_id'])
#         pairs=parse_position_id(sig['trade_id'])[1]
#         order=dict(
#             result='SUCCESSFUL',
#             data=dict(
#                 trade_id=sig['trade_id'],
#                 close=get_position_price(sig['trade_id'], 
#                                     ny_now().strftime('%Y-%m-%d')),
#                 long=[x[1] for x in pairs if x[0] == 'L'],
#                 short=[x[1] for x in pairs if x[0] == 'S'],
#                 quantity=sig['quantity'],
#             ),
#             signal_id=sig['signal_id'],
#             direction=SignalTypes.LONG.value if size > 0 else SignalTypes.SHORT.value,
#             map_signal_id=map_signal_id,
#         )
#         close_order_list[meta['ticker']] = order

#     return close_order_list



# def get_open_orders(bkt:OptionSignalBacktest, t1: str) -> dict:
#     """
#     Extract orders from the backtest object for a specific date.
#     Args:
#         bkt (OptionSignalBacktest): The backtest object containing the risk manager.
#         t1 (str): The date for which to extract orders, formatted as 'YYYY-MM-DD'.
#     Returns:
#         dict: A dictionary where keys are tickers and values are dictionaries containing order details.
#     """
#     if not isinstance(t1, str):
#         if isinstance(t1, datetime):
#             t1 = t1.strftime('%Y-%m-%d')
#         else:
#             raise ValueError("t1 must be a string in 'YYYY-MM-DD' format or a datetime object.")

#     orders = bkt.risk_manager.order_cache.get(t1, {})
#     unadjusted_trades = bkt.unadjusted_trades
#     for order in orders.values():
#         opt_signal_id = order['signal_id']
#         order['map_signal_id'] = unadjusted_trades[unadjusted_trades['signal_id'] == opt_signal_id]['PT_BKTEST_SIG_ID'].unique()[0]
#     return orders


# def get_open_orders_meta(bkt:OptionSignalBacktest, T_1: str) -> List[dict]:
#     """
#     Get open orders metadata 
#     for the backtest on a specific date.
#     Args:
#         bkt (OptionSignalBacktest): The backtest object containing the risk manager.
#         T_1 (str): The date for which to extract open orders, formatted as 'YYYY-MM-DD'.
#     Returns:
#     """
#     save_to_df = []
#     unadjusted_trades = bkt.unadjusted_trades
#     _open_orders = get_open_orders(bkt, T_1)

#     ## Open Orders
#     for tick, order in _open_orders.items():
#         meta=dict(
#             signal_id=unadjusted_trades[unadjusted_trades['signal_id'] == order['signal_id']]['PT_BKTEST_SIG_ID'].unique()[0],
#             position_id=order['data']['trade_id'],
#             submitted_timestamp=ny_now(),
#             ticker=tick,
#             direction=order['direction'],
#             order_type='GTC',
#             quantity=order['data']['quantity'],
#             limit_price=order['data']['close'],
#             fill_ts=None,
#             fill_price=None,
#             filled_qty=None,
#             position_effect='OPEN',
#             strategy_name='LongBBandsTrend_SL'
#         )
#         save_to_df.append(meta)
#     return save_to_df

# def get_actions(bkt:OptionSignalBacktest, T_1: str) -> dict:
#     """
#     Get actions from the risk manager for a specific date.
#     Args:
#         bkt (OptionSignalBacktest): The backtest object containing the risk manager.
#         T_1 (str): The date for which to extract actions, formatted as 'YYYY-MM-DD'.
#     Returns:
#         dict: A dictionary where keys are tickers and values are dictionaries containing action details.
#     """
#     actions = [v for x, v in bkt.risk_manager._actions.items() if x.strftime('%Y-%m-%d') == T_1]
#     actions = actions[0] if actions else {}
#     return actions

# ## Save Utils
# def save_orders_to_database(open_orders_meta: List[dict], 
#                             close_order_meta: List[dict]) -> pd.DataFrame:
#     """
#     Save open and exit orders to the database.
#     Args:
#         open_orders_meta (List[dict]): List of dictionaries containing open orders metadata.
#         gotten from get_open_orders_meta function.
#         close_order_meta (List[dict]): List of dictionaries containing close orders metadata.
#         gotten from get_close_orders_meta function.
#     Returns:
#         pd.DataFrame: DataFrame containing the saved orders metadata.
        
#     """
#     save_to_df=open_orders_meta+close_order_meta
#     db=DatabaseAdapter()
#     db.save_to_database(
#         data=pd.DataFrame(save_to_df),
#         table_name='orders',
#         db='portfolio_data',
#         filter_data=False,
#         _raise=True)
#     return pd.DataFrame(save_to_df)

# def _save_to_database_helper(
#         consumption_trades: pd.DataFrame,
#         strat_name: str,
#         bkt: OptionSignalBacktest,
#         T_1: str
# ):
#     """
#     Helper function to save orders to the database.
#     Args:
#         consumption_trades (pd.DataFrame): DataFrame containing trades data.
#         strat_name (str): Name of the strategy.
#         bkt (OptionSignalBacktest): The backtest object.
#         T_1 (str): The date for which to extract orders, formatted as 'YYYY-MM-DD'.
#     Returns:
#         None
#     """
#     close_today_order_meta = get_close_orders_meta(consumption_trades, strat_name)
#     open_orders_meta = get_open_orders_meta(bkt, T_1)
    
#     if not open_orders_meta and not close_today_order_meta:
#         print("No open or close orders to save.")
#         return
#     print(f"Saving {len(open_orders_meta)} open orders and {len(close_today_order_meta)} close orders to database.")
#     if not open_orders_meta:
#         print("No open orders to save.")
#     if not close_today_order_meta:
#         print("No close orders to save.")
#     # Save to database
#     return save_orders_to_database(open_orders_meta, close_today_order_meta)


# def make_fill_meta(
#         signal_id: str,
#         strategy_name: str,
#         position_id: str,
#         fill_price: float,
#         fill_timestamp: datetime,
#         quantity: int|float,
#         position_effect: str,
#         direction: str,
#         ticker: str,
#         order_type: str,
#         limit_price: float,
#         filled_qty: int|float
# ):
#     """
#     Create a fill meta dictionary for the order.
#     """
#     assert position_effect in ['OPEN', 'CLOSE'], "position_effect must be either 'OPEN' or 'CLOSE'"
#     assert direction in ['LONG', 'SHORT'], "direction must be either 'LONG' or 'SHORT'"
#     return {
#         'signal_id': signal_id,
#         'strategy_name': strategy_name,
#         'position_id': position_id,
#         'fill_price': fill_price,
#         'fill_timestamp': fill_timestamp,
#         'quantity': quantity,
#         'position_effect': position_effect,
#         'direction': direction,
#         'ticker': ticker,
#         'order_type': order_type,
#         'limit_price': limit_price,
#         'filled_qty': filled_qty
#     }

# def save_fills_to_database(fills_list_meta: List[dict]):
#     """
#     Save fills to the database.
#     """
#     db=DatabaseAdapter()
#     db.save_to_database(
#         data=pd.DataFrame(fills_list_meta),
#         table_name='fills',
#         db='portfolio_data',
#         filter_data=False,
#         _raise=True
#     )

#     return True



# ########## NEW FUNCTIONS ##########
# from ..positions.loaders.configs import get_configs
# from EventDriven.riskmanager._order_validator import build_inputs_with_config, OrderInputs, OrderSchema
# from EventDriven.riskmanager.market_data import get_timeseries_obj, OPTION_TIMESERIES_START_DATE
# CONFIGS = get_configs()

# def load_position_actions(slug:str, test:bool = False) -> pd.DataFrame:
#     trades = generate_trades_data(slug)[0]
#     if not test:
#         return trades[trades.ACTION == 'OPEN'],trades[trades.ACTION == 'CLOSE']
#     else:
#         print("WARNING: TEST MODE. USING INCORRECT INFORMATION IN load_position_actions function")
#         close = trades.copy()
#         close['ACTION'] = 'CLOSE'
#         close['EntryTime'] = datetime.now() - BDay(1)
#         close['ExitTime'] = datetime.now() - BDay(1)
#         opens =  trades.copy()
#         opens['ACTION'] = 'OPEN'
#         opens['EntryTime'] = datetime.now() - BDay(1)
#         opens['ExitTime'] = datetime.now() - BDay(1)
#         return opens, close

# def load_timeseries_for_trades(sym_list: List[str], force=False) -> None:
#     timeseries = get_timeseries_obj()
#     for sym in sym_list:
#         load_bool = sym not in timeseries.spot \
#             or sym not in timeseries.chain_spot \
#             or sym not in timeseries.dividends \
#             or force
        
#         if load_bool:
#             timeseries.load_timeseries(sym, OPTION_TIMESERIES_START_DATE, datetime.now(), force=force)


# def get_max_cash_for_symbol(sym: str, slug: str) -> float:
#     return CONFIGS.get_configs(slug).cash_map[sym]


# def build_inputs(slug: str, 
#                  row: pd.Series, 
#                  tick: str) -> tuple[OrderSchema, OrderInputs]:
#     """
#     Builds the inputs for the order selection engine based on the strategy slug and trade row.
#     Args:
#         slug (str): The strategy slug.
#         row (pd.Series): The trade row containing trade details. Expected keys: ['PT_BKTEST_SIG_ID', 'Size', 'EntryTime']
#         tick (str): The stock ticker symbol.
#         date (str|datetime): The date for the order selection.

#     Returns:
#         Tuple[OrderSchema, OrderInputs]: A tuple containing the OrderSchema and OrderInputs dataclass.
#     """

#     ## Build Config for the strategy slug
#     config = CONFIGS.get_configs(slug)

#     ## This is the max price for the order search engine
#     max_close = get_max_cash_for_symbol(tick, slug)