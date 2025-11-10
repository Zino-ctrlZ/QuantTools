raise DeprecationWarning("This module is deprecated. Use algo.strategies.utils instead.")
# import os
# import yaml
# import importlib
# from dateutil.relativedelta import relativedelta
# from pathlib import Path
# from datetime import datetime, timedelta
# import pandas as pd
# from dbase.database.SQLHelpers import (
#     get_engine,
#     list_tables_from_db,
#     create_table_from_schema,
#     dynamic_batch_update,
#     DatabaseAdapter
# )
# from dbase.DataAPI.ThetaData import (
#     retrieve_quote,
#     retrieve_quote_rt
# )
# from trade.assets.Calculate import Calculate
# from trade.helpers.helper import (
#     parse_option_tick,
#     binomial_implied_vol,
#     CustomCache
# )
# from EventDriven.riskmanager.market_data import (
#     OPTION_TIMESERIES_START_DATE,
#     MarketTimeseries
# )
# from EventDriven.riskmanager.utils import parse_position_id
# from algo.strategies.enums import Action


# ## 1). Setup data source. CustomCache location
# DATA_LOCATION = Path(f"{os.environ['GEN_CACHE_PATH']}")
# CUSTOM_CACHE = None

# ## 2). Object for live data. Reloads after 3 mins
# LIVE_TIMESERIES = None
# def get_live_timeseries_obj() -> MarketTimeseries:
#     """
#     Get a MarketTimeseries object that refreshes data every 3 minutes.
#     """
#     global LIVE_TIMESERIES
    
#     ## Start is 2 weeks from today
#     end = datetime.now()
#     start = end - relativedelta(weeks=2)
#     if LIVE_TIMESERIES is None:
#         LIVE_TIMESERIES = MarketTimeseries(_refresh_delta=timedelta(minutes=3),
#                                           _start=start.strftime('%Y-%m-%d'),
#                                           _end=end.strftime('%Y-%m-%d'))

#     ## No need for refresh here. It is handled in the class during at_index call
#     return LIVE_TIMESERIES

# def get_custom_cache(loc=DATA_LOCATION):
#     """
#     Get a CustomCache instance for storing trading data.
#     This function initializes a CustomCache with a specified location and settings.
#     """
#     global CUSTOM_CACHE
#     if CUSTOM_CACHE is None:
#         CUSTOM_CACHE = CustomCache(
#             loc,
#             fname="bot_prod_data",
#             expiry_days=500,
#             clear_on_exit=False
#         )
#     return CUSTOM_CACHE

# def get_option_price_theta_data(opttick:str,
#                                 as_of:str|datetime) -> float|None:
#     """
#     Retrieve option price data from ThetaData
#     """
#     as_of = pd.to_datetime(as_of)
#     meta = parse_option_tick(opttick)
#     data = retrieve_quote(
#         symbol=meta['ticker'],
#         start_date=as_of - timedelta(days=7),
#         end_date=as_of + timedelta(days=1),
#         strike=meta['strike'],
#         right= meta['put_call'],
#         exp=meta['exp_date'],
#         print_url = False,
#         interval='1d')
#     if data is None or data.empty:
#         return None
    
#     if as_of.date() not in data.index.date:
#         raise ValueError(f"Data for {as_of.date()} not found in the retrieved data.")
#     return data.loc[data.index.date == as_of.date()]['Midpoint'].values[0]


# def get_option_price(_id:str, date:str, force = False) -> float|None:
#     """
#     Get the position price for a given position ID and date.

#     Args:
#     force (bool): If True, force refresh the price from ThetaData.
#     _id (str): Position ID.
#     date (str): Date in 'YYYY-MM-DD' format.
#     Returns:
#     float|None: The position price if available, otherwise None.
#     """
#     if _id not in get_custom_cache() or force: ## If not in cache or force refresh, get from ThetaData
#         return get_option_price_theta_data(_id, date)
#     data = get_custom_cache()[_id]
#     if date not in data.index:
#         print(f"No data for {_id} on {date}")
#         return get_option_price_theta_data(_id, date)
#     return data.loc[date, 'Midpoint']


# def get_option_realtime_quote(_id:str) -> float|None:
#     """
#     Get the position quote for a given position ID and date.
#     """
#     meta = parse_option_tick(_id)
#     return retrieve_quote_rt(
#     symbol=meta['ticker'],
#     exp=meta['exp_date'],
#     right=meta['put_call'],
#     strike=meta['strike'],
#     )['Midpoint'][0]

# def get_position_price(_id:str, date:str, force = False) -> float|None:
#     """
#     Get the position price for a given position ID and date.
#     Args:
#     force (bool): If True, force refresh the price from ThetaData.
#     _id (str): Position ID.
#     date (str): Date in 'YYYY-MM-DD' format.

#     Returns:
#     float|None: The position price if available, otherwise None.
#     """
#     opt_ids_list = parse_position_id(_id)[1]
#     prices = []
#     for leg, opt_id in opt_ids_list:

#         price = get_option_price(opt_id, date, force=force)
#         if price is not None:
#             prices.append(price if leg == 'L' else -price)
#         else:
#             print(f"No price found for {opt_id} on {date}")
#     if not prices:
#         print(f"No prices found for {_id} on {date}")
#         return None
#     return sum(prices) 


# def get_position_realtime_quote(_id:str) -> float|None:
#     """
#     Get the position quote for a given position ID.
#     Args:
#         _id (str): Position ID.
#     Returns:
#         float|None: The position quote if available, otherwise None.
#     """
#     opt_ids_list = parse_position_id(_id)[1]
#     quotes = []
#     for leg, opt_id in opt_ids_list:
#         quote = get_option_realtime_quote(opt_id)
#         if quote is not None:
#             quotes.append(quote if leg == 'L' else -quote)
#         else:
#             print(f"No quote found for {opt_id}")
#     if not quotes:
#         print(f"No quotes found for {_id}")
#         return None
#     return sum(quotes)



# def live_calculate_option_delta(opttick:str, date:str|datetime) -> float:
#     """
#     Calculate the delta of an option using the binomial model. This is a live calculation that fetches the necessary data.
    
#     """
#     tick, option_type, exp, strike = parse_option_tick(opttick).values()
#     option_price = get_option_price(opttick, date, force=True)
#     timeseries = get_live_timeseries_obj()
#     if tick not in timeseries.spot:
#         timeseries.load_timeseries(tick, OPTION_TIMESERIES_START_DATE, datetime.now(), force=True)

#     at_index = timeseries.get_at_index(tick, date)
#     s = at_index.chain_spot.close
#     y = at_index.dividends
#     r = at_index.rates.annualized
#     vol = binomial_implied_vol(
#         price=option_price,
#         S=s,
#         K=strike,
#         r=r,
#         exp_date=exp,
#         option_type=option_type.lower(),
#         pricing_date=date,
#         dividend_yield=y)
    


#     return Calculate.delta(
#         S=s,
#         K=strike,
#         r=r,
#         sigma=vol,
#         start=date,
#         flag=option_type.lower(),
#         exp=exp,
#         y=y,
#         model='binomial'
#     )

# def live_calculate_position_delta(position_id: str, date:str|datetime) -> float:
#     ids =parse_position_id(position_id)[1]
#     pos_delta = 0
#     for side, opttick in ids:
#         if side.upper() == 'L':
#             sign = 1
#         elif side.upper() == 'S':
#             sign = -1
#         else:
#             raise ValueError("Invalid side in position ID. Must be 'LONG' or 'SHORT'.")
#         delta = live_calculate_option_delta(opttick, date)
#         pos_delta += (sign * delta)
#     return pos_delta

# def create_max_cash_map(weights: dict, 
#                         cash: int|float, 
#                         threshold_map: dict) -> dict:
#     """
#     weights: dict of symbol -> weight (numeric)
#     cash: scalar. This is initial cash for the portfolio. Not total cash as at a time.
#     threshold_map: dict where keys are numeric thresholds and value is the assigned value,
#                    plus an optional 'else' key for fallback.
#                    Example: {500:4, 300:3, 200:2, 100:1, 'else': 0.5}
#     Returns: dict symbol -> assigned max_cash
#     """
#     # Extract numeric thresholds and sort descending
#     numeric_thresholds = sorted(
#         (k for k in threshold_map if isinstance(k, (int, float))),
#         reverse=True
#     )
#     fallback = threshold_map.get('else')
#     result = {}
#     for s, w in weights.items():
#         amount = w * cash
#         # find first threshold that amount exceeds
#         assigned = None
#         for thresh in numeric_thresholds:
#             if amount > thresh:
#                 assigned = threshold_map[thresh]
#                 break
#         if assigned is None:
#             assigned = fallback
#         result[s] = assigned
#     return result

# def load_eod_tasks() -> list:
#     """
#     Loads functions for eod task. Note any scheduled task must work with taking no keywords.
#     ENSURE IT IS A FUNCTION THAT TAKES NO ARGUMENTS.
#     """
#     with open(f"{os.environ['ALGO_DIR']}/algo/strategies/eod_tasks.yaml") as f:
#         tasks = yaml.safe_load(f)['tasks']

#     run_tasks = []
#     for task in tasks:
#         module = importlib.import_module(task['module'])
#         func = getattr(module, task['name'], None)
#         enabled = task['enabled']
#         if func is not None and enabled:
#             run_tasks.append(func)
#     return run_tasks


# def get_prod_last_run(strat_name:str) -> pd.DataFrame:
#     """
#     Retrieve the last run information for all strategies from the prod_last_run table.
    
#     Returns
#     -------
#     pd.DataFrame
#         The last run information for all strategies.
#     """
#     db = DatabaseAdapter()
#     data = db.query_database(
#         db = 'strategy_trades_signals',
#         table_name= 'prod_last_run',
#         query= "SELECT * FROM strategy_trades_signals.prod_last_run WHERE strat_name = '%s'" % strat_name
#     )
#     return data.run_date.max()

# def update_prod_last_run(strat_name:str, run_date: str) -> None:
#     """
#     Update a new entry to the prod_last_run table.
    
#     Parameters
#     ----------
#     strat_name : str
#         The name of the strategy.
#     run_date : str
#         The date and time of the run in 'YYYY-MM-DD HH:MM:SS' format.
#     """

#     dynamic_batch_update(
#         db = 'strategy_trades_signals',
#         table_name = 'prod_last_run',
#         update_values= {
#             'run_date': pd.to_datetime(run_date).date(),
#         },
#         condition={
#             'strat_name': strat_name
#         },
#     )
    
# def add_strat_to_prod_last_run(strat_name:str, run_date: str) -> None:
#     """
#     Add a new strategy to the prod_last_run table if it does not already exist.
    
#     Parameters
#     ----------
#     strat_name : str
#         The name of the strategy.
#     run_date : str
#         The date and time of the run in 'YYYY-MM-DD HH:MM:SS' format.
#     """
#     db = DatabaseAdapter()
#     existing = db.query_database(
#         db = 'strategy_trades_signals',
#         table_name= 'prod_last_run',
#         query= "SELECT * FROM strategy_trades_signals.prod_last_run WHERE strat_name = '%s'" % strat_name
#     )
#     if existing.empty:
#         df = pd.DataFrame({
#             'strat_name': [strat_name],
#             'run_date': [run_date]
#         })
#         db.save_to_database(
#             db = 'strategy_trades_signals',
#             table_name = 'prod_last_run',
#             data = df,
#         )


# def create_strategy_signals_table(strategy_slug: str) -> None:

#     """
#     Create a table for the specified strategy if it does not already exist.
    
#     Parameters
#     ----------
#     strategy_slug : str
#         The slug of the strategy for which to create the table.
#     """
#     actions = [action.value for action in Action]
#     engine = get_engine('strategy_trades_signals')  ## Location db
#     tables = list_tables_from_db('strategy_trades_signals') ## Table in location db

#     # Check if the table already exists
#     if strategy_slug not in tables:
#         print(f"Creating table for strategy {strategy_slug}...")
#         create_table_from_schema(
#             engine,
#             {
#                 'table_name': strategy_slug,
#                 'columns':
#                 [
#                     {'name': "Ticker", 'type': "String", 'length': 50, 'nullable': False},
#                     {'name': 'Size', 'type': 'Integer', 'nullable': False},
#                     {'name': 'SIGNAL_ORIGINAL_ENTRY_TIME', 'type': 'DateTime', 'nullable': False},
#                     {'name': 'SIGNAL_ORIGINAL_EXIT_TIME', 'type': 'DateTime', 'nullable': False},
#                     {'name': 'SIGNAL_ID', 'type': 'String', 'length': 50, 'nullable': False},
#                     {'name': 'OPEN_TODAY', 'type': 'Boolean', 'nullable': False},
#                     {'name': 'CLOSE_TODAY', 'type': 'Boolean', 'nullable': False},
#                     {'name': 'POSITION_PREV_OPENED', 'type': 'Boolean', 'nullable': False},
#                     {'name': 'POSITION_ACTIVE', 'type': 'Boolean', 'nullable': False},
#                     {'name': 'POSITION_CLOSED', 'type': 'Boolean', 'nullable': False},
#                     {'name': 'SIGNAL_CLOSED', 'type': 'Boolean', 'nullable': False},
#                     {'name': 'ACTION', 'type': 'Enum', 'values': actions, 'nullable': False},
#                     {'name': 'RATIONALE', 'type': 'String', 'length': 255, 'nullable': True},
#                     {'name': 'NEW_ENTRY_TIME', 'type': 'DateTime', 'nullable': False},
#                     {'name': 'NEW_EXIT_TIME', 'type': 'DateTime', 'nullable': False},
#                 ]
#             }
#         )

#     else:
#         print(f"Table for strategy {strategy_slug} already exists. Skipping creation.")
