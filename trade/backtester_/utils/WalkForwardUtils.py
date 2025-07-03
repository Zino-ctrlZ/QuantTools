import os
import sys
from trade.backtester_.backtester_ import PTBacktester, PTDataset
from trade.assets.Stock import Stock
import pandas as pd
from pandas.tseries.offsets import BDay
from datetime import datetime
import yfinance as yf
from trade.backtester_.Universe import universe
UNIVERSE = universe



def create_datasate(stocks: list, start: str,interval: str, engine: str = 'yf', timewidth = None, timeframe = None, end: str = datetime.today(), return_object = False ):
    dataset = []
    raw_dataset = {}
    data_range = pd.date_range(start, end, freq = 'B')
    if engine.lower() == 'yf':
        from datetime import datetime
        for stock in stocks:
            data2 = yf.download(stock, start = start, end = end, interval=interval, progress = False)
            if  pd.isna(data2.Open.max()):
                pass
            else:
                raw_dataset[stock] = data2
                dataset.append(PTDataset(stock, data2))
    else:
        for stk in stocks:
            stock = Stock(stk)
            data = stock.spot(ts = True, ts_start = '2018-01-01', ts_timeframe=tmframe)
            data.rename(columns = {x:x.capitalize() for x in data.columns}, inplace= True)
            data['Timestamp'] = pd.to_datetime(data['Timestamp'], format = '%Y-%m-%d')
            data2 = data.set_index('Timestamp')
            data2 = data2.asfreq('W', method = 'ffill')
            data2 = data2.fillna(0)
            data2['Next_Day_Open'] = data2.Open.shift(-1)
            data2['EMA'] = ta.ma('ema', data2.Close, length = 21).fillna(0)
            dataset.append(PTDataset(stk, data2))
            raw_dataset[stock] = data2
    return dataset if return_object else raw_dataset

def prev_monday(date):
    date = pd.to_datetime(date)
    day_of_week_ = date.day_of_week
    date = date.replace(day = date.day - day_of_week_)
    return date






def annualize_net_profit(net_profit, initial_investment, days, value = True):
    annualized_profit = ((1 + net_profit / initial_investment) ** (260 / days)) - 1
    return annualized_profit * initial_investment if value else annualized_profit *100



def split_window(
        stocks,
        strategy,
        data_end,
        data_length_str, 
        warmup_bars,
        lookback_bars=28*1440,
        validation_bars=7*1440,
        anchored = False,
        interval = '1d'):
    
    validation_run = 1
    data_start = eval(f'datetime.today()-relativedelta({data_length_str})')
    tester = ['^GSPC']
    data_bank = create_datasate(tester, data_start, interval,end = data_end , return_object=False)
    data_dict = create_datasate(stocks, data_start, interval,end = data_end , return_object=False)
    data_full = data_bank[tester[0]]
    split_data = {}
    split_window = {}
    train_window = {}
    test_window = {}
    test_data = {}
    train_data = {}
    for i in range(lookback_bars+warmup_bars, len(data_full)-validation_bars, validation_bars):
        s = i -lookback_bars - warmup_bars if not anchored else 0
        length_filter = 300
        sample_datas = []
        validation_datas = []

        pass_val_names = []
        ## I NEED TO CREATE A LIST OF SAMPLE DATA PTDATASET 
        for name, data in data_bank.items():
            off_start_train = pd.to_datetime(data_full.iloc[s+warmup_bars].name)
            start = pd.to_datetime(data_full.iloc[s].name).strftime('%Y-%m-%d')
            end = pd.to_datetime(data_full.iloc[i].name ).strftime('%Y-%m-%d')
            
            for stk in stocks:
                temp = data_dict[stk]
                temp = temp[(temp.index >= start) & (temp.index <= end)]
                # if len(temp) <i-(i- lookback_bars-warmup_bars)+1:
                if len(temp) <= length_filter:
                    pass_val_names.append(stk)
                    continue
                else:
                    sample_datas.append(PTDataset(stk, temp))
        ## GET VALIDATION DATA
                    valStart =  pd.to_datetime(data_full.iloc[i-warmup_bars].name).strftime('%Y-%m-%d')
                    valEnd = pd.to_datetime(data_full.iloc[i+validation_bars].name ).strftime('%Y-%m-%d')
                    off_start = pd.to_datetime(data_full.iloc[i].name ) - BDay(1)
                    temp = data_dict[stk]
                    temp = temp[(temp.index >= valStart) & (temp.index <= valEnd)]
                    validation_datas.append(PTDataset(stk, temp))

        train_window[str(validation_run)] = {'off_start': off_start_train, 'Start': start, 'End': end}
        test_window[str(validation_run)] = {'off_start': off_start,'Start': valStart, 'End': valEnd}
        train_data[str(validation_run)] = sample_datas
        test_data[str(validation_run)] = validation_datas
        validation_run += 1

    split_window['train_window'] = train_window
    split_window['test_window'] = test_window
    split_data['train_window'] = train_data
    split_data['test_window'] = test_data
    return split_window, split_data


def position_train(
        strategy,
        sample_datas,
        optimize_params,
        off_start,
        optimize_str = 'Return [%]', 
        cash=1_000, 
        commission=0.002,
        baseRun = True,
        printHeaders = False):
    
    packaged_data = {}
    # Carry out training 
    strategy.prev_open_positions_dict = None
    strategy.official_start = off_start
    strategy.open_positions = []
    strategy.is_test_backtest = False
    strategy.is_train_backtest = True
    bt_training = PTBacktester(sample_datas, strategy, cash=cash, commission=commission)
    stats = bt_training.run()
    agg1 = bt_training.aggregate()
    print('Pre optimize CAGR: ', agg1['CAGR [%]'], ' Return', agg1['Return [%]']) if printHeaders else None
    if not baseRun:
        # Optimize training data & pick best parameters

        optimized = bt_training.position_optimize(optimize_params, maximize = optimize_str)
        strategy_settings = optimized.to_dict('index')
        bt_training = PTBacktester(sample_datas, strategy, cash=cash, commission=commission, strategy_settings= strategy_settings)
        bt_training.run()

        packaged_data['strategy_settings'] = strategy_settings
        packaged_data['agg'] = bt_training.aggregate()
        packaged_data['stats_by_tick'] = stats
        print('Post optimize CAGR: ', packaged_data['agg']['CAGR [%]'], ' Return', packaged_data['agg']['Return [%]']) if printHeaders else None
    
    strategy.reset_variables()
    return packaged_data

def cross_train(
        strategy,
        sample_datas,
        optimize_params,
        off_start,
        optimize_list = ['rtrn'], 
        cash=1_000, 
        commission=0.002,
        baseRun = True,
        printHeaders = False):
    
    packaged_data = {}
    # Carry out training 
    strategy.prev_open_positions_dict = None
    strategy.official_start = off_start
    strategy.open_positions = []
    param_dict = {}
    strategy.is_test_backtest = False
    strategy.is_train_backtest = True
    bt_training = PTBacktester(sample_datas, strategy, cash=cash, commission=commission)
    stats = bt_training.run()
    agg1 = bt_training.aggregate()
    print('Pre optimize CAGR: ', agg1['CAGR [%]'], ' Return', agg1['Return [%]']) if printHeaders else None
    if not baseRun:

        optimized = bt_training.optimize(optimize_params, optimize_list)
        optimized.sort_values(optimize_list[0], ascending = False, inplace = True)
        optimized_page = optimized.head(1)
        for attr in optimize_params.keys():
            param_dict[attr] = optimized_page[attr].values[0]
    

        packaged_data['agg'] = bt_training.aggregate()
        packaged_data['stats_by_tick'] = stats
        packaged_data['strategy_settings'] = param_dict 
        print('Post optimize CAGR: ', packaged_data['agg']['CAGR [%]'], ' Return', packaged_data['agg']['Return [%]']) if printHeaders else None
    
    strategy.reset_variables()
    return packaged_data



def position_test(
        strategy,
        off_start,
        strategy_settings,
        validation_datas, 
        cash, 
        commission,
        baseRun = False,
        anchored = False,
        plot_positions = False):

    """
    This function carries out a test run with per position method. 

    Parameters:
    _____________

    strategy: backtesting.backtesting.Strategy object
    off_start (pd.Timestamp): Official Start date. This is assuming WFO would be a class in this strategy
    strategy_settings (dict): Params as a dict to optimize by. In per ticker format
    validation_data (List[PTDataset]): list of PTDataset
    cash (int, float, dict): Cash
    commission(float): Commission
    baseRun (bool): This is assuming there will be no optimizing if True
    anchored (bool): True to not move validation period forward, false to move forward
    plot_position (bool): True to plot each positions charts


    Returns:
    __________

    dict: 
        agg: Aggregate data from PTBacktester
        stats_by_tick: Stats by Ticker from backtesting.py
    
    
    """
    
    #Run Validation backtest
    packaged_data = {}
    strategy.official_start =off_start
    strategy.is_train_backtest = False
    strategy.open_positions = []
    strategy.prev_open_positions_dict = {} # pre_open_positions_dict has be an empty dict in the first run because there are no previous positions
    if baseRun:
        bt_validation = PTBacktester(validation_datas, strategy, cash=cash, commission=commission)
    else:
        bt_validation = PTBacktester(validation_datas, strategy, cash=cash, commission=commission, strategy_settings = strategy_settings)
    stats_validation = bt_validation.run()
    agg_validation = bt_validation.aggregate()
    packaged_data['agg'] = agg_validation
    packaged_data['stats_by_tick'] = stats_validation
    #Plot each validation run
    if plot_positions:
        for d in bt_validation.datasets:
            name = d.name
            bt_validation.plot_position(name, filename = f"No Optimization-{off_start.strftime('%Y-%m-%d')}_{name}")
    
    return packaged_data


def cross_test(
        strategy,
        off_start,
        strategy_settings,
        validation_datas, 
        cash, 
        commission,
        baseRun = False,
        anchored = False,
        plot_positions = False):
    

    """
    This function carries out a across positions without changing params per ticker. 

    Parameters:
    _____________

    strategy: backtesting.backtesting.Strategy object
    off_start (pd.Timestamp): Official Start date. This is assuming WFO would be a class in this strategy
    strategy_settings (dict): Params as a dict to optimize by. In per params format
    validation_data (List[PTDataset]): list of PTDataset
    cash (int, float, dict): Cash
    commission(float): Commission
    baseRun (bool): This is assuming there will be no optimizing if True
    anchored (bool): True to not move validation period forward, false to move forward
    plot_position (bool): True to plot each positions charts


    Returns:
    __________

    dict: 
        agg: Aggregate data from PTBacktester
        stats_by_tick: Stats by Ticker from backtesting.py
    
    
    """


    #Run Validation backtest
    packaged_data = {}
    strategy.official_start =off_start
    strategy.is_train_backtest = False
    strategy.open_positions = []
    strategy.prev_open_positions_dict = {} #pre_open_positions_dict has be an empty dict in the first run because there are no previous positions
    if baseRun:
        bt_validation = PTBacktester(validation_datas, strategy, cash=cash, commission=commission)
    else:
        for attr, value in strategy_settings.items():
            setattr(strategy, attr, value)
        bt_validation = PTBacktester(validation_datas, strategy, cash=cash, commission=commission)
    stats_validation = bt_validation.run()
    agg_validation = bt_validation.aggregate()
    packaged_data['agg'] = agg_validation
    packaged_data['stats_by_tick'] = stats_validation
    #Plot each validation run
    if plot_positions:
        for d in bt_validation.datasets:
            name = d.name
            bt_validation.plot_position(name, filename = f"No Optimization-{off_start.strftime('%Y-%m-%d')}_{name}")
    
    return packaged_data