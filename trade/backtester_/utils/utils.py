from typing import Union, Dict, Optional, List, Callable
from itertools import product
from collections.abc import Callable as callable_func
import random
import inspect
from typing import Union, Dict, Optional, List, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from backtesting import Backtest
import pandas as pd
import sys
import os
sys.path.append(
    os.environ.get('WORK_DIR'))
from trade.assets.Stock import Stock
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from itertools import product
from collections.abc import Callable as callable_func
import random
import inspect
from pathos.multiprocessing import ProcessingPool as Pool
import threading






def get_class_attributes(cls):
    return [attr for attr in dir(cls) if not callable(getattr(cls, attr)) and not attr.startswith("__")]


def plot_portfolio(_tr: pd.DataFrame,
                    _eq: pd.DataFrame,
                    _dd: pd.DataFrame,
                    _bnch: Optional[pd.DataFrame] = None,
                    plot_bnchmk: Optional[bool] = True,
                    return_plot: Optional[bool] = False,
                    **kwargs):
    """
    Plots a graph of current porfolio metrics. These graphs are Equity Curve, Portfolio Drawdown, Trades, Periodic returns
    Plotting function is plotly. Through **kwargs, you can edit the subplot
    
    Parameters:
    _tr (pd.DataFrame): Dataframe containing all trades. Necessary Columns: 'ReturnPct', 'EntryTime', 'Ticker', 'Size'. Index should be a DateTimeIndex
    _eq (pd.DataFrame): Dataframe containing daily equity values. Necessary Columns: 'Total'. Index should be a DateTimeIndex
    _dd (pd.DataFrame): Dataframe containing daily drawdown values. Necessary Columns: 'dd' which is drawdown value. Index should be a DateTimeIndex
    _bnch Optional[pd.DataFrame]: Dataframe containing daily Close values for chosen benchmark. Necessary Columns: 'Close'
    plot_bnchmk (Optional[bool]): Optionality to plot a benchmark or not
    return_plot Optional [bool]: Returns the plot object. User may opt for this if they plan to make further editing beyond **kwargs functionality. 
                                Note, best to designate this to a variable to avoid being displayed twice

    Returns: 
    Plot: For further editing by the user
    """

    import plotly.io as pio
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _tr['Color'] = _tr.ReturnPct.apply(lambda x: 'red' if x <= 0 else 'green')
    specs = kwargs.pop('specs', [[{'colspan': 2}, None],
            [{'colspan': 2}, None],
            [{'colspan': 2}, None],
            [{},{}]
    
    ])
    row_heights = kwargs.pop('row_heights', [0.6,0.2,0.3,0.2])
    subplot_titles=kwargs.pop('subplot_titles',['EQUITY CURVE', 'TRADES','PORTFOLIO DRAWDOWN','WEEKLY PORTFOLIO RTRNS', 'MONTHLY PORTFOLIO RTRNS'])
    shared_xaxes = kwargs.pop('shared_xaxes',True)
    vertical_spacing = kwargs.pop('vertical_spacing',0.05)
    horizontal_spacing = kwargs.pop('horizontal_spacing',0.05)
    
    title = dict(text = 'Equity Curve', x= 0.5, xanchor = 'center', font = dict(size = 20))
 
    fig = make_subplots(rows = 4, cols = 2, subplot_titles= subplot_titles, 
                        shared_xaxes= True, vertical_spacing=0.05, horizontal_spacing= 0.05, specs = specs, row_heights=row_heights)

    # hovertemplate =  '<b>%{text}</b><br><br>' +
    # 'Date: %{x}<br>' +
    # 'Size: %{marker.size:.2f}<br>' +
    # 'PnL: %{y}<br>'

    fig.add_trace(go.Scatter( x= _eq.index, y = (round(_eq.Total, 2)/_eq.Total[0]), showlegend= True, name = 'Equity Curve'),
    row = 1, col = 1
    )
    if plot_bnchmk:
        fig.add_trace(go.Scatter( x= _bnch.index, y = (_bnch.Close.pct_change()+1).cumprod(), showlegend= True, name = 'Benchmark'),
        row = 1, col = 1
        )


    fig.add_trace(go.Scatter(x = _tr.EntryTime, y = _tr.ReturnPct, text = _tr.Ticker, mode = 'markers', name = 'Ticker', customdata=_tr['Size'], hovertemplate=
        '<b>%{text}</b><br><br>' +
        'Date: %{x}<br>' +
        'Size: %{customdata}<br>' +
        'PnL: %{y}<br>', marker = dict(color= _tr['Color']), showlegend= False),
    row = 2, col = 1
    )


    fig.add_trace(go.Scatter( x= _dd.index, y = _dd.dd, line = dict(color = 'red'), showlegend= False, name = 'Drawdown', fill = 'tozeroy'),
    row = 3, col = 1
    )

    fig.add_trace(go.Histogram(y = _eq.resample('W').last().Total.pct_change().unique(), histnorm = 'percent' , marker = dict(color = 'purple'), showlegend= False ),
    row = 4, col = 1
    )

    fig.add_trace(go.Histogram(y = _eq.resample('M').last().Total.pct_change().unique(), histnorm = 'percent', marker = dict(color = 'purple'), showlegend= False ),
    row = 4, col = 2
    )
   
    height = kwargs.pop('height', 1050)
    width = kwargs.pop('width',1200)
    hovermode = kwargs.pop('hovermode', 'x unified')
    paper_bgcolor = kwargs.pop('paper_bgcolor', "WHITE")
    margin = kwargs.pop('margin',dict(l = 15, r= 15, t = 50, b = 20))
    layout_kwargs = {key: value for key, value in kwargs.items() if key in fig.layout and key not in [height, width,hovermode,paper_bgcolor, margin]}
    fig.update_layout(height = height, width = width, hovermode=hovermode,paper_bgcolor="WHITE", margin = margin, **layout_kwargs)
    fig.show()
    return fig if return_plot else None


def optimize_(object: object, 

            optimize_var: dict,
            maximize: Union[List[Callable], str],
            max_tries: Union[int, float] = None,
            constraint: Callable = None
):

    """
    Returns a dataframe containing all values corresponding to optimized parameter
    Parameters:
    object: Backtesting engine. Preferably PTBacktester
    optimize_var (dict): Containing VARIABLE NAME AS IS NAMED IN THE STRATEGY CLASS for Keys, and a list of possible ranges for values. 
                        If name does not exist in strategy variables, will raise an error.
    maximize (List[Callable]): List of functions to calculalte values for and thus maximize. This parameter takes a list of items. There are only 3 allowed class of items.
                              Self Method pass as a string, Custom Function & Self Method pass as a function obj
    max_tries: Is the maximal number of strategy runs. If max_tries is between [0,1], this sets the number of runs as a fraction of total grid
               if the number is an integer, for grid_search method, it randomizes the search, but is constrained to the max amount of tries.
               For Grid Search, this is defaulted to exhaustive, while for skopt, defaulted to 200
    constraint: Function or Str evaluating the values IN THE OPTIMIZE VAR
                Best Practice for Constraints:
                 - Function or list must contain what is exactly in optimize_var.
                 Examples:
                    str: 'entry_ma > exit_ma'
                    lamda: entry_ma : entry_ma > 10 



    """
    max_ = 0
    combo = pd.DataFrame(columns = optimize_var.keys())
    cart_plane = np.array(list(product(*optimize_var.values())))

    #INITIATE MAX_TRIES FOR THE CARTESIAN PLANE
    if isinstance(max_tries, float) and max_tries > 0 and max_tries < 1:
        fitr = int(len(cart_plane)* max_tries)
        max_try_index = random.sample(range(len(cart_plane)), fitr) 
    elif isinstance(max_tries, int) and max_tries >1:
        max_try_index = random.sample(range(len(cart_plane)), max_tries) 
    elif not max_tries:
        if max_tries is None:
            max_try_index = list(range(len(cart_plane)))
        elif max_tries <= 0:
            raise ValueError(f'{max_tries} for max_tries is invalid. Either choose a float between 0 and 1 (greater than 0) or int greater than 1')
    else:
        raise ValueError(f'{max_tries} for max_tries is invalid. Either choose a float between 0 and 1 or int greater than 1')
    cart_plane = cart_plane[max_try_index]
    plane_dict = {}



    #CREATE RESET VARIABLES TO RETURB TO INITIAL STATE
    reset_dict = {}
    for name in optimize_var.keys():
        reset_dict[name] = [getattr(dataset.backtest._strategy, name) for dataset in object.datasets]



    #BEGIN EXHAUSTIVE GRID SEARCH. LOOPING THROUGH EACH PLANE AND HANDLING OTHER ISSUES SUCH AS CONTRAINTS
    for j, plane in enumerate(cart_plane): #LOOP 1, LOOP THROUGH ALL CARTISIAN PLANES

        for i, attr in enumerate(optimize_var.keys()): # LOOP 2. LOOP THROUGH OPTIMIZABLE VALUES AND SETS ATTRIBUTES OF EACH GIVEN ATTR BASED ON THE CURRENT CARTISIAN PLANE

            for n in range(len(plane)): #LOOP 3. LOOP THROUGH THE PLANE TO PLACE THEM IN THEIR CORRESPONDING INDEX OF A DATAFRAME
                plane_dict[attr] = plane[i]
                combo.at[j, list(optimize_var.keys())[i]] = plane[i]
            for dataset in object.datasets: # LOOP 4. LOOP OVER EACH DATASETS (PORTFOLIO POSITIONS) AND SET ATTRIBUTES
                setattr(dataset.backtest._strategy, attr, plane[i])
        if isinstance(constraint, str):
            constraint_flag = eval(constraint, {}, plane_dict)
            
    
        elif callable(constraint):
            params = inspect.signature(constraint).parameters
            lambda_args = [plane_dict[param] for param in params]
            constraint_flag = constraint(*lambda_args)
        elif constraint is None:
            constraint_flag = True
        if constraint_flag:
            object.run() #RE-RUN BACKTEST BASED ON EDITIED PARAMETERS



    #LOOP 5. LOOP THROUGH EACH PROVIDED FUNCTIONS
            for method in maximize: 
                if isinstance(method, str):
                    func = getattr(object, method) ## REPLACE tt WITH SELF. FOR INSTANCES WHEN USER PASSES A STRING THAT IS PART OF THE CLASS
                elif callable(method):
                    if hasattr(method, '__self__'): #and method.__self__ is self THIS UNCOMMENTED PART CHECKS IF THE FUNCTION PASSED IS PART OF THE UNDERLYING BACKTESTING INSTANCE
                        func = method
                    else: 
                        func = lambda x:method(object) #REPLACE WITH SELF
                combo.at[j, func.__name__] = func()
        else:
            pass
    for name, values in reset_dict.items():
        for i, dataset in enumerate(object.datasets):
            setattr(dataset.backtest._strategy, name, values[i])

    object.run()
    return combo.dropna()







def evaluate_plane(plane, optimize_var, object, maximize, constraint):
    plane_dict_local = {}
    for i, attr in enumerate(optimize_var.keys()):
        for dataset in object.datasets:
            setattr(dataset.backtest._strategy, attr, plane[i])
        plane_dict_local[attr] = plane[i]

    if isinstance(constraint, str):
        constraint_flag = eval(constraint, {}, plane_dict_local)
    elif callable(constraint):
        params = inspect.signature(constraint).parameters
        lambda_args = [plane_dict_local[param] for param in params]
        constraint_flag = constraint(*lambda_args)
    elif constraint is None:
        constraint_flag = True

    if constraint_flag:
        object.run()  # RE-RUN BACKTEST BASED ON EDITED PARAMETERS

        result = {}
        for method in maximize:
            if isinstance(method, str):
                func = getattr(object, method)
            elif callable(method):
                if hasattr(method, '__self__'):
                    func = method
                else:
                    func = lambda x: method(object)
            result[func.__name__] = func()
        for i, attr in enumerate(optimize_var.keys()):
            result[attr] = plane_dict_local[attr]
        return result
    else:
        return None




def optimize(object: 'PTBacktester', 
              optimize_var: dict,
              maximize: Union[List[Callable], str],
              max_tries: Union[int, float] = None,
              constraint: Callable = None,
              num_workers: int = None):

    """
    Returns a dataframe containing all values corresponding to optimized parameter
    Parameters:
    optimize_var (dict): Containing VARIABLE NAME AS IS NAMED IN THE STRATEGY CLASS for Keys, and a list of possible ranges for values. 
                        If name does not exist in strategy variables, will raise an error.
    maximize (List[Callable]): List of functions to calculalte values for and thus maximize. This parameter takes a list of items. There are only 3 allowed class of items.
                              Self Method pass as a string, Custom Function & Self Method pass as a function obj
    max_tries: Is the maximal number of strategy runs. If max_tries is between [0,1], this sets the number of runs as a fraction of total grid
               if the number is an integer, for grid_search method, it randomizes the search, but is constrained to the max amount of tries.
               For Grid Search, this is defaulted to exhaustive, while for skopt, defaulted to 200
    constraint: Function or Str evaluating the values IN THE OPTIMIZE VAR
                Best Practice for Constraints:
                 - Function or list must contain what is exactly in optimize_var.
                 Examples:
                    str: 'entry_ma > exit_ma'
                    lamda: entry_ma : entry_ma > 10 
    num_workers: Number of worker processes to use for multiprocessing
    """
    
    
    import threading
    import os
    from functools import partial
    # from pathos.multiprocessing import ProcessingPool as Pool
    max_ = 0
    combo = pd.DataFrame(columns=optimize_var.keys())
    cart_plane = np.array(list(product(*optimize_var.values())))
    print("Using Multi Processing")
    # INITIATE MAX_TRIES FOR THE CARTESIAN PLANE
    if isinstance(max_tries, float) and max_tries > 0 and max_tries < 1:
        fitr = int(len(cart_plane) * max_tries)
        max_try_index = random.sample(range(len(cart_plane)), fitr)
    elif isinstance(max_tries, int) and max_tries > 1:
        max_try_index = random.sample(range(len(cart_plane)), max_tries)
    elif not max_tries:
        if max_tries is None:
            max_try_index = range(len(cart_plane))
        elif max_tries <= 0:
            raise ValueError(f'{max_tries} for max_tries is invalid. Either choose a float between 0 and 1 (greater than 0) or int greater than 1')
    else:
        raise ValueError(f'{max_tries} for max_tries is invalid. Either choose a float between 0 and 1 or int greater than 1')
    cart_plane = cart_plane[max_try_index]
    plane_dict = {}

    # CREATE RESET VARIABLES TO RETURN TO INITIAL STATE
    reset_dict = {}
    for name in optimize_var.keys():
        reset_dict[name] = [getattr(dataset.backtest._strategy, name) for dataset in object.datasets]
    
    partial_funct = partial(evaluate_plane, object = object, optimize_var = optimize_var, maximize = maximize, constraint = constraint )
    pool = Pool()
    pool.restart()
    results = pool.map(partial_funct, cart_plane )
    pool.close()
    pool.join()



    # Reset to original state
    for name, values in reset_dict.items():
        for i, dataset in enumerate(object.datasets):
            setattr(dataset.backtest._strategy, name, values[i])
    object.run()

    results = [res for res in results if res is not None]
    return pd.DataFrame(results).dropna()





def pf_value_ts(PortStats):
    port_equity_data = pd.DataFrame()
    for tick, data in PortStats.items():
        ## EQUITY CURVE
        equity_curve = data['_equity_curve']['Equity']
        equity_curve.name = tick
        port_equity_data = pd.concat([port_equity_data, equity_curve], axis = 1)


    port_equity_data['Total'] = port_equity_data.sum(axis = 1)
    port_equity_data.index = pd.DatetimeIndex(port_equity_data.index)
    return port_equity_data

def dates_(port_stats: dict, start: bool = True) -> pd.Timestamp: #YES
    """
    Returns the either the start or end date of the portfolio

    Parameters:
    start (bool): This determins whether to return a start or end date
    port_stats(dict): A dict containing the stats as values and anything as key

    Returns:
    pd.Timestamp: Corresponding date
    
    """

    start_list = []
    end_list = []
    duration_list = []
    for tick, data in port_stats.items():
        start_list.append(data['Start'])
        end_list.append(data['End'])
        duration_list.append(data['Duration'])

    return min(start_list) if start else max(end_list)


def peak_value_func(equity_timeseries: pd.DataFrame,value = True): #YES
    """
    Returns the either the Peak value or a dict holding peak date as key and peak value as value

    Parameters:
    equity_timeseries (pd.DataFrame): This is the timeseries of the periodic equity values
    value (bool): A boolean value. If true returns ONLY peak value if false returns dict

    Returns:
    float: Peak value or 
    dict: {peak date: peak value}
    
    """
    ts = equity_timeseries
    peak_value = ts['Total'].max()
    peak_date = ts[ts['Total'] == peak_value].index[0]
    peak_dict = {peak_date: round(peak_value, 2)}
    return peak_value if value else peak_dict


def final_value_func(equity_timeseries:pd.DataFrame):
    """
    Returns the final value of the equity_timeseries

    Parameters:
    equity_timeseries (pd.DataFrame): This is the timeseries of the periodic equity values

    Returns:
    float: Final value
    
    """
    ts = equity_timeseries
    final_val = round(ts['Total'][-1],2)
    return final_val


def rtrn(equity_timeseries: pd.DataFrame):
    """
    Returns the rtrn for equity timeseries

    Parameters:
    equity_timeseries (pd.DataFrame): This is the timeseries of the periodic equity values

    Returns:
    float: Return of timeseries from start to end

    """
    ts = equity_timeseries
    rtrn = (ts['Total'][-1]/ts['Total'][0])-1
    return rtrn*100

def buyNhold(PortStats: dict):
    """
    Calculate the percent return a buyNhold strategy would give

    Parameters:
    PortStats: Dict holding portfolio statistics

    Returns:
    float: percent return
    """
    assert isinstance(PortStats, dict), f"Incorrect Data Type {type(PortStats)}. Please Pass Dictionary Containing backtesting.py output"
    initial_val = np.ones(len(PortStats)).sum() 
    return_vals = np.zeros(len(PortStats))
    for i, (k, v) in enumerate(PortStats.items()):
        rtrn = v['Buy & Hold Return [%]']/100
        return_vals[i] = (1+rtrn)
    bNh_rtrn = round(((return_vals.sum()/initial_val)-1)*100,2)
    return bNh_rtrn

def cagr(equity_timeseries: pd.DataFrame):
    """
    Calculate the Cumualtive Avg Growth Rate

    Parameters:
    equity_timeseries (pd.DataFrame): This is the timeseries of the periodic equity values

    Returns:
    float: CAGR value
    """
    ts = equity_timeseries
    begin_val = ts['Total'][0]
    end_val = ts['Total'][-1]
    days = (ts.index.max() - ts.index.min()).days
    return ((end_val/begin_val)**(252/days) -1)*100

def vol_annualized(equity_timeseries: pd.DataFrame, downside: bool = False, MAR: Union[int,float] = None):
    """
    Calculate the Annualized Volatility of the portfolio

    Parameters:
    equity_timeseries (pd.DataFrame): This is the timeseries of the periodic equity values
    downside (bool): Calculate only downside vol or all vol
    MAR (Union[int, float]): Minimum Acceptable return expressed as 0.01 for 1%

    Returns:
    float: Annualized Volatility
    """

    ts = equity_timeseries
    if not downside:
        return round(np.std(ts['Total'].pct_change(), ddof =1) *np.sqrt(252) *100, 6)
    else:
        if not MAR:
            MAR = 0

        ts = ts['Total'].pct_change() - MAR
        ts_d  = ts[ts < 0]
        return round(np.std(ts_d, ddof =1) *np.sqrt(252) *100, 6)


def daily_rtrns(equity_timeseries: pd.DataFrame):
    """
    Calculate the Portfolio Daily Returns

    Parameters:
    equity_timeseries (pd.DataFrame): This is the timeseries of the periodic equity values
    

    Returns:
    pd.Series: Portfolio Daily returns
    """
    ts = equity_timeseries
    return ts['Total'].pct_change()


def sharpe(risk_free_rate: float, equity_timeseries: pd.DataFrame):
    """
    Calculate the Sharpe Ratio of the Strategy

    Parameters:
    equity_timeseries (pd.DataFrame): This is the timeseries of the periodic equity values
    

    Returns:
    float: Sharpe Ratio
    """

    ## ANNUALIZED MEAN EXCESS RETURN / ANNUALIZED VOLATILITY
    daily_rfrate = (1+risk_free_rate)**(1/252) - 1 
    annualized_vol = vol_annualized(equity_timeseries)/100
    excess_retrns = np.mean(daily_rtrns() - daily_rfrate)*252
    return excess_retrns/annualized_vol



def sortino(risk_free_rate: float, MAR = None): #YES
    
    ## ANNUALIZED MEAN EXCESS RETURN / ANNUALIZED VOLATILITY
    if not MAR:
        MAR = (1+risk_free_rate)**(1/252) - 1 
    daily_rfrate = (1+risk_free_rate)**(1/252) - 1 
    annualized_vol = vol_annualized(True, MAR)/100
    excess_retrns = np.mean(daily_rtrns() - daily_rfrate)*252
    return excess_retrns/annualized_vol

def dd(full = False):
    ts = pf_value_ts()
    data = pd.DataFrame()
    data['Total'] = ts['Total']
    data['Running_max'] = data.Total.cummax()
    data['dd'] = (data.Total/data.Running_max)-1
    if full:
        return data
    else:
        return data['dd']

def mdd(): #YES
    dd_ = dd()
    return dd_.min()*100

def calmar(): #YES
    return abs(cagr()/mdd())

def avg_dd_percent(): #YES
    return round(dd().mean()*100,6)

def mdd_value(): #YES
    dd_ = dd(True)
    return round((dd_['Total'] - dd_['Running_max']).min(),2)

def mdd_duration(full = False): #YES
    from datetime import timedelta
    dd_ = dd(True)

    for i, (index, row) in enumerate(dd_.iterrows()):
        total = row['Total']
        running_max = row['Running_max']
        date = index
        running_max_date = dd_[dd_['Total'] == running_max].index[0]
        dd_.at[index, 'timedelta'] = (date - running_max_date )

    if full:
        return dd_
    else:
        return dd_.timedelta.max()

def avg_dd_duration(): #YES
    dd_duration = mdd_duration(True)
    return dd_duration.timedelta.mean()

def trades(PortStats): 
    trades_df = pd.DataFrame()
    for k, v in PortStats.items():
        holder = v['_trades']
        holder['Ticker'] = k
        trades_df = pd.concat([trades_df, holder])
    return trades_df

def numOfTrades(PortStats: Union[dict, pd.DataFrame]): #YES
    return len(trades(PortStats)) if isinstance(PortStats, dict) else len(PortStats)

def winRate(PortStats:  Union[dict, pd.DataFrame]): #YES
    trades_ = trades(PortStats) if isinstance(PortStats, dict) else PortStats
    return round(((trades_.ReturnPct >0).sum()/(trades_.ReturnPct ).count())*100 , 2)


def lossRate(PortStats: Union[dict, pd.DataFrame]): #YES
    return round((100 - winRate(PortStats)),2)


def avgPnL(PortStats, Type_: str, value = True): #YES
    assert Type_.upper() in ['W', 'L', 'A'],  f"Invalid type: {Type_}. Must be 'L', 'W' or 'A."
    trades_ = trades(PortStats) if isinstance(PortStats, dict) else PortStats
    PnL = trades_.PnL if value else trades_.ReturnPct
    WPnL = PnL[PnL>0] if Type_.upper() == 'W' else PnL[PnL<=0] if Type_.upper() == 'L' else PnL   
    return WPnL.mean() *100


def bestTrade(PortStats: Union[dict, pd.DataFrame]): #YES
    return trades(PortStats).ReturnPct.max()*100 if isinstance(PortStats, dict) else PortStats.ReturnPct.max()*100



def worstTrade(PortStats: Union[dict, pd.DataFrame]): #YES
    return trades(PortStats).ReturnPct.min()*100 if isinstance(PortStats, dict) else PortStats.ReturnPct.min()*100


def profitFactor(PortStats: Union[dict, pd.DataFrame]): #YES
    tr = trades(port_stats) if isinstance(PortStats, dict) else PortStats
    tot_loss = tr[tr['ReturnPct'] <= 0]['PnL'].sum()
    tot_gain = tr[tr['ReturnPct'] > 0]['PnL'].sum()
    return round(abs(tot_gain/tot_loss),6)

def Expectancy(PortStats): #YES
    tr = trades(port_stats) if isinstance(PortStats, dict) else PortStats

    return (avgPnL(tr, 'W', False) * winRate(tr)) + (avgPnL(tr, 'L', False) *lossRate(tr))

def SQN(PortStats): #YES
    trades_ = trades(PortStats) if isinstance(PortStats, dict) else PortStats
    return((trades_.ReturnPct.mean() * np.sqrt(len(trades_)))/np.std(trades_.ReturnPct))

def ExposureDays(PortStats): #YES
    time_in = pd.DataFrame(index = pf_value_ts().index)
    time_in['position'] = 0
    tr = trades(PortStats) if isinstance(PortStats, dict) else PortStats
    for index, row in tr.iterrows():
        entry = row['EntryTime']
        exit_ = row['ExitTime']
        time_in['position'].loc[(time_in.index >= entry) & (time_in.index <= exit_)] = 1
    return round((time_in['position'] == 1).sum()/len(time_in)*100, 2)


def yearly_retrns() -> dict: #YES
    ts = pf_value_ts()
    ts['Year'] = ts.index.year
    unq_year = ts.Year.unique()
    rtrn_d = {}
    for year in unq_year:
        data = ts[ts['Year'] == year].sort_index()
        ret = ((data.loc[data.index.max(), 'Total']/data.loc[data.index.min(), 'Total'])-1)*100
        rtrn_d[year] = ret

    return rtrn_d

def avg_holding_period(PortStats: Union[pd.Series, dict], Type_: str = 'W') -> pd.Timedelta: #YES
    """ 
    Returns the average holding period of the portfolio based on Trades data

    Parameters:
    Type_ (flaot): Which holding period are we looking for. Available options are
        'W': For winning holding period
        'L': For losing holding period
        'A': For Holding period based on all data

    PortStats (pd.Series): The Stats data obtained from backtesting.py simulation

    Returns:
    pd.Timedelta: Corresponding Value
    """
    assert Type_.upper() in ['A', 'W', 'L'], f"Invalid type: {Type_}. Must be 'L', 'W' or 'A."
    trades_ = trades(PortStats) if isinstance(PortStats, dict) else PortStats
    if Type_.upper() == 'W':
        trades_ = trades_[trades_['ReturnPct'] > 0]
    elif Type_.upper() == 'L':
        trades_ = trades_[trades_['ReturnPct'] <= 0]
    
    return trades_.Duration.mean()


def max_holding_period(PortStats: Union[pd.Series, dict], Type_: str = 'W') -> pd.Timedelta: #YES
    """ 
    Returns the max holding period of the portfolio based on Trades data

    Parameters:
    Type_ (flaot): Which holding period are we looking for. Available options are
        'W': For winning holding period
        'L': For losing holding period
        'A': For Holding period based on all data

    PortStats (pd.Series): The Stats data obtained from backtesting.py simulation

    Returns:
    pd.Timedelta: Corresponding Value
    """
    assert Type_.upper() in ['A', 'W', 'L'], f"Invalid type: {Type_}. Must be 'L', 'W' or 'A."
    trades_ = trades(PortStats) if isinstance(PortStats, dict) else PortStats
    if Type_.upper() == 'W':
        trades_ = trades_[trades_['ReturnPct'] > 0]
    elif Type_.upper() == 'L':
        trades_ = trades_[trades_['ReturnPct'] <= 0]
    
    return trades_.Duration.max()


def streak(PortStats: Union[pd.Series, dict], Type_: str = 'W') -> int: #YES
    """ 
    Returns the Losing/Winning Streak based on Trades data

    Parameters:
    Type_ (flaot): Which holding period are we looking for. Available options are
        'W': For winning holding period
        'L': For losing holding period

    PortStats (pd.Series): The Stats data obtained from backtesting.py simulation

    Returns:
    int: Corresponding Value
    """
    assert Type_.upper() in ['L', 'W'], f"Invalid type: {Type_}. Must be 'L' or 'W'."
    t = trades(PortStats) if isinstance(PortStats, dict) else PortStats
    t['Is_Loss'] = int
    t['Is_Loss'] = t['ReturnPct'] <= 0 if Type_.upper() =='L' else t['ReturnPct'] > 0 
    t['Loss_Streak'] = (t['Is_Loss'] != t['Is_Loss'].shift()).cumsum()
    streak_lengths = t.groupby('Loss_Streak')['Is_Loss'].sum()
    return streak_lengths[t.groupby('Loss_Streak')['Is_Loss'].first()].max()


def aggregate(PortStats: dict, 
    risk_free_rate: float = 0.05,
    MAR: float = None) -> pd.Series:
    """ 

    Returns aggregated Data for the Porftolio

    Parameters:

    PortStats (pd.Series): The Stats data obtained from backtesting.py simulation
    risk_free_rate (float): Current Risk Free Rate Value (Annual Rate, not daily)
    Returns:
    int: Corresponding Value
    """
    MAR = 0.0 if not MAR else MAR
    tr = trades(PortStats)
    assert isinstance(MAR, float), f"Recieved MAR of type {type(MAR)} instead of Type float"
    series1 = pd.Series({
        'Start': dates_(True),
        'End': dates_(False),
        'Duration': dates_(False) - dates_(True),
        'Exposure Time [%]': ExposureDays(tr),
        'Equity Final [$]': final_value_func(),
        'Equity Peak [$]': peak_value_func(),
        'Return [%]': rtrn(),
        'Buy & Hold Return [%]': buyNhold(PortStats),
        'CAGR [%]' :  cagr(),
        'Volatility Ann. [%]': vol_annualized(),
        'Sharpe Ratio': sharpe(risk_free_rate),
        'Sortino Ratio': sortino(risk_free_rate, MAR),
        'Calmar Ratio': calmar(),
        'Max. Drawdown [%]': mdd(),
        'Max. Drawdown Value [$]': mdd_value(),
        'Avg. Drawdown [%]': avg_dd_percent(),
        'Max. Drawdown Duration': mdd_duration(),
        'Avg Dradown Duration': avg_dd_duration(),
        '# Trades': numOfTrades(tr),
        'Win Rate [%]': winRate(tr),
        'Lose Rate [%]': lossRate(tr),
        'Avg. Trade [%]' : avgPnL(tr, 'A', False),
        'Avg. Winning Trade [%]' : avgPnL(tr, 'W', False),
        'Avg. Losing Trade [%]' : avgPnL(tr, 'L', False),
        'Best Trade [%]': bestTrade(tr),
        'Worst Trade [%]': worstTrade(tr),
        'Avg Trade Duration': avg_holding_period(tr, 'A'),
        'Avg Win Trade Duration': avg_holding_period(tr, 'W'),
        'Avg Lose Duration': avg_holding_period(tr, 'L'),
        'Max Trade Duration': max_holding_period(tr, 'A'),
        'Max Win Trade Duration': max_holding_period(tr, 'W'),
        'Max Lose Duration': max_holding_period(tr, 'L'),
        'Profit Factor': profitFactor(tr),
        'Expectancy [%]': Expectancy(tr),
        'SQN': SQN(tr)
        
    })

    rtrn_dict = yearly_retrns()
    rtrn_series = pd.Series({f"{year} Return [%]" : value for year, value in rtrn_dict.items()})

    series3 = pd.Series({
        'Winning Streak': streak(PortStats, 'W'),
        'Losing Streak': streak(PortStats, 'L'),
        '_strategy': list(PortStats.values())[0]['_strategy'],
        'equity_curve': pf_value_ts(),
        '_trades': trades(PortStats),
        '_tickers': stocks

    })
    return pd.concat([series1, rtrn_series, series3])





