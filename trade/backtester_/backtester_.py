from typing import Union, Dict, Optional, List, Callable
import yfinance as yf
from typing import Union, Dict, Optional, List, Callable
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
from trade.backtester_.utils.utils import plot_portfolio, optimize
import plotly
from trade.backtester_.utils.aggregators import *


def get_class_attributes(cls):
    return [attr for attr in dir(cls) if not callable(getattr(cls, attr)) and not attr.startswith("__")]


class PTDataset:
    """
    Custom dataset holding ticker name, ticker timeseries & backtest object from backtesting.py
    
    """
    name = None
    data = None
    backtest = None
    cash = None

    def __init__(self, name: str, data: pd.DataFrame, param_settings: dict = None):
        self.__param_settings = param_settings ## Making param_settings private
        self.name = name
        self.data = data
    
    def __repr__(self):
        return f"PTDataset({self.name})"
    
    def __str__(self):
        return f"PTDataset({self.name})"
    
    @property
    def param_settings(self):
        """Getter for param settings"""
        return self.__param_settings
    
    @param_settings.setter
    def param_settings(self, value: dict):
        """Setter for param_settings with type checking"""
        self.__param_settings = value



class PTBacktester(AggregatorParent):
    """ Responsible for running backtests on multiple datasets. It is a wrapper class for the backtesting.py library.
        It is done iteratively for each dataset in the list. It also allows for optimization of parameters for each dataset.
    """

    def __init__(self, 
                datalist: list , 
                strategy,
                cash,
                strategy_settings: dict = None, 
                **kwargs) -> None:
        """
        Initializes the PTBacktester class with the following parameters:
        datalist (list): List of PTDataset objects containing the ticker name and timeseries data
        strategy (class): Strategy class to be used for backtesting
        cash (Union[int, float, dict]): Initial cash to be used for backtesting. If dict, must contain all tickers in the datalist
        strategy_settings (dict): Dictionary containing the tick as key and settings as value. Eg: {AAPL: {"entry_ma":10}}
        **kwargs: Additional keyword arguments to be passed to the backtesting.py library

        Returns:
        None
            
        """
        
        self.datasets = []
        self.strategy = strategy
        self.__port_stats = None
        self._trades = None
        self._equity = None
        self.strategy_settings = strategy_settings
        self.default_settings = None
        self._names = [d.name for d in datalist]
        self.update_settings(datalist) if self.strategy_settings else None

        assert isinstance(cash, dict) or isinstance(cash, int) or isinstance(cash, float), "Cash must be of type float, int, dict"
        if isinstance(cash, dict):
            assert all(name in list(cash.keys()) for name in self._names ), "If passing a dict for cash, all names in the dict must be present in all names in the dataset list"

        for d in datalist:
            if isinstance(cash, dict):
                cash_ = cash[d.name]
            elif isinstance(cash, int):
                cash_ = cash
            elif isinstance(cash, float):
                cash_ = cash
            
            d.backtest = Backtest(d.data, strategy = self.strategy, cash = cash_, **kwargs)
            self.datasets.append(d)
        self.cash = cash

    
    def update_settings(self, datalist) -> None:
        assert isinstance(self.strategy_settings, dict), 'Please pass a dictionary containing the tick as key and settings as value. Eg: {AAPL: {"entry_ma":10}}'
        no_settings_names = []
        updated_settings = list({list(x.keys())[0] for x in self.strategy_settings.values()})
        
        ## Get default setting for strategy to reset with
        default_setting = {}
        for setting in updated_settings:
            default_setting[setting] = getattr(self.strategy, setting)

        for name in self._names:
            dataset_obj = [x for x in datalist if x.name == name][0]
            try:
                param_setting = self.strategy_settings[name]
                
                ## Assign default setting to param_setting if another ticker has the setting but one doesn't
                not_present_settings = [x for x in updated_settings if x not in param_setting.keys()]
                for setting in not_present_settings:
                    param_setting[setting] = getattr(self.strategy, setting)

                if not isinstance(param_setting, dict):
                    raise ValueError(f'For datasets settings, please assign a dictionary containing parameters as key and values, got {type(param_setting)}')    
                dataset_obj = [x for x in datalist if x.name == name][0]
                dataset_obj.param_settings = param_setting
            except:
                ## Use default settings as settings in not given names
                dataset_obj.param_settings = default_setting
                no_settings_names.append(name)
        self.default_settings = default_setting
        

    def reset_settings(self):
        for settings, value in self.default_settings.items():
            setattr(self.strategy, settings, value)


    def run(self) -> pd.DataFrame:
        results = []
        for i, d in enumerate(self.datasets):
            d.backtest._strategy._name = d.name
            d.backtest._strategy._runIndex = i
            if d.param_settings:
                # print(d.name, "param_settings", d.param_settings)
                if d.param_settings:
                    for setting, value in d.param_settings.items():                    
                        setattr(self.strategy, setting, value)
                        # print('Pre Run', setting, getattr(self.strategy, setting))
            stats = d.backtest.run()
            self.reset_settings() if d.param_settings else None
            try:
                del d.backtest._strategy._name
            except: 
                pass
            try:
                del d.backtest._strategy._runIndex
            except:
                pass
            results.append(stats)
        names = [d.name for d in self.datasets]
        names = [d.name for d in self.datasets]
        self.__port_stats = {name:results[i] for i, name in enumerate(names)}
        self._trades = self.trades()
        self._equity = self.pf_value_ts()
        dataframe = pd.DataFrame(results).transpose()
        dataframe.columns = names
        return dataframe

    def get_port_stats(self):
        return self.__port_stats


    def pf_value_ts(self) -> pd.DataFrame:
        """
        Returns Timeseries of periodic portfolio value
        """
        PortStats = self.__port_stats
        date_range = pd.date_range(start= self.dates_(True), end = self.dates_(False), freq = 'B')
        start = self.dates_(True)
        end = self.dates_(False)
        port_equity_data = pd.DataFrame(index = date_range)
        for tick, data in PortStats.items():
            equity_curve = data['_equity_curve']['Equity']
            if isinstance(self.cash, dict):
                cash = self.cash[tick]
            elif isinstance(self.cash, int) or isinstance(self.cash, float):
                cash = self.cash
            
            equity_curve.name = tick
            tick_start = min(equity_curve.index)
            if tick_start > start:
                temp = pd.DataFrame(index = pd.date_range(start = start, end =equity_curve.index.min(), freq = 'B' ))
                temp[tick] = cash
                equity_curve = pd.concat([equity_curve, temp], axis = 0)
        
            port_equity_data = port_equity_data.join(equity_curve)

        port_equity_data = port_equity_data.dropna(how = 'all')
        port_equity_data = port_equity_data.fillna(method = 'ffill')
        port_equity_data['Total'] = port_equity_data.sum(axis = 1)
        port_equity_data.index = pd.DatetimeIndex(port_equity_data.index)
        return port_equity_data


    def plot_portfolio(self,
                    benchmark: Optional[str] = 'SPY',
                    plot_bnchmk: Optional[bool] = True,
                    return_plot: Optional[bool] = False,
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

        
        stock = Stock(benchmark)
        data = stock.spot(ts = True, ts_start = '2018-01-01')
        data.rename(columns = {x:x.capitalize() for x in data.columns}, inplace= True)
        data['Timestamp'] = pd.to_datetime(data['Timestamp'], format = '%Y-%m-%d')
        data2 = data.set_index('Timestamp')
        data2 = data2.asfreq('B', method = 'ffill')
        _bnch = data2.fillna(0)
        return plot_portfolio(self._trades, self._equity, self.dd(True), _bnch,plot_bnchmk=plot_bnchmk, return_plot=return_plot, **kwargs)


    def plot_position(self,
                     name: str,
                     **kwargs) -> None:
        """
        Plots a HMTL graph of a named position.
        name: Tick name of the position in the portfolio
        **kwargs: Variables available in backtest.plot()
        """
        name = name.upper()
        assert name in [dataset.name for dataset in self.datasets], f"Name {name} is not part of the portfolio. Please pick a name from {[dataset.name for dataset in self.datasets]}"
        for dataset in self.datasets:
            if dataset.name == name:
                break
        dataset.backtest.plot(**kwargs)

    def optimize(self, 
            optimize_var: dict,
            maximize: Union[List[Callable], str],
            max_tries: Union[int, float] = None,
            constraint: Callable = None

            

        
        ):
        
        """
        Returns a dataframe containing all values corresponding to optimized parameter
        Parameters:
        object (self): The PTBacktester Instance. Not passable

        optimize_var (dict): Containing PARAMETER NAMES (must be represented with object variables/attributes names) AS IS FOUND IN THE STRATEGY CLASS for Keys, 
                             and a list of possible ranges for values. 
                             If name does not exist in strategy variables, will raise an error.

        maximize (List[Callable]): List of functions to calculalte values for and thus maximize. This parameter takes a list of items. There are only 3 allowed types of items.
                                   Self Method pass as a string, Custom Function & Self Method pass as a function obj
                                   (Refer to dir(object instance) for available functions)

        max_tries  (Union[int, float]): Is the maximum number of strategy runs. If max_tries is between [0,1], this sets the number of runs as a fraction of total grid
                                        if the number is an integer, for grid_search method, it randomizes the search, but is constrained to the max amount of tries.
                                        For Grid Search, this is defaulted to exhaustive, while for skopt, defaulted to 200
                                        
        constraint: Function or Str evaluating the values IN THE OPTIMIZE VAR
                    Best Practice for Constraints:
                    - Function or list must contain what is exactly in optimize_var.
                    Examples:
                        str: 'entry_ma > exit_ma'
                        lamda: entry_ma : entry_ma > 10 

            """
            
        return optimize(self, optimize_var, maximize, max_tries, constraint)
    
    def position_optimize(self,
        param_kwargs: dict,
        **kwargs
        ): 
        """
        Optimize position based on given parameters.

        Parameters:
        param_kwargs (dict): A dictionary of required parameters for optimization.
        **kwargs: Additional optional keyword arguments.

        Returns:
        Optimization results.

        For more information, visit the 
        `backtesting.py documentation <https://kernc.github.io/backtesting.py/doc/backtesting/backtesting.html#backtesting.backtesting.Backtest.optimize>`_

        """
        assert isinstance(param_kwargs, dict), f'param_kwargs must be a dict containing params as key & list of values as value, recieved {type(param_kwargs)}'
        datasets = self.datasets
        # Create Dataframe to hold Optimized Values & heat
        optimized = pd.DataFrame()
        portfolio_hm = pd.DataFrame()

        # Check if heatmap is passed
        return_heatmap = kwargs.get('return_heatmap') if 'return_heatmap' in kwargs else False
        
        ## Save default params before optimize
        default_params = {}
        for param in param_kwargs.keys():
            default_params[param] = getattr(self.strategy, param)
            
        ## Loop through each datasets backtest, optimize & append to optimized dataframe
        for dataset in self.datasets:
            
            if return_heatmap:
                opt, hm = dataset.backtest.optimize(**param_kwargs, **kwargs)
                for param in param_kwargs.keys():
                    optimized.at[dataset.name, param] = getattr(opt._strategy,param)
                hhm = pd.DataFrame(hm)
                hhm['Name'] = dataset.name
                cols = ['Name'] + list(param_kwargs.keys())
                hhm.reset_index(inplace = True)
                hhm.set_index(cols, inplace = True)
                portfolio_hm = pd.concat([portfolio_hm, hhm], axis = 0)


            else: 
                opt = dataset.backtest.optimize(**param_kwargs, **kwargs)
                for param in param_kwargs.keys():
                    optimized.at[dataset.name, param] = getattr(opt._strategy,param)
        
        ## Reset default params before optimize
        for param in param_kwargs.keys():
            setattr(self.strategy, param, default_params[param])
        
        return (optimized, portfolio_hm) if return_heatmap else optimized
