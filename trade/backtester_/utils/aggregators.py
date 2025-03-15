## To-Do: This was written with Long in mind. Need to add Short functionality
## To-do: Add Thorp Expectancy (from building a winning system book)
## To-do: Add skew
## To-do: Is sharpe annualized?
## To-Do: Change BuyNHold to the function in EventDrive.portfolio.OptionSignalPortfolio

import sys
import os
sys.path.append(
    os.environ.get('WORK_DIR'))
from trade.helpers.helper import copy_doc_from,filter_inf,filter_zeros
from trade.assets.Stock import Stock
from abc import ABC, abstractmethod
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly
from typing import Union, Dict, Optional, List, Callable
from itertools import product
from collections.abc import Callable as callable_func
import random
import inspect
from typing import Union, Dict, Optional, List, Callable
import numpy as np
from backtesting import Backtest
import pandas as pd

def pf_value_ts(port_stats: dict, cash: Union[dict, int, float]) -> pd.DataFrame:
    """
    Parameters:
    port_stats (dict): A dict holding aggregated pd.Series as values and anything as keys
    cash (Union[dict, int, float]): Starting cash. A dict would have tick as name and cash as values. Whereas int/float would be used for all names


    Returns Timeseries of periodic portfolio value
    """

    PortStats = port_stats
    date_range = pd.date_range(start=dates_(True), end=dates_(False), freq='B')
    start = dates_(True)
    end = dates_(False)
    port_equity_data = pd.DataFrame(index=date_range)
    for tick, data in PortStats.items():
        equity_curve = data['_equity_curve']['Equity']
        if isinstance(cash, dict):
            cash = cash[tick]
        elif isinstance(cash, int) or isinstance(cash, float):
            cash = cash

        equity_curve.name = tick
        tick_start = min(equity_curve.index)
        if tick_start > start:
            temp = pd.DataFrame(index=pd.date_range(
                start=start, end=equity_curve.index.min(), freq='B'))
            temp[tick] = cash
            equity_curve = pd.concat([equity_curve, temp], axis=0)

        port_equity_data = port_equity_data.join(equity_curve)

    port_equity_data = port_equity_data.dropna(how='all')
    port_equity_data = port_equity_data.fillna(method='ffill')
    port_equity_data['Total'] = port_equity_data.sum(axis=1)
    port_equity_data.index = pd.DatetimeIndex(port_equity_data.index)
    return port_equity_data


def short_returns(t0, t1):
    return 1 - (t1/t0)



def dates_(port_stats: dict, start: bool = True) -> pd.Timestamp:
    """
    Returns the either the start or end date of the portfolio

    Parameters:
    start (bool): This determins whether to return a start or end date
    port_stats (dict): A dict holding aggregated pd.Series as values and anything as keys

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


def peak_value_func(equity_timeseries: pd.DataFrame, value: bool = True) -> Union[float, Dict]:
    """
    Returns the peak value of the portfolio and has the option to return corresponding date

    Parameters:
    value (bool): This determins whether to return a value (if true) or a dict with Date as key and peak value as value (if true)
    equity_timeseries (pd.DataFrame): This is the timeseries of the periodic equity values
    Returns:
    Union[float, dict]: Corresponding date


    """
    ts = equity_timeseries
    peak_value = ts['Total'].max()
    peak_date = ts[ts['Total'] == peak_value].index[0]
    peak_dict = {peak_date: round(peak_value, 2)}
    return peak_value if value else peak_dict


def final_value_func(equity_timeseries: pd.DataFrame) -> float:
    """
    Returns the final value of the portfolio
    equity_timeseries (pd.DataFrame): This is the timeseries of the periodic equity values

    Returns:
    float


    """
    ts = equity_timeseries
    final_val = round(ts['Total'][-1], 2)
    return final_val


def rtrn(equity_timeseries: pd.DataFrame, use_col = 'Total', long = True) -> float:
    """
    Parameters:

    equity_timeseries (pd.DataFrame): This is the timeseries of the periodic equity values

    Returns:
    float: Returns returns of portfolio from initial date to final date
    """
    ts = equity_timeseries
    rtrn = (ts[use_col][-1]/ts[use_col][0])-1 if long else 1 - (ts[use_col][-1]/ts[use_col][0])
    return rtrn*100


def buyNhold(port_stats: dict) -> float:
    PortStats = port_stats
    initial_val = np.ones(len(PortStats)).sum()
    return_vals = np.zeros(len(PortStats))
    for i, (k, v) in enumerate(PortStats.items()):
        rtrn = v['Buy & Hold Return [%]']/100
        return_vals[i] = (1+rtrn)
    bNh_rtrn = round(((return_vals.sum()/initial_val)-1)*100, 2)
    return bNh_rtrn


def cagr(equity_timeseries: pd.DataFrame) -> float:
    """
    Parameters:
    equity_timeseries (pd.DataFrame): This is the timeseries of the periodic equity values


    Returns:
    float: Returns average annualize retruns for the portfolio. Cumulative Annual Growth Rate
    """
    ts = equity_timeseries
    begin_val = ts['Total'].iloc[0]
    end_val = ts['Total'].iloc[-1]
    if isinstance(ts.index, pd.DatetimeIndex):
        days = (ts.index.max() - ts.index.min()).days
    elif isinstance(ts.index, pd.RangeIndex):
        days = (ts.index.max() - ts.index.min())
    return ((end_val/begin_val)**(365/days) - 1)*100


def vol_annualized(equity_timeseries: pd.DataFrame, downside: Optional[bool] = False, MAR: Optional[Union[int, float]] = 0) -> float:
    """
    Returns the annualized volatility of the portfolio, which is calculated from the Portfolio Timeseries Value

    Parameters: 
    equity_timeseries (pd.DataFrame): Timeseries of the periodic equity values
    downside (Optional[bool]): False for regular volatility, True to calculate downside volatility
    MAR (Optional[Union[int, float]]): Minimum Acceptable Return

    Returns:
    float: Annualized Volatility
    """
    ts = equity_timeseries.fillna(0)
    annual_trading_days = 365
    ts_date_width = (ts.index.to_series().diff()).dt.days.mean()
    if not downside:
        return round(np.std(filter_zeros(ts['Total']).pct_change(), ddof=1) * np.sqrt(annual_trading_days/ts_date_width) * 100, 6)
    else:
        if not MAR:
            MAR = 0

        ts = ts['Total'].pct_change() - MAR
        ts_d = ts[ts < 0]
        return round(np.std(ts_d, ddof=1) * np.sqrt(252) * 100, 6)


def daily_rtrns(equity_timeseries: pd.DataFrame, long = True) -> pd.Series:
    """
    Parameters:
    equity_timeseries (pd.DataFrame): This is the timeseries of the periodic equity values

    Returns:
    pd.Series: Utility method. Returns timeseries of daily portfolio returns
    """
    ts = filter_zeros(equity_timeseries['Total']).pct_change()
    return ts if long else -ts


def sharpe(equity_timeseries: pd.DataFrame, risk_free_rate: float = 0.055, long = True) -> float:
    """
    Returns the Sharpe ratio of the portfolio
    Parameters: 
    risk_free_rate (float): A single value representing the risk free rate. This should be an annualized value
    equity_timeseries (pd.DataFrame): This is the timeseries of the periodic equity values

    Returns:
    float: Sharpe Ratio
    """

    # ANNUALIZED MEAN EXCESS RETURN / ANNUALIZED VOLATILITY
    annual_trading_days = 365
    ts_date_width = (equity_timeseries.index.to_series().diff()).dt.days.mean()
    annual_period = annual_trading_days/ts_date_width
    equity_timeseries = filter_zeros(equity_timeseries)
    daily_rfrate = (1+risk_free_rate)**(1/252) - 1
    annualized_vol = vol_annualized(equity_timeseries)/100
    excess_retrns = np.mean(daily_rtrns(equity_timeseries, long) - daily_rfrate)*annual_period
    return excess_retrns/annualized_vol


# YES
def sortino(equity_timeseries: pd.DataFrame, risk_free_rate: float, MAR: Optional[float] = None) -> float:
    """
    Returns the Sortino ratio of the portfolio

    Parameters: 
    risk_free_rate (float): A single value representing the risk free rate. This should be an annualized value
    MAR: Minimum Acceptable Return. A Value to compare with returns to ascertain true downside returns. Eg can be inflation rate, to show that real returns can be negative even though nominal is positive
    equity_timeseries (pd.DataFrame): Timeseries of the periodic equity values

    Returns:
    float: Sortino Ratio
    """

    # ANNUALIZED MEAN EXCESS RETURN / ANNUALIZED VOLATILITY
    if not MAR:
        MAR = (1+risk_free_rate)**(1/252) - 1
    daily_rfrate = (1+risk_free_rate)**(1/252) - 1
    annualized_vol = vol_annualized(equity_timeseries, True, MAR)/100
    excess_retrns = np.mean(daily_rtrns(equity_timeseries) - daily_rfrate)*252
    return excess_retrns/annualized_vol


def dd(equity_timeseries: pd.DataFrame, full: bool = False) -> Union[pd.DataFrame, pd.Series]:
    """
    Returns portfolio DrawDrown timeseires

    Parameters:
    full (bool): Whether to return timeseries for Drawdown Percent or full Dataframe containing Daily Value, Running Max & Percent Drawdown
    equity_timeseries (pd.DataFrame): Timeseries of the periodic equity values
    """
    ts = equity_timeseries
    data = pd.DataFrame()
    data['Total'] = ts['Total']
    data['Running_max'] = data.Total.cummax()
    data['dd'] = (data.Total/data.Running_max)-1
    if full:
        return data
    else:
        return data['dd']


def mdd(equity_timeseries: pd.DataFrame) -> float:
    """
    Parameters:
    equity_timeseries (pd.DataFrame): Timeseries of the periodic equity values

    Returns Max Drawdown
    """
    dd_ = dd(equity_timeseries)
    return dd_.min()*100


def calmar(equity_timeseries: pd.DataFrame) -> float:
    """
    params:
    equity_timeseries (pd.DataFrame): Timeseries of the periodic equity values


    Returns calmar Ratio
    """
    return abs(cagr(equity_timeseries)/mdd(equity_timeseries))


def avg_dd_percent(equity_timeseries: pd.DataFrame) -> float:
    """
    params:
    equity_timeseries (pd.DataFrame): Timeseries of the periodic equity values

    Returns avg Drawdown %
    """
    return round(dd(equity_timeseries).mean()*100, 6)


def mdd_value(equity_timeseries: pd.DataFrame) -> float:
    """
    params:
    equity_timeseries (pd.DataFrame): Timeseries of the periodic equity values

    Returns Maximum Drawdown value
    """
    dd_ = dd(equity_timeseries, True)
    return round((dd_['Total'] - dd_['Running_max']).min(), 2)


def mdd_duration(equity_timeseries: pd.DataFrame, full: bool = False) -> pd.Timedelta:
    """
    Max Draw Down Duration. The max time it took to return from a trough to a peak

    params:
    equity_timeseries (pd.DataFrame): Timeseries of the periodic equity values
    full (bool): True to return the whole dataframe, False to return the time delta

    Returns:
    maximum drawdown duration
    """
    from datetime import timedelta
    dd_ = dd(equity_timeseries, True)

    for i, (index, row) in enumerate(dd_.iterrows()):
        total, running_max, date = row['Total'], row['Running_max'], index
        running_max_date = dd_[dd_['Total'] == running_max].index[0]
        dd_.at[index, 'timedelta'] = (date - running_max_date)

    if full:
        return dd_
    else:
        return dd_.timedelta.max()


def avg_dd_duration(equity_timeseries: pd.DataFrame) -> pd.Timedelta:
    """

    params:
    equity_timeseries (pd.DataFrame): Timeseries of the periodic equity values

    Returns the average amount of time to return to peak from  trough
    """
    dd_duration = mdd_duration(equity_timeseries, True)
    return dd_duration.timedelta.mean()


def trades(port_stats) -> pd.DataFrame:
    """
    Returns a dataframe containing all trades taken

    """
    trades_df = pd.DataFrame()
    for k, v in port_stats.items():
        holder = v['_trades']
        holder['Ticker'] = k
        trades_df = pd.concat([trades_df, holder])
    return trades_df.sort_values(['EntryTime', 'ExitTime']).reset_index(drop = True)


def numOfTrades(trades_df) -> int:
    """
    params:
    trades_df (pd.DataFrame): DataFrame Contatining Trades
    """
    return len(trades_df)


def winRate(trades_df: pd.DataFrame) -> float:
    """
    params:
    trades_df (pd.DataFrame): DataFrame Contatining Trades
    """
    trades_ = trades_df
    return round(((trades_.ReturnPct > 0).sum()/(trades_.ReturnPct).count())*100, 2)


def lossRate(trades_df: pd.DataFrame) -> float:
    """
    params:
    trades_df (pd.DataFrame): DataFrame Contatining Trades
    """
    return round((100 - winRate(trades_df)), 2)


def avgPnL(trades_df: pd.DataFrame, Type_: str, value=True) -> float:
    """
    params:

    trades_df (pd.DataFrame): DataFrame Contatining Trades & PnL or ReturnPct column
    Type_ (str): 'W', 'L', 'A'. Win, Loss or All
    value (bool): True to return 
    """

    assert Type_.upper() in [
        'W', 'L', 'A'],  f"Invalid Type_: '{Type_}'. Must be 'L', 'W' or 'A."
    assert 'PnL' in trades_df.columns or 'ReturnPct' in trades_df.columns, f"Please pass a dataframe holding trades and ensure it has either 'PnL' or 'ReturnPct' in the columns. Current Columns {trades_df.columns}"
    trades_ = trades_df
    PnL = trades_.PnL if value else trades_.ReturnPct
    WPnL = PnL[PnL > 0] if Type_.upper() == 'W' else PnL[PnL <=
                                                         0] if Type_.upper() == 'L' else PnL

    return WPnL.mean() * 100 if not WPnL.empty else 0


def bestTrade(trades_df: pd.DataFrame) -> float:

    return trades_df.ReturnPct.max()*100


def worstTrade(trades_df) -> float:
    return trades_df.ReturnPct.min()*100


def profitFactor(trades_df: pd.DataFrame) -> float:
    """
    params:
    trades_df (pd.DataFrame): DataFrame Contatining Trades & PnL or ReturnPct column

    Returns the profit factor of the strategy. Synonymous to R/R
    """
    tr = trades_df
    tot_loss = tr[tr['ReturnPct'] <= 0]['PnL'].sum()
    tot_gain = tr[tr['ReturnPct'] > 0]['PnL'].sum()
    return round(abs(tot_gain/tot_loss), 6)


def Expectancy(trades_df: pd.DataFrame) -> float:
    """
    Returns the expected %pnl based on portfolio data
    """
    tr = trades_df
    return (avgPnL(tr, 'W', False) * (winRate(tr)/100)) + (avgPnL(tr, 'L', False) * (lossRate(tr)/100))


def SQN(trades_df: pd.DataFrame) -> float:
    """
    params:
    trades_df (pd.DataFrame): DataFrame Contatining Trades & PnL or ReturnPct column

    System Quality Number. Used to guage how good a system is
    """
    trades_ = trades_df
    return ((trades_.ReturnPct.mean() * np.sqrt(len(trades_)))/np.std(trades_.ReturnPct))


def ExposureDays(equity_timeseries: pd.DataFrame, trades_df: pd.DataFrame) -> float:
    """
    params:
    trades_df (pd.DataFrame): DataFrame Contatining Trades & PnL or ReturnPct column
    equity_timeseries (pd.DataFrame): Timeseries of daily equity, with dates as index


    Returns the percent of days the portfolio had exposure
    """
    time_in = pd.DataFrame(index=equity_timeseries.index)
    time_in['position'] = 0
    tr = trades_df
    tr.dropna(subset=['EntryTime', 'ExitTime'], inplace=True)
    for index, row in tr.iterrows():
        entry = pd.to_datetime(row['EntryTime']).date()
        exit_ = pd.to_datetime(row['ExitTime']).date()
        time_in['position'].loc[(time_in.index.date >= entry)
                                & (time_in.index.date <= exit_)] = 1


    return round((time_in['position'] == 1).sum()/len(time_in)*100, 2)


def yearly_retrns(equity_timeseries: pd.DataFrame) -> dict:
    """
    params:
    equity_timeseries (pd.DataFrame): Timeseries of daily equity, with dates as index


    Returns yearly returns as a dict
    """

    ts = equity_timeseries
    ts['Year'] = ts.index.year
    ts.drop_duplicates(inplace=True)
    unq_year = ts.Year.unique()
    rtrn_d = {}
    for year in unq_year:
        data = ts[ts['Year'] == year].sort_index()
        ret = ((data.loc[data.index.max(), 'Total'] /
               data.loc[data.index.min(), 'Total'])-1)*100
        rtrn_d[year] = ret
    return rtrn_d


def holding_period(trades_df: pd.DataFrame, aggfunc: Callable, Type_: str = 'W') -> pd.Timedelta:
    """ 
    Returns the average or max holding period of the portfolio based on Trades data

    Parameters:
    aggfunc (Callable): A callable that performs an arithmetic calculation. Eg: Mean, Median, Mode, Max, Min 
    trades_df (pd.DataFrame): DataFrame Contatining Trades & PnL or ReturnPct column

    Type_ (float): Which holding period are we looking for. Available options are
        'W': For winning holding period
        'L': For losing holding period
        'A': For Holding period based on all data


    Returns:
    pd.Timedelta: Corresponding Value
    """
    assert aggfunc.__name__.lower() in [
        'mean', 'max'], f"Function of type '{aggfunc.__name__}' cannot be used for this method. Please use a mean or max function"
    assert Type_.upper() in [
        'A', 'W', 'L'], f"Invalid type: {Type_}. Must be 'L', 'W' or 'A."
    # assert aggfunc.upper() in ['A', 'M'], f"Invalid type: {aggfunc}. Must be 'M' for Max or 'A' for Avg."
    trades_ = trades_df
    if Type_.upper() == 'W':
        trades_ = trades_[trades_['ReturnPct'] > 0]
    elif Type_.upper() == 'L':
        trades_ = trades_[trades_['ReturnPct'] <= 0]

    # return trades_.Duration.mean() if aggfunc.upper() == 'A' else trades_.Duration.max()
    return aggfunc(trades_.Duration)


def streak(trades_df: pd.DataFrame, Type_: str = 'W') -> int:
    """ 
    Returns the Losing/Winning Streak based on Trades data

    Parameters:
    Type_ (flaot): Which holding period are we looking for. Available options are
        'W': For winning holding period
        'L': For losing holding period

    trades_df (pd.DataFrame): DataFrame Contatining Trades & PnL or ReturnPct column


    Returns:
    int: Corresponding Value
    """
    assert Type_.upper() in [
        'L', 'W'], f"Invalid type: {Type_}. Must be 'L' or 'W'."
    t = trades_df
    t['Is_Loss'] = int
    t['Is_Loss'] = t['ReturnPct'] <= 0 if Type_.upper() == 'L' else t['ReturnPct'] > 0
    t['Loss_Streak'] = (t['Is_Loss'] != t['Is_Loss'].shift()).cumsum()
    streak_lengths = t.groupby('Loss_Streak')['Is_Loss'].sum()
    return streak_lengths[t.groupby('Loss_Streak')['Is_Loss'].first()].max()


class AggregatorParent(ABC):

    def __init__(self):
        self._equity = None
        self.__port_stats = None
        self._trades = None

    @copy_doc_from(dates_)
    def dates_(self, start: bool):
        try:
            overwrite = pd.to_datetime(getattr(self, 'start_overwrite')).date()
        except AttributeError:
            overwrite = None

        if overwrite:
            return overwrite if start else dates_(self.get_port_stats(), start).date()
        else:
            return dates_(self.get_port_stats(), start).date() if start else dates_(self.get_port_stats(), start).date()


    @abstractmethod
    def get_port_stats(self):
        """
        Abstract method to get the port_stats attribute.
        Any subclass must implement this method.
        """
        pass

    def peak_value_func(self, value: bool = True) -> Union[float, Dict]:
        """
        Returns the peak value of the portfolio and has the option to return corresponding date

        Parameters:
        value (bool): This determins whether to return a value (if true) or a dict with Date as key and peak value as value (if true)
        Returns:
        Union[float, dict]: Corresponding date


        """

        return peak_value_func(self._equity, value)

    def final_value_func(self) -> float:
        """
        Returns the final value of the portfolio

        Returns:
        float
        """

        return final_value_func(self._equity)

    def rtrn(self) -> float:
        """
        Returns:
        float: Returns returns of portfolio from initial date to final date
        """
        assert self._equity is not None, f'Portfolio Equity is empty'
        return rtrn(self._equity)

    def buyNhold(self) -> float:
        return buyNhold(self.get_port_stats())

    def cagr(self) -> float:
        """
        Returns:
        float: Returns average annualize retruns for the portfolio. Cumulative Annual Growth Rate
        """
        return cagr(self._equity)

    def vol_annualized(self, downside: Optional[bool] = False, MAR: Optional[Union[int, float]] = 0) -> float:
        """
        Returns the annualized volatility of the portfolio, which is calculated from the Portfolio Timeseries Value

        Parameters: 
        downside (Optional[bool]): False for regular volatility, True to calculate downside volatility
        MAR (Optional[Union[int, float]]): Minimum Acceptable Return

        Returns:
        float: Annualized Volatility
        """
        return vol_annualized(self._equity, downside, MAR)

    def daily_rtrns(self) -> pd.Series:
        """

        Returns:
        pd.Series: Utility method. Returns timeseries of daily portfolio returns
        """

        return daily_rtrns(self._equity)

    def sharpe(self, risk_free_rate: float = 0.055) -> float:
        """
        Returns the Sharpe ratio of the portfolio
        Parameters: 
        risk_free_rate (float): A single value representing the risk free rate. This should be an annualized value
        equity_timeseries (pd.DataFrame): This is the timeseries of the periodic equity values

        Returns:
        float: Sharpe Ratio
        """

        # ANNUALIZED MEAN EXCESS RETURN / ANNUALIZED VOLATILITY
        return sharpe(self._equity, risk_free_rate)

    # YES
    def sortino(self, risk_free_rate: float, MAR: Optional[float] = None) -> float:
        """
        Returns the Sortino ratio of the portfolio

        Parameters: 
        risk_free_rate (float): A single value representing the risk free rate. This should be an annualized value
        MAR: Minimum Acceptable Return. A Value to compare with returns to ascertain true downside returns. Eg can be inflation rate, to show that real returns can be negative even though nominal is positive

        Returns:
        float: Sharpe Ratio
        """

        # ANNUALIZED MEAN EXCESS RETURN / ANNUALIZED VOLATILITY
        return sortino(self._equity, risk_free_rate, MAR)

    def dd(self, full: bool = False) -> Union[pd.DataFrame, pd.Series]:
        """
        Returns portfolio DrawDrown timeseires

        Parameters:
        full (bool): Whether to return timeseries for Drawdown Percent or full Dataframe containing Daily Value, Running Max & Percent Drawdown
        """

        if full:
            return dd(self._equity, full)
        else:
            return dd(self._equity, full)

    def mdd(self) -> float:
        """
        Returns Max Drawdown
        """
        return mdd(self._equity)

    def calmar(self) -> float:
        """
        Returns calmar Ratio
        """
        return calmar(self._equity)

    def avg_dd_percent(self) -> float:
        """
        Returns avg Drawdown %
        """
        return avg_dd_percent(self._equity)

    def mdd_value(self) -> float:
        """
        Returns Maximum Drawdown value
        """
        return mdd_value(self._equity)

    def mdd_duration(self, full: bool = False) -> pd.Timedelta:
        """
        Max Draw Down Duration. The max time it took to return from a trough to a peak

        params:
        full (bool): True to return the whole dataframe, False to return the time delta

        Returns:
        maximum drawdown duration
        """
        from datetime import timedelta
        return mdd_duration(self._equity, full)

    def avg_dd_duration(self) -> pd.Timedelta:
        """
        Returns the average amount of time to return to peak from  trough
        """
        return avg_dd_duration(self._equity)

    def trades(self) -> pd.DataFrame:
        """
        Returns a dataframe containing all trades taken

        """
        return trades(self.get_port_stats())

    def numOfTrades(self) -> int:
        """
        params:
        """
        return numOfTrades(self._trades)

    def winRate(self) -> float:
        return winRate(self._trades)

    def lossRate(self) -> float:
        return round((100 - winRate(self._trades)), 2)

    def avgPnL(self, Type_: str, value=True) -> float:
        """
        params:

        Type_ (str): 'W', 'L', 'A'. Win, Loss or All
        value (bool): True to return 
        """
        return avgPnL(self._trades, Type_, value)

    def bestTrade(self) -> float:

        return bestTrade(self._trades)

    def worstTrade(self) -> float:
        return worstTrade(self._trades)

    def profitFactor(self) -> float:
        """
        Returns the profit factor of the strategy. Synonymous to R/R
        """

        return profitFactor(self._trades)

    def Expectancy(self) -> float:
        """
        Returns the expected %pnl based on portfolio data
        """

        return Expectancy(self._trades)

    def SQN(self) -> float:
        """
        System Quality Number. Used to guage how good a system is
        """
        return SQN(self._trades)

    def ExposureDays(self) -> float:
        """
        Returns the percent of days the portfolio had exposure
        """

        return ExposureDays(self._equity, self._trades)

    def yearly_retrns(self) -> dict:
        """
        Returns yearly returns as a dict
        """

        return yearly_retrns(self._equity)

    def holding_period(self, aggfunc: Callable, Type_: str = 'W') -> pd.Timedelta:
        """ 
        Returns the average or max holding period of the portfolio based on Trades data

        Parameters:
        aggfunc (Callable): A callable that performs an arithmetic calculation. Eg: Mean, Median, Mode, Max, Min 

        Type_ (float): Which holding period are we looking for. Available options are
            'W': For winning holding period
            'L': For losing holding period
            'A': For Holding period based on all data


        Returns:
        pd.Timedelta: Corresponding Value
        """

        return holding_period(self._trades, aggfunc, Type_)

    def streak(self, Type_: str = 'W') -> int:
        """ 
        Returns the Losing/Winning Streak based on Trades data

        Parameters:
        Type_ (flaot): Which holding period are we looking for. Available options are
            'W': For winning holding period
            'L': For losing holding period

        Returns:
        int: Corresponding Value
        """
        return streak(self._trades, Type_)

## FIXME: This is ridiculously slow.
    def aggregate(self,
                  risk_free_rate: float = 0.0,
                  MAR: float = 0) -> pd.Series:
        """ 

        Returns aggregated Data for the Porftolio

        Parameters:

        PortStats (pd.Series): The Stats data obtained from backtesting.py simulation
        risk_free_rate (float): Current Risk Free Rate Value (Annual Rate, not daily)
        Returns:
        int: Corresponding Value
        """
        assert self.get_port_stats(), f"Run Portfolio Backtest before aggregating"
        MAR = 0.0 if not MAR else MAR
        
        ## Extending to ensure useability in other places. It was originally designed for PTBacktest,
        ## But can be used in other places

        ## Re-implement dates_. This is very specific to PTBacktest


        assert isinstance(
            MAR, float), f"Recieved MAR of type {type(MAR)} instead of Type float"
        
        try:
            start_overwrite = getattr(self, 'start_overwrite')
        except AttributeError:
            start_overwrite = None

        try:
            strategy = list(self.get_port_stats().values())[0]['_strategy']
        except AttributeError:
            strategy = None

        try: 
            equity = self.pf_value_ts()
        except AttributeError:
            equity = self._equity
        except Exception:
            raise Exception('Either implement pf_value_ts method or self.equity')

        try:
            ## Function call for trades
            trades = self.trades()
        except TypeError:
            ## If trades is not implemented, we can use the self._trades attribute
            trades = self._trades
        except Exception:
            raise Exception('Either implement trades method or self._trades')

        try:
            tickers = [dataset.name for dataset in self.datasets]
        except AttributeError:
            tickers = self.symbol_list
        except Exception:
            raise Exception('Either implement datasets attribute with PTDataset or self.symbol_list')

        rtrn_ = self.rtrn()
        series1 = pd.Series({
            'Start': start_overwrite if start_overwrite else self.dates_(True),
            'End': self.dates_(False),
            'Duration': self.dates_(False) - self.dates_(True),
            'Exposure Time [%]': self.ExposureDays(),
            'Equity Final [$]': self.final_value_func(),
            'Equity Peak [$]': self.peak_value_func(),
            'Return [%]': rtrn_,
            'Buy & Hold Return [%]': self.buyNhold(),
            'CAGR [%]':  self.cagr(),
            'Volatility Ann. [%]': self.vol_annualized(),
            'Sharpe Ratio': self.sharpe(risk_free_rate),
            'Sortino Ratio': self.sortino(risk_free_rate, MAR),
            'Skew': self._equity.Total.pct_change().skew(),
            'Calmar Ratio': self.calmar(),
            'Max. Drawdown [%]': self.mdd(),
            'Max. Drawdown Value [$]': self.mdd_value(),
            'Avg. Drawdown [%]': self.avg_dd_percent(),
            'Max. Drawdown Duration': self.mdd_duration(),
            'Avg Dradown Duration': self.avg_dd_duration(),
            '# Trades': self.numOfTrades(),
            'Win Rate [%]': self.winRate(),
            'Lose Rate [%]': self.lossRate(),
            'Avg. Trade [%]': self.avgPnL('A', False),
            'Avg. Winning Trade [%]': self.avgPnL('W', False),
            'Avg. Losing Trade [%]': self.avgPnL('L', False),
            'Best Trade [%]': self.bestTrade(),
            'Worst Trade [%]': self.worstTrade(),
            'Avg Trade Duration': self.holding_period(np.mean, 'A'),
            'Avg Win Trade Duration': self.holding_period(np.mean, 'W'),
            'Avg Lose Duration': self.holding_period(np.mean, 'L'),
            'Max Trade Duration': self.holding_period(np.max, 'A'),
            'Max Win Trade Duration': self.holding_period(np.max, 'W'),
            'Max Lose Duration': self.holding_period(np.max, 'L'),
            'Profit Factor': self.profitFactor(),
            'Expectancy [%]': self.Expectancy(),
            'SQN': self.SQN()

        })

        rtrn_dict = self.yearly_retrns()
        rtrn_series = pd.Series(
            {f"{year} Return [%]": value for year, value in rtrn_dict.items()})

        series3 = pd.Series({
            'Winning Streak': self.streak('W'),
            'Losing Streak': self.streak('L'),
            '_strategy': strategy,
            'equity_curve': equity,
            '_trades': trades,
            '_tickers': tickers

        })
        return pd.concat([series1, rtrn_series, series3])