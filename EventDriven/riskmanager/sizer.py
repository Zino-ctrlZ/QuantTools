from __future__ import annotations
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any
from trade.helpers.helper import CustomCache, setup_logger
from typing import TYPE_CHECKING
from EventDriven.riskmanager.utils import parse_signal_id
from trade.assets.helpers.utils import swap_ticker
import math
import pandas as pd
from dateutil.relativedelta import relativedelta
from pandas.tseries.offsets import BDay
import numpy as np
import os
from pathlib import Path


if TYPE_CHECKING:
    from EventDriven.riskmanager.base import RiskManager
    from EventDriven.portfolio import OptionSignalPortfolio


logger = setup_logger('QuantTools.EventDriven.riskmanager.sizer')
BASE = Path(os.environ["WORK_DIR"])/ ".riskmanager_cache"

class BaseSizer(ABC):
    """
    Abstract base class for size calculation.

    This class provides a framework for calculating position sizes based on available cash and other parameters.

    Attributes:
    CASH_USE_RULES (dict): A dictionary defining the rules for cash base when calculating position sizes and greek limits.
    """
    __CASH_USE_RULES = {
        1: 'USE_CASH_EVERY_NEW_POSITION_ID',
        2: 'USE_CASH_EVERY_NEW_TICK',
        3: 'USE_CASH_EVERY_NEW_SIGNAL_ID',
    }

    def __init__(self, 
                 pm: OptionSignalPortfolio, 
                 rm: RiskManager, 
                 sizing_lev=1.0):
        """
        Initialize the BaseSizer with a PowerManager and ResourceManager.
        Args:
            pm (OptionSignalPortfolio): The PowerManager instance managing the portfolio.
            rm (RiskManager): The ResourceManager instance managing resources.
            sizing_lev (float): The sizing level for position sizing. Default is 1.0.
        """
        self.pm = pm
        self.rm = rm
        self.sizing_lev = sizing_lev
        self.signal_starting_cash = {}
        self.position_id_starting_cash = {} ## This is helpful to track the starting cash for each position ID, especially when calculating limits
        self.ticker_starting_cash = pm.allocated_cash_map.copy()
        self.cash_rule = self.__CASH_USE_RULES[1]
        self.re_update_on_roll = True  # Flag to control re-updating on roll
        self.delta_limit_log = {}
    
    @abstractmethod
    def update_delta_limit(self, *args, **kwargs):
        """
        Abstract method to calculate the delta limit for a given position ID and date.
        """
        pass

    @abstractmethod
    def calculate_position_size(self, *args, **kwargs):
        """
        Abstract method to calculate the position size for a given position ID and date.
        """
        pass

    @abstractmethod
    def pre_analyze_task(self, *args, **kwargs):
        """
        Abstract method for daily updates.
        """
        pass

    @abstractmethod
    def post_analyze_task(self, *args, **kwargs):
        """
        Abstract method for daily updates.
        """
        pass

    def get_cash(self, tick:str, signal_id:str, position_id:str) -> float:
        """
        Get the available cash for a given position ID and date based on the cash rule.
        This method determines the cash available for position sizing based on the specified cash rule.
        Args:
            tick (str): The ticker symbol for the asset.
            signal_id (str): The ID of the signal associated with the position.
            position_id (str): The ID of the position.
        """
        if self.cash_rule == self.__CASH_USE_RULES[1]:
            return self.position_id_starting_cash.get(position_id, self.pm.allocated_cash_map[tick])
        elif self.cash_rule == self.__CASH_USE_RULES[2]:
            return self.ticker_starting_cash[tick]
        elif self.cash_rule == self.__CASH_USE_RULES[3]:
            return self.signal_starting_cash.get(signal_id, self.pm.allocated_cash_map[tick])
        else:
            raise ValueError(f"Unknown cash rule: {self.cash_rule}")
    
    def set_cash_rule(self, rule: int) -> None:
        """
        Set the cash rule for position sizing.
        Args:
            rule (int): The cash rule to set. Must be one of the keys in CASH_USE_RULES.
        """
        if rule in self.__CASH_USE_RULES:
            self.cash_rule = self.__CASH_USE_RULES[rule]
        else:
            raise ValueError(f"Invalid cash rule: {rule}. Available rules: {self.__CASH_USE_RULES.keys()}")
    
    def register_signal_starting_cash(self, signal_id:str, cash:float) -> None:
        """
        Register the starting cash for a specific signal ID.
        Args:
            signal_id (str): The ID of the signal.
            cash (float): The starting cash amount for the signal.
        """
        if signal_id in self.signal_starting_cash:
            logger.warning(f"Signal ID {signal_id} already has starting cash registered. Overwriting previous value.")
            return
        self.signal_starting_cash[signal_id] = cash

    def register_position_id_starting_cash(self, position_id:str, cash:float) -> None:
        """
        Register the starting cash for a specific position ID.
        Args:
            position_id (str): The ID of the position.
            cash (float): The starting cash amount for the position.
        """
        if position_id in self.position_id_starting_cash:
            logger.warning(f"Position ID {position_id} already has starting cash registered. Overwriting previous value.")
            return
        self.position_id_starting_cash[position_id] = cash

    def log_daily_delta_limit(self, symbol:str, date:datetime, delta_limit:float) -> None:
        """
        Log the daily delta limit for a specific symbol and date.
        Args:
            symbol (str): The ticker symbol for the asset.
            date (datetime): The date for which the delta limit is logged.
            delta_limit (float): The delta limit value to log.
        """
        if symbol not in self.delta_limit_log:
            self.delta_limit_log[symbol] = {}
        self.delta_limit_log[symbol][date] = delta_limit


    @property
    def CASH_USE_RULES(self):
        """
        Get the available cash use rules.
        """
        return self.__CASH_USE_RULES
    
    @CASH_USE_RULES.setter
    def CASH_USE_RULES(self, value):
        """
        Set the available cash use rules.
        """
        raise AttributeError("CASH_USE_RULES is a read-only property.")

    def __repr__(self):
        """
        String representation of the BaseSizer class.
        """
        return f"{self.__class__.__name__}(sizing_lev={self.sizing_lev}, cash_rule={self.cash_rule})"


class DefaultSizer(BaseSizer):
    """
    Default implementation of the BaseSizer.
    Calculates position size based on delta limits and available cash.
    """

    def __init__(self, pm, rm, sizing_lev):
        super().__init__(pm, rm, sizing_lev)
        
    def get_daily_delta_limit(self, signal_id:str, position_id:str, date:str|datetime) -> float:
        """
        Returns the delta limit for a given signal ID and date.
        This method retrieves the delta limit for a specific signal ID and date, 
        calculating it if it doesn't already exist in the risk manager's greek limits.
        
        Args:
            signal_id (str): The ID of the signal.
            position_id (str): The ID of the position.
            date (str|datetime): The date for which the delta limit is calculated.
        
        Returns:
            float: The delta limit for the specified signal ID and date.
        """
 
        logger.info(f"DefaultSizer: Calculating Delta Limit for Signal ID: {signal_id} and Position ID: {position_id} on Date: {date}\n")
        id_details = parse_signal_id(signal_id)
        self.register_signal_starting_cash(signal_id, self.pm.allocated_cash_map[id_details['ticker']])  ## Register the starting cash for the signal
        self.register_position_id_starting_cash(position_id, self.pm.allocated_cash_map[id_details['ticker']])  ## Register the starting cash for the position ID


        logger.info(f"Updating Greek Limits for Signal ID: {signal_id} and Position ID: {position_id}")
        starting_cash = self.get_cash(id_details['ticker'], signal_id, position_id)  ## Get the cash available for the ticker based on the cash rule
        logger.info(f"Starting Cash for {id_details['ticker']} on {date}: {starting_cash} vs Current Cash: {self.pm.allocated_cash_map[id_details['ticker']]}")
        delta_at_purchase = self.rm.position_data[position_id]['Delta'][date]  ## This is the delta at the time of purchase
        s0_at_purchase = self.rm.position_data[position_id]['s'][date]  ## This is the delta at the time of purchase## As always, we use the chain spot data to account for splits
        equivalent_delta_size = (starting_cash * self.sizing_lev) / (s0_at_purchase * 100) if s0_at_purchase != 0 else 0
        logger.info(f"Spot Price at Purchase: {s0_at_purchase} at time {date}  ## This is the delta at the time of purchase")
        logger.info(f"Delta at Purchase: {delta_at_purchase}")
        logger.info(f"Equivalent Delta Size: {equivalent_delta_size}, with Cash Available: {starting_cash}, and Leverage: {self.sizing_lev}")
        return equivalent_delta_size
        


    def update_delta_limit(self, signal_id:str, position_id:str, date:str|datetime) -> float:
        """
        Updates the limits associated with a signal
        ps: This should only be updated on first purchase of the signal
            Limits are saved in absolute values to account for both long and short positions
        
        Limit Calc: (allocated_cash * lev)/(S0 * 100)
        """
        equivalent_delta_size = self.get_daily_delta_limit(signal_id, position_id, date)  ## Get the delta limit for the signal ID and date
        self.rm.greek_limits['delta'][signal_id] = equivalent_delta_size  ## Update the delta limit in the risk manager's greek limits
        return equivalent_delta_size

    def calculate_position_size(self, signal_id:str, position_id:str, opt_price:str, date:str|datetime) -> float:
        """
        Returns the quantity of the position that can be bought based on the sizing type
        Args:
            signal_id (str): The ID of the signal.
            position_id (str): The ID of the position.
            opt_price (float): The price of the option.
            date (str|datetime): The date for which the position size is calculated.
        """
        self.update_delta_limit(signal_id,position_id, date) ## Always calculate the delta limit first
        logger.info(f"Calculating Quantity for Position ID: {position_id} and Signal ID: {signal_id} on Date: {date}")
        if position_id not in self.rm.position_data: ## If the position data isn't available, calculate the greeks
            self.rm.calculate_position_greeks(position_id, date)
        
        ## First get position info and ticker
        position_dict, _ = self.rm.parse_position_id(position_id)
        key = list(position_dict.keys())[0]
        ticker = swap_ticker(position_dict[key][0]['ticker'])

        ## Now calculate the max size cash can buy
        cash_available = self.pm.allocated_cash_map[ticker]
        purchase_date = pd.to_datetime(date)
        s0_at_purchase = self.rm.position_data[position_id]['s'][purchase_date]  ## s -> chain spot, s0_close -> adjusted close
        logger.info(f"Spot Price at Purchase: {s0_at_purchase} at time {purchase_date}")
        logger.info(f"Cash Available: {cash_available}, Option Price: {opt_price}, Cash_Available/OptPRice: {(cash_available/(opt_price*100))}")
        max_size_cash_can_buy = abs(math.floor(cash_available/(opt_price*100))) ## Assuming Allocated Cash map is already in 100s
          
        delta = self.rm.position_data[position_id]['Delta'][purchase_date]
        target_delta = self.rm.greek_limits['delta'][signal_id]
        logger.info(f"Target Delta: {target_delta}")
        delta_size = (math.floor(target_delta/abs(delta)))
        logger.info(f"Delta from Full Cash Spend: {max_size_cash_can_buy * delta}, Size: {max_size_cash_can_buy}")
        logger.info(f"Delta with Size Limit: {delta_size * delta}, Size: {delta_size}")
        return delta_size if abs(delta_size) <= abs(max_size_cash_can_buy) else max_size_cash_can_buy
    
    def daily_update(self):
        """
        Daily update method to refresh the position data and limits.
        """
        pass

    def pre_analyze_task(self, *args, **kwargs):
        """
        Abstract method for daily updates.
        """
        pass

    def post_analyze_task(self, *args, **kwargs):
        """
        Abstract method for daily updates.
        """
        pass






class ZscoreRVolSizer(BaseSizer):
    """
    Sizer that calculates position size based on the zscore of the realized volatility.
    """
    VOL_TYPES = ['mean', 'weighted_mean', 'window']

    def __init__(self, 
                 pm: OptionSignalPortfolio, 
                 rm: RiskManager, 
                 sizing_lev:float=1.0, 
                 rvol_window:int|tuple=None, 
                 rolling_window:int=100,
                 weights = (0.5, 0.3, 0.2),
                 vol_type:str='mean',
                 *args, **kwargs):
        """
        Initialize the ZscoreRVolSizer with a PowerManager and ResourceManager.
        Args:
            pm (OptionSignalPortfolio): The PowerManager instance managing the portfolio.
            rm (RiskManager): The ResourceManager instance managing resources.
            sizing_lev (float): The sizing level for position sizing. Default is 1.0.
            rvol_window (int): The window size for realized volatility calculation. Default is 30 days.
            rolling_window (int): The window size for rolling calculations. Default is 100 days.
            weights (tuple): Weights for the weighted mean calculation. Default is (0.5, 0.3, 0.2). Corresponding to 5D, 20D, and 63D realized volatility respectively.
        """
        super().__init__(pm, rm, sizing_lev)
        
        if isinstance(weights, (list)): weights = tuple(weights)  ## Ensure weights is a tuple
        if isinstance(rvol_window, (list)): rvol_window = tuple(rvol_window)  ## Ensure rvol_window is a tuple if provided as a list
        
        rvol_window = self.__rvol_window_assert(vol_type, rvol_window)  ## Assert that the rvol_window is valid based on the vol_type
        assert vol_type in self.VOL_TYPES, f"vol_type must be one of {self.VOL_TYPES}, got {self.vol_type}"
        assert isinstance(weights, tuple) and len(weights) == 3, "weights must be a tuple of length 3"
        assert sum(weights) == 1, "weights must sum to 1"
        assert rolling_window > 0, "rolling_window must be a positive integer"

        self.__rvol_window = rvol_window
        self.__rolling_window = rolling_window
        self.rvol_timeseries = {} 
        self.z_i = {}
        self.scaler = {}
        self.vol_type = vol_type
        self.norm_constant = kwargs.get('norm_constant', 1.0)  ## Normalization constant for the scaler

        self.weights = weights  ## Weights for the weighted mean calculation

    def __rvol_window_assert(self, vol_type:str, rvol_window:int|tuple=None) -> int|tuple:
        """
        Assert that the rvol_window is a positive integer or a tuple of three integers.
        """
        if rvol_window is None: ## If rvol_window is not provided, set it to default
            rvol_window = 30 if vol_type == 'window' else (5, 20, 63)

        if isinstance(rvol_window, int): ## If rvol_window is an int, it must be a positive integer and vol_type must be 'window'
            assert vol_type == 'window', "rvol_window must be an int only when vol_type is 'window'."
            assert rvol_window > 0, "rvol_window must be a positive integer."

        elif isinstance(rvol_window, tuple): ## If rvol_window is a tuple, it must be of length 3 and vol_type must be 'weighted_mean' or 'mean'
            assert vol_type == 'weighted_mean' or vol_type == 'mean', "rvol_window must be a tuple only when vol_type is 'weighted_mean' or 'mean'."
            assert len(rvol_window) == 3, "rvol_window must be a tuple of length 3 for weighted mean calculation."

        return rvol_window

    @property
    def rvol_window(self):
        return self.__rvol_window
    
    @rvol_window.setter
    def rvol_window(self, value):
        self.__rvol_window_assert(self.vol_type, value)  ## Assert that the rvol_window is valid based on the vol_type
        
        if value != self.__rvol_window: ## Clear the cache if the window changes
            logger.info(f"Setting realized volatility window to {value} days")
            self.rvol_timeseries = {}  # Clear the cache if the window changes
        self.__rvol_window = value

    @property
    def rolling_window(self):
        return self.__rolling_window

    @rolling_window.setter
    def rolling_window(self, value):
        if not isinstance(value, int) or value <= 0:
            raise ValueError("rolling_window must be a positive integer.")
        self.__rolling_window = value

    def get_daily_delta_limit(self, signal_id:str, position_id:str, date:str|datetime) -> float:
        """
        Calculate the delta limit based on the percentile of the realized volatility.
        This method is called to get the delta limit for a given signal and position.
        Args:
            signal_id (str): The ID of the signal.
            position_id (str): The ID of the position.
            date (str|datetime): The date for which the delta limit is calculated.

        Returns:
            float: The scaled delta size based on the realized volatility scaler.

        Limit Calc: (allocated_cash * lev)/(S0 * 100) * 1/(1+Zscore(rvol(30), window))
        """

        logger.info(f"ZscoreRVolSizer: Calculating Delta Limit for Signal ID: {signal_id} and Position ID: {position_id} on Date: {date}\n")
        id_details = parse_signal_id(signal_id)
        symbol = swap_ticker(id_details['ticker'])
        
        ## Check if the scaler for the symbol is already calculated
        if symbol not in self.scaler:
            self.calculate_scaler(symbol)

        ## Get the scaler for the symbol and date
        scaler = self.scaler[symbol][date]

        id_details = parse_signal_id(signal_id)
        starting_cash = self.get_cash(id_details['ticker'], signal_id, position_id) 
        logger.info(f"Starting Cash for {id_details['ticker']} on {date}: {starting_cash} vs Current Cash: {self.pm.allocated_cash_map[id_details['ticker']]}")
        delta_at_purchase = self.rm.position_data[position_id]['Delta'][date]  
        s0_at_purchase = self.rm.position_data[position_id]['s'][date]  
        equivalent_delta_size = ((starting_cash * self.sizing_lev) / (s0_at_purchase * 100)) if s0_at_purchase != 0 else 0
        scaled_delta_size = equivalent_delta_size * scaler
        logger.info(f"Scaler for {symbol} on {date}: {scaler}")
        logger.info(f"Spot Price at Purchase: {s0_at_purchase} at time {date}  ## This is the delta at the time of purchase")
        logger.info(f"Delta at Purchase: {delta_at_purchase}")
        logger.info(f"Equivalent Delta Size: {equivalent_delta_size}, with Cash Available: {starting_cash}, Leverage: {self.sizing_lev} and Scaler: {scaler}")
        logger.info(f"Scaled Delta Size: {scaled_delta_size}")
        return scaled_delta_size
    
    def update_delta_limit(self, signal_id:str, position_id:str, date:str|datetime) -> float:
        """
        Calculate the delta limit based on the percentile of the realized volatility.
        This method is called to update the delta limit for a given signal and position.
        Args:
            signal_id (str): The ID of the signal.
            position_id (str): The ID of the position.
            date (str|datetime): The date for which the delta limit is calculated.

        This method checks if the delta limit for the signal ID is already updated. If not, it calculates the delta limit based on the realized volatility and updates the risk manager's greek limits.
        """

        id_details = parse_signal_id(signal_id)
        symbol = swap_ticker(id_details['ticker'])
        self.register_signal_starting_cash(signal_id, self.pm.allocated_cash_map[symbol])  ## Register the starting cash for the signal
        self.register_position_id_starting_cash(position_id, self.pm.allocated_cash_map[symbol])  ## Register the starting cash for the position ID
        logger.info(f"Updating Greek Limits for Signal ID: {signal_id} and Position ID: {position_id}")
        scaled_delta_size = self.get_daily_delta_limit(signal_id, position_id, date)
        self.rm.greek_limits['delta'][signal_id] = abs(scaled_delta_size)
        return scaled_delta_size

    def calculate_position_size(self, signal_id:str, position_id:str, opt_price:float, date:str|datetime) -> float:
        """
        Calculate the position size based on the percentile of the realized volatility.
        This method calculates the position size for a given signal ID and position ID based on the available cash and the delta limit.
        Args:
            signal_id (str): The ID of the signal.
            position_id (str): The ID of the position.
            opt_price (float): The price of the option.
            date (str|datetime): The date for which the position size is calculated.


        """
        self.update_delta_limit(signal_id,position_id, date) ## Always calculate the delta limit first
        logger.info(f"Calculating Quantity for Position ID: {position_id} and Signal ID: {signal_id} on Date: {date}")
        if position_id not in self.rm.position_data: ## If the position data isn't available, calculate the greeks
            self.rm.calculate_position_greeks(position_id, date)
        
        ## First get position info and ticker
        position_dict, _ = self.rm.parse_position_id(position_id)
        key = list(position_dict.keys())[0]
        ticker = swap_ticker(position_dict[key][0]['ticker'])

        ## Now calculate the max size cash can buy
        cash_available = self.pm.allocated_cash_map[ticker]
        purchase_date = pd.to_datetime(date)
        s0_at_purchase = self.rm.position_data[position_id]['s'][purchase_date]  ## s -> chain spot, s0_close -> adjusted close
        logger.info(f"Spot Price at Purchase: {s0_at_purchase} at time {purchase_date}")
        logger.info(f"Cash Available: {cash_available}, Option Price: {opt_price}, Cash_Available/OptPRice: {(cash_available/(opt_price*100))}")
        max_size_cash_can_buy = abs(math.floor(cash_available/(opt_price*100))) ## Assuming Allocated Cash map is already in 100s

        delta = self.rm.position_data[position_id]['Delta'][purchase_date]
        target_delta = self.rm.greek_limits['delta'][signal_id]
        logger.info(f"Target Delta: {target_delta}")
        delta_size = (math.floor(target_delta/abs(delta)))
        logger.info(f"Delta from Full Cash Spend: {max_size_cash_can_buy * delta}, Size: {max_size_cash_can_buy}")
        logger.info(f"Delta with Size Limit: {delta_size * delta}, Size: {delta_size}")
        return max(delta_size if abs(delta_size) <= abs(max_size_cash_can_buy) else max_size_cash_can_buy, 1)

    def calculate_scaler(self, symbol:str)-> None:
        """
        Calculate the scaler for the realized volatility timeseries.
        """
        if symbol not in self.rvol_timeseries:
            self.load_rvol_timeseries(symbol)
        
        rvol = self.rvol_timeseries[symbol]
        rolling_mean = rvol.rolling(window=self.rolling_window).mean()
        rolling_std = rvol.rolling(window=self.rolling_window).std()
        z_i = (rvol - rolling_mean) / rolling_std
        scaler = self.norm_constant/(1+z_i.abs())
        self.z_i[symbol] = z_i
        self.scaler[symbol] = scaler

    def load_rvol_timeseries(self, symbol:str)-> None:
        """
        Load the realized volatility timeseries for a given symbol.
        """
        logger.info(f"Loading realized volatility timeseries for {symbol} with vol_type: {self.vol_type} and rvol_window: {self.rvol_window}")
        def weighted_mean(series: pd.Series) -> pd.Series:
            """
            Calculate the weighted mean of a series.
            """
            logger.info(f"Calculating weighted mean for {symbol} with weights: {self.weights} and rvol_window: {self.rvol_window}")
            w1, w2, w3 = self.weights
            win1, win2, win3 = self.rvol_window 
            rvol_series1 = self.pm.get_underlier_data(symbol).rvol(
                                                ts_start = pd.to_datetime(self.rm.start_date) - relativedelta(years=1),
                                                ts_end = pd.to_datetime(self.rm.end_date) + BDay(1),
                                                window = win1,
                                                log = True)[f'rvol_{win1}D']
            rvol_series2 = self.pm.get_underlier_data(symbol).rvol(
                                                ts_start = pd.to_datetime(self.rm.start_date) - relativedelta(years=1),
                                                ts_end = pd.to_datetime(self.rm.end_date) + BDay(1),
                                                window = win2,
                                                log = False)[f'rvol_{win2}D']
            rvol_series3 = self.pm.get_underlier_data(symbol).rvol(
                                                ts_start = pd.to_datetime(self.rm.start_date) - relativedelta(years=1),
                                                ts_end = pd.to_datetime(self.rm.end_date) + BDay(1),
                                                window = win3,
                                                log = False)[f'rvol_{win3}D']
            rvol_series = pd.concat([rvol_series1 * w1, rvol_series2 * w2, rvol_series3 * w3], axis=1).sum(axis=1)
            return rvol_series.rename(symbol)

        def mean(series: pd.Series) -> pd.Series:
            """
            Calculate the weighted mean of a series.
            """
            logger.info(f"Calculating mean for {symbol} with rvol_window: {self.rvol_window}")
            win1, win2, win3 = self.rvol_window 
            rvol_series1 = self.pm.get_underlier_data(symbol).rvol(
                                                ts_start = pd.to_datetime(self.rm.start_date) - relativedelta(years=1),
                                                ts_end = pd.to_datetime(self.rm.end_date) + BDay(1),
                                                window = win1,
                                                log = True)[f'rvol_{win1}D']
            rvol_series2 = self.pm.get_underlier_data(symbol).rvol(
                                                ts_start = pd.to_datetime(self.rm.start_date) - relativedelta(years=1),
                                                ts_end = pd.to_datetime(self.rm.end_date) + BDay(1),
                                                window = win2,
                                                log = False)[f'rvol_{win2}D']
            rvol_series3 = self.pm.get_underlier_data(symbol).rvol(
                                                ts_start = pd.to_datetime(self.rm.start_date) - relativedelta(years=1),
                                                ts_end = pd.to_datetime(self.rm.end_date) + BDay(1),
                                                window = win3,
                                                log = False)[f'rvol_{win3}D']
            rvol_series = pd.concat([rvol_series1, rvol_series2, rvol_series3], axis=1).mean(axis=1)
            return rvol_series.rename(symbol)
        
        def window(series: pd.Series) -> pd.Series:
            """
            Calculate the windowed mean of a series.
            """
            logger.info(f"Calculating windowed mean for {symbol} with rvol_window: {self.rvol_window}")
            rvol_series1 = self.pm.get_underlier_data(symbol).rvol(
                                                ts_start = pd.to_datetime(self.rm.start_date) - relativedelta(years=1),
                                                ts_end = pd.to_datetime(self.rm.end_date) + BDay(1),
                                                window = self.rvol_window,
                                                log = True)[f'rvol_{self.rvol_window}D']
            return rvol_series1.rename(symbol)
        if symbol not in self.rvol_timeseries:
            if self.vol_type == 'mean':
                rvol_series = mean(pd.Series())
            elif self.vol_type == 'weighted_mean':
                rvol_series = weighted_mean(pd.Series())
            elif self.vol_type == 'window':
                rvol_series = window(pd.Series())
            else:
                raise ValueError(f"Unknown vol_type: {self.vol_type}. Available types: {self.VOL_TYPES}")
            self.rvol_timeseries[symbol] = rvol_series

    def daily_update(self):


        ### Consider implementing:
        ### - Increase & Reduce position delta based on scaler
        ### - Reduce only using min(limit, today_delta_limit)
        pass

    def pre_analyze_task(self, *args, **kwargs):
        """
        Abstract method for daily updates.
        """
        pass

    def post_analyze_task(self, *args, **kwargs):
        """
        Abstract method for daily updates.
        """
        pass

        