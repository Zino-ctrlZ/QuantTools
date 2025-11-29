from datetime import datetime
import math
from dataclasses import dataclass, field
from typing import Literal, ClassVar
import pandas as pd
from EventDriven.riskmanager.utils import logger
from ..._vars import Y2_LAGGED_START_DATE
from ..market_data import get_timeseries_obj



def default_delta_limit(
    cash_available: float,
    sizing_lev: float,
    underlier_price_at_time: float,
):
    """
    This function calculates the equivalent delta size based on the cash available * sizing leverage.
    Equivalent delta size is the amount of delta exposure that can be taken on based on the cash available if buying Delta 1.
    Sizing leverage allows for increasing the amount of delta exposure that can be taken on.
    args:
    -------
    cash_available: Cash available for trading
    sizing_lev: Sizing leverage. This is a multiplier that increases the equivalent delta size
    underlier_price_at_time: Price of the underlier at the time of calculation
    returns:
    ---------
    equivalent_delta_size: Equivalent delta size based on cash available and sizing leverage
    """
    equivalent_delta_size = ((cash_available * sizing_lev) / \
                             (underlier_price_at_time * 100)) if underlier_price_at_time != 0 else 0
    return equivalent_delta_size

def zscore_rvol_delta_limit(
    cash_available: float,
    sizing_lev: float,
    underlier_price_at_time: float,
    _scaler: float
) -> float:
    """
    This function calculates the equivalent delta size based on the cash available * sizing leverage and a z-score scaler.
    Equivalent delta size is the amount of delta exposure that can be taken on based on the cash available if buying Delta 1.
    Sizing leverage allows for increasing the amount of delta exposure that can be taken on.
    The z-score scaler adjusts the equivalent delta size based on the realized volatility of the underlier.
    args:
    --------
    cash_available: Cash available for trading
    sizing_lev: Sizing leverage. This is a multiplier that increases the equivalent delta size
    underlier_price_at_time: Price of the underlier at the time of calculation
    scaler: Z-score scaler based on the realized volatility of the underlier
    returns:
    ----------
    equivalent_delta_size: Equivalent delta size based on cash available, sizing leverage and z-score scaler
    """

    equiv =  default_delta_limit(
        cash_available=cash_available,
        sizing_lev=sizing_lev,
        underlier_price_at_time=underlier_price_at_time
    ) 
    return equiv * _scaler

def delta_position_sizing(
        cash_available: float,
        option_price_at_time: float,
        delta: float,
        delta_limit: float,
) -> int:
    """
    Calculate the position size based on delta and cash available.
    Args:
        cash_available (float): The cash available for trading.
        option_price_at_time (float): The price of the option at the time of calculation.
        delta (float): The delta of the option.
        delta_limit (float): The maximum delta exposure allowed.
    Returns:
        int: The calculated position size."""
    ## TODO: Add docstring
    ## TODO: Raise error if delta is 0 or cash_available is <= 0
    if delta == 0 or cash_available <= 0 or option_price_at_time <= 0:
        logger.critical(f"Delta is 0 or cash_available is <= 0 or option_price_at_time <= 0. delta: {delta}, cash_available: {cash_available}, option_price_at_time: {option_price_at_time}. This is intended to be long only sizing. Returning 0.")
        return 0
    delta_size = (math.floor(delta_limit/abs(delta)))
    max_size_cash_can_buy = abs(math.floor(cash_available/(option_price_at_time*100)))
    # size = max(delta_size if abs(delta_size) <= abs(max_size_cash_can_buy) else max_size_cash_can_buy, 1)

    ## Opting to return 0 if size is 0
    size = delta_size if abs(delta_size) <= abs(max_size_cash_can_buy) else max_size_cash_can_buy
    return size

def raise_none(param, name):
    if param is None:
        raise ValueError(f"{name} cannot be None")

def realized_vol(series: pd.Series, window: int) -> pd.Series:
    return series.pct_change().rolling(window).std() * (252 ** 0.5)

def weighted_realized_vol(series: pd.Series, windows: tuple, weights: tuple) -> pd.Series:
    vols = [realized_vol(series, w) for w in windows]
    weighted_vol = sum(w * v for w, v in zip(weights, vols))
    return weighted_vol

def mean_realized_vol(series: pd.Series, windows: tuple) -> pd.Series:
    vols = [realized_vol(series, w) for w in windows]
    mean_vol = sum(vols) / len(vols)
    return mean_vol

def scaler(realized_vol_series: pd.Series, factor: float, window:int) -> pd.Series:
    rolling_mean =realized_vol_series.rolling(window=window).mean()
    rolling_std = realized_vol_series.rolling(window=window).std()
    z = (realized_vol_series - rolling_mean) / rolling_std
    return factor/((1+z.abs()))



@dataclass
class ZcoreScalar:
    rvol_window: int|tuple
    rolling_window: int
    weights: tuple
    vol_type: Literal['mean', 'weighted', 'window']
    norm_constant: int = 1.0
    rvol_timeseries: dict = field(default_factory=dict)
    VOL_TYPES: ClassVar[set] = {'mean', 'weighted_mean', 'window'} ## TODO: Align ZscoreRvolSizer to this
    syms: list = field(default_factory=list)
    interval: str = '1d'
    scalers: dict = field(default_factory=dict)

    def __post_init__(self):
        ## Ensure all lists are tuples

        if self.interval != '1d':
            raise NotImplementedError("Not allowed to change interval yet.")
        
        if isinstance(self.weights, list):
            self.weights = tuple(self.weights)

        if isinstance(self.rvol_window, list):
            self.rvol_window = tuple(self.rvol_window)

        ## Assert rvol_window is valid
        self.rvol_window = self.__rvol_window_assert(self.vol_type, self.rvol_window)
        assert self.vol_type in self.VOL_TYPES, f"vol_type must be one of {self.VOL_TYPES}, got {self.vol_type}"
        assert isinstance(self.weights, tuple) and len(self.weights) == 3, "weights must be a tuple of length 3"
        assert sum(self.weights) == 1, "weights must sum to 1"
        assert self.rolling_window > 0, "rolling_window must be a positive integer"

    def reload(self):
        """
        Reload the scalers for all symbols in self.syms
        """
        self.scalers = {}




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
    
    def __repr__(self) -> str:
        return f"ZcoreScalar(rvol_window={self.rvol_window}, syms={self.syms})"
    
    def load_scalers(self, syms: list = None, force=False) -> None:
        """
        Load the z-score scalers for the specified symbols.
        args:
        -------
        syms: List of symbols to load. If None, load all symbols in self.syms
        force: If True, reload scalers even if they already exist
        Note: syms must be in self.syms. No loading unassociated syms
        Raises:
        -------
        ValueError: If any symbol in syms is not in self.syms
        TypeError: If syms is not a list
        Returns:
        -------
        None: Values are stored in self.scalers
        """

        ## Get timeseries object
        timeseries = get_timeseries_obj()

        ## If syms is None, use the existing syms
        ## This is to avoid reloading timeseries if already loaded
        if syms is None:
            syms = self.syms

        if not isinstance(syms, list):
            raise TypeError(f"syms must be a list, got {type(syms)}")
        
        ## Raise error for syms not in self.syms. No loading unassociated syms
        unknown_syms = [s for s in syms if s not in self.syms]
        if unknown_syms:
            raise ValueError(f"Unknown syms: {unknown_syms}. Please add them to self.syms first.")

        ## Ensure syms is added to self.syms
        if not force:
            in_syms_bool = all(s in self.scalers.keys() for s in syms) 
            syms = list(set(syms)  - set(self.scalers.keys())) ## Only load syms that are not already in self.scalers
        else:
            in_syms_bool = False

        
        ## If all syms are already in self.syms, do nothing
        ## Add new syms to self.syms
        if not in_syms_bool:
            self.syms.extend(syms)

        ## Load timeseries for each symbol and calculate the z-score scaler
        for sym in syms:
            timeseries.load_timeseries(sym=sym, 
                                       start_date=Y2_LAGGED_START_DATE,
                                       end_date=datetime.now(),
                                       interval=self.interval)
            ts = timeseries.get_timeseries(sym=sym, interval=self.interval).spot['close']
            
            if self.vol_type == 'window':
                func = lambda x: realized_vol(x, self.rvol_window)
            elif self.vol_type == 'weighted_mean':
                func = lambda x: weighted_realized_vol(x, self.rvol_window, self.weights)
            elif self.vol_type == 'mean':
                func = lambda x: mean_realized_vol(x, self.rvol_window)
            else:
                raise ValueError(f"Unknown vol_type: {self.vol_type}")
            
            

            def _callable(sym, func=func, ts=ts):
                self.rvol_timeseries[sym] = func(ts)
                return scaler(self.rvol_timeseries[sym], self.norm_constant, self.rolling_window)
            self.scalers[sym] = _callable(sym)
            

    def append_syms(self, syms: list|str, ignore_load:bool = False):
        """
        Add syms to list of syms
        
        args:
        -------

        syms: List of syms, or a single sym
        ignore_load: ignores the load function if true
        """

        if isinstance(syms, str):
            if syms not in self.syms:
                self.syms.append(syms)
            syms = [syms]
        
        elif isinstance(syms, list):
            syms = [s for s in syms if s not in self.syms]
            self.syms.extend(syms)

        else:
            raise TypeError(f"Unknown format: {type(syms)}, expected [list, str]")

        if not ignore_load:
            self.load_scalers(syms=syms)

    def get_scaler(self, sym: str) -> pd.Series:
        """
        Get the z-score scaler for a specific symbol.
        
        args:
        -------
        sym: Symbol to get the scaler for
        
        returns:
        --------
        pd.Series: Z-score scaler for the symbol
        """
        if sym not in self.scalers:
            self.append_syms(sym)
        
        return self.scalers[sym]
    
    def get_scaler_on_date(self, sym: str, date: pd.Timestamp|str|datetime) -> float:
        """
        Get the z-score scaler for a specific symbol on a specific date.
        """
        if isinstance(date, str):
            date = pd.Timestamp(date)
        
        scaler_ts = self.get_scaler(sym)
        if date not in scaler_ts.index:
            raise ValueError(f"Date {date} not found in scaler_ts index for symbol {sym}.")
        
        return scaler_ts.loc[date]
    