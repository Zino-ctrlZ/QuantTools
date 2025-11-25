"""
Responsible for loading and managing market timeseries data for equities, including spot prices, chain prices, and dividends.
Utilizes OpenBB for data retrieval and supports additional data processing through user-defined callables.
"""
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Literal, Optional, Tuple
import pandas as pd
from pandas.tseries.offsets import BDay
from openbb import obb
from dbase.DataAPI.ThetaData import resample
from trade.helpers.helper import (
    retrieve_timeseries, 
    ny_now, 
    CustomCache,
    get_missing_dates)
from trade.helpers.Logging import setup_logger
from trade.assets.rates import get_risk_free_rate_helper
from EventDriven._vars import OPTION_TIMESERIES_START_DATE, load_riskmanager_cache
from EventDriven.exceptions import UnaccessiblePropertyError


logger = setup_logger('EventDriven.riskmanager.market_data', stream_log_level="WARNING")

## TODO: This var is from optionlib. Once ready, import from there.
## TODO: Implement interval handling to have multiple intervals

OPTIMESERIES: Optional['MarketTimeseries'] = None
DIVIDEND_CACHE: CustomCache = load_riskmanager_cache(target="dividend_timeseries")
SPOT_CACHE: CustomCache = load_riskmanager_cache(target="spot_timeseries")
CHAIN_SPOT_CACHE: CustomCache = load_riskmanager_cache(target="chain_spot_timeseries")


@dataclass 
class AtIndexResult:
    """Dataclass to hold the result of retrieving market data at a specific index (date)."""
    sym: str
    date: pd.Timestamp
    spot: pd.Series
    chain_spot: pd.Series
    rates: pd.Series
    dividends: pd.Series
    additional: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"AtIndexResult(sym={self.sym}, date={self.date})"

@dataclass
class TimeseriesData:
    """Class to hold timeseries data for a specific symbol."""
    spot: pd.Series
    chain_spot: pd.Series
    dividends: pd.Series
    additional_data: Dict[str, pd.Series] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"TimeseriesData(spot={self.spot is not None}, chain_spot={self.chain_spot is not None}, dividends={self.dividends is not None}, additional_data_keys={list(self.additional_data.keys())})"


@dataclass
class MarketTimeseries:
    """Class to manage market timeseries data for equities."""
    additional_data: Dict[str, Any] = field(default_factory=dict)
    rates: pd.DataFrame = field(default_factory=get_risk_free_rate_helper)
    DEFAULT_NAMES: ClassVar[List[str]] = ['spot', 'chain_spot', 'dividends']
    _refresh_delta: Optional[timedelta] = timedelta(minutes=30)
    _last_refresh: Optional[datetime] = field(default_factory=ny_now)
    _start: str = OPTION_TIMESERIES_START_DATE
    _end: str = (datetime.now() - BDay(1)).strftime('%Y-%m-%d')
    should_refresh: bool = True

    @property
    def spot(self) -> dict:
        raise UnaccessiblePropertyError("The 'spot' property is not accessible directly. Use 'get_timeseries' method instead. Or access via 'get_at_index' method.")
    
    @property
    def chain_spot(self) -> dict:
        raise UnaccessiblePropertyError("The 'chain_spot' property is not accessible directly. Use 'get_timeseries' method instead. Or access via 'get_at_index' method.")
    
    @property
    def dividends(self) -> dict:
        raise UnaccessiblePropertyError("The 'dividends' property is not accessible directly. Use 'get_timeseries' method instead. Or access via 'get_at_index' method.")

    @property
    def _spot(self) -> CustomCache:
        return SPOT_CACHE
    
    @property
    def _chain_spot(self) -> CustomCache:
        return CHAIN_SPOT_CACHE
    
    @property
    def _dividends(self) -> CustomCache:
        return DIVIDEND_CACHE

    def _already_loaded(self, 
                       sym: str, 
                       interval: str = '1d', 
                       start: str|datetime = None, 
                       end: str|datetime = None) -> Tuple[bool, List[pd.Timestamp]]:
        """
        Check if the timeseries for a given symbol and interval is already loaded.
        Hidden method that also returns missing dates if not fully loaded.
        """
        start = start or self._start
        end = end or self._end
        sym_available = sym in self._spot.get(interval, {})
        interval_available = interval in self._spot
        all_dates_present = False
        
        data_to_check = [
            self._spot.get(interval, {}).get(sym),
            self._chain_spot.get(interval, {}).get(sym),
            self._dividends.get(interval, {}).get(sym)
        ]
        
        missing_dates_set = set()
        all_dates_present = False
        for data in data_to_check:
            if data is not None:
                missing_dates = get_missing_dates(data, start, end)
                missing_dates_set.update(missing_dates)
                if not missing_dates:
                    all_dates_present = True
                else:
                    all_dates_present = False
        ## If all dates not present, return missing dates
        return_dates = list(missing_dates_set)
        if not all_dates_present:
            
            ## If missing dates is empty, return start and end
            if not return_dates:
                return_dates = [pd.Timestamp(start), pd.Timestamp(end)]
            else:
                return_dates = [min(return_dates), max(return_dates)]
        
        ## If all dates present, return empty list
        else:
            return_dates = []
            
                    
        return (sym_available and interval_available and all_dates_present), return_dates
    
    def already_loaded(self, 
                       sym: str, 
                       interval: str = '1d', 
                       start: str|datetime = None, 
                       end: str|datetime = None) -> bool:
        """
        Public method to check if the timeseries for a given symbol and interval is already loaded.
        """
        already_loaded, _ = self._already_loaded(sym, interval, start, end)
        return already_loaded

            


    def load_timeseries(self, 
                        sym: str, 
                        start_date: str|datetime = None, 
                        end_date: str|datetime = None, 
                        interval='1d',
                        force: bool = False) -> None:
        already_loaded, dt_range = self._already_loaded(sym, interval, start_date, end_date)
        
        if already_loaded and not force:
            logger.info("Timeseries for %s already loaded. Use force=%s to reload.", sym, force)
            return
        
        start_date = min(dt_range)
        end_date = max(dt_range)
        
        spot = retrieve_timeseries(sym, start_date, end_date, interval)
        chain_spot = retrieve_timeseries(sym, start_date, end_date, interval, spot_type='chain_price')
        try:
            divs = obb.equity.fundamental.dividends(symbol=sym, provider='yfinance').to_df()
            divs.set_index('ex_dividend_date', inplace = True)
        except Exception:
            logger.error("Failed to retrieve dividends for symbol %s", sym)
            divs = pd.DataFrame({'amount':[0]}, index = pd.bdate_range(start=self._start, end=self._end, freq=interval))
        
        ## Ensure datetime index
        divs.index = pd.to_datetime(divs.index)
        divs = divs.reindex(pd.bdate_range(start=self._start, end=self._end, freq=interval), method='ffill')
        divs = resample(divs['amount'], method='ffill', interval=interval)


        ## Interval Dict
        spot_dict = self._spot.get(interval, {})
        chain_spot_dict = self._chain_spot.get(interval, {})
        div_dict = self._dividends.get(interval, {})

        ## Current Data
        current_spot = spot_dict.get(sym)
        current_chain_spot = chain_spot_dict.get(sym)
        current_divs = div_dict.get(sym)

        ## We are moving from overwritting prev data to merging new data
        if current_spot is not None:
            spot = pd.concat([current_spot, spot]).sort_index()
            spot = spot[~spot.index.duplicated(keep='last')]
        else:
            logger.info("No previous spot data for symbol %s, adding new data.", sym)
        if current_chain_spot is not None:
            chain_spot = pd.concat([current_chain_spot, chain_spot]).sort_index()
            chain_spot = chain_spot[~chain_spot.index.duplicated(keep='last')]
        if current_divs is not None:
            divs = pd.concat([current_divs, divs]).sort_index()
            divs = divs[~divs.index.duplicated(keep='last')]

        
        ## Assign data to regular dicts. Caveat (Potentially large data in memory)
        spot_dict[sym] = spot
        chain_spot_dict[sym] = chain_spot
        div_dict[sym] = divs
        
        ## Store data in caches. CustomCaches do not support in-place assignments
        ## So we have to retrieve the dict, modify it, and then reassign it.
        self._spot[interval] = spot_dict
        self._chain_spot[interval] = chain_spot_dict
        self._dividends[interval] = div_dict


    def get_at_index(self, sym: str, index: pd.Timestamp, interval: str = '1d') -> AtIndexResult:
        """
        Retrieve the spot price, chain spot price, and dividends for a given symbol at a specific index (date).
        Args:
            sym (str): The stock symbol.
            index (pd.Timestamp or str): The date for which to retrieve the data.
        Returns:
            AtIndexResult: A dataclass containing spot price, chain spot price, and dividends."""
        
        already_available = self.already_loaded(sym, interval)

        if not already_available:
            print("Reloading timeseries data for symbol %s.", sym)
            self.load_timeseries(
                sym=sym,
                start_date=self._start,
                end_date=self._end,
                force=True
            )
            self._last_refresh = ny_now()
                    

        if isinstance(index, (str, datetime)):
            index = pd.Timestamp(index)
        if sym not in self._spot.get(interval, {}):
            raise ValueError(f"Symbol {sym} not found in timeseries data.")
        if not (isinstance(index, pd.Timestamp) or isinstance(index, datetime)):
            raise ValueError("Index must be a pandas Timestamp or datetime object.")
        index = index.strftime('%Y-%m-%d')
        spot = self._spot[interval][sym].loc[index] if sym in self._spot.get(interval, {}) else None
        chain_spot = self._chain_spot[interval][sym].loc[index] if sym in self._chain_spot.get(interval, {}) else None
        dividends = self._dividends[interval][sym].loc[index] if sym in self._dividends.get(interval, {}) else None
        rates = self.rates.loc[index] if self.rates is not None else None
        return AtIndexResult(spot=spot, chain_spot=chain_spot, dividends=dividends, sym=sym, date=index, rates=rates)
    
    def calculate_additional_data(self,
                             factor: Literal['spot', 'chain_spot', 'dividends'],
                             sym: str,
                             additional_data_name: str,
                             _callable: Any,
                             column:Optional[str]='close',
                             force_add:bool=False) -> None:
        """
        Load additional data for a given factor (spot, chain_spot, dividends) using a callable function.

        Process:
        Callable passed should only take in a pd.Series and return a pd.Series. 
        It manipulates the timeseries data for the specified factor and appends the result to the additional_data dictionary.
        The schema of additional_data: {additional_data_name: {sym: pd.Series}}

        Args:
            factor (Literal['spot', 'chain_spot', 'dividends']): The factor to process.
            sym (str): The stock symbol.
            additional_data_name (str): The name under which to store the additional data.
            _callable (Any): A callable function that processes the pd.Series.
            column (Optional[str]): The column to use from the factor data. Defaults to 'close'.
            force_add (bool): If True, will overwrite existing additional data for the given name and symbol.

        Raises:
            ValueError: If the factor is not recognized or if the symbol is not found in the timeseries data.
        """

        ## Raise error if factor not recognized
        if factor not in self.DEFAULT_NAMES:
            raise ValueError(f"Factor {factor} not recognized. Must be one of ['spot', 'chain_spot', 'dividends'].")
        
        ## Get the data for the specified factor and symbol
        factor_data = getattr(self, factor).get(sym)

        ## Raise error if symbol not found
        if factor_data is None:
            raise ValueError(f"No data found for factor {factor} and symbol {sym}.")
        
        ## If column specified, ensure it exists in the DataFrame
        if column and isinstance(factor_data, (pd.DataFrame, pd.Series)):
            if column not in factor_data.columns:
                raise ValueError(f"Column {column} not found in data for factor {factor} and symbol {sym}.")
            factor_data = factor_data[column]
        
        ## Process the data using the provided callable
        processed_data = _callable(factor_data)
        if additional_data_name not in self.additional_data:
            self.additional_data[additional_data_name] = {}
        
        ## Check if data already exists and force_add is not set
        exists = sym in self.additional_data.get(additional_data_name, {})
        if exists and not force_add:
            logger.info("Additional data for %s and symbol %s already exists. Use force_add=True to overwrite.", additional_data_name, sym)
            return
        
        self.additional_data[additional_data_name][sym] = processed_data

    def get_timeseries(self,
                       sym: str,
                       factor: Literal['spot', 'chain_spot', 'dividends', 'additional']=None,
                       interval: str = '1d',
                       additional_data_name: Optional[str] = None,
                       ) -> TimeseriesData:
        """
        Retrieve the timeseries data for a given symbol and factor.
        Args:
            sym (str): The stock symbol.
            factor (Literal['spot', 'chain_spot', 'dividends', 'additional']): The factor to retrieve.
            additional_data_name (Optional[str]): The name of the additional data if factor is 'additional'.
        Returns:
            TimeseriesData: A dataclass containing the requested timeseries data.
        """
        if factor not in self.DEFAULT_NAMES + ['additional', None]:
            raise ValueError(f"Factor {factor} not recognized. Must be one of {self.DEFAULT_NAMES + ['additional']}.")
        if factor == 'additional':
            if additional_data_name is None:
                raise ValueError("additional_data_name must be provided when factor is 'additional'.")
            data = self.additional_data.get(additional_data_name, {}).get(sym)
            if data is None:
                raise ValueError(f"No additional data found for name {additional_data_name} and symbol {sym}.")
            return TimeseriesData(spot=None, chain_spot=None, dividends=None, additional_data={additional_data_name: data})
        
        elif factor in self.DEFAULT_NAMES:
            factor = '_'+factor
            data = getattr(self, factor).get(interval, {}).get(sym)
            if data is None:
                raise ValueError(f"No data found for factor {factor} and symbol {sym}.")
            if factor == '_spot':
                return TimeseriesData(spot=data, chain_spot=None, dividends=None)
            elif factor == '_chain_spot':
                return TimeseriesData(spot=None, chain_spot=data, dividends=None)
            elif factor == '_dividends':
                return TimeseriesData(spot=None, chain_spot=None, dividends=data)
            else:
                raise ValueError(f"Unhandled factor {factor}.")
        
        elif factor is None:
            spot = self._spot.get(interval, {}).get(sym)
            chain_spot = self._chain_spot.get(interval, {}).get(sym)
            dividends = self._dividends.get(interval, {}).get(sym)
            return TimeseriesData(spot=spot, chain_spot=chain_spot, dividends=dividends)


    def __repr__(self) -> str:
        return f"MarketTimeseries(symbols: {list(self._spot.keys())}, intervals: {list(self._spot.keys())})"

    

def get_timeseries_obj() -> MarketTimeseries:
    global OPTIMESERIES
    if OPTIMESERIES is None:
        OPTIMESERIES = MarketTimeseries()

    return OPTIMESERIES

def reset_timeseries_obj() -> None:
    global OPTIMESERIES
    OPTIMESERIES = None