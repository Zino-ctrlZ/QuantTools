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
from trade.helpers.decorators import timeit
from trade.helpers.Logging import setup_logger
from trade.assets.rates import get_risk_free_rate_helper
from EventDriven._vars import OPTION_TIMESERIES_START_DATE, load_riskmanager_cache
from EventDriven.exceptions import UnaccessiblePropertyError
from trade import register_signal


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
    _today: str = datetime.now().strftime('%Y-%m-%d')
    should_refresh: bool = True

    def __post_init__(self):
        register_signal(signum=15, signal_func=self._on_exit_sanitize)
        register_signal(signum=0, signal_func=self._on_exit_sanitize)

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
    
    def _on_exit_sanitize(self):
        """Remove today's data from all stored timeseries data."""

        for sym in self._spot.keys():
            d = self._spot[sym]
            d = d[d.index < self._today]
            self._spot[sym] = d
        for sym in self._chain_spot.keys():
            d = self._chain_spot[sym]
            d = d[d.index < self._today]
            self._chain_spot[sym] = d
        for sym in self._dividends.keys():
            d = self._dividends[sym]
            d = d[d.index < self._today]
            self._dividends[sym] = d


    @timeit
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
        sym_available = sym in self._spot
        all_dates_present = False
        
        data_to_check = [
            self._spot.get(sym),
            self._chain_spot.get(sym),
            self._dividends.get(sym)
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
        
        return (sym_available and all_dates_present), return_dates
    
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
    
    @timeit
    def _remove_today_data(self, data: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
        """Remove today's data from the given DataFrame or Series."""
        today_str = ny_now().strftime('%Y-%m-%d')
        if isinstance(data, pd.DataFrame):
            return data[data.index < today_str]
        elif isinstance(data, pd.Series):
            return data[data.index < today_str]
        else:
            raise ValueError("Data must be a pandas DataFrame or Series. Got type: {}".format(type(data)))
        
    @timeit
    def _sanitize_today_data(self) -> None:
        """Remove today's data from all stored timeseries data."""
        
        for sym in self._spot.keys():
            self._spot[sym] = self._remove_today_data(self._spot[sym])
        for sym in self._chain_spot.keys():
            self._chain_spot[sym] = self._remove_today_data(self._chain_spot[sym])
        for sym in self._dividends.keys():
            self._dividends[sym] = self._remove_today_data(self._dividends[sym])

    
    @timeit
    def _sanitize_data(self) -> None:
        """
        Sanitize all stored timeseries data by removing today's data.
        Dropping duplicates, ensuring datetime index, and sorting.
        """
        self._sanitize_today_data()

        for sym in self._spot.keys():
            data = self._spot[sym]
            data.index = pd.to_datetime(data.index)
            data = data[~data.index.duplicated(keep='last')]
            data = data.sort_index()
            self._spot[sym] = data
        
        for sym in self._chain_spot.keys():
            data = self._chain_spot[sym]
            data.index = pd.to_datetime(data.index)
            data = data[~data.index.duplicated(keep='last')]
            data = data.sort_index()
            self._chain_spot[sym] = data
        
        for sym in self._dividends.keys():
            data = self._dividends[sym]
            data.index = pd.to_datetime(data.index)
            data = data[~data.index.duplicated(keep='last')]
            data = data.sort_index()
            self._dividends[sym] = data

            


    def _pre_sanitize_load_timeseries(self, 
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


        ## Current Data
        current_spot = self._spot.get(sym)
        current_chain_spot = self._chain_spot.get(sym)
        current_divs = self._dividends.get(sym)

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

        
        ## Assign data directly to cache
        ## We remove today's data to avoid situations where it was loaded intraday and remains in database
        ## This ensures only historical data is stored.
        self._spot[sym] = spot
        self._chain_spot[sym] = chain_spot
        self._dividends[sym] = divs

    def load_timeseries(self,
                        sym: str, 
                        start_date: str|datetime = None, 
                        end_date: str|datetime = None, 
                        interval='1d',
                        force: bool = False) -> None:
        """
        Public method to load timeseries data for a given symbol and interval.
        """
        self._pre_sanitize_load_timeseries(sym, start_date, end_date, interval, force)

    def _is_date_in_index(self, sym: str, date: pd.Timestamp, interval: str = '1d') -> bool:
        """
        Check if a specific date is present in the timeseries index for a given symbol and interval.
        Args:
            sym (str): The stock symbol.
            date (pd.Timestamp or str): The date to check.
            interval (str): The interval of the timeseries data. Defaults to '1d'.
        Returns:
            bool: True if the date is present, False otherwise.
        """
        all_data = [
            self._spot.get(sym),
            self._chain_spot.get(sym),
            self._dividends.get(sym)
        ]

        for data in all_data:
            date = pd.to_datetime(date).date()
            if data is not None and date in data.index.date:
                continue
            else:
                return False
        return True

    def get_at_index(self, sym: str, index: pd.Timestamp, interval: str = '1d') -> AtIndexResult:
        """
        Retrieve the spot price, chain spot price, and dividends for a given symbol at a specific index (date).
        Args:
            sym (str): The stock symbol.
            index (pd.Timestamp or str): The date for which to retrieve the data.
        Returns:
            AtIndexResult: A dataclass containing spot price, chain spot price, and dividends."""
        
        ## Only load date if not available. Not loading all unavailable dates
        already_available = self._is_date_in_index(sym, index, interval)

        if not already_available:
            logger.critical("Reloading timeseries data for symbol %s.", sym)
            prev_day = (pd.Timestamp(index) - BDay(1)).strftime('%Y-%m-%d')
            self._pre_sanitize_load_timeseries(
                sym=sym, start_date=prev_day, end_date=index, interval=interval, force=True
            )
                    

        ## OPTIMIZATION: Consolidate type checks and conversions (Task #3)
        if not isinstance(index, pd.Timestamp):
            index = pd.Timestamp(index)
        
        if sym not in self._spot:
            raise ValueError(f"Symbol {sym} not found in timeseries data.")
        
        index_str = index.strftime('%Y-%m-%d')
        spot = self._spot[sym].loc[index_str] if sym in self._spot else None
        chain_spot = self._chain_spot[sym].loc[index_str] if sym in self._chain_spot else None
        dividends = self._dividends[sym].loc[index_str] if sym in self._dividends else None
        rates = self.rates.loc[index_str] if self.rates is not None else None
        return AtIndexResult(spot=spot, chain_spot=chain_spot, dividends=dividends, sym=sym, date=index_str, rates=rates)
    
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
        if not self.already_loaded(sym, interval):
            logger.critical("Timeseries for symbol %s not loaded. Loading now.", sym)
            self._pre_sanitize_load_timeseries(sym, interval=interval, force=True)
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
            data = getattr(self, factor).get(sym)
            if data is None:
                raise ValueError(f"No data found for factor {factor} and symbol {sym}.")
            if factor == '_spot':
                ts =  TimeseriesData(spot=data, chain_spot=None, dividends=None)
            elif factor == '_chain_spot':
                ts = TimeseriesData(spot=None, chain_spot=data, dividends=None)
            elif factor == '_dividends':
                ts = TimeseriesData(spot=None, chain_spot=None, dividends=data)
            else:
                raise ValueError(f"Unhandled factor {factor}.")
        
        elif factor is None:
            spot = self._spot.get(sym)
            chain_spot = self._chain_spot.get(sym)
            dividends = self._dividends.get(sym)
            ts = TimeseriesData(spot=spot, chain_spot=chain_spot, dividends=dividends)

        return ts


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