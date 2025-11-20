"""
Responsible for loading and managing market timeseries data for equities, including spot prices, chain prices, and dividends.
Utilizes OpenBB for data retrieval and supports additional data processing through user-defined callables.
"""
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Literal, Optional
import pandas as pd
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
    _spot: dict = field(default_factory=dict)
    _chain_spot: dict = field(default_factory=dict)
    _dividends: dict = field(default_factory=dict)
    additional_data: Dict[str, Any] = field(default_factory=dict)
    rates: pd.DataFrame = field(default_factory=get_risk_free_rate_helper)
    DEFAULT_NAMES: ClassVar[List[str]] = ['spot', 'chain_spot', 'dividends']
    _refresh_delta: Optional[timedelta] = timedelta(minutes=30)
    _last_refresh: Optional[datetime] = field(default_factory=ny_now)
    _start: str = OPTION_TIMESERIES_START_DATE
    _end: str = datetime.now().strftime('%Y-%m-%d')
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

    def already_loaded(self, 
                       sym: str, 
                       interval: str = '1d', 
                       start: str|datetime = None, 
                       end: str|datetime = None) -> bool:
        """
        Check if the timeseries for a given symbol and interval is already loaded.
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

        for data in data_to_check:
            if data is not None:
                missing_dates = get_missing_dates(data, start, end)
                if not missing_dates:
                    all_dates_present = True
                else:
                    all_dates_present = False
                    
        return sym_available and interval_available and all_dates_present   

            


    def load_timeseries(self, 
                        sym: str, 
                        start_date: str|datetime = None, 
                        end_date: str|datetime = None, 
                        interval='1d',
                        force: bool = False) -> None:
        already_loaded = self.already_loaded(sym, interval, start_date, end_date)
        
        if already_loaded and not force:
            logger.info("Timeseries for %s already loaded. Use force=%s to reload.", sym, force)
            return
        
        start_date = start_date or self._start
        end_date = end_date or self._end
        
        spot = retrieve_timeseries(sym, start_date, end_date, interval)
        chain_spot = retrieve_timeseries(sym, start_date, end_date, interval, spot_type='chain_price')
        try:
            divs = obb.equity.fundamental.dividends(symbol=sym, provider='yfinance').to_df()
            divs.set_index('ex_dividend_date', inplace = True)
        except Exception:
            logger.error("Failed to retrieve dividends for symbol %s", sym)
            divs = pd.DataFrame({'amount':[0]}, index = pd.bdate_range(start=self._start, end=self._end, freq='1Q'))
        
        ## Ensure datetime index
        divs.index = pd.to_datetime(divs.index)
        divs = divs.reindex(pd.bdate_range(start=self._start, end=self._end, freq='1D'), method='ffill')
        divs = resample(divs['amount'], method='ffill', interval=interval)
        self._spot[interval] = {}
        self._chain_spot[interval] = {}
        self._dividends[interval] = {}
        self._spot[interval][sym] = spot
        self._chain_spot[interval][sym] = chain_spot
        self._dividends[interval][sym] = divs

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
            logger.info("Reloading timeseries data for symbol %s.", sym)
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
            data = getattr(self, factor).get(interval, {}).get(sym)
            if data is None:
                raise ValueError(f"No data found for factor {factor} and symbol {sym}.")
            if factor == 'spot':
                return TimeseriesData(spot=data, chain_spot=None, dividends=None)
            elif factor == 'chain_spot':
                return TimeseriesData(spot=None, chain_spot=data, dividends=None)
            elif factor == 'dividends':
                return TimeseriesData(spot=None, chain_spot=None, dividends=data)
        
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