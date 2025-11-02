"""
Responsible for loading and managing market timeseries data for equities, including spot prices, chain prices, and dividends.
Utilizes OpenBB for data retrieval and supports additional data processing through user-defined callables.
"""
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Literal, Optional
import pandas as pd
from openbb import obb
from dbase.DataAPI.ThetaData import resample
from trade.helpers.helper import retrieve_timeseries, ny_now
from trade.helpers.Logging import setup_logger
from trade.assets.rates import get_risk_free_rate_helper


logger = setup_logger('EventDriven.riskmanager.market_data')

## TODO: This var is from optionlib. Once ready, import from there.
## TODO: Implement interval handling to have multiple intervals

OPTION_TIMESERIES_START_DATE: str|datetime = '2017-01-01'
Y1_LAGGED_START_DATE: str|datetime = (pd.to_datetime(OPTION_TIMESERIES_START_DATE) - relativedelta(years=1)).strftime('%Y-%m-%d')
Y2_LAGGED_START_DATE: str|datetime = (pd.to_datetime(OPTION_TIMESERIES_START_DATE) - relativedelta(years=2)).strftime('%Y-%m-%d')
Y3_LAGGED_START_DATE: str|datetime = (pd.to_datetime(OPTION_TIMESERIES_START_DATE) - relativedelta(years=3)).strftime('%Y-%m-%d')
TIMESERIES: Optional['MarketTimeseries'] = None


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
class MarketTimeseries:
    """Class to manage market timeseries data for equities."""
    spot: dict = field(default_factory=dict)
    chain_spot: dict = field(default_factory=dict)
    dividends: dict = field(default_factory=dict)
    additional_data: Dict[str, Any] = field(default_factory=dict)
    rates: pd.DataFrame = field(default_factory=get_risk_free_rate_helper)
    DEFAULT_NAMES: ClassVar[List[str]] = ['spot', 'chain_spot', 'dividends']
    _refresh_delta: Optional[timedelta] = timedelta(minutes=30)
    _last_refresh: Optional[datetime] = field(default_factory=ny_now)
    _start: str = OPTION_TIMESERIES_START_DATE
    _end: str = datetime.now().strftime('%Y-%m-%d')

    def load_timeseries(self, 
                        sym: str, 
                        start_date: str|datetime = None, 
                        end_date: str|datetime = None, 
                        interval='1d',
                        force: bool = False) -> None:
        
        start_date = start_date or self._start
        end_date = end_date or self._end
        if sym in self.spot and not force:
            logger.info("Timeseries for %s already loaded. Use force=True to reload.", sym)
            return
        
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
        self.spot[sym] = spot
        self.chain_spot[sym] = chain_spot
        self.dividends[sym] = divs

    def get_at_index(self, sym: str, index: pd.Timestamp) -> AtIndexResult:
        """
        Retrieve the spot price, chain spot price, and dividends for a given symbol at a specific index (date).
        Args:
            sym (str): The stock symbol.
            index (pd.Timestamp or str): The date for which to retrieve the data.
        Returns:
            AtIndexResult: A dataclass containing spot price, chain spot price, and dividends."""
        
        ## Adding refresh check every at_index call
        refresh = (
            self._refresh_delta is not None and ## Check if refresh is set
            self._last_refresh is not None and ## Check if last refresh is set
            (ny_now() - self._last_refresh) > self._refresh_delta ## Check if refresh is due
        )

        ## If refresh is due, reload all timeseries
        if refresh:
            logger.info("Refreshing market timeseries data...")
            for s in self.spot.keys():
                self.load_timeseries(
                    sym=s,
                    start_date=self._start,
                    end_date=self._end,
                    force=True
                )
            self._last_refresh = ny_now()
                    

        if isinstance(index, (str|datetime)):
            index = pd.Timestamp(index)
        if sym not in self.spot:
            raise ValueError(f"Symbol {sym} not found in timeseries data.")
        if not (isinstance(index, pd.Timestamp) or isinstance(index, datetime)):
            raise ValueError("Index must be a pandas Timestamp or datetime object.")
        index = index.strftime('%Y-%m-%d')
        spot = self.spot[sym].loc[index] if sym in self.spot else None
        chain_spot = self.chain_spot[sym].loc[index] if sym in self.chain_spot else None
        dividends = self.dividends[sym].loc[index] if sym in self.dividends else None
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

    def __repr__(self) -> str:
        return f"MarketTimeseries(symbols: {list(self.spot.keys())})"

    

def get_timeseries_obj() -> MarketTimeseries:
    global TIMESERIES
    if TIMESERIES is None:
        TIMESERIES = MarketTimeseries()

    return TIMESERIES

def reset_timeseries_obj() -> None:
    global TIMESERIES
    TIMESERIES = None