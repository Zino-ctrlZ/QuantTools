from typing import TypedDict
from enum import Enum
from trade.helpers.exception import SymbolChangeError

class OptionTickMetaData(TypedDict):
    ticker: str
    exp_date: str
    put_call: str
    strike: float
    
class PositionData(TypedDict): 
    long: list[str]
    short: list[str]



class OptionModelAttributes(Enum):
    S0 = 'unadjusted_S0'
    K = 'K'
    exp_date = 'exp'
    sigma = 'sigma'
    y = 'y'
    put_call = 'put_call'
    r = 'rf_rate'
    start = 'end_date'
    spot_type = 'chain_price'



class TickerMap(dict):
    invalid_tickers = {'FB': 'META'}
    def __getitem__(self, key):
        if key in self.invalid_tickers:
            raise SymbolChangeError(f"Tick name changed from {key} to {self.invalid_tickers[key]}, access the new tick instead")
        return super().__getitem__(key)
    