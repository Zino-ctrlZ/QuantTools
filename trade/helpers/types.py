from typing import TypedDict
from enum import Enum

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
    