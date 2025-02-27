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


class ResultsEnum(Enum):
    SUCCESSFUL = 'SUCCESSFUL'
    MONEYNESS_TOO_TIGHT = 'MONEYNESS_TOO_TIGHT'
    NO_ORDERS = 'NO_ORDERS'
    UNSUCCESSFUL = 'UNSUCCESSFUL'
    IS_HOLIDAY = 'IS_HOLIDAY'
    UNAVAILABLE_CONTRACT = 'NO LISTED CONTRACTS' 
    MAX_PRICE_TOO_LOW = 'MAX_PRICE_TOO_LOW'
    TOO_ILLIQUID = 'TOO_ILLIQUID'
    NO_TRADED_CLOSE = 'NO_TRADED_CLOSE'


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
    