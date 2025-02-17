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
    