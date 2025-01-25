from typing import TypedDict

class OptionTickMetaData(TypedDict):
    ticker: str
    exp_date: str
    put_call: str
    strike: float
    
class PositionData(TypedDict): 
    long: list[str]
    short: list[str]
    