from pydantic.dataclasses import dataclass as pydantic_dataclass
from pydantic import ConfigDict
from typing import Union, Literal
from datetime import datetime, date
import numbers


@pydantic_dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class OrderRequest:
    """
    Dataclass representing an order request generated for Order Picker to process.

    """

    date: Union[datetime, str, date]
    symbol: str
    option_type: str
    max_close: numbers.Number
    tick_cash: numbers.Number
    direction: Literal["LONG", "SHORT"]
    signal_id: str
    option_type: Literal["c", "p"]
    spot: numbers.Number = None
    chain_spot: numbers.Number = None

    


# @pydantic_dataclass(config=ConfigDict(arbitrary_types_allowed=True))
# class RiskManagerOrderRequest:
#     """
#     Dataclass representing an order request generated for risk manager to process.
#     """

#     symbol: str
#     date: Union[datetime, str]
#     signal_id: str
#     max_close: numbers.Number
#     direction: str
#     tick_cash: numbers.Number
#     option_type: str
