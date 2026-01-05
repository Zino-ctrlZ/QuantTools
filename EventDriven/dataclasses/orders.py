from pydantic.dataclasses import dataclass
from pydantic import ConfigDict
from typing import Union, Literal
from datetime import datetime, date
import numbers


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
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
    is_tick_cash_scaled: bool = False
