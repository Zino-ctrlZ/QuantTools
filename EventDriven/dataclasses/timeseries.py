from datetime import datetime
from typing import Union, Dict, Any
import numbers
from pydantic.dataclasses import dataclass as pydantic_dataclass
from pydantic import ConfigDict, Field


@pydantic_dataclass(config=ConfigDict(arbitrary_types_allowed=True), kw_only=True)
class AtTimeBaseData:
    """
    Base dataclass for at-time data representations.
    This is solely for market data at a specific time for options and positions.
    """

    date: Union[datetime, str]
    close: numbers.Number
    bid: numbers.Number
    ask: numbers.Number
    midpoint: numbers.Number
    delta: numbers.Number
    gamma: numbers.Number
    vega: numbers.Number
    theta: numbers.Number

    ## USE_{ITEMS} for choosing btwn bid/ask/close/midpoint
    use_price: str = "midpoint"

    def get_price(self) -> numbers.Number:
        """
        Get the price based on the specified use_price attribute.
        """
        if self.use_price == "close":
            return self.close
        elif self.use_price == "bid":
            return self.bid
        elif self.use_price == "ask":
            return self.ask
        elif self.use_price == "midpoint":
            return self.midpoint
        else:
            raise ValueError(f"Invalid use_price value: {self.use_price}")


@pydantic_dataclass(config=ConfigDict(arbitrary_types_allowed=True), kw_only=True)
class AtTimeOptionData(AtTimeBaseData):
    opttick: str


@pydantic_dataclass(config=ConfigDict(arbitrary_types_allowed=True), kw_only=True)
class AtTimePositionData(AtTimeBaseData):
    position_id: str
    skips: Dict[str, Any] = Field(default_factory=dict)
