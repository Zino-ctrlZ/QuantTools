from datetime import datetime
from typing import Union, Dict, Any
import numbers
from pydantic.dataclasses import dataclass as pydantic_dataclass
from pydantic import ConfigDict, Field


@pydantic_dataclass(config=ConfigDict(arbitrary_types_allowed=True), kw_only=True, frozen=True)
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
    is_qty_scaled: bool = False

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
    
    def get_spread(self) -> numbers.Number:
        """
        Get the spread based on the bid and ask prices.
        """
        if self.bid <= 0 or self.ask <= 0 or self.ask < self.bid:
            return 0.0  # Invalid spread
        return self.ask - self.bid


@pydantic_dataclass(config=ConfigDict(arbitrary_types_allowed=True), kw_only=True, frozen=True)
class AtTimeOptionData(AtTimeBaseData):
    opttick: str


@pydantic_dataclass(config=ConfigDict(arbitrary_types_allowed=True), kw_only=True, frozen=True)
class AtTimePositionData(AtTimeBaseData):
    position_id: str
    skips: Dict[str, Any] = Field(default_factory=dict)
