from datetime import datetime
from typing import Union, Dict, Any
import numbers
from pydantic.dataclasses import dataclass as pydantic_dataclass
from pydantic import ConfigDict, Field

@pydantic_dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class AtTimeOptionData:
    opttick: str
    date: Union[datetime, str]
    close: numbers.Number
    bid: numbers.Number
    ask: numbers.Number
    midpoint: numbers.Number


@pydantic_dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class AtTimePositionData:
    position_id: str
    date: Union[datetime, str]
    close: numbers.Number
    bid: numbers.Number
    ask: numbers.Number
    midpoint: numbers.Number
    skips: Dict[str, Any] = Field(default_factory=dict)
