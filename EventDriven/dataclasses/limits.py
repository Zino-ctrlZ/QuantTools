from pydantic.dataclasses import dataclass as pydantic_dataclass
from pydantic import ConfigDict
from typing import Optional, Union
from datetime import datetime, date

@pydantic_dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class PositionLimits:
    """
    Dataclass to hold position limits information.
    """
    creation_date: Optional[Union[str, datetime, date]] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    vega: Optional[float] = None
    theta: Optional[float] = None
    dte: Optional[int] = None
    moneyness: Optional[float] = None
    exercise: bool = False
