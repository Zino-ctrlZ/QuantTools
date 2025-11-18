from pydantic.dataclasses import dataclass as pydantic_dataclass
from pydantic import ConfigDict
from typing import Optional

@pydantic_dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class PositionLimits:
    """
    Dataclass to hold position limits information.
    """

    delta: Optional[float] = None
    gamma: Optional[float] = None
    vega: Optional[float] = None
    theta: Optional[float] = None
    dte: Optional[int] = None
    moneyness: Optional[float] = None
    exercise: bool = False
