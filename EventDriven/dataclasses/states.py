from pydantic.dataclasses import dataclass as pydantic_dataclass
from pydantic import ConfigDict, Field
from typing import Optional, Dict, Any
from datetime import datetime
from EventDriven.types import Order
from EventDriven.dataclasses.orders import OrderRequest
from EventDriven.riskmanager.market_data import AtIndexResult
from EventDriven.dataclasses.timeseries import AtTimePositionData

@pydantic_dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class NewPositionState:
    trade_id: str
    order: Order
    request: OrderRequest
    at_time_data: AtTimePositionData
    undl_at_time_data: AtIndexResult

    def __repr__(self):
        return f"NewPositionState(trade_id={self.trade_id})"

@pydantic_dataclass(config=ConfigDict(arbitrary_types_allowed=True), frozen=True)
class PortfolioState:
    """
    Dataclass to hold portfolio state information.
    """

    cash: float
    positions: Dict[str, Any] = Field(default_factory=dict)
    pnl: float = 0.0
    total_value: float = 0.0
    last_updated: Optional[datetime] = None