from pydantic.dataclasses import dataclass as pydantic_dataclass
from pydantic import ConfigDict, Field
from typing import Optional, List
from datetime import datetime
from EventDriven.types import Order
from EventDriven.dataclasses.orders import OrderRequest
from EventDriven.riskmanager.market_data import AtIndexResult
from EventDriven.dataclasses.timeseries import AtTimePositionData
from EventDriven.dataclasses.limits import PositionLimits
from EventDriven.riskmanager.actions import RMAction

@pydantic_dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class NewPositionState:
    trade_id: str
    order: Order
    request: OrderRequest
    at_time_data: AtTimePositionData
    undl_at_time_data: AtIndexResult
    limits: Optional[PositionLimits] = None

    def __repr__(self):
        return f"NewPositionState(trade_id={self.trade_id})"
    
@pydantic_dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class PortfolioMetaInfo:
    """
    Dataclass to hold portfolio backtest information.
    """

    portfolio_name: Optional[str] = None
    initial_cash: Optional[float] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    t_plus_n: Optional[int] = None
    is_backtest: Optional[bool] = True
    

@pydantic_dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class PositionState:
    """
    Dataclass to hold position state information.
    """

    trade_id: str
    signal_id: str
    underlier_tick: str
    quantity: int
    entry_price: float
    current_position_data: AtTimePositionData
    current_underlier_data: AtIndexResult
    pnl: float
    last_updated: datetime
    action: Optional[RMAction] = None

    def __repr__(self):
        return f"PositionState(date={self.last_updated}, trade_id={self.trade_id}, quantity={self.quantity}, pnl={self.pnl}, signal_id={self.signal_id})"

@pydantic_dataclass(config=ConfigDict(arbitrary_types_allowed=True), frozen=True)
class PortfolioState:
    """
    Dataclass to hold portfolio state information.
    """

    cash: float
    positions: List[PositionState] = Field(description="List of current position states in the portfolio.")
    pnl: float = 0.0
    total_value: float = 0.0
    last_updated: Optional[datetime] = None


@pydantic_dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class PositionAnalysisContext:
    """
    Read-only snapshot of everything a cog needs to reason about positions.
    This is passed into every cog on each analysis run.
    """

    date: datetime
    portfolio: PortfolioState
    portfolio_meta: PortfolioMetaInfo

@pydantic_dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class CogActions:
    """
    Holds all actions recommended by a cog for a list of positions.
    """
    date: datetime
    source_cog: str
    opinions: List[PositionState]

    def __repr__(self):
        return f"CogActions(source={self.source_cog}, date={self.date}, num_opinions={len(self.opinions)})"
    

@pydantic_dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class StrategyChangeMeta:
    """
    Metadata about changes made to a strategy's positions during analysis.
    """

    date: datetime
    actionables: List[PositionState] 

    def __repr__(self):
        actionables = [x for x in self.actionables if x.action.type.value != "HOLD"]
        return f"StrategyChangeMeta(date={self.date}, num_actions={len(actionables)})"