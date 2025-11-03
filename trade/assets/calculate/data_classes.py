from typing import Optional, Dict, Tuple
from datetime import datetime
import pandas as pd
from pydantic.dataclasses import dataclass as pydantic_dataclass
from pydantic import Field, ConfigDict
from EventDriven.types import PositionEffect
from trade.helpers.Logging import setup_logger
from trade.assets.calculate.enums import AttributionModel
logger = setup_logger('trade.assets.calculate.data_classes')

SYMBOL_PAYLOADS: Dict[Tuple[str, datetime, datetime], "SymbolPayload"] = {}

@pydantic_dataclass
class TradePnlInfo:
    """
    Trade PnL Information Data Class
    """
    position_effect_close: float
    effect_date: datetime
    tmin0_close: float
    tmin1_close: float
    position_effect: PositionEffect
    quantity: float = Field(default=1.0)
    position_entry_price: float = Field(default=0.0)
    trade_pnl: Optional[float] = None

    def calculate_trade_pnl(self) -> float:
        """
        Calculate Trade PnL based on position effect
        Returns:
            float: Calculated Trade PnL
        """
        ## Calculate Trade PnL based on position effect
        if self.position_effect == PositionEffect.OPEN:
            ## If it is an open position, trade pnl is T-1 close - entry price
            ## We're using this because the original attribution function calculates DoD
            ## T-1 Close - Entry Price adjustment is equivalent to Entry Price - T-0 Close for T-0 pnl
            self.trade_pnl = (self.tmin1_close - self.position_effect_close) * self.quantity

        elif self.position_effect == PositionEffect.CLOSE:
            ## If it is a close position, trade pnl is position effect close - position_entry_price
            ## This reflects the actual realized pnl on closing the position
            self.trade_pnl = (self.position_effect_close - self.position_entry_price) * self.quantity
        else:
            raise ValueError("Invalid Position Effect")
        # self.trade_pnl = 
        return self.trade_pnl
    
    def __repr__(self) -> str:
        return f"<TradePnlInfo(effect_date={self.effect_date}, position_effect={self.position_effect}, trade_pnl={self.trade_pnl})>"
    

@pydantic_dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class SymbolPayload:
    """
    Data class to hold symbol payload data.
    """
    symbol: str
    datetime: datetime
    spot: pd.Series

    def __repr__(self):
        return f"SymbolPayload(symbol={self.symbol}, datetime={self.datetime.date()})"


@pydantic_dataclass(
    config=ConfigDict(arbitrary_types_allowed=True)
)
class OptionPnlPayload:
    """
    Data class to hold option payload data.

    Attributes:
        opttick (str): The option ticker symbol.
        date (datetime): The date of the payload.
        vol (pd.DataFrame | pd.Series): The volatility data.
        spot (pd.DataFrame | pd.Series): The spot price data.
        greeks (pd.DataFrame): The Greeks data.
        asset_payload (SymbolPayload): The associated asset payload.
        rates_payload (SymbolPayload): The associated rates payload.
        dod_change (Optional[pd.DataFrame]): Day-over-day change data.
        attribution (Optional[pd.DataFrame]): Attribution data.
    """
    opttick: str
    date: datetime
    vol: pd.DataFrame | pd.Series
    spot: pd.DataFrame | pd.Series
    greeks: pd.DataFrame
    asset_payload: SymbolPayload
    rates_payload: SymbolPayload
    attribution_model: AttributionModel = AttributionModel.UNDEFINED
    dod_change: Optional[pd.DataFrame] = Field(default_factory=pd.DataFrame)
    attribution: Optional[pd.DataFrame] = Field(default_factory=pd.DataFrame)
    vol_model_dynamics: Optional[pd.DataFrame] = Field(default_factory=pd.DataFrame)

    def __repr__(self):
        return f"{self.__class__.__name__}(opttick={self.opttick}, date={self.date.date()})"
    
    def __hash__(self):
        return hash((self.opttick, self.date))