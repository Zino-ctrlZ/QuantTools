from datetime import datetime, date
from enum import Enum

class ResultsEnum(Enum):
    SUCCESSFUL = 'SUCCESSFUL'
    MONEYNESS_TOO_TIGHT = 'MONEYNESS_TOO_TIGHT'
    NO_ORDERS = 'NO_ORDERS'
    UNSUCCESSFUL = 'UNSUCCESSFUL'
    IS_HOLIDAY = 'IS_HOLIDAY'
    UNAVAILABLE_CONTRACT = 'NO LISTED CONTRACTS' 
    MAX_PRICE_TOO_LOW = 'MAX_PRICE_TOO_LOW'
    TOO_ILLIQUID = 'TOO_ILLIQUID'
    NO_TRADED_CLOSE = 'NO_TRADED_CLOSE'
    IS_WEEKEND = 'IS_WEEKEND'
    NO_CONTRACTS_FOUND = 'NO_CONTRACTS_FOUND'
    
class EventTypes(Enum): 
  SIGNAL = 'SIGNAL'
  ORDER = 'ORDER'
  FILL = 'FILL'
  MARKET = 'MARKET'
  EXERCISE = 'EXERCISE'
  ROLL = 'ROLL'
  
class SignalTypes(Enum):
  LONG = 'LONG'
  SHORT = 'SHORT'
  CLOSE = 'CLOSE'
  
class FillDirection(Enum):
  BUY = 'BUY'
  SELL = 'SELL'
  EXERCISE = 'EXERCISE'

class OpenPositionAction(Enum):
   ROLL = 'ROLL'
   EXERCISE = 'EXERCISE'
   HOLD = 'HOLD'
   ADJUST = 'ADJUST'
   CLOSE = 'CLOSE'

class PositionAdjustmentReason(Enum):
   DTE_ROLL = 'DTE_ROLL'
   MONEYNESS_ROLL = 'MONEYNESS_ROLL'
   LIMIT_BREACH = 'LIMIT_BREACH'


from dataclasses import dataclass
from typing import Any, Dict



@dataclass
class OrderData:
    trade_id: str
    long: [str]
    short: [str]
    close: float
    quantity: int

    @staticmethod 
    def from_dict(d: Dict[str, Any]) -> 'OrderData':
        """Convert dictionary to OrderData dataclass"""
        return OrderData(**d)

    def to_dict(self) -> Dict[str, Any]: 
        """Convert OrderData dataclass to dictionary"""
        return {
            'trade_id': self.trade_id,
            'long': self.long,
            'short': self.short,
            'close': self.close,
            'quantity': self.quantity
        }


@dataclass
class Order:
    result: str
    signal_id: str
    map_signal_id: str
    date: date
    data: OrderData

    def __getitem__(self, key):
        """Get item like a dict, dict[key]"""
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        """Set item like a dict, dict[key] = value"""
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        """Get item like a dict, dict.get()"""
        return getattr(self, key, default)
    
    def keys(self): 
        """Return keys like a dict"""
        return [field.name for field in self.__dataclass_fields__.values()]

    def items(self):
        """Return items like a dict"""
        return [(key, getattr(self, key)) for key in self.keys()]

    def to_dict(self) -> Dict[str, Any]:
        """Convert Order dataclass to dictionary"""
        # Convert the nested OrderData object to dict
        data_dict = {
            'trade_id': self.data.trade_id,
            'long': self.data.long,
            'short': self.data.short,
            'close': self.data.close,
            'quantity': self.data.quantity
        }
        
        # Return the main dictionary
        return {
            'result': self.result,
            'signal_id': self.signal_id,
            'map_signal_id': self.map_signal_id,
            'date': self.date,
            'data': data_dict
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'Order':
        """Convert dictionary to Order dataclass"""
        # Extract the nested data dict
        data_dict = d['data']
        
        # Convert nested data dict to OrderData object
        order_data = OrderData(
            trade_id=data_dict['trade_id'],
            long=data_dict['long'],
            short=data_dict['short'],
            close=data_dict['close'],
            quantity=data_dict['quantity']
        )
        
        # Create and return Order object
        return Order(
            result=d['result'],
            signal_id=d['signal_id'],
            map_signal_id=d['map_signal_id'],
            date=d['date'],
            data=order_data
        )