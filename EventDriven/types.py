from datetime import datetime, date
from enum import Enum
import pandas as pd
from dataclasses import dataclass
from typing import Any, Dict

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
  ADJUST = 'ADJUST'
  CLOSE = 'CLOSE'
  OPEN = 'OPEN'
  HOLD = 'HOLD'

class OrderStatus(Enum):
    FILLED = 'FILLED'
    CANCELLED = 'CANCELLED'
    EXPIRED = 'EXPIRED'
    FAILED = 'FAILED'
    CONFIRMED = 'CONFIRMED'
class PositionEffect(Enum):
  OPEN = 'OPEN'
  CLOSE = 'CLOSE'
  
class SignalTypes(Enum):
    LONG = 'LONG'
    SHORT = 'SHORT'
    CLOSE = 'CLOSE'

class FillDirection(Enum):
    BUY = 'BUY'
    SELL = 'SELL'
    EXERCISE = 'EXERCISE'

class PositionAdjustmentReason(Enum):
   DTE_ROLL = 'DTE_ROLL'
   MONEYNESS_ROLL = 'MONEYNESS_ROLL'
   LIMIT_BREACH = 'LIMIT_BREACH'


@dataclass
class OrderData:
    """
        Represents detailed execution data for a trading order.
    
    This class contains the specific trade execution details including
    position information, pricing, and quantities. It's used as a
    nested data structure within the Order class to organize trade-specific
    information.
    
    Attributes:
        trade_id (str): Unique identifier for the trade execution
        long (List[str]): List of symbols/positions for long positions
        short (List[str]): List of symbols/positions for short positions
        close (float): Closing price or execution price for the trade
        quantity (int): Number of shares/contracts in the trade
    
    Example:
        >>> order_data = OrderData(
        ...     trade_id='&L:BA20260515C285&S:BA20260515C290',
        ...     long=['BA20260515C285'],
        ...     short=['BA20260515C290'],
        ...     close=0.4250,
        ...     quantity=1
        ... )
    """
    trade_id: str
    long: [str]
    short: [str]
    close: float
    quantity: int

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
    """
    Represents a trading order with signal information and execution data.
    
    This class encapsulates all the information related to a trading order,
    including the signal that generated it, execution results, and detailed
    trade data. It provides dictionary-like access methods for compatibility
    with existing code that expects dictionary objects.
    
    Attributes:
        result (str): The execution result of the order (e.g., 'SUCCESSFUL', 'FAILED')
        signal_id (str): Unique identifier for the signal that generated this order
        map_signal_id (str): Mapped signal identifier for tracking purposes
        date (date): The date when the signal was generated
        data (OrderData): Detailed trade execution data including positions and pricing
    
    Methods:
        __getitem__(key): Get attribute value using dictionary-style access
        __setitem__(key, value): Set attribute value using dictionary-style access
        get(key, default): Get attribute value with optional default (dict-like)
        keys(): Return list of attribute names (dict-like)
        items(): Return list of (key, value) pairs (dict-like)
        to_dict(): Convert Order object to dictionary representation
        from_dict(d): Create Order object from dictionary (static method)
    
    Example:
        >>> order = Order(
        ...     result='SUCCESSFUL',
        ...     signal_id='BA20250701LONG',
        ...     map_signal_id='BA20250701LONG',
        ...     date=date(2025, 10, 7),
        ...     data=OrderData(...)
        ... )
        >>> order['signal_id']  # Dictionary-style access
        'BA20250701LONG'
        >>> order_dict = order.to_dict()  # Convert to dict
        >>> restored_order = Order.from_dict(order_dict)  # Convert back
    """
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
            date=pd.to_datetime(d['date']).date(),
            data=order_data
        )