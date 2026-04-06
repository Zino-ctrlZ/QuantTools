from datetime import date
from enum import Enum
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict
from EventDriven.helpers import parse_signal_id, generate_signal_id, parse_position_id
from trade.helpers.Logging import setup_logger

logger = setup_logger("EventDriven.types")


class OptionFloat(float):
    """Custom float type for option-related values to allow for future extensions or validations."""

    def __new__(cls, value):
        return super().__new__(cls, value)

    def __init__(self, value, dollar_normalized=False):
        super().__init__()
        self.dollar_normalized = dollar_normalized


class Metrics(TypedDict):
    spread_pct_ratio: Optional[float]
    spread_oi: Optional[float]
    min_dte: Optional[int]
    max_dte: Optional[int]
    min_moneyness: Optional[float]
    max_moneyness: Optional[float]


class Scores(TypedDict):
    moneyness_score: Optional[float]
    dte_score: Optional[float]
    mid_score: Optional[float]
    pct_spread_score: Optional[float]
    oi_score: Optional[float]
    theta_burden_score: Optional[float]


class SignalID(str):
    """Unique identifier for a trading signal.

    Format:
        {TICKER}{YYYYMMDD}{SIGNAL_TYPE}(::_{STRATEGY_SLUG}) (optional strategy slug prefix)
    """

    __slots__ = ("ticker", "date", "direction", "strategy_slug")

    def __new__(cls, signal_id: str) -> "SignalID":
        return super().__new__(cls, signal_id)

    def __init__(self, signal_id: str) -> None:
        if "::" in signal_id:
            strategy_slug, signal_id = signal_id.split("::", 1)  # Remove strategy slug if present
            self.strategy_slug = strategy_slug
        else:
            self.strategy_slug = None
        parsed = parse_signal_id(signal_id)
        self.ticker = parsed["ticker"]
        self.date = parsed["date"]
        self.direction = parsed["direction"]

    def parse(self) -> Dict[str, Any]:
        return parse_signal_id(self)

    @staticmethod
    def generate(underlier: str, date: pd.Timestamp, signal_type: str, strategy_slug: str = None) -> "SignalID":
        signal_id = generate_signal_id(underlier, date, signal_type)
        if strategy_slug:
            signal_id = strategy_slug + "::" + signal_id
        return SignalID(signal_id)

    def __repr__(self) -> str:
        return f"SignalID({str(self)})"

    def __str__(self):
        return super().__str__()


class TradeID(str):
    """Unique identifier for a trade execution.

    Format:
        &L:{LONG_LEG_1}&L:{LONG_LEG_2}...&S:{SHORT_LEG_1}&S:{SHORT_LEG_2}...
    """

    __slots__ = ("meta", "legs")

    def __new__(cls, trade_id: str) -> "TradeID":
        return super().__new__(cls, trade_id)

    def __init__(self, trade_id: str) -> None:
        self.meta, self.legs = parse_position_id(trade_id)

    def __repr__(self) -> str:
        return f"TradeID({str(self)})"

    def __str__(self):
        return super().__str__()


class OrderDataDict(TypedDict):
    trade_id: str
    long: List[str]
    short: List[str]
    close: float
    quantity: int


class OrderDict(TypedDict):
    result: str
    signal_id: str
    map_signal_id: str
    date: date
    data: OrderDataDict
    metrics: Metrics | None
    scores: Scores | None


class PositionsDict(TypedDict):
    position: OrderDataDict
    quantity: int
    entry_price: float
    market_value: float
    signal_id: str


class ResultsEnum(Enum):
    SUCCESSFUL = "SUCCESSFUL"
    MONEYNESS_TOO_TIGHT = "MONEYNESS_TOO_TIGHT"
    NO_ORDERS = "NO_ORDERS"
    UNSUCCESSFUL = "UNSUCCESSFUL"
    IS_HOLIDAY = "IS_HOLIDAY"
    UNAVAILABLE_CONTRACT = "NO LISTED CONTRACTS"
    MAX_PRICE_TOO_LOW = "MAX_PRICE_TOO_LOW"
    TOO_ILLIQUID = "TOO_ILLIQUID"
    NO_TRADED_CLOSE = "NO_TRADED_CLOSE"
    IS_WEEKEND = "IS_WEEKEND"
    NO_CONTRACTS_FOUND = "NO_CONTRACTS_FOUND"
    POSITION_SIZE_ZERO = "POSITION_SIZE_ZERO"


class EventTypes(Enum):
    SIGNAL = "SIGNAL"
    ORDER = "ORDER"
    FILL = "FILL"
    MARKET = "MARKET"
    EXERCISE = "EXERCISE"
    ROLL = "ROLL"
    ADJUST = "ADJUST"
    CLOSE = "CLOSE"
    OPEN = "OPEN"
    HOLD = "HOLD"


class OrderStatus(Enum):
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"
    FAILED = "FAILED"
    CONFIRMED = "CONFIRMED"


class PositionEffect(Enum):
    OPEN = "OPEN"
    CLOSE = "CLOSE"


class SignalTypes(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    CLOSE = "CLOSE"


class FillDirection(Enum):
    BUY = "BUY"
    SELL = "SELL"
    EXERCISE = "EXERCISE"


class PositionAdjustmentReason(Enum):
    DTE_ROLL = "DTE_ROLL"
    MONEYNESS_ROLL = "MONEYNESS_ROLL"
    LIMIT_BREACH = "LIMIT_BREACH"


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
    long: List[str]
    short: List[str]
    close: float
    quantity: int

    def __getitem__(self, key):
        """Get item like a dict, dict[key]"""
        return getattr(self, key)

    def __setitem__(self, key, value):
        """Set item like a dict, dict[key] = value"""
        setattr(self, key, value)

    def __repr__(self):
        """String representation of OrderData"""
        return f"OrderData(trade_id={self.trade_id}, quantity={self.quantity})"

    def get(self, key: str, default: Any = None) -> Any:
        """Get item like a dict, dict.get()"""
        return getattr(self, key, default)

    def keys(self):
        """Return keys like a dict"""
        return self.__dict__.keys()

    def items(self):
        """Return items like a dict"""
        return [(key, getattr(self, key)) for key in self.keys()]

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "OrderData":
        """Convert dictionary to OrderData dataclass"""
        return OrderData(**d)

    def to_dict(self) -> OrderDataDict:
        """Convert OrderData dataclass to dictionary"""
        return OrderDataDict(
            trade_id=self.trade_id, long=self.long, short=self.short, close=self.close, quantity=self.quantity
        )


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
    metrics: Metrics = None
    scores: Scores = None

    def __getitem__(self, key):
        """Get item like a dict, dict[key]"""
        return getattr(self, key)

    def __setitem__(self, key, value):
        """Set item like a dict, dict[key] = value"""
        setattr(self, key, value)

    def __repr__(self):
        """String representation of Order"""
        return f"Order(signal_id={self.signal_id}), data={self.data}, result={self.result}, metrics={self.metrics}, scores={self.scores})"

    def get(self, key: str, default: Any = None) -> Any:
        """Get item like a dict, dict.get()"""
        return getattr(self, key, default)

    def keys(self):
        """Return keys like a dict"""
        return self.__dict__.keys()

    def items(self):
        """Return items like a dict"""
        return [(key, getattr(self, key)) for key in self.keys()]

    def to_dict(self) -> OrderDict:
        """Convert Order dataclass to dictionary"""
        # Convert the nested OrderData object to dict
        data_dict = {
            "trade_id": self.data.trade_id,
            "long": self.data.long,
            "short": self.data.short,
            "close": self.data.close,
            "quantity": self.data.quantity,
        }
        data_dict = OrderDataDict(**data_dict)
        order_dict = OrderDict(
            result=self.result,
            signal_id=self.signal_id,
            map_signal_id=self.map_signal_id,
            date=self.date,
            data=data_dict,
            metrics=self.metrics,
            scores=self.scores,
        )

        # Return the main dictionary
        return order_dict

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Order":
        """Convert dictionary to Order dataclass"""
        # Extract the nested data dict
        data_dict = d["data"]
        metrics = d.get("metrics", None)
        scores = d.get("scores", None)

        # Convert nested data dict to OrderData object
        if data_dict is None:
            data_dict = {"trade_id": None, "long": None, "short": None, "close": None, "quantity": None}

        if metrics is not None:
            d["metrics"] = Metrics(
                spread_pct_ratio=metrics["spread_pct_ratio"],
                spread_oi=metrics["spread_oi"],
                min_dte=metrics.get("min_dte", None),
                max_dte=metrics.get("max_dte", None),
                min_moneyness=metrics.get("min_moneyness", None),
                max_moneyness=metrics.get("max_moneyness", None),
            )
        else:
            d["metrics"] = None

        if scores is not None:
            d["scores"] = Scores(
                moneyness_score=scores.get("moneyness_score", None),
                dte_score=scores.get("dte_score", None),
                mid_score=scores.get("mid_score", None),
                pct_spread_score=scores.get("pct_spread_score", None),
                oi_score=scores.get("oi_score", None),
                theta_burden_score=scores.get("theta_burden_score", None),
            )
        else:
            d["scores"] = None

        order_data = OrderData(
            trade_id=data_dict["trade_id"],
            long=data_dict.get("long", []),
            short=data_dict.get("short", []),
            close=data_dict.get("close", np.nan),
            quantity=data_dict.get("quantity", 0),
        )

        # Create and return Order object
        raw_date = d.get("date")
        date = pd.to_datetime(raw_date).date() if raw_date is not None else None
        return Order(
            result=d["result"],
            signal_id=d["signal_id"],
            map_signal_id=d["map_signal_id"],
            date=date,
            data=order_data,
            metrics=d["metrics"],
            scores=d["scores"],
        )
