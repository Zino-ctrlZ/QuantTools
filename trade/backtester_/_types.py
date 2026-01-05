from enum import Enum

class Side(str, Enum):
    """Enum for order sides."""

    ## Side should be LONG/SHORT. Meanwhilem PositionEffect is OPEN/CLOSE
    BUY = "LONG"
    SELL = "SHORT"
    LONG = "LONG"
    SHORT = "SHORT"


class SideInt(int, Enum):
    """Enum for order sides as integers."""

    BUY = 1
    SELL = -1
