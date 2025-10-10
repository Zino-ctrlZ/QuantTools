from enum import Enum

ALPHA_LEVELS = {
    1: 0.25,   # very passive
    2: 0.325,
    3: 0.50,   # mid
    4: 0.75,
    5: 1.00    # very aggressive
}

def get_tick_increments(mid: float, volume: float | None)-> float:
    """Return tick size based on mid price."""
    if volume is None or volume < 200:
        if mid < 3.00:
            return 0.05
        else:
            return 0.10
    else:
        return 0.01
    
class Side(str, Enum):
    """Enum for order sides."""
    BUY = "LONG"
    SELL = "SHORT"

class SideInt(int, Enum):
    """Enum for order sides as integers."""
    BUY = 1
    SELL = -1

class PositionEffect(str, Enum):
    """Enum for position effects."""
    OPEN = "OPEN"
    CLOSE = "CLOSE"

def limit_price(
    side: Side,
    bid: float,
    ask: float,
    level: int,                 # 1..5 (1=passive, 3=mid, 5=aggressive)
    *,
    day_volume: float | None = None,  # very optional
    tick: float = None                 # price increment
) -> float:
    """Side-aware, novice-friendly execution price."""
    # Basic sanity
    tick = tick or get_tick_increments((bid + ask) / 2, day_volume)
    if bid <= 0 or ask <= 0 or ask < bid:
        return round((bid + ask) / 2 / tick) * tick

    # 1) Side-aware endpoints
    # BUY: passive=bid, aggressive=ask; SELL: passive=ask, aggressive=bid
    passive, aggressive = (bid, ask) if side == Side.BUY else (ask, bid)

    # 2) Base alpha from level (1..5 → 0.00, 0.25, 0.50, 0.75, 1.00)
    lvl = max(1, min(5, int(level)))
    alpha = ALPHA_LEVELS[lvl]

    # 3) Simple nudge (optional): look at spread & volume only
    spread_pct = (ask - bid) / ((ask + bid) / 2)
    tight = spread_pct <= 0.1        # Tight spread <= 10%
    wide  = spread_pct > 0.20         # Wide spread > 20%
    vol_ok   = (day_volume or 0) >= 200   # Volume OK if >= 200 options
    vol_low  = (day_volume or 0) <= 50    # Volume Low if <= 50 options


    ## No implementation for tight & vol_ok yet
    ## if wide & vol_low, be more aggressive by increasing lvl by 1
    if wide and vol_low and lvl > 1:
        alpha = ALPHA_LEVELS[min(lvl+1, 5)]   # drop one step

    # 4) Final price = passive + alpha * (aggressive - passive)
    raw = passive + alpha * (aggressive - passive)

    # 5) Tick-round
    return round((raw / tick) * tick, 2)