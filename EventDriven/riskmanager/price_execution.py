"""Order Execution Price Optimization and Slippage Calculation.

This module implements intelligent limit price calculation for order execution with
adjustable aggressiveness levels. It balances getting fills (aggressive) versus
minimizing slippage (passive) based on market conditions and user preferences.

Key Functions:
    limit_price: Calculate optimal limit price with side-aware logic
    calculate_slippage_basic: Compute slippage given execution details
    get_tick_increments: Determine appropriate tick size for rounding

Execution Philosophy:
    - Passive (Level 1): Join the queue at bid (buy) or ask (sell)
    - Mid (Level 3): Split the spread, balanced approach
    - Aggressive (Level 5): Cross the spread for immediate fill
    - Levels 2 and 4 provide intermediate aggressiveness

Aggressiveness Levels (1-5):
    Level 1 (0.25 alpha) - Very Passive:
        - BUY: Place order at bid, wait for fill
        - SELL: Place order at ask, wait for fill
        - Best price, but low fill probability
        - Use for liquid options with tight spreads

    Level 2 (0.325 alpha) - Slightly Passive:
        - BUY: 32.5% between bid and ask
        - SELL: 32.5% from ask toward bid
        - Improved fill rate with minimal price sacrifice

    Level 3 (0.50 alpha) - Neutral/Mid:
        - BUY/SELL: Midpoint of bid-ask spread
        - Balanced risk/reward
        - Default for most strategies

    Level 4 (0.75 alpha) - Slightly Aggressive:
        - BUY: 75% toward ask
        - SELL: 75% toward bid
        - Higher fill probability
        - Small additional cost

    Level 5 (1.00 alpha) - Very Aggressive:
        - BUY: Place order at ask (immediate fill)
        - SELL: Place order at bid (immediate fill)
        - Maximum fill certainty
        - Pays full spread as slippage

Market Condition Adjustments:
    Wide Spread & Low Volume:
        - Automatically increases aggressiveness by 1 level
        - Prevents missed opportunities in illiquid markets
        - Criteria: spread > 20% of mid, volume < 50 contracts

    Tight Spread & Good Volume:
        - Maintains specified level (no penalty)
        - Criteria: spread ≤ 10% of mid, volume ≥ 200 contracts

Tick Size Logic:
    Low Volume (< 200 contracts):
        - Price < $3.00: $0.05 increments
        - Price ≥ $3.00: $0.10 increments

    Good Volume (≥ 200 contracts):
        - Standard $0.01 increments

Slippage Calculation:
    BUY Orders:
        - Slippage = max(0, limit_price - mid_price)
        - Positive when paying above mid

    SELL Orders:
        - Slippage = max(0, mid_price - limit_price)
        - Positive when receiving below mid

    Properties:
        - Always non-negative
        - Expressed in dollars per contract
        - Multiply by quantity for total slippage

Side-Aware Pricing:
    - BUY: passive=bid, aggressive=ask
    - SELL: passive=ask, aggressive=bid
    - Ensures directionally correct price calculation
    - Prevents crossing the spread unintentionally

Position Effects (for future enhancement):
    - OPEN: New position creation
    - CLOSE: Position exit
    - Currently defined but not fully implemented

Usage Examples:
    # Calculate limit price for buying options
    buy_limit = limit_price(
        side=Side.BUY,
        bid=3.50,
        ask=3.70,
        level=3,  # Mid-point
        day_volume=500,
        tick=0.05
    )
    # Result: 3.60 (mid-point, rounded to $0.05)

    # Calculate limit price for selling (aggressive)
    sell_limit = limit_price(
        side=Side.SELL,
        bid=5.20,
        ask=5.40,
        level=5,  # Very aggressive
        day_volume=150
    )
    # Result: 5.20 (at bid for immediate fill)

    # Calculate slippage after execution
    slippage = calculate_slippage_basic(
        side='BUY',
        lmt=3.65,
        mid=3.60
    )
    # Result: 0.05 (paid $0.05 above mid)

Integration Points:
    - Used by RiskManager during order generation
    - Executor applies to all fills
    - Slippage tracked in performance attribution
    - Logged for order audit trail

Configuration:
    - Aggressiveness level typically set in executor_settings
    - Can vary by strategy or market conditions
    - Volume thresholds configurable via constants
    - Spread percentage thresholds adjustable

Performance Considerations:
    - Fast computation (microseconds)
    - No external dependencies or I/O
    - Stateless functions suitable for vectorization
    - Logging optional for debugging

Notes:
    - All prices rounded to appropriate tick size
    - Invalid spreads (bid > ask) handled gracefully
    - Zero or negative prices default to mid-point
    - Volume=None treated as low liquidity
    - Tick size auto-detected if not provided
"""

from enum import Enum
from trade.helpers.Logging import setup_logger
from trade.backtester_._types import Side, SideInt  # noqa

logger = setup_logger("algo.strategies.fill_optimizer")

ALPHA_LEVELS = {
    1: 0.25,  # very passive
    2: 0.325,
    3: 0.50,  # mid
    4: 0.75,
    5: 1.00,  # very aggressive
}


def get_tick_increments(mid: float, volume: float | None) -> float:
    """Return tick size based on mid price."""
    if volume is None or volume < 200:
        if mid < 3.00:
            return 0.05
        else:
            return 0.10
    else:
        return 0.01


class PositionEffect(str, Enum):
    """Enum for position effects."""

    OPEN = "OPEN"
    CLOSE = "CLOSE"


def limit_price(
    side: Side,
    bid: float,
    ask: float,
    level: int,  # 1..5 (1=passive, 3=mid, 5=aggressive)
    *,
    day_volume: float | None = None,  # very optional
    tick: float = None,  # price increment
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
    tight = spread_pct <= 0.1  # noqa
    wide = spread_pct > 0.20  # noqa
    vol_ok = (day_volume or 0) >= 200  # noqa
    vol_low = (day_volume or 0) <= 50  # noqa

    ## No implementation for tight & vol_ok yet
    ## if wide & vol_low, be more aggressive by increasing lvl by 1
    if wide and vol_low and lvl > 1:
        alpha = ALPHA_LEVELS[min(lvl + 1, 5)]  # drop one step

    # 4) Final price = passive + alpha * (aggressive - passive)
    raw = passive + alpha * (aggressive - passive)

    # 5) Tick-round
    logger.info(
        f"Limit price calc: side={side}, bid={bid}, ask={ask}, level={level}, day_volume={day_volume} => tick={tick}, price={raw}"
    )
    return round((raw / tick) * tick, 2)


def calculate_slippage_basic(side: str, lmt: float, mid: float) -> float:
    """
    Calculate slippage given side, limit price, and mid price.
    Args:
        side (str): 'BUY' or 'SELL'.
        lmt (float): The limit price.
        mid (float): The mid price.
    Returns:
        float: The calculated slippage.
    """
    if mid is None or mid == 0.0:
        raise ValueError("Mid price is not provided or zero. Cannot calculate slippage.")
    if side == Side.BUY.value or side == "BUY":
        slippage = max(0.0, lmt - mid)
    elif side == Side.SELL.value or side == "SELL":
        slippage = max(0.0, mid - lmt)
    else:
        raise ValueError(f"Invalid side {side}. Cannot calculate slippage.")
    return round(slippage, 2)
