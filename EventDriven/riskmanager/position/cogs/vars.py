"""Shared Constants and Configuration for Position Analysis.

This module defines global constants, enumerations, and configurations used across
the position analysis system. It centralizes definitions for Greek measures,
action priorities, and actionable event types to ensure consistency.

Constants:
    MEASURES: tuple[str, ...]
        - Tuple of supported Greek measures: ('delta', 'gamma', 'vega', 'theta')
        - Defines standard risk metrics tracked across positions
        - Used for limit validation and enforcement
        - Order matters for display and reporting

    MEASURES_SET: frozenset[str]
        - Frozen set version of MEASURES for O(1) lookup
        - Used in filtering and validation operations
        - Immutable for safety in multi-threaded contexts
        - Optimizes "measure in MEASURES_SET" checks

    ACTIONABLE_ANALYSIS: List[EventTypes]
        - List of event types requiring position changes
        - Includes: EXERCISE, ROLL, ADJUST, CLOSE
        - Excludes HOLD (no action needed)
        - Used to filter analysis results for execution

    ACTION_PRIORITY: Dict[EventTypes, int]
        - Priority ranking for conflicting actions
        - Lower number = higher priority
        - CLOSE (1) > EXERCISE (2) > ROLL (3) > ADJUST (4) > HOLD (5)
        - Used in action reconciliation when multiple cogs disagree

    ACTIONABLE_ANALYSIS_STR: List[str]
        - String versions of actionable event types
        - Used for database queries and string comparisons
        - Matches EventTypes.value attributes

Action Priority Rationale:
    1. CLOSE (highest):
       - Risk limit breach requires immediate exit
       - Overrides all other recommendations
       - Prevents further loss accumulation

    2. EXERCISE:
       - Time-sensitive at expiration
       - ITM exercise captures intrinsic value
       - Higher priority than routine adjustments

    3. ROLL:
       - Position maintenance strategy
       - Preferred over simple adjustment
       - Maintains strategy continuity

    4. ADJUST:
       - Fine-tuning of position size
       - Lower urgency than roll/close
       - Can be deferred if needed

    5. HOLD (lowest):
       - Default when no action required
       - Never conflicts with other actions
       - Indicates position within all limits

Greek Measures:
    Delta:
        - Directional exposure (dollars per $1 underlying move)
        - Primary risk metric for most strategies
        - Limits enforce position size constraints

    Gamma:
        - Convexity exposure (delta change per $1 underlying move)
        - Important for managing large price swings
        - Higher near expiration and ATM strikes

    Vega:
        - Volatility exposure (dollars per 1% IV change)
        - Critical for vol-sensitive strategies
        - Higher for longer-dated options

    Theta:
        - Time decay (daily P&L from time passage)
        - Important for income strategies
        - Accelerates near expiration

Usage:
    from EventDriven.riskmanager.position.cogs.vars import (
        MEASURES,
        MEASURES_SET,
        ACTION_PRIORITY,
        ACTIONABLE_ANALYSIS
    )

    # Check if measure is valid
    if measure in MEASURES_SET:
        # O(1) lookup
        process_measure(measure)

    # Reconcile conflicting actions
    if ACTION_PRIORITY[action1] < ACTION_PRIORITY[action2]:
        chosen_action = action1  # Higher priority

    # Filter for actionable events
    actions_to_execute = [
        action for action in all_actions
        if action.type in ACTIONABLE_ANALYSIS
    ]

Integration:
    - Used by all cogs for consistent measure handling
    - PositionAnalyzer uses ACTION_PRIORITY for reconciliation
    - Database queries filter by ACTIONABLE_ANALYSIS_STR
    - Reporting systems display MEASURES in defined order

Notes:
    - MEASURES tuple is immutable to prevent accidental modification
    - MEASURES_SET provides performance optimization
    - ACTION_PRIORITY must cover all EventTypes
    - Adding new measures requires updates across cogs
"""

from typing import Dict, List
from trade.helpers.Logging import setup_logger
from EventDriven.types import EventTypes
from dbase.database.SQLHelpers import DatabaseAdapter


logger = setup_logger("algo.positions.analyze")
db = DatabaseAdapter()
MEASURES = ("delta", "gamma", "vega", "theta")
MEASURES_SET = frozenset(MEASURES)  # O(1) lookup for filtering performance
ACTIONABLE_ANALYSIS: List[EventTypes] = [
    EventTypes.EXERCISE,
    EventTypes.ROLL,
    EventTypes.ADJUST,
    EventTypes.CLOSE,
]

ACTION_PRIORITY: Dict[EventTypes, int] = {
    EventTypes.HOLD: 5,
    EventTypes.ADJUST: 4,
    EventTypes.ROLL: 3,
    EventTypes.EXERCISE: 2,
    EventTypes.CLOSE: 1,
}

ACTIONABLE_ANALYSIS_STR: List[str] = [action.value for action in ACTIONABLE_ANALYSIS]
