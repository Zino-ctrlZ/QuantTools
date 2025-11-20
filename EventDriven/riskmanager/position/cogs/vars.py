from typing import Dict, List
from trade.helpers.Logging import setup_logger
from EventDriven.types import EventTypes
from dbase.database.SQLHelpers import DatabaseAdapter


logger = setup_logger("algo.positions.analyze")
db = DatabaseAdapter()
MEASURES = ("delta", "gamma", "vega", "theta")
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
