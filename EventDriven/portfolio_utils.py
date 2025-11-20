from typing import List, Dict, Union
from EventDriven.event import OrderEvent, RollEvent
from EventDriven.riskmanager.utils import parse_signal_id
from EventDriven.types import PositionsDict, EventTypes
from EventDriven.dataclasses.states import PositionState
from EventDriven.exceptions import BacktestNotImplementedError

ORDER_TYPES = [
    EventTypes.OPEN,
    EventTypes.CLOSE,
    EventTypes.ADJUST,
]

ROLL_EVENT_TYPES = [
    EventTypes.ROLL,
]

DO_NOTHING_EVENT_TYPES = [
    EventTypes.HOLD,
]

NOT_IMPLEMENTED_EVENT_TYPES = [
    EventTypes.EXERCISE,
]

def extract_events(
    actionables: List[PositionState], current_positions: Dict[str, Dict[str, PositionsDict]]
) -> List[Union[OrderEvent, RollEvent]]:
    """
    Extract events from actionables based on current positions.

    Parameters:
    actionables (List[PositionState]): List of actionable position states.
    current_positions (Dict[str, Dict[str, PositionsDict]]): Current positions in the portfolio.

    Returns:
    List[Event]: List of extracted events.
    """
    events = []
    for act in actionables:
        ## Find position in pm.current_positions
        trade_id = act.trade_id
        position = current_positions.get(act.underlier_tick, {}).get(act.signal_id, {})
        if not position:
            print(
                f"No position found for trade_id: {trade_id}, underlier: {act.underlier_tick}, signal_id: {act.signal_id}"
            )
            continue
        sym = act.underlier_tick
        place_date = act.action.effective_date
        qty_diff = act.action.action.get("quantity_diff", 0)
        event_expected_position = position["position"]

        if act.action.type in ORDER_TYPES:
            event = OrderEvent(
                symbol=sym,
                datetime=place_date,
                order_type="MKT",
                quantity=abs(qty_diff),
                direction="SELL" if qty_diff < 0 else "BUY",
                position=position["position"],
                signal_id=act.signal_id,
            )
        if act.action.type in ROLL_EVENT_TYPES:
            event = RollEvent(
                datetime=place_date,
                symbol=sym,
                signal_type=parse_signal_id(act.signal_id)["direction"],
                position=event_expected_position,
                signal_id=act.signal_id,
            )
        if act.action.type in NOT_IMPLEMENTED_EVENT_TYPES:
            raise BacktestNotImplementedError(f"{act.action.type} action not implemented in backtest.")
        if act.action.type in DO_NOTHING_EVENT_TYPES:
            continue
        events.append(event)
    return events
