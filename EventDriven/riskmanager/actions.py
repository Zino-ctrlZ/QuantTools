## Actions classes
from datetime import datetime
from EventDriven.types import EventTypes
from typing_extensions import TypedDict, Union
class Changes(TypedDict):
    quantity_diff: int
    new_quantity: int

action_hint = Union[str, Changes]
class RMAction:
    def __init__(self, trade_id: str, action: action_hint = None):
        self.trade_id = trade_id
        self.action = action if action is not None else {}
        self.type: EventTypes = None
        self.name: str = None
        self.reason: str = None
        self.event: EventTypes = None
        self.analysis_date: datetime | str = None
        self.effective_date: datetime | str = None
        self.verbose_info: str = None



class HOLD(RMAction):
    def __init__(self, trade_id: str, action: action_hint = None):
        super().__init__(trade_id, action)
        self.name = EventTypes.HOLD.value
        self.reason = None
        self.type = EventTypes("HOLD")

    def __repr__(self):
        return f'HOLD({self.trade_id}) Reason: {self.reason})'

class CLOSE(RMAction):
    def __init__(self, trade_id: str, action: action_hint = None):
        super().__init__(trade_id, action)
        self.name = EventTypes.CLOSE.value
        self.type = EventTypes("CLOSE")
        self.reason = None

    def __repr__(self):
        return f'CLOSE({self.trade_id}), Reason: {self.reason})'

class ROLL(RMAction):
    def __init__(self, trade_id: str, action: action_hint = None):
        super().__init__(trade_id, action)
        self.name = EventTypes.ROLL.value
        self.type = EventTypes("ROLL")
        self.reason = None
        self.quantity_change = action.get('quantity_diff', None)
    
    def __repr__(self):
        return f'ROLL({self.trade_id}, Quantity Change: {self.quantity_change}), Reason: {self.reason})'


class ADJUST(RMAction):
    def __init__(self, trade_id: str, action: action_hint = None):
        super().__init__(trade_id, action)
        self.quantity_change = action['quantity_diff']
        self.name = EventTypes.ADJUST.value
        self.type = EventTypes("ADJUST")
        self.reason = None
    
    def __repr__(self):
        return f'ADJUST({self.trade_id}, Quantity Change: {self.action["quantity_diff"]}), Reason: {self.reason})'
    

class EXERCISE(RMAction):
    def __init__(self, trade_id: str, action: action_hint = None):
        super().__init__(trade_id, action)
        self.name = EventTypes.EXERCISE.value
        self.type = EventTypes("EXERCISE")
        self.reason = None
    
    def __repr__(self):
        return f'EXERCISE({self.trade_id}, Reason: {self.reason})'