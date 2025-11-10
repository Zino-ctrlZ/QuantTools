## Actions classes
from enum import Enum
from abc import ABC, abstractmethod
from EventDriven.types import EventTypes

class RMAction(ABC):
    def __init__(self, trade_id: str, action: str|dict):
        self.trade_id = trade_id
        self.action = action
        self.type: EventTypes = None
        self.name: str = None
        self.reason: str = None
        self.event = None



class HOLD(RMAction):
    def __init__(self, trade_id: str, action: str = 'hold'):
        super().__init__(trade_id, action)
        self.name = EventTypes.HOLD.value
        self.reason = None
        self.type = EventTypes("HOLD")

    def __repr__(self):
        return f'HOLD({self.trade_id}) Reason: {self.reason})'

class CLOSE(RMAction):
    def __init__(self, trade_id: str, action: str = 'close'):
        super().__init__(trade_id, action)
        self.name = EventTypes.CLOSE.value
        self.type = EventTypes("CLOSE")
        self.reason = None

    def __repr__(self):
        return f'CLOSE({self.trade_id}), Reason: {self.reason})'

class ROLL(RMAction):
    def __init__(self, trade_id: str, action: dict):
        super().__init__(trade_id, action)
        self.name = EventTypes.ROLL.value
        self.type = EventTypes("ROLL")
        self.reason = None
        self.quantity_change = action.get('quantity_diff', None)
    
    def __repr__(self):
        return f'ROLL({self.trade_id}, Quantity Change: {self.quantity_change}), Reason: {self.reason})'


class ADJUST(RMAction):
    def __init__(self, trade_id: str, action: dict):
        super().__init__(trade_id, action)
        self.quantity_change = action['quantity_diff']
        self.name = EventTypes.ADJUST.value
        self.type = EventTypes("ADJUST")
        self.reason = None
    
    def __repr__(self):
        return f'ADJUST({self.trade_id}, Quantity Change: {self.action["quantity_diff"]}), Reason: {self.reason})'
    

class EXERCISE(RMAction):
    def __init__(self, trade_id: str, action: dict):
        super().__init__(trade_id, action)
        self.name = EventTypes.EXERCISE.value
        self.type = EventTypes("EXERCISE")
        self.reason = None
    
    def __repr__(self):
        return f'EXERCISE({self.trade_id}, Reason: {self.reason})'