## Actions classes
from EventDriven.types import OpenPositionAction
from abc import ABC, abstractmethod

class RMAction(ABC):
    def __init__(self, trade_id: str, action: str|dict):
        self.trade_id = trade_id
        self.action = action
        self.event = None

class HOLD(RMAction):
    def __init__(self, trade_id: str, action: str = 'hold'):
        super().__init__(trade_id, action)
        self.name = OpenPositionAction.HOLD.value
        self.reason = None

    def __repr__(self):
        return f'HOLD({self.trade_id}) Reason: {self.reason})'

class CLOSE(RMAction):
    def __init__(self, trade_id: str, action: str = 'close'):
        super().__init__(trade_id, action)
        self.name = OpenPositionAction.CLOSE.value
        self.reason = None

    def __repr__(self):
        return f'CLOSE({self.trade_id}), Reason: {self.reason})'

class ROLL(RMAction):
    def __init__(self, trade_id: str, action: dict):
        super().__init__(trade_id, action)
        self.name = OpenPositionAction.ROLL.value
        self.reason = None
    
    def __repr__(self):
        return f'ROLL({self.trade_id}, Reason: {self.reason})'


class ADJUST(RMAction):
    def __init__(self, trade_id: str, action: dict):
        super().__init__(trade_id, action)
        self.quantity_change = action['quantity_diff']
        self.name = OpenPositionAction.ADJUST.value
        self.reason = None
    
    def __repr__(self):
        return f'ADJUST({self.trade_id}, Quantity Change: {self.action["quantity_diff"]}), Reason: {self.reason})'
    

class EXERCISE(RMAction):
    def __init__(self, trade_id: str, action: dict):
        super().__init__(trade_id, action)
        self.name = OpenPositionAction.EXERCISE.value
        self.reason = None
    
    def __repr__(self):
        return f'EXERCISE({self.trade_id}, Reason: {self.reason})'