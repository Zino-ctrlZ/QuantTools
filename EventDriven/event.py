#Event indicate a change in the state of the strategy, market, portfolio or execution system.

from datetime import datetime

from EventDriven.types import EventTypes, SignalTypes
from trade.helpers.helper import parse_option_tick
class Event(object):
    """
    Event is base class providing an interface for all subsequent 
    (inherited) events, that will trigger further events in the 
    trading infrastructure.   
    """
    pass    



class MarketEvent(Event):
    """
    Handles the event of receiving a new market update with 
    corresponding bars.
    """

    def __init__(self, datetime):
        """
        Initialises the MarketEvent.
        """
        self.type = 'MARKET'
        self.datetime = datetime
        
    def __str__(self):
        return f"MarketEvent date:{self.datetime}"

class SignalEvent(Event):
    """
    Handles the event of sending a Signal from a Strategy object.
    This is received by a Portfolio object and acted upon.
    """
    
    def __init__(self, symbol, datetime, signal_type: SignalTypes, signal_id: str = None, max_contract_price:int = None, order_settings = None):
        """
        Initialises the SignalEvent.

        Parameters:
        symbol - The ticker symbol, e.g. 'GOOG'.
        datetime - The timestamp at which the signal was generated.
        signal_type - 'LONG', 'SHORT' or 'CLOSE'.
        signal_id- A unique identifier for the signal 
        max_contract_price - The maximum price for the contract
        order_settings - specifically for Order signals, to specify the kind of contract to generate, 
            example: {'type': 'naked',
                        'specifics': [{'direction': 'long',
                        'rel_strike': .900,
                        'dte': 365,
                        'moneyness_width': 0.15},
                        {'direction': 'short',
                        'rel_strike': .80,
                        'dte': 365,
                        'moneyness_width': 0.15}],

                        'name': 'vertical_spread'}
        """
        
        self.type = 'SIGNAL'
        self.symbol = symbol
        self.datetime = datetime
        self.signal_type = signal_type
        self.signal_id = signal_id
        self.max_contract_price = max_contract_price
        self.order_settings = order_settings
        
    def __str__(self):
        return f"SignalEvent type:{self.signal_type}, symbol={self.symbol}, date:{self.datetime}, Order Settings={self.order_settings},Max Contract Price:{self.max_contract_price} , signal_id:{self.signal_id}"

class OrderEvent(Event):
    """
    Handles the event of sending an Order to an execution system.
    The order contains a symbol (e.g. GOOG), a type (market or limit),
    quantity and a direction.
    """

    def __init__(self, symbol, datetime, order_type,  direction, cash:int | float = None ,quantity: int |float = None,position = None, signal_id: str = None):
        """
        Initialises the order type, setting whether it is
        a Market order ('MKT') or Limit order ('LMT'), has
        a quantity (integral) and its direction ('BUY' or
        'SELL').

        Parameters:
        symbol - The instrument to trade.
        order_type - 'MKT' or 'LMT' for Market or Limit.
        cash - The cash available to spend on the order
        quantity - Non-negative integer for quantity.
        direction - 'BUY' or 'SELL' for long or short.
        position - A dict with 'long' and 'short' keys, just long if position is a naked option
        signal_id - A unique identifier for the signal that generated the order
        """
        
        self.type = 'ORDER'
        self.datetime = datetime
        self.symbol = symbol
        self.order_type = order_type
        self.cash = cash
        self.quantity = quantity
        self.direction = direction
        self.position = position #a dict with 'long' and 'short' keys 
        self.signal_id = signal_id

    def __str__(self):
        """
        Outputs the values within the Order.
        """
        return f"OrderEvent type={self.order_type}, symbol={self.symbol}, date:{self.datetime}, cash:{self.cash}, quantity={self.quantity}, direction={self.direction}, position={self.position}, signal_id={self.signal_id}"
        
class FillEvent(Event):
    """
    Encapsulates the notion of a Filled Order, as returned
    from a brokerage. Stores the quantity of an instrument
    actually filled and at what price. In addition, stores
    the commission of the trade from the brokerage.
    """

    def __init__(self, datetime : datetime | str, symbol : str, exchange : str, quantity : int,
                 direction: str, fill_cost: float, market_value: float = None, commission : float =None, slippage : float = None , position = None, signal_id: str = None):
        """
        Initialises the FillEvent object. Sets the symbol, exchange,
        quantity, direction, cost of fill and an optional 
        commission.

        If commission is not provided, the Fill object will
        calculate it based on the trade size and Interactive
        Brokers fees.

        Parameters:
        datetime - The bar-resolution when the order was filled.
        symbol - The instrument which was filled.
        exchange - The exchange where the order was filled.
        quantity - The filled quantity.
        direction - The direction of fill ('BUY' or 'SELL')
        fill_cost - The holdings value in dollars.
        commission - An optional commission sent from IB.
        signal_id - A unique identifier for the signal that generated the order
        """
        
        self.type = 'FILL'
        self.datetime = datetime
        self.symbol = symbol
        self.exchange = exchange
        self.quantity = quantity
        self.direction = direction
        self.fill_cost = fill_cost
        self.position = position
        self.market_value = market_value
        self.slippage = slippage
        self.signal_id = signal_id

        # Calculate commission
        if commission is None:
            self.commission = self.calculate_ib_commission()
        else:
            self.commission = commission

    def __str__(self):
        return f"FillEvent symbol={self.symbol}, date:{self.datetime}, exchange={self.exchange}, quantity={self.quantity}, direction={self.direction}, fill_cost={self.fill_cost}, commission={self.commission}, market_value={self.market_value}, slippage={self.slippage}, position={self.position}, signal_id={self.signal_id}"        
class ExerciseEvent(Event): 
    """
    Encapsulates the notion of an exercise event, as returned from a brokerage. 
    Stores the quantity of an instrument actually exercised and at what price. In addition, stores the commission of the trade from the brokerage.
    """
    
    def __init__(self, datetime : datetime | str, symbol : str, quantity : int, entry_date: datetime| str, spot: float, long_premiums: dict, short_premiums:dict, position = None, signal_id: str = None):
        """
        Initialises the ExerciseEvent object. Sets the symbol, exchange, quantity, direction, cost of fill and an optional commission.
        
        Parameters:
        datetime - The bar-resolution when the order was filled.
        symbol - The instrument which was filled.
        quantity - The filled quantity.
        position - A dict with 'long' and 'short' keys, just long if position is a naked option
        signal_id - A unique identifier for the signal that generated the order
        """
        
        self.type = 'EXERCISE'
        self.datetime = datetime
        self.symbol = symbol
        self.quantity = quantity
        self.spot = spot
        self.position = position
        self.signal_id = signal_id
        self.long_premiums = long_premiums
        self.short_premiums = short_premiums
        self.entry_date = entry_date
        
    def __str__(self):
        return f"ExerciseEvent symbol={self.symbol}, date:{self.datetime} quantity={self.quantity}, long_premiums={self.long_premiums}, short_premiums={self.short_premiums}, position={self.position}, signal_id={self.signal_id}"
    
    
class RollEvent(Event): 
    """
    Encapsulates the notion of a roll event, this event simply tells the portfolio to close its current position and open anew position
    """
    
    def __init__(self, datetime: datetime | str, symbol: str, signal_type: SignalTypes, position: dict, signal_id: str = None):
        """
        Initialises the RollEvent object. Sets the symbol, exchange, signal_type, direction, cost of fill and an optional commission.
        
        Parameters:
        datetime - The bar-resolution when the order was filled.
        symbol - The instrument which was filled.
        signal_type - 'LONG' or 'SHORT'.
        position - A dict with 'long' and 'short' keys, just long if position is a naked option
        signal_id - A unique identifier for the signal that generated the order
        """
        
        self.type = 'ROLL'
        self.datetime = datetime
        self.symbol = symbol
        self.signal_type = signal_type
        self.position = position
        self.signal_id = signal_id
    
    def __str__(self):
        return f"RollEvent symbol={self.symbol}, date:{self.datetime}, signal_type={self.signal_type}, position={self.position}, signal_id={self.signal_id}"   
    