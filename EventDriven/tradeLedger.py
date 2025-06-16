"""
This module serves as a ledger for a trade, containing critical information on the trade
"""
import pandas as pd

from EventDriven.helpers import normalize_dollar_amount_to_decimal, normalize_dollar_amount
from EventDriven.event import FillEvent

class TradeLedger:
    """
    This class serves as a ledger for a trade, containing critical information on the trade.
    Stores fill event data with datetime as keys.
    
    Attributes:
        id (str): Unique identifier for the trade ledger.
        avg_price (float): The average price of all trades. This is an average value.
        quantity (int): The total quantity of assets traded. This is a running sum.
        commission (float): The total commission incurred across all trades. This is a running sum.
        slippage (float): The total slippage incurred across all trades. This is a running sum.
        ledger (dict): A dictionary storing trade entries, with datetime as keys.
        ledger_df (pd.DataFrame): A DataFrame representation of the ledger for easier analysis.
        market_value (float): The total market value of the trades. This is a running sum.
        avg_total_cost (float): The average total cost of all trades. This is an average value.
        aux_cost (float): The total auxiliary costs (commission + slippage) incurred. This is a running sum.
    """
    
    def __init__(self, id:str) -> None:
        self.id = id
        self.avg_price = 0.0
        self.quantity = 0
        self.commission= 0.0
        self.slippage= 0.0
        self.ledger = []
        self.ledger_df = None
        self.market_value = 0.0
        self.avg_total_cost = 0.0
        self.aux_cost = 0.0
    
    
    def add_entry(self, fill_event: FillEvent):
        """
        Adds an entry to the ledger using datetime as key
        """
        # Use the datetime as key for the ledger
        entry_time = fill_event.datetime
        entry = {}
        trade_id = fill_event.position['trade_id']
        uid = f'{trade_id}_{fill_event.signal_id}_{entry_time}'
        entry = {
            'datetime': fill_event.datetime,
            'uid': uid, 
            'price': normalize_dollar_amount(fill_event.fill_cost/fill_event.quantity),
            'quantity': fill_event.quantity,
            'symbol': fill_event.symbol,
            'commission': 0.0 if fill_event.direction == 'EXERCISE' else normalize_dollar_amount(fill_event.commission),
            'market_value': normalize_dollar_amount(fill_event.market_value),
            'slippage': 0.0 if fill_event.direction == 'EXERCISE' else normalize_dollar_amount(fill_event.slippage),
            'total_cost': normalize_dollar_amount(fill_event.fill_cost),
            'aux_cost': 0.0 if fill_event.direction == 'EXERCISE' else normalize_dollar_amount(abs(fill_event.commission) + abs(fill_event.slippage)),
            'direction': fill_event.direction
        }
            
        
        # Update aggregated metrics
        self.avg_price = ((self.avg_price * self.quantity) + 
                         (self.ledger[entry_time]['price'] * fill_event.quantity)) / (self.quantity + fill_event.quantity)
        # self.avg_total_cost = ((self.avg_total_cost * self.quantity) + 
        #                        (self.ledger[entry_time]['total_cost'] * fill_event.quantity)) / (self.quantity + fill_event.quantity)
        self.avg_total_cost = self.avg_total_cost + (self.ledger[entry_time]['total_cost'] ) ## Assuming fill_cost is the total cost for the fill event
        self.aux_cost += self.ledger[entry_time]['aux_cost']
        self.quantity += self.ledger[entry_time]['quantity']
        self.commission += self.ledger[entry_time]['commission']
        self.slippage += self.ledger[entry_time]['slippage']
        self.market_value += self.ledger[entry_time]['market_value']
        
        self.ledger.append(entry)
        self.ledger_df = pd.DataFrame(self.ledger)