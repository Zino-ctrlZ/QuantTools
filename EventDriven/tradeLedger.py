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
        trade_id = fill_event.position['trade_id']
        self._add_entry_common(
            entry_time=fill_event.datetime,
            trade_id=trade_id,
            signal_id=fill_event.signal_id,
            fill_cost=fill_event.fill_cost,
            quantity=fill_event.quantity,
            symbol=fill_event.symbol,
            commission=fill_event.commission,
            market_value=fill_event.market_value,
            slippage=fill_event.slippage,
            direction=fill_event.direction,
            normalize=True,
        )

    def _add_entry_kw(
        self,
        *,
        entry_time,
        trade_id: str,
        signal_id: str,
        fill_cost: float,
        quantity: int,
        symbol: str,
        commission: float,
        market_value: float,
        slippage: float,
        direction: str,
        normalize: bool = False,
    ):
        """
        Adds an entry using keyword args instead of a FillEvent.
        """
        self._add_entry_common(
            entry_time=entry_time,
            trade_id=trade_id,
            signal_id=signal_id,
            fill_cost=fill_cost,
            quantity=quantity,
            symbol=symbol,
            commission=commission,
            market_value=market_value,
            slippage=slippage,
            direction=direction,
            normalize=normalize,
        )

    def _add_entry_common(
        self,
        *,
        entry_time,
        trade_id: str,
        signal_id: str,
        fill_cost: float,
        quantity: int,
        symbol: str,
        commission: float,
        market_value: float,
        slippage: float,
        direction: str,
        normalize: bool,
    ):
        uid = f'{trade_id}_{signal_id}_{entry_time}'
        # Normalize monetary fields unless explicitly disabled
        price_val = normalize_dollar_amount(fill_cost / quantity) if normalize else fill_cost / quantity
        commission_val = 0.0 if direction == 'EXERCISE' else (normalize_dollar_amount(commission) if normalize else commission)
        market_value_val = normalize_dollar_amount(market_value) if normalize else market_value
        slippage_val = 0.0 if direction == 'EXERCISE' else (normalize_dollar_amount(slippage) if normalize else slippage)
        per_unit_slippage_val = 0.0 if direction == 'EXERCISE' else (
            normalize_dollar_amount_to_decimal(slippage / quantity) if normalize else slippage / quantity
        )
        total_cost_val = normalize_dollar_amount(fill_cost) if normalize else fill_cost
        aux_cost_val = 0.0 if direction == 'EXERCISE' else (
            normalize_dollar_amount(abs(commission) + abs(slippage)) if normalize else abs(commission) + abs(slippage)
        )
        entry = {
            'datetime': entry_time,
            'uid': uid,
            'price': price_val,
            'quantity': quantity,
            'symbol': symbol,
            'commission': commission_val,
            'market_value': market_value_val,
            'slippage': slippage_val,
            'per_unit_slippage': per_unit_slippage_val,
            'total_cost': total_cost_val,
            'aux_cost': aux_cost_val,
            'direction': direction
        }

        self.avg_price = ((self.avg_price * self.quantity) +
                         (entry['price'] * quantity)) / (self.quantity + quantity)
        self.avg_total_cost = ((self.avg_total_cost * self.quantity) +
                               (entry['total_cost'] * quantity)) / (self.quantity + quantity)
        self.aux_cost += entry['aux_cost']
        self.quantity += entry['quantity']
        self.commission += entry['commission']
        self.slippage += entry['slippage']
        self.market_value += entry['market_value']

        self.ledger.append(entry)
        self.ledger_df = pd.DataFrame(self.ledger)
