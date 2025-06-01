"""
This module defines the Trade class for tracking buy and sell transactions for a trade.
"""
import pandas as pd
from copy import deepcopy
from EventDriven.tradeLedger import TradeLedger
from EventDriven.event import FillEvent

class Trade:
    """
    Class to track buy and sell transactions for a specific trade.
    Has separate ledgers for buy and sell events.
    """
    
    def __init__(self, trade_id:str, symbol:str):
        self.trade_id = trade_id
        self.symbol= symbol
        self.buy_ledger= TradeLedger(f"{trade_id}_buy")
        self.sell_ledger = TradeLedger(f"{trade_id}_sell")
        self.entry_date = None
        self.exit_date = None
        self.current_price = None
        self.stats = None
    
    def update(self, fill_event: FillEvent):
        """
        Update the appropriate ledger based on the fill event direction
        """
        if fill_event.direction == 'BUY':
            self.buy_ledger.add_entry(fill_event)
            if self.entry_date is None:
                self.entry_date = fill_event.datetime
                
        elif fill_event.direction in ['SELL', 'EXERCISE']:
            self.sell_ledger.add_entry(fill_event)
            # Update exit date if position is fully closed
            if self.is_closed():
                self.exit_date = fill_event.datetime
                
        self.stats = pd.DataFrame([self.aggregate()])
            
            
    def is_closed(self):
        """
        Check if the trade is closed (buy quantity - sell quantity = 0)
        """
        return self.buy_ledger.quantity - self.sell_ledger.quantity == 0
    
    def get_position_size(self):
        """
        Get the current position size (buy - sell)
        """
        return self.buy_ledger.quantity - self.sell_ledger.quantity
    
    def aggregate(self):
        """
        Generate aggregated statistics for the trade
        """
        stats = {}
        stats['trade_id'] = self.trade_id
        stats['symbol'] = self.symbol
        stats['entry_date'] = self.entry_date
        stats['exit_date'] = self.exit_date
        
        # Calculate metrics for buy transactions
        stats['avg_entry_price'] = self.buy_ledger.avg_price
        stats['total_entry_commission'] = self.buy_ledger.commission
        stats['total_entry_slippage'] = self.buy_ledger.slippage
        stats['total_entry_quantity'] = self.buy_ledger.quantity
        stats['total_aux_entry_cost'] = self.buy_ledger.aux_cost
        stats['avg_entry_cost'] = self.buy_ledger.avg_total_cost
        
        # Calculate metrics for sell transactions
        stats['avg_exit_price'] = self.sell_ledger.avg_price
        stats['total_exit_commission'] = self.sell_ledger.commission
        stats['total_exit_slippage'] = self.sell_ledger.slippage
        stats['total_exit_quantity'] = self.sell_ledger.quantity
        stats['total_aux_exit_cost'] = self.sell_ledger.aux_cost
        stats['avg_exit_cost'] = self.sell_ledger.avg_total_cost
        
        # Calculate PnL metrics if we have both buy and sell transactions
        if stats['total_entry_quantity'] > 0 and stats['total_exit_quantity'] > 0:
            # Calculate realized PnL for closed portion
            stats['closed_quantity'] = stats['total_exit_quantity']
            stats['closed_pnl'] = (stats['avg_exit_price'] - stats['avg_entry_price']) * stats['total_exit_quantity']
            
            # Calculate commission and slippage impact
            stats['total_commission'] = stats['total_entry_commission'] + stats['total_exit_commission']
            stats['total_slippage'] = stats['total_entry_slippage'] + stats['total_exit_slippage']
            stats['total_aux_cost'] = stats['total_commission'] + stats['total_slippage']
            
            # Calculate unrealized PnL for open position
            open_quantity = stats['total_entry_quantity'] - stats['total_exit_quantity']
            stats['open_quantity'] = open_quantity
            
            if open_quantity > 0 and self.current_price is not None:
                stats['unrealized_pnl'] = (self.current_price - stats['avg_entry_price']) * open_quantity
            else:
                stats['unrealized_pnl'] = 0
                
            # Calculate total PnL
            stats['total_pnl'] = stats['closed_pnl'] + stats['unrealized_pnl']
            
            # Calculate return percentage
            if stats['avg_entry_cost'] > 0:
                stats['return_pct'] = (stats['total_pnl'] / stats['avg_entry_cost'])
            else:
                stats['return_pct'] = 0
                
            # Calculate duration in days if trade is closed
            if self.is_closed() and self.exit_date and self.entry_date:
                stats['duration_days'] = (self.exit_date - self.entry_date).days
            else:
                stats['duration_days'] = None
        
        return stats
    
    def update_current_price(self, price):
        """
        Update the current market price for calculating unrealized PnL
        """
        self.current_price = price
    
    def entries(self):
        """
        Return a combined dataframe of buy and sell transactions
        """
        buy_df = self.buy_ledger.ledger_df
        sell_df = self.sell_ledger.ledger_df
        
        combined_df = pd.DataFrame()
        
        if buy_df is not None and not buy_df.empty:
            buy_df = buy_df.copy()
            buy_df['transaction_type'] = 'BUY'
            combined_df = pd.concat([combined_df, buy_df], ignore_index=True)
            
        if sell_df is not None and not sell_df.empty:
            sell_df = sell_df.copy()
            sell_df['transaction_type'] = 'SELL'
            combined_df = pd.concat([combined_df, sell_df], ignore_index=True)
        
        if not combined_df.empty:
            # Sort by datetime
            combined_df.sort_values('datetime', inplace=True)
            
        return combined_df