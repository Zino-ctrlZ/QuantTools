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
    
    def __init__(self, trade_id:str, symbol:str, signa_id: str = None):
        self.trade_id = trade_id
        self.symbol= symbol
        self.buy_ledger= TradeLedger(f"{trade_id}_buy")
        self.sell_ledger = TradeLedger(f"{trade_id}_sell")
        self.entry_date = None
        self.exit_date = None
        self.current_price = None
        self.stats = None
        self.signal_id = signa_id
    
    def __getitem__(self, key):
        """
        Allows access to the stats dictionary using the key.
        """
        if self.stats is not None and key in self.stats.columns:
            return self.stats[key].iloc[0]
        elif key in ['trade_id', 'symbol', 'entry_date', 'exit_date', 'current_price']:
            return getattr(self, key)
        else:
            raise KeyError(f"Key '{key}' not found in stats.")
        
    def __setitem__(self, key, value):
        """
        Allows setting values in the stats dictionary using the key.
        """
        if  key in ['trade_id', 'symbol', 'entry_date', 'exit_date', 'current_price']:
            setattr(self, key, value)
        else:
            raise KeyError(f"Key '{key}' not found in stats.")
        
        
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
        Generate aggregated statistics for the trade with standardized key naming
        that matches the trades_data objects.
        """
        stats = {}
        stats['TradeID'] = self.trade_id
        stats['SignalID'] = self.signal_id
        stats['Ticker'] = self.symbol
        stats['EntryTime'] = self.entry_date
        stats['ExitTime'] = self.exit_date
        
        # Calculate metrics for buy transactions
        stats['EntryPrice'] = self.buy_ledger.avg_price
        stats['EntryCommission'] = self.buy_ledger.commission
        stats['EntrySlippage'] = self.buy_ledger.slippage
        stats['EntryQuantity'] = self.buy_ledger.quantity
        stats['EntryAuxilaryCost'] = self.buy_ledger.aux_cost
        stats['TotalEntryCost'] = self.buy_ledger.avg_total_cost# * self.buy_ledger.quantity ## Patch fix for now
        
        # Calculate metrics for sell transactions
        # stats['ExitPrice'] = self.sell_ledger.avg_price
        stats['ExitPrice'] = self.sell_ledger.avg_price #if self.sell_ledger.avg_price else self.current_price
        stats['ExitCommission'] = self.sell_ledger.commission
        stats['ExitSlippage'] = self.sell_ledger.slippage
        stats['ExitQuantity'] = self.sell_ledger.quantity
        stats['ExitAuxilaryCost'] = self.sell_ledger.aux_cost
        stats['TotalExitCost'] = self.sell_ledger.avg_total_cost# * self.sell_ledger.quantity ## Patch fix for now
        
        stats['Quantity']  = stats['ExitQuantity']

        # Calculate PnL metrics if we have both buy and sell transactions
        if stats['EntryQuantity'] > 0 and stats['ExitQuantity'] > 0:
            # Calculate realized PnL for closed portion
            stats['ClosedQuantity'] = stats['ExitQuantity']
            stats['ClosedPnL'] = (stats['ExitPrice'] - stats['EntryPrice']) * stats['ExitQuantity']
            
            # Calculate commission and slippage impact
            stats['TotalCommission'] = stats['EntryCommission'] + stats['ExitCommission']
            stats['TotalSlippage'] = stats['EntrySlippage'] + stats['ExitSlippage']
            stats['TotalAuxilaryCost'] = -abs(stats['TotalCommission']) - abs(stats['TotalSlippage'])
            
            # Calculate unrealized PnL for open position
            open_quantity = stats['EntryQuantity'] - stats['ExitQuantity']
            stats['OpenQuantity'] = open_quantity
            if open_quantity > 0 and self.current_price is not None:
                stats['UnrealizedPnL'] = (self.current_price - stats['EntryPrice']) * open_quantity
            else:
                stats['UnrealizedPnL'] = 0
                
            # Calculate total PnL
            stats['PnL'] = stats['ClosedPnL'] + stats['UnrealizedPnL'] #- abs(stats['TotalSlippage']) - abs(stats['TotalCommission']) ## Exit & Entry Price already includes slippage and commission
            
            # Calculate return percentage
            if stats['TotalEntryCost'] > 0:
                stats['ReturnPct'] = (stats['PnL'] / stats['TotalEntryCost']) 
            else:
                stats['ReturnPct'] = 0
                
            # Calculate duration in days if trade is closed
            if self.is_closed() and self.exit_date and self.entry_date:
                stats['Duration'] = (self.exit_date - self.entry_date).days
            else:
                stats['Duration'] = None
        
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