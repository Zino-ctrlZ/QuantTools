from typing import Tuple, Optional
import math
import pandas as pd
from EventDriven.configs.core import BaseCogConfig
from EventDriven.dataclasses.states import NewPositionState
from EventDriven.dataclasses.orders import OrderRequest
from trade.helpers.Logging import setup_logger
from EventDriven.riskmanager.position.base import BaseCog
from EventDriven.dataclasses.states import (
    PositionAnalysisContext,
    CogActions,
    PositionState
)
from EventDriven.riskmanager.actions import CLOSE, ROLL, HOLD
from pydantic.dataclasses import dataclass as pydantic_dataclass
logger = setup_logger("EventDriven.riskmanager.position.cogs.pnl_monitor", stream_log_level="INFO")

@pydantic_dataclass
class PnlMonitorConfig(BaseCogConfig):
    """
    Configuration dataclass for PnLMonitorCog.
    """
    
    name: Optional[str] = "PnLMonitorCog"
    enabled: bool = True


class PnLMonitorCog(BaseCog):
    """
    Cog to monitor the PnL of open positions and trigger alerts or actions based on predefined thresholds.
    """
    default_config = PnlMonitorConfig()

    def __init__(self, config: Optional[PnlMonitorConfig] = None):
        if config is None:
            config = PnlMonitorConfig()
        super().__init__(config)
        self.config = config

    def on_new_position(self, new_position_state: NewPositionState) -> None:
        """
        Don't do anything on new position.
        """
        pass

    def on_new_order_request(self, new_request_state: OrderRequest) -> None:
        """
        Don't do anything on new order request.
        """
        logger.info(f"Received new order request for {new_request_state.symbol} with signal ID {new_request_state.signal_id}. Monitoring PnL for this request.")
        

        pnl = new_request_state.symbol_total_pnl
        tick_cash = new_request_state.tick_cash

        ## Scale to dollar amount for easier interpretation
        tick_cash = tick_cash * 100 if not new_request_state.is_tick_cash_scaled else tick_cash

        ## If have profits, add 25% of the profits to the tick cash to scale up the position and lock in profits.
        if pnl is not None and pnl > 0:
            additional_tick_cash = (pnl * 0.25)  ## Add 25% of the profits to the tick cash to scale up the position and lock in profits.
            
            ## Undo pnl from tick cash to avoid double counting, then add the additional tick cash to lock in profits.
            undone_pnl_tick_cash = tick_cash - pnl
            new_tick_cash = undone_pnl_tick_cash + additional_tick_cash
            logger.info(f"Details: PnL: {pnl:.2f}, Original Tick Cash: {tick_cash:.2f}, Undone PnL Tick Cash: {undone_pnl_tick_cash:.2f}, Additional Tick Cash to Lock in Profits: {additional_tick_cash:.2f}, New Tick Cash: {new_tick_cash:.2f}")
            logger.info(f"Adding {additional_tick_cash:.2f} to tick cash for {new_request_state.symbol} to lock in profits. Original Tick Cash: {tick_cash:.2f}, New Tick Cash: {new_tick_cash:.2f}, PnL: {pnl:.2f}")
            new_request_state.tick_cash = new_tick_cash
            new_request_state.is_tick_cash_scaled = True
            
        else:
            logger.info(f"No profits to lock in for {new_request_state.symbol}. Tick Cash remains at {tick_cash:.2f}. PnL: {pnl:.2f}")

    def _analyze_impl(self, portfolio_context: PositionAnalysisContext) -> CogActions:
        """
        Analyze the current position context and return any actions to be taken.
        For this PnL monitor, we will not take any specific actions, but we could log or trigger alerts if needed.
        """
        logger.info(f"PnLMonitor received position analysis context {portfolio_context}. Analyzing PnL for open positions.")
        opinions = []

        ## Rules:
        ## 1. If PnL is greater than 50% of entry price, close half the position if quantity > 1 to lock in profits. 
        ## 2. If quantity is 1 and PnL is greater than 150% of entry price, ROLL the position.
        ##  2a. If have closed before, and remainding quantity is > 1, ROLL when PnL is greater than 150% of entry price to lock in profits.
        positions = portfolio_context.portfolio.positions
        bkt_info = portfolio_context.portfolio_meta
        t_plus_n = bkt_info.t_plus_n
        portfolio_state = portfolio_context.portfolio
        last_updated = portfolio_state.last_updated
        t_plus_n_timedelta = pd.Timedelta(days=t_plus_n)
        for pos_state in positions:
            pl_pct = pos_state.pnl / (pos_state.entry_price * pos_state.quantity) if pos_state.entry_price * pos_state.quantity != 0 else 0
            if pl_pct > 0.5 and pos_state.quantity > 1:
                qdiff = math.ceil(pos_state.quantity / 2)
                new_q = pos_state.quantity - qdiff
                logger.info(f"Position {pos_state.trade_id} has PnL of {pl_pct:.2%}. Closing half the position to lock in profits. Quantity change: {qdiff}, New Quantity: {new_q}")
                action = CLOSE(
                    trade_id=pos_state.trade_id,
                    action={"quantity_diff": qdiff, "new_quantity": new_q}
                )
                action.analysis_date = portfolio_context.date
                action.reason = f"PnL is {pl_pct:.2%} which is greater than 50% of entry price. Closing half the position to lock in profits."
                action.verbose_info = f"Position {pos_state.trade_id} has PnL of {pl_pct:.2%}. Closing half the position to lock in profits. Quantity change: {qdiff}, New Quantity: {new_q}"
                tplus_n_timedelta = max(t_plus_n_timedelta, pd.Timedelta(days=1))  # Ensure at least 1 day is added to move to the next trading day
                action.effective_date = last_updated + tplus_n_timedelta
                pos_state.action = action
                opinions.append(pos_state)
            
            elif pl_pct > 1.0:
                if pos_state.quantity == 1 or (self._has_closed_before(pos_state.trade_id, pos_state) and pos_state.quantity > 1):
                    logger.info(f"Position {pos_state.trade_id} has PnL of {pl_pct:.2%}. Rolling the position to lock in profits.")
                    action = ROLL(
                        trade_id=pos_state.trade_id,
                        action={"quantity_diff": 0, "new_quantity": pos_state.quantity}
                    )
                    action.analysis_date = portfolio_context.date
                    action.reason = f"PnL is {pl_pct:.2%} which is greater than 150% of entry price. Rolling the position to lock in profits."
                    action.verbose_info = f"Position {pos_state.trade_id} has PnL of {pl_pct:.2%}. Rolling the position to lock in profits."
                    tplus_n_timedelta = max(t_plus_n_timedelta, pd.Timedelta(days=1))  # Ensure at least 1 day is added to move to the next trading day
                    action.effective_date = last_updated + tplus_n_timedelta
                    pos_state.action = action
                    opinions.append(pos_state)
            else:
                logger.info(f"Position {pos_state.trade_id} has PnL of {pl_pct:.2%}. No action taken.")
                pos_state.action = HOLD(trade_id=pos_state.trade_id)
                pos_state.action.reason = f"PnL is {pl_pct:.2%}. No action taken."
                pos_state.action.verbose_info = f"Position {pos_state.trade_id} has PnL of {pl_pct:.2%}. No action taken."
                opinions.append(pos_state)
            
                
                
                

        return CogActions(opinions=opinions, strategy_id=self.config.run_name, date=portfolio_context.date, source_cog=self.name)
    
    def _get_current_close_open_quantity(self, pos_state: PositionState) -> Tuple[int, int]:
        """
        Helper method to aggregate the total open and close quantities for a position based on its trade entries.
        """
        entries = pos_state.trades.entries() if pos_state.trades is not None else pd.DataFrame()
        if entries.empty:
            return 0, 0
        open_qty = entries[entries["direction"] == "BUY"]["quantity"].sum()
        close_qty = entries[entries["direction"].isin(["SELL", "EXERCISE"])]["quantity"].sum() if not entries[entries["direction"].isin(["SELL", "EXERCISE"])]["quantity"].empty else 0
        return open_qty, close_qty

    
    def _has_closed_before(self, trade_id: str, portfolio_context: PositionState) -> bool:
        """
        Helper method to check if a position has been closed before based on its trade ID.

        True if there has been at least one close transaction (SELL or EXERCISE) for the given trade ID, False otherwise.
        False if there are no transactions for the trade ID or if all transactions are BUY.
        """
        open_qty, close_qty = self._get_current_close_open_quantity(portfolio_context)
        if open_qty == 0:
            return False
        return close_qty > 0

