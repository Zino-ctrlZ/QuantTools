"""PnL monitoring cog for profit-locking and risk-off actions.

Provides a position-level monitoring cog that evaluates realized/unrealized
profit and loss as part of the PositionAnalyzer pipeline, then emits
actionable opinions (HOLD, CLOSE, ROLL) using rule-based thresholds.

Core Classes:
    PnLMonitorCog: Applies PnL-driven management rules to open positions.

Core Functions:
    on_new_order_request: Re-scales requested tick cash when profitable.
    _analyze_impl: Evaluates per-position PnL percent and produces actions.
    _get_current_close_open_quantity: Aggregates BUY vs SELL/EXERCISE flow.
    _has_closed_before: Detects whether a trade has partial close history.

Processing Flow:
    1. Receive a position analysis context from PositionAnalyzer.
    2. Compute per-position PnL ratio: pnl / (entry_price * quantity).
    3. Apply rule set in priority order:
       - Take-profit partial close when PnL ratio > 50% and quantity > 1.
       - Roll when PnL ratio > 100% and roll criteria are satisfied.
       - Optional stop-loss close when PnL ratio <= -70%.
       - Otherwise HOLD.
    4. Stamp analysis metadata (date/reason/effective_date) on actions.
    5. Return CogActions with the resulting opinions.

Risk/Assumptions:
    - PnL thresholds are static and percentage-based.
    - Effective dates are delayed by business days using t_plus_n.
    - Stop-loss branch is disabled by default (enable_stop_loss=False).
    - Quantity adjustments assume position quantities are integer-like.
    - Trade history interpretation relies on direction labels:
      BUY, SELL, EXERCISE.

Rule Summary:
    Profit lock (partial close):
        If pnl_ratio > 0.5 and quantity > 1, close approximately half.

    Roll for large gains:
        If pnl_ratio > 1.0 and either:
        - quantity == 1, or
        - the trade has prior close activity and quantity > 1,
        then emit ROLL.

    Stop-loss (optional):
        If pnl_ratio <= -0.7 and stop loss is enabled, close fully.

Order Request Tick-Cash Adjustment:
    During on_new_order_request, when symbol_total_pnl > 0:
        - Normalize tick_cash to scaled dollars.
        - Remove embedded pnl component from tick_cash.
        - Add back 25% of pnl to lock profits while resizing.
        - Mark is_tick_cash_scaled=True.

Usage:
    >>> cog = PnLMonitorCog()
    >>> # Analyzer invokes this internally with PositionAnalysisContext
    >>> # actions = cog.analyze(context)

Notes:
    - This cog is opinion-generating; execution remains downstream.
    - Action reasons and verbose_info are populated for observability.
    - Configuration defaults come from PnlMonitorConfig.
"""

from typing import Tuple, Optional
import math
import pandas as pd
from EventDriven.dataclasses.states import NewPositionState
from EventDriven.dataclasses.orders import OrderRequest
from trade.helpers.Logging import setup_logger
from EventDriven.riskmanager.position.base import BaseCog
from EventDriven.dataclasses.states import PositionAnalysisContext, CogActions, PositionState
from EventDriven.riskmanager.actions import CLOSE, ROLL, HOLD
from EventDriven.configs.core import PnlMonitorConfig, PnLMonitorConfigConfigurable

logger = setup_logger("EventDriven.riskmanager.position.cogs.pnl_monitor", stream_log_level="INFO")


class PnLMonitorCog(BaseCog):
    """PnL-aware position management cog.

    This cog converts position-level PnL states into risk-manager opinions. It is
    designed to run inside the position analysis cycle and produce deterministic
    action proposals based on configurable threshold rules:

    1. Partial profit-taking (CLOSE part of the position).
    2. Profit-protecting roll logic (ROLL the position).
    3. Optional stop-loss behavior (full CLOSE).
    4. HOLD when no trigger is met.

    In addition, it can adjust incoming order request capital (`tick_cash`) when
    symbol-level PnL is positive, so downstream sizing preserves locked profits.
    """

    default_config = PnlMonitorConfig()

    def __init__(
        self,
        config: Optional[PnlMonitorConfig | PnLMonitorConfigConfigurable] = None,
    ):
        """Initialize the PnL monitor cog.

        Args:
            config: Optional runtime configuration. If not provided, a default
                `PnlMonitorConfig` is created.

        Notes:
            - `enable_stop_loss` is initialized to ``False`` so the stop-loss
              branch is opt-in.
            - `default_config` remains available for framework-level defaults.
        """
        if config is not None and not isinstance(config, (PnlMonitorConfig, PnLMonitorConfigConfigurable)):
            raise TypeError("Config must be of type PnlMonitorConfig or PnLMonitorConfigConfigurable")
        if config is None:
            config = PnlMonitorConfig()
        super().__init__(config)
        self.config: PnlMonitorConfig | PnLMonitorConfigConfigurable = config

    def on_new_position(self, new_position_state: NewPositionState) -> None:
        """Handle a newly created position state.

        Args:
            new_position_state: Newly created position container provided by the
                risk manager workflow.

        Returns:
            None. This cog intentionally performs no mutation at creation time;
            PnL-based decisions are deferred to analysis cycles.
        """
        pass

    def on_new_order_request(self, new_request_state: OrderRequest) -> None:
        """Adjust request cash using cap and profit-lock rules.

        Processing steps:
            1. Normalize `tick_cash` to dollar units when needed.
            2. If `max_trade_dollar_size` is set, cap cash first.
            3. If symbol pnl is positive and profit-lock mode is active,
               add `profit_lock_in_pct * pnl` to a mode-specific base cash.

        Profit-lock mode activation:
            - Active when max is set and `profit_lock_in_lvl == 2`.
            - Active when max is not set and `profit_lock_in_lvl == 1`.

        Effective behavior by combination:
            - max=None, lvl=1: profit-lock adjustment runs.
            - max=None, lvl=2: no adjustment (invalid by config validation).
            - max=set,  lvl=1: constant capped size (no profit addition).
            - max=set,  lvl=2: capped base plus profit increment.

        Notes:
            - Profit increment is applied only when `symbol_total_pnl > 0`.
            - Updates are applied in-place to `new_request_state`.

        Args:
            new_request_state: Incoming order request that may be resized prior
                to downstream execution and sizing.

        Returns:
            None.
        """
        logger.info(
            f"Received new order request for {new_request_state.symbol} with signal ID {new_request_state.signal_id}. Monitoring PnL for this request."
        )

        pnl = new_request_state.symbol_total_pnl
        tick_cash = new_request_state.tick_cash

        ## Scale to dollar amount for easier interpretation
        tick_cash = tick_cash * 100 if not new_request_state.is_tick_cash_scaled else tick_cash

        ## If max_trade_dollar_size is set, cap the tick_cash to prevent excessive allocation
        if self.config.max_trade_dollar_size is not None:
            logger.info(
                f"Max trade dollar size is set to ${self.config.max_trade_dollar_size:.2f}. Original tick cash: ${tick_cash:.2f}."
            )

            ## If tick_cash > max_trade_dollar_size, tick_cash = max_trade_dollar_size
            ## If tick_cash <= max_trade_dollar_size, tick_cash remains unchanged
            tick_cash = min(tick_cash, self.config.max_trade_dollar_size)
            new_request_state.tick_cash = tick_cash
            new_request_state.is_tick_cash_scaled = True

        ## PL adjustment decider
        should_adjust = (
            ## If max_trade_dollar_size is set & lvl == 2
            (self.config.max_trade_dollar_size is not None and self.config.profit_lock_in_lvl == 2)
            ## Or if max_trade_dollar_size is not set and lvl == 1 (default)
            or (self.config.max_trade_dollar_size is None and self.config.profit_lock_in_lvl == 1)
        )

        ## If have profits, add 25% of the profits to the tick cash to scale up the position and lock in profits.
        if pnl is not None and pnl > 0 and should_adjust:
            additional_tick_cash = (
                pnl * self.config.profit_lock_in_pct
            )  ## Add 25% of the profits to the tick cash to scale up the position and lock in profits.

            ## Undo pnl from tick cash to avoid double counting, then add the additional tick cash to lock in profits.
            undone_pnl_tick_cash = tick_cash - pnl

            ## Either use the undone_pnl_tick_cash for base cash
            ## Or use the original tick_cash for base cash if profit_lock_in_lvl is 2
            base_tick_cash = undone_pnl_tick_cash if self.config.profit_lock_in_lvl == 1 else tick_cash
            new_tick_cash = base_tick_cash + additional_tick_cash

            logger.info(
                f"Details: PnL: {pnl:.2f}, Original Tick Cash: {tick_cash:.2f}, Base Tick Cash: {base_tick_cash:.2f}, Additional Tick Cash to Lock in Profits: {additional_tick_cash:.2f}, New Tick Cash: {new_tick_cash:.2f}"
            )
            logger.info(
                f"Adding {additional_tick_cash:.2f} to tick cash for {new_request_state.symbol} to lock in profits. Original Tick Cash: {tick_cash:.2f}, New Tick Cash: {new_tick_cash:.2f}, PnL: {pnl:.2f}"
            )

            new_request_state.tick_cash = new_tick_cash
            new_request_state.is_tick_cash_scaled = True

        else:
            pnl_str = f"{pnl:.2f}" if pnl is not None else "None"
            logger.info(
                f"No profits to lock in for {new_request_state.symbol}. Tick Cash remains at {tick_cash:.2f}. PnL: {pnl_str}"
            )

    def _analyze_impl(self, portfolio_context: PositionAnalysisContext) -> CogActions:
        """Evaluate each open position and emit HOLD, CLOSE, or ROLL opinions.

        Decision flow per position:
            1. Compute pnl ratio against cost basis.
            2. If pnl ratio is above lock-in threshold and quantity > 1,
               partially CLOSE to secure gains.
            3. Else if pnl ratio is above roll threshold and roll conditions
               pass, emit ROLL.
            4. Else if stop-loss is enabled and stop condition is hit,
               fully CLOSE.
            5. Otherwise emit HOLD.

        Stop-loss supports two mutually exclusive modes:
            - pct mode: compare pnl ratio to stop_loss_pct
            - cash mode: compare signal-level pnl to stop_loss_cash_threshold

        Args:
            portfolio_context: Portfolio snapshot and metadata for the current
                analysis date, including positions and backtest parameters.

        Returns:
            CogActions: Opinion bundle generated by this cog for the current
            analysis cycle.
        """
        # logger.info(f"PnLMonitor received position analysis context {portfolio_context}. Analyzing PnL for open positions.")
        opinions = []

        ## Rules:
        ## 1. If PnL is greater than lock_in_profit_threshold of entry price, close half the position if quantity > 1 to lock in profits.
        ## 2. If quantity is 1 and PnL is greater than roll_profit_threshold of entry price, ROLL the position.
        ##  2a. If have closed before, and remainding quantity is > 1, ROLL when PnL is greater than roll_profit_threshold of entry price to lock in profits.
        positions = portfolio_context.portfolio.positions
        bkt_info = portfolio_context.portfolio_meta
        t_plus_n = bkt_info.t_plus_n
        portfolio_state = portfolio_context.portfolio
        last_updated = portfolio_state.last_updated
        t_plus_n_bdays = pd.offsets.BusinessDay(max(t_plus_n, 1))
        is_cash_stop_loss = self.config.stop_loss_cash_threshold is not None
        for pos_state in positions:
            pl_pct = (
                pos_state.pnl / (pos_state.entry_price * pos_state.quantity)
                if pos_state.entry_price * pos_state.quantity != 0
                else 0
            )
            signal_pnl = pos_state.signal_total_pnl if pos_state.signal_total_pnl is not None else 0
            if pl_pct > self.config.lock_in_profit_threshold and pos_state.quantity > 1:
                qdiff = -math.ceil(pos_state.quantity / 2)
                new_q = pos_state.quantity + qdiff
                logger.info(
                    f"Position {pos_state.trade_id} has PnL of {pl_pct:.2%}. Closing half the position to lock in profits. Quantity change: {qdiff}, New Quantity: {new_q}"
                )
                action = CLOSE(trade_id=pos_state.trade_id, action={"quantity_diff": qdiff, "new_quantity": new_q})
                action.analysis_date = portfolio_context.date
                action.reason = f"PnL is {pl_pct:.2%} which is greater than {self.config.lock_in_profit_threshold:.2%} of entry price. Closing half the position to lock in profits."
                action.verbose_info = f"Position {pos_state.trade_id} has PnL of {pl_pct:.2%}. Closing half the position to lock in profits. Quantity change: {qdiff}, New Quantity: {new_q}"
                action.effective_date = last_updated + t_plus_n_bdays
                pos_state.action = action
                opinions.append(pos_state)

            elif pl_pct > self.config.roll_profit_threshold:
                if pos_state.quantity == 1 or (
                    self._has_closed_before(pos_state.trade_id, pos_state) and pos_state.quantity > 1
                ):
                    logger.info(
                        f"Position {pos_state.trade_id} has PnL of {pl_pct:.2%} which is greater than {self.config.roll_profit_threshold:.2%} of entry price. Rolling the position to lock in profits."
                    )
                    # quantity_diff = -abs(qty) signals close-first for live; new_quantity is post-roll size.
                    action = ROLL(
                        trade_id=pos_state.trade_id,
                        action={
                            "quantity_diff": -abs(pos_state.quantity),
                            "new_quantity": pos_state.quantity,
                        },
                    )
                    action.analysis_date = portfolio_context.date
                    action.reason = f"PnL is {pl_pct:.2%} which is greater than {self.config.roll_profit_threshold:.2%} of entry price. Rolling the position to lock in profits."
                    action.verbose_info = f"Position {pos_state.trade_id} has PnL of {pl_pct:.2%}. Rolling the position to lock in profits."
                    action.effective_date = last_updated + t_plus_n_bdays
                    pos_state.action = action
                    opinions.append(pos_state)
                else:
                    logger.info(
                        f"Position {pos_state.trade_id} exceeded roll threshold but roll criteria were not met. No action taken."
                    )
                    pos_state.action = HOLD(trade_id=pos_state.trade_id)
                    pos_state.action.reason = (
                        f"PnL is {pl_pct:.2%}. Roll threshold exceeded but roll criteria were not met. No action taken."
                    )
                    pos_state.action.verbose_info = (
                        f"Position {pos_state.trade_id} exceeded roll threshold but did not qualify for rolling."
                    )
                    opinions.append(pos_state)

            ## Stop loss branch: close <= stop_loss_pct to prevent further losses. This branch is disabled by default and can be enabled by setting enable_stop_loss to True.
            elif self.config.enable_stop_loss:
                if not is_cash_stop_loss:
                    stop_hit = self.config.stop_loss_pct is not None and pl_pct <= self.config.stop_loss_pct
                else:
                    stop_hit = (
                        self.config.stop_loss_cash_threshold is not None
                        and signal_pnl <= self.config.stop_loss_cash_threshold
                    )
                if stop_hit:
                    cash_threshold_msg = (
                        f"{self.config.stop_loss_cash_threshold:.2f}"
                        if self.config.stop_loss_cash_threshold is not None
                        else "N/A"
                    )
                    pct_threshold_msg = (
                        f"{self.config.stop_loss_pct:.2%}" if self.config.stop_loss_pct is not None else "N/A"
                    )
                    logger_msg = (
                        f"Position {pos_state.trade_id} has PnL of {pl_pct:.2%}. Signal-level PnL is {signal_pnl:.2f}. "
                        f"Cash threshold is {cash_threshold_msg}. "
                        f"pct stop loss threshold is {pct_threshold_msg}. "
                    )
                    logger.info(logger_msg)
                    action = CLOSE(
                        trade_id=pos_state.trade_id, action={"quantity_diff": -pos_state.quantity, "new_quantity": 0}
                    )
                    action.analysis_date = portfolio_context.date
                    if is_cash_stop_loss:
                        action.reason = (
                            f"Signal-level PnL is {signal_pnl:.2f}, below cash stop-loss threshold "
                            f"{cash_threshold_msg}. Closing the position to prevent further losses."
                        )
                    else:
                        action.reason = (
                            f"PnL is {pl_pct:.2%} which is less than or equal to {pct_threshold_msg} "
                            f"of entry price. Closing the position to prevent further losses."
                        )
                    action.verbose_info = f"Position {pos_state.trade_id} has PnL of {pl_pct:.2%}. Closing the position to prevent further losses."
                    action.effective_date = last_updated + t_plus_n_bdays
                    pos_state.action = action
                    opinions.append(pos_state)

            else:
                logger.info(
                    f"Position {pos_state.trade_id} has PnL of {pl_pct:.2%}. No action taken. PnL is less than both lock-in threshold of {self.config.lock_in_profit_threshold:.2%} and roll threshold of {self.config.roll_profit_threshold:.2%}, and stop-loss condition is not met."
                )
                pos_state.action = HOLD(trade_id=pos_state.trade_id)
                pos_state.action.reason = f"PnL is {pl_pct:.2%}. No action taken."
                pos_state.action.verbose_info = (
                    f"Position {pos_state.trade_id} has PnL of {pl_pct:.2%}. No action taken."
                )
                opinions.append(pos_state)

        return CogActions(opinions=opinions, date=portfolio_context.date, source_cog=self.name)

    def _get_current_close_open_quantity(self, pos_state: PositionState) -> Tuple[int, int]:
        """Aggregate opened and closed quantity from trade entries.

        Args:
            pos_state: Position state whose trade ledger entries are inspected.

        Returns:
            Tuple[int, int]:
                - open quantity from BUY entries
                - close quantity from SELL/EXERCISE entries

        Notes:
            Empty or missing trade ledgers return ``(0, 0)``.
        """
        entries = pos_state.trades.entries() if pos_state.trades is not None else pd.DataFrame()
        if entries.empty:
            return 0, 0
        open_qty = entries[entries["direction"] == "BUY"]["quantity"].sum()
        close_qty = (
            entries[entries["direction"].isin(["SELL", "EXERCISE"])]["quantity"].sum()
            if not entries[entries["direction"].isin(["SELL", "EXERCISE"])]["quantity"].empty
            else 0
        )
        return open_qty, close_qty

    def _has_closed_before(self, trade_id: str, portfolio_context: PositionState) -> bool:
        """Determine whether the position has prior close activity.

        Args:
            trade_id: Trade identifier associated with the position.
            portfolio_context: Position state containing the trade ledger.

        Returns:
            bool: ``True`` if at least one SELL/EXERCISE entry exists after at
            least one BUY entry; otherwise ``False``.

        Notes:
            `trade_id` is accepted for API clarity and logging consistency with
            caller logic, while the computation currently relies on the provided
            `portfolio_context` entries.
        """
        open_qty, close_qty = self._get_current_close_open_quantity(portfolio_context)
        if open_qty == 0:
            return False
        return close_qty > 0
