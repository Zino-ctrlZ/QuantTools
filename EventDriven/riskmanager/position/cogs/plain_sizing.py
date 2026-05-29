"""Plain sizing cog for default delta-limit based position management.

Provides a lightweight sizing and analysis cog that:
- Sizes new positions using default_delta_limit.
- Applies a one-contract affordability fallback when quantity is zero.
- Emits DTE-based ROLL opinions during analysis.
- Skips sizing and analysis checks for excluded strategy slug tokens.

Core Classes:
    PlainSizingCog: Position sizing and DTE-based roll analysis implementation.

Configuration:
    PlainSizingCogConfig: Runtime knobs for sizing leverage, DTE checks,
        and strategy-slug exclusion tokens.

Usage:
    >>> cog = PlainSizingCog()
    >>> cog.on_new_position(new_position_state)
"""

from typing import Dict, Optional

import pandas as pd

from EventDriven.configs.core import PlainSizingCogConfig
from EventDriven.dataclasses.limits import PositionLimits
from EventDriven.dataclasses.states import CogActions, NewPositionState, PositionAnalysisContext
from EventDriven.riskmanager.actions import Changes, ROLL
from EventDriven.riskmanager.position.base import BaseCog
from EventDriven.riskmanager.position.cogs.analyze_utils import get_dte_and_moneyness_from_trade_id
from EventDriven.riskmanager.position.cogs.limits import _LimitsMetaData
from EventDriven.riskmanager.sizer._utils import default_delta_limit, delta_position_sizing
from EventDriven.types import SignalID
from trade.helpers.Logging import setup_logger

logger = setup_logger("EventDriven.riskmanager.position.cogs.plain_sizing", stream_log_level="INFO")


class PlainSizingCog(BaseCog):
    """Default delta-limit sizing cog with DTE-based roll recommendations."""

    default_config = PlainSizingCogConfig()

    def __init__(self, config: Optional[PlainSizingCogConfig] = None):
        """Initialize the plain sizing cog.

        Args:
            config: Optional runtime configuration for sizing, roll checks,
                and exclusion behavior.
        """
        if config is None:
            config = PlainSizingCogConfig()
        super().__init__(config)
        self.position_limits: Dict[str, PositionLimits] = {}
        self.position_metadata: Dict[str, _LimitsMetaData] = {}
        self.config: PlainSizingCogConfig = config

    def _is_excluded_strategy(self, signal_id: str) -> bool:
        """Return whether a signal should be excluded from checks.

        Args:
            signal_id: Raw signal identifier.

        Returns:
            True if any configured exclusion token is contained in the
            strategy slug; otherwise False.
        """
        try:
            slug = SignalID(signal_id).strategy_slug
        except Exception:
            logger.warning(f"Unable to parse signal id {signal_id} for exclusion check.", exc_info=True)
            return False

        tokens = self.config.exclude_strategy_slug_tokens or []
        return any(token and token in slug for token in tokens)

    def on_new_position(self, state: NewPositionState) -> None:
        """Size a newly created position with default delta limits.

        Process:
            1. Skip if strategy slug matches exclusion tokens.
            2. Compute default delta limit from available cash.
            3. Compute quantity via delta_position_sizing.
            4. If quantity is zero and one contract is affordable, set quantity to 1.
            5. Store limits and metadata on state.

        Args:
            state: New position state to size.

        Returns:
            None. Updates are applied in place on ``state``.
        """
        order = state.order
        request = state.request

        if self._is_excluded_strategy(order.signal_id):
            logger.debug(
                f"Skipping plain sizing for excluded strategy signal {order.signal_id}. "
                f"Trade ID: {order['data']['trade_id']}"
            )
            return
        else:
            logger.info(f"Applying plain sizing for signal {order.signal_id}. Trade ID: {order['data']['trade_id']}")

        undl_data = state.undl_at_time_data
        option_chain = state.at_time_data
        chain_spot = undl_data.chain_spot["close"]
        opt_price = option_chain.get_price()
        delta = option_chain.delta

        cash_available = request.tick_cash if request.is_tick_cash_scaled else request.tick_cash * 100

        limit = default_delta_limit(
            cash_available=cash_available,
            underlier_price_at_time=chain_spot,
            sizing_lev=self.config.sizing_lev,
        )

        q = delta_position_sizing(
            cash_available=cash_available,
            option_price_at_time=opt_price,
            delta=delta,
            delta_limit=limit,
        )
        logger.info(
            "Plain sizing calculated values for trade %s | signal %s | cash_available=%s | "
            "tick_cash=%s | tick_cash_scaled=%s | underlier_price=%s | option_price=%s | "
            "option_delta=%s | sizing_lev=%s | delta_limit=%s | quantity=%s",
            order["data"]["trade_id"],
            order["signal_id"],
            cash_available,
            request.tick_cash,
            request.is_tick_cash_scaled,
            chain_spot,
            opt_price,
            delta,
            self.config.sizing_lev,
            limit,
            q,
        )

        if q == 0 and cash_available >= opt_price * 100:
            logger.info(
                f"Quantity was zero for trade {order['data']['trade_id']} but one contract is affordable. "
                "Setting quantity to 1."
            )
            q = 1

        order["data"]["quantity"] = q

        pos_lmts = PositionLimits(
            delta=limit,
            dte=self.config.dte_threshold,
            creation_date=request.date,
        )
        state.limits = pos_lmts
        self.position_limits[order["data"]["trade_id"]] = pos_lmts

        metadata = _LimitsMetaData(
            trade_id=order["data"]["trade_id"],
            date=order["date"],
            signal_id=order["signal_id"],
            scalar=1.0,
            sizing_lev=self.config.sizing_lev,
            delta_per_contract=delta,
            option_price=opt_price,
            undl_price=chain_spot,
            delta_lmt=limit,
            new_quantity=q,
        )
        logger.info(f"Storing plain sizing metadata for trade_id {order['data']['trade_id']}: {metadata}")
        self.position_metadata[order["data"]["trade_id"]] = metadata

    def _analyze_impl(self, context: PositionAnalysisContext) -> CogActions:
        """Analyze open positions and emit DTE-based roll recommendations.

        Process:
            1. Return no opinions when DTE checks are disabled.
            2. Skip excluded strategy slugs.
            3. Compute DTE via trade-id parser helper.
            4. Emit ROLL opinion when DTE is below threshold.

        Args:
            context: Portfolio snapshot for the current analysis cycle.

        Returns:
            CogActions containing roll opinions generated by this cog.
        """
        opinions = []

        if not self.config.dte_limit_enabled:
            return CogActions(date=context.date, source_cog=self.name, opinions=opinions)

        positions = context.portfolio.positions
        portfolio_meta = context.portfolio_meta
        last_updated = context.portfolio.last_updated
        t_plus_n = portfolio_meta.t_plus_n
        t_plus_n_bdays = pd.offsets.BusinessDay(max(t_plus_n, 1))
        backtest_start = portfolio_meta.start_date

        for pos_state in positions:
            if self._is_excluded_strategy(pos_state.signal_id):
                continue

            try:
                dte, _ = get_dte_and_moneyness_from_trade_id(
                    trade_id=pos_state.trade_id,
                    check_date=pos_state.last_updated,
                    check_price=pos_state.current_underlier_data.chain_spot["close"],
                    start=backtest_start,
                    is_backtest=portfolio_meta.is_backtest,
                )

                if dte < self.config.dte_threshold:
                    action = ROLL(
                        trade_id=pos_state.trade_id,
                        action=Changes(quantity_diff=0, new_quantity=pos_state.quantity),
                    )
                    action.reason = (
                        f"DTE {dte} below threshold {self.config.dte_threshold}. Rolling to extend duration."
                    )
                    action.analysis_date = last_updated
                    action.effective_date = last_updated + t_plus_n_bdays
                    action.verbose_info = (
                        f"Analysis: {action.analysis_date} | Effective: {action.effective_date} | "
                        f"Trade: {pos_state.trade_id} | DTE: {dte} | Threshold: {self.config.dte_threshold}"
                    )
                    pos_state.action = action
                    opinions.append(pos_state)

            except Exception as exc:
                logger.warning(
                    f"Error calculating DTE for position {pos_state.trade_id}: {exc}. Skipping roll check.",
                    exc_info=True,
                )

        return CogActions(date=context.date, source_cog=self.name, opinions=opinions)
