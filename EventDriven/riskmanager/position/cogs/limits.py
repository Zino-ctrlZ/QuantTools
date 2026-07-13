"""Position Limits and Sizing Cog for Risk Management.

This module implements the LimitsAndSizingCog, a critical component of the position
analysis system that enforces Greek-based risk limits and calculates position sizes.
It integrates with various sizer implementations to provide dynamic risk management
adapted to market conditions.

Core Class:
    LimitsAndSizingCog: Enforces limits and sizes positions

Key Responsibilities:
    Limit Enforcement:
        - Monitor Greek exposures (delta, gamma, vega, theta)
        - Compare against configured thresholds
        - Generate CLOSE/ADJUST actions on violations
        - Track position-level and portfolio-level limits

    Position Sizing:
        - Calculate optimal contract quantities
        - Apply sizer-specific logic (fixed, vol-adjusted)
        - Respect capital constraints
        - Ensure integer contract quantities

    Metadata Tracking:
        - Store position-specific limit information
        - Track delta limits per position
        - Maintain sizing scalars (for vol adjustment)
        - Log all limit calculations

Limit Types:
    Delta Limits:
        - Primary directional risk constraint
        - Calculated per position based on capital
        - Can be fixed or volatility-adjusted
        - Enforced at position and portfolio level

    Gamma Limits (planned):
        - Controls convexity exposure
        - Important for large moves
        - Typically proportional to delta limit

    Vega Limits (planned):
        - Controls volatility exposure
        - Critical for vol-sensitive strategies
        - Scaled by position notional

    Theta Limits (planned):
        - Controls time decay exposure
        - Important for income strategies
        - Daily P&L impact tracking

Sizer Integration:
    DefaultSizer:
        - Fixed delta limits
        - Simple capital-based sizing
        - No market regime adjustment
        - Predictable and stable

    ZscoreRVolSizer:
        - Volatility-adjusted limits
        - Z-score based scaling
        - Adapts to market conditions
        - Reduces risk in volatile periods

Position Lifecycle:
    New Position Creation:
        1. on_new_position() called by analyzer
        2. _calculate_limits() determines delta limit
        3. _update_position_quantity() sizes contracts
        4. _create_position_metadata() stores details
        5. PositionLimits stored in position_limits dict

    Ongoing Monitoring:
        1. _analyze_impl() called on each cycle
        2. Iterate through all positions
        3. Check Greeks against limits
        4. Generate actions for violations
        5. Return CogActions with recommendations

Analysis Process:
    Per-Position Evaluation:
        - Retrieve current position data
        - Get option Greeks at current date
        - Compare to stored limits
        - Determine if action needed
        - Generate appropriate RMAction

    Action Generation:
        - CLOSE: Limit violated beyond threshold
        - ADJUST: Minor violation, reduce size
        - HOLD: Within acceptable ranges
        - Reason captured for audit

Configuration:
    LimitsEnabledConfig:
        - enabled: Master switch for cog
        - name: Cog identifier ('limits')
        - delta_lmt_type: 'default' or 'zscore'
        - enabled_limits: List of active limit types
        - violation_threshold: Tolerance before action

    SizerConfigs:
        - DefaultSizerConfigs: For fixed sizing
        - ZscoreSizerConfigs: For vol-adjusted sizing
        - Auto-selected based on delta_lmt_type

Position Limits Structure:
    PositionLimits dataclass:
        - delta: Delta limit (contracts * delta per contract)
        - gamma: Gamma limit (optional)
        - vega: Vega limit (optional)
        - theta: Theta limit (optional)
        - notional: Maximum position notional
        - quantity: Maximum contract quantity

Metadata Structure:
    _LimitsMetaData dataclass:
        - trade_id: Position identifier
        - date: Limit calculation date
        - signal_id: Originating signal
        - scalar: Vol adjustment scalar (1.0 for default)
        - sizing_lev: Leverage applied
        - delta_lmt: Calculated delta limit
        - delta: Actual position delta
        - option_price: Price at time of sizing
        - undl_price: Underlying price at time

Limit Calculation:
    Default (Fixed) Limits:
        delta_limit = (cash * sizing_lev) / spot_price

    ZScore (Vol-Adjusted) Limits:
        base_limit = (cash * sizing_lev) / spot_price
        scalar = 1 / (1 + |zscore| * sensitivity)
        delta_limit = base_limit * scalar

    Both ensure:
        - Proportional to available capital
        - Scaled by leverage setting
        - Inversely related to underlying price

Position Quantity Calculation:
    Formula:
        quantity = floor(delta_limit / (delta * option_price))

    Constraints:
        - Minimum: 1 contract
        - Maximum: Based on cash available
        - Integer contracts only
        - Respects delta limit strictly

Limit Violations:
    Detection:
        - Current delta > position_limits.delta * threshold
        - Other Greeks checked if enabled
        - Portfolio aggregates checked separately

    Response:
        - Generate CLOSE action for severe violations
        - Generate ADJUST for minor overages
        - Include violation details in action.reason
        - Log for audit and debugging

Usage Example:
    # Initialize cog
    limits_config = LimitsEnabledConfig(
        enabled=True,
        delta_lmt_type='zscore',
        enabled_limits=['delta', 'vega']
    )

    sizer_config = ZscoreSizerConfigs(
        sizing_lev=1.5,
        sensitivity=0.5,
        lookback=252
    )

    limits_cog = LimitsAndSizingCog(
        config=limits_config,
        sizer_configs=sizer_config,
        underlier_list=['AAPL', 'MSFT', 'GOOGL']
    )

    # Add to analyzer
    analyzer.add_cog(limits_cog)

    # On new position
    new_pos_state = limits_cog.on_new_position(new_position_state)
    # Now includes calculated limits and quantity

    # During analysis
    context = PositionAnalysisContext(...)
    cog_actions = limits_cog.analyze(context)
    # Returns actions for any limit violations

Integration Points:
    - Called by PositionAnalyzer on each cycle
    - Receives NewPositionState from RiskManager
    - Returns CogActions with limit-based opinions
    - Sizer interfaces with portfolio manager
    - Limits stored and tracked per position

Performance:
    - Limit calculations cached per position
    - Metadata stored for quick lookup
    - Efficient Greek comparisons
    - Minimal overhead per analysis cycle

Error Handling:
    - Invalid delta_lmt_type raises EVBacktestError
    - Missing underlier_list defaults to empty
    - Zero deltas handled gracefully
    - Logging of all calculation steps

Notes:
    - Position limits set once on creation
    - Limits can be recalculated on rolls (configurable)
    - Metadata includes scalar for transparency
    - All calculations logged to delta_limit_log
    - Sizer type determined by delta_lmt_type
    - Config changes trigger sizer reload
"""

import math

import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime
from EventDriven.configs.core import LimitsEnabledConfig
from EventDriven.dataclasses.states import NewPositionState
from EventDriven.types import Order
from EventDriven.exceptions import EVBacktestError
from typing import List, Optional, Union, Dict
from trade.helpers.Logging import setup_logger
from trade.helpers.helper import to_datetime
from EventDriven.riskmanager.position.base import BaseCog
from EventDriven.riskmanager.sizer._sizer import DefaultSizer, BaseSizer, ZscoreRVolSizer, default_delta_limit  # noqa
from EventDriven.configs.core import ZscoreSizerConfigs, DefaultSizerConfigs
from EventDriven.dataclasses.limits import PositionLimits
from EventDriven.dataclasses.states import (
    PositionAnalysisContext,
    CogActions,
)
from .analyze_utils import (
    analyze_position,
    get_dte_and_moneyness_from_trade_id,
)
from .vars import MEASURES_SET

logger = setup_logger("EventDriven.riskmanager.position.cogs.limits", stream_log_level="INFO")


@dataclass
class _LimitsMetaData:
    trade_id: str
    date: datetime
    signal_id: str
    scalar: float
    sizing_lev: float
    delta_lmt: float
    delta_per_contract: Optional[float] = None
    option_price: Optional[float] = None
    undl_price: Optional[float] = None
    prev_quantity: Optional[int] = None
    new_quantity: Optional[int] = None
    rvol: Optional[float] = None


def metadata_from_store_payload(payload: object) -> Optional[_LimitsMetaData]:
    """Convert a store payload back to ``_LimitsMetaData``.

    Args:
        payload: Dataclass, dict, or ``None`` value from a metadata store.

    Returns:
        Reconstructed metadata object, or ``None`` when ``payload`` is ``None``.
    """
    if payload is None:
        return None
    if isinstance(payload, _LimitsMetaData):
        return payload
    if not isinstance(payload, dict):
        return None

    raw_date = payload.get("date")
    if isinstance(raw_date, str):
        raw_date = to_datetime(raw_date)

    return _LimitsMetaData(
        trade_id=payload["trade_id"],
        date=raw_date,
        signal_id=payload["signal_id"],
        scalar=payload["scalar"],
        sizing_lev=payload["sizing_lev"],
        delta_lmt=payload["delta_lmt"],
        delta_per_contract=payload.get("delta_per_contract"),
        option_price=payload.get("option_price"),
        undl_price=payload.get("undl_price"),
        prev_quantity=payload.get("prev_quantity"),
        new_quantity=payload.get("new_quantity"),
        rvol=payload.get("rvol"),
    )


class LimitsAndSizingCog(BaseCog):
    """
    Class for Managing Position Limits and Sizing based on various risk metrics
    """

    default_config = LimitsEnabledConfig()

    def __init__(
        self,
        config: Optional[LimitsEnabledConfig] = None,
        sizer_configs: Optional[Union[DefaultSizerConfigs, ZscoreSizerConfigs]] = None,
        underlier_list: Optional[List[str]] = None,
    ):
        self._sizer_configs: Optional[Union[DefaultSizerConfigs, ZscoreSizerConfigs]] = sizer_configs
        self.position_limits: Dict[str, PositionLimits] = {}
        self.position_metadata: Dict[str, _LimitsMetaData] = {}
        self.allow_buffer = True  # Whether to allow a buffer to the delta limit if the calculated position size is very low (e.g. <=2 contracts).
        self.underlier_list = list(set(underlier_list if underlier_list is not None else []))
        if config is None:
            config = LimitsEnabledConfig()

        super().__init__(config)
        self._load_sizers_and_configs()

    def _load_sizers_and_configs(self):
        """Initialize configuration and sizer objects."""
        ## Initialize Config
        if self.config is None:
            self.config = LimitsEnabledConfig()

        ## Initialize Sizer Configs
        if self._sizer_configs is None:
            if self.config.delta_lmt_type == "default":
                self._sizer_configs = DefaultSizerConfigs()
            elif self.config.delta_lmt_type == "zscore":
                self._sizer_configs = ZscoreSizerConfigs()
            else:
                raise EVBacktestError(f"Delta limit type {self.config.delta_lmt_type} not recognized")

        ## Initialize Sizer based on delta limit type
        if self.config.delta_lmt_type == "default":
            self.sizer: BaseSizer = DefaultSizer(**self._sizer_configs.__dict__)
        elif self.config.delta_lmt_type == "zscore":
            self.sizer: BaseSizer = ZscoreRVolSizer(
                **self._sizer_configs.__dict__, underlier_list=self.underlier_list, pm=None, rm=None
            )
        else:
            raise EVBacktestError(f"Delta limit type {self.config.delta_lmt_type} not recognized")

    @property
    def sizer_configs(self) -> Union[DefaultSizerConfigs, ZscoreSizerConfigs]:
        return self._sizer_configs

    @sizer_configs.setter
    def sizer_configs(self, value: Union[DefaultSizerConfigs, ZscoreSizerConfigs]) -> None:
        assert isinstance(value, (DefaultSizerConfigs, ZscoreSizerConfigs)), (
            "sizer_configs must be of type DefaultSizerConfigs or ZscoreSizerConfigs"
        )
        self._sizer_configs = value

        ## Update config delta limit type to match sizer configs
        if self.config.delta_lmt_type != value.delta_lmt_type:
            logger.info(
                f"Updating LimitsCog config delta_lmt_type from {self.config.delta_lmt_type} to {value.delta_lmt_type}"
            )
        self.config.delta_lmt_type = value.delta_lmt_type

        ## Reload sizer with new configs if applicable
        self._load_sizers_and_configs()

    def on_new_position(self, new_pos_state: NewPositionState) -> NewPositionState:
        """
        Handle new position event and update limits accordingly.
        This is where daily limits would be recalculated based on new positions.
        Args:
            new_pos_state (NewPositionState): The new position state containing order, request, and position data.
            request (OrderRequest): The order request associated with the new position.

        What this does:
            - It sets up the maximum delta a position can take based on the sizer's daily delta limit calculation.
            - It also updates the position quantity in the order data.
            - To get info on how limits are calculated, refer to the sizer's `get_daily_delta_limit` method.
        Returns:
            NewPositionState: The updated position state with limits applied.
        """
        self._calculate_limits(new_pos_state)
        self._update_position_quantity(new_pos_state)
        self._create_position_metadata(new_pos_state)
        self._on_new_position_failsafe(
            new_pos_state
        )  # Ensure limits and quantity are set to reasonable values even if there are issues in the main logic
        return new_pos_state

    def _on_new_position_failsafe(self, new_pos_state: NewPositionState) -> NewPositionState:
        """
        Failsafe method ensures the quantity size is never 0 as long as cash can buy at least 1 contract, and that limits are set to some default value, even
        if there are errors in the main on_new_position logic.

        When quantity is 0, may bump quantity to 1 and persist a buffered delta limit (delta * 1.15) only when
        ``at_time_data.delta`` is finite and positive; otherwise the sizer CREATE limit is kept and not saved.
        """
        if new_pos_state.limits is not None and new_pos_state.order["data"]["quantity"] != 0:
            return new_pos_state
        request = new_pos_state.request
        option_price = new_pos_state.at_time_data.get_price()
        delta = new_pos_state.at_time_data.delta
        tick_cash = request.tick_cash
        chain_spot = new_pos_state.undl_at_time_data.chain_spot["close"]
        order = new_pos_state.order
        default_delta_lmt = default_delta_limit(
            cash_available=tick_cash,
            underlier_price_at_time=chain_spot,
            sizing_lev=self.sizer_configs.sizing_lev,
        )
        if new_pos_state.limits is None:
            logger.warning(
                f"Limits were not set for trade_id {new_pos_state.order['data']['trade_id']}. Setting to default delta limit of {default_delta_lmt}."
            )
            new_pos_state.limits = PositionLimits(
                delta=default_delta_lmt, dte=self.config.default_dte, moneyness=self.config.default_moneyness
            )
        if new_pos_state.order["data"]["quantity"] == 0:
            max_size_cash_can_buy = abs(math.floor(tick_cash / (option_price * 100)))
            if max_size_cash_can_buy >= 1:
                logger.warning(
                    f"Quantity was calculated as 0 for trade_id {new_pos_state.order['data']['trade_id']}. However, based on the available cash of {tick_cash} and option price of {option_price}, the strategy could afford to buy up to {max_size_cash_can_buy} contracts. Setting quantity to 1."
                )
                order_dict = new_pos_state.order.to_dict()
                order_dict["data"]["quantity"] = 1
                new_pos_state.order = Order.from_dict(order_dict)
            else:
                logger.warning(
                    f"Quantity was calculated as 0 for trade_id {new_pos_state.order['data']['trade_id']}, and even a single contract cannot be afforded based on the available cash of {tick_cash} and option price of {option_price}. Quantity will remain at 0."
                )

            ## Update limits to set to delta + buffer to avoid repeated sizing issues if the issue was with limit calculation
            if delta is not None and math.isfinite(delta) and delta > 0:
                logger.warning(
                    f"Delta limit for trade_id {new_pos_state.order['data']['trade_id']} was calculated as {new_pos_state.limits.delta}, which is below the default delta per contract of {delta}. Setting delta limit to {delta * 1.15} to allow for at least 1 contract to be sized in the next cycle, and to avoid repeated sizing issues if the issue was with limit calculation."
                )
                new_pos_state.limits.delta = (
                    delta * 1.15
                )  # Set delta limit to 15% above the delta of a single contract to allow for at least 1 contract to be sized in the next cycle, and to avoid repeated sizing issues if the issue was with limit calculation
                pos_lmts = PositionLimits(
                    delta=new_pos_state.limits.delta,
                    dte=self.config.default_dte,
                    moneyness=self.config.default_moneyness,
                    creation_date=request.date,
                )
                self._save_position_limits(order["data"]["trade_id"], order["signal_id"], pos_lmts)
            else:
                logger.warning(
                    f"Delta for trade_id {new_pos_state.order['data']['trade_id']} is non-finite ({delta!r}); "
                    f"skipping failsafe delta limit overwrite and keeping sizer CREATE limit of {new_pos_state.limits.delta}."
                )

            ## Update metadata as well to reflect the new limits and quantity
            metadata = self._get_metadata(new_pos_state.order["data"]["trade_id"])
            if metadata is not None:
                metadata.delta_lmt = new_pos_state.limits.delta
                metadata.new_quantity = new_pos_state.order["data"]["quantity"]
                ## Re-persist so live DB/metadata stores see the failsafe bump
                self._store_metadata(metadata)
                logger.warning(
                    f"Updated metadata for trade_id {new_pos_state.order['data']['trade_id']} to reflect new delta limit of {metadata.delta_lmt} and new quantity of {metadata.new_quantity}."
                )

        return new_pos_state

    def _create_position_metadata(self, new_pos_state: NewPositionState) -> None:
        """
        Create and store metadata for the new position.
        """
        order = new_pos_state.order
        request = new_pos_state.request
        undl_data = new_pos_state.undl_at_time_data
        option_price = new_pos_state.at_time_data.get_price()
        delta = new_pos_state.at_time_data.delta
        scalar = (
            1
            if isinstance(self.sizer, DefaultSizer)
            else self.sizer.scaler.get_scaler_on_date(sym=new_pos_state.symbol, date=request.date)
        )
        rvol = (
            None
            if isinstance(self.sizer, DefaultSizer)
            else self.sizer.scaler.get_rvol_on_date(sym=new_pos_state.symbol, date=request.date)
        )
        metadata = _LimitsMetaData(
            trade_id=order["data"]["trade_id"],
            date=request.date,
            signal_id=order["signal_id"],
            scalar=scalar,
            sizing_lev=self.sizer_configs.sizing_lev,
            delta_per_contract=delta,
            option_price=option_price,
            undl_price=undl_data.chain_spot["close"],
            delta_lmt=new_pos_state.limits.delta,
            new_quantity=order["data"]["quantity"],
            rvol=rvol,
        )
        logger.info(f"Storing position metadata: {metadata}")
        self._store_metadata(metadata)

    def _store_metadata(self, metadata: _LimitsMetaData) -> None:
        """
        Store the given metadata in the position_metadata dictionary.
        """
        self.position_metadata[metadata.trade_id] = metadata

    def _get_metadata(self, trade_id: str) -> Optional[_LimitsMetaData]:
        """
        Retrieve metadata for a given trade ID.
        """
        return self.position_metadata.get(trade_id, None)

    def _calculate_limits(self, new_pos_state: NewPositionState) -> float:
        """
        Calculate the position limits for a new position state.
        """
        logger.info(f"New position event received: {new_pos_state}")
        order = new_pos_state.order
        request = new_pos_state.request
        undl_data = new_pos_state.undl_at_time_data
        limits = self.sizer.get_daily_delta_limit(
            signal_id=order["signal_id"],
            position_id=order["data"]["trade_id"],
            date=request.date,
            current_cash=request.tick_cash,
            underlier_price_at_time=undl_data.chain_spot["close"],
        )
        logger.info(
            f"Calculated limits for position {order['data']['trade_id']}: {limits}. Details: cash={request.tick_cash}, undl_price={undl_data.chain_spot['close']}"
        )
        pos_lmts = PositionLimits(
            delta=limits,
            dte=self.config.default_dte,
            moneyness=self.config.default_moneyness,
            creation_date=request.date,
        )
        self._save_position_limits(order["data"]["trade_id"], order["signal_id"], pos_lmts)
        new_pos_state.limits = pos_lmts
        return pos_lmts.delta

    def _save_position_limits(self, trade_id: str, signal_id: str, limits: PositionLimits) -> None:
        """
        Save the position limits for a given trade ID.
        """
        self.position_limits[trade_id] = limits

    def _get_position_limits(self, trade_id: str, signal_id: str) -> PositionLimits:
        """
        Retrieve the position limits for a given trade ID.
        """
        return self.position_limits.get(trade_id, None)

    def _update_position_quantity(self, new_position_state: NewPositionState) -> None:
        """
        Update the position quantity in the order data.
        """
        delta_lmt = new_position_state.limits.delta
        request = new_position_state.request
        option_price = new_position_state.at_time_data.get_price()
        delta = new_position_state.at_time_data.delta
        q = self.sizer.calculate_position_size(
            current_cash=request.tick_cash, option_price_at_time=option_price, delta=delta, delta_limit=delta_lmt
        )
        order = new_position_state.order
        order_dict = order.to_dict()
        order_dict["data"]["quantity"] = q
        if q == 0:
            logger.warning(
                f"Calculated position size is 0 for order {order['data']['trade_id']}. Delta per contract ({delta}) exceeds limit {delta_lmt}."
            )

        ## Add buffer to delta limit if quantity is <=2 to avoid repeatedly hitting delta limit.
        elif q <= 2 and self.allow_buffer:
            logger.warning(
                f"Calculated position size is {q} for order {order['data']['trade_id']}, which is very low and may indicate that the position is hitting the delta limit. Delta per contract is {delta} and delta limit is {delta_lmt}. Consider reviewing the position or adjusting the delta limit to allow for more flexibility."
            )
            if isinstance(self.sizer, ZscoreRVolSizer):
                rvol_z = self.sizer.scaler.get_rvol_on_date(sym=new_position_state.symbol, date=request.date)
                multiplier = min(1 + 0.15 * max(rvol_z, 0), 1.20)
            else:
                multiplier = 1.15
            new_delta_lmt = delta_lmt * multiplier
            lmts = PositionLimits(
                delta=new_delta_lmt,
                dte=self.config.default_dte,
                moneyness=self.config.default_moneyness,
                creation_date=request.date,
            )
            self._save_position_limits(order["data"]["trade_id"], order["signal_id"], lmts)
            logger.warning(
                f"Delta limit for order {order['data']['trade_id']} has been increased from {delta_lmt} to {new_delta_lmt} to allow for a larger position size and to avoid repeatedly hitting the delta limit. This adjustment is based on a multiplier of {multiplier} applied to the original delta limit."
            )

        logger.info(f"Updated position quantity to {q} for order {order['data']['trade_id']}.")
        new_position_state.order = Order.from_dict(order_dict)

    def _analyze_impl(self, portfolio_context: PositionAnalysisContext) -> CogActions:
        """
        Analyze the given portfolio state and print relevant metrics.
        """
        positions = portfolio_context.portfolio.positions
        portfolio_state = portfolio_context.portfolio
        bkt_info = portfolio_context.portfolio_meta
        opinions = []

        ## Pre-compute commonly used values (Task #5 optimization)
        strat_enabled_limits = self.config.enabled_limits
        bkt_start_date = bkt_info.start_date
        t_plus_n = bkt_info.t_plus_n
        last_updated = portfolio_state.last_updated
        t_plus_n_bdays = pd.offsets.BusinessDay(max(t_plus_n, 1))

        for position in positions:
            trade_id = position.trade_id
            scaling_qty = position.quantity if not position.current_position_data.is_qty_scaled else 1
            dte_limit = self._get_position_limits(trade_id, position.signal_id).dte
            moneyness_limit = self._get_position_limits(trade_id, position.signal_id).moneyness

            ## Optimized dictionary filtering (Task #3)
            ## Direct key access with MEASURES_SET for O(1) lookup
            greeks_limit = {
                k: v
                for k, v in self._get_position_limits(trade_id, position.signal_id).__dict__.items()
                if k in MEASURES_SET
            }
            greek_exposure = {
                k: v * scaling_qty for k, v in position.current_position_data.__dict__.items() if k in MEASURES_SET
            }

            qty = position.quantity

            ## Use combined function to parse trade_id only once (performance optimization)
            dte, moneyness_list = get_dte_and_moneyness_from_trade_id(
                trade_id=trade_id,
                check_date=position.last_updated,
                check_price=position.current_underlier_data.chain_spot["close"],
                start=bkt_start_date,
                is_backtest=bkt_info.is_backtest,
            )

            ## Analyze position
            action = analyze_position(
                trade_id=trade_id,
                dte=dte,
                position_greek_limit=greeks_limit,
                dte_limit=dte_limit,
                moneyness_limit=moneyness_limit,
                greeks=greek_exposure,
                qty=qty,
                moneyness_list=moneyness_list,
                strategy_enabled_actions=strat_enabled_limits,
                t_plus_n=t_plus_n,
                tick=position.underlier_tick,
                signal_id=position.signal_id,
            )

            ## Update analysis_date
            action.analysis_date = last_updated

            ## Update effective date to be the next trading day after last_updated + t_plus_n
            ## This is because analysis is based on EOD data at last_updated. Execution therefore has to start from the next trading day.
            ## If t_plus_n is 0, then effective date will be the next trading day after last_updated, which is the expected behavior.
            ## If t_plus_n is > 0, then effective date will be last_updated + t_plus_n, but if that falls on a non-trading day, we need to move it to the next trading day. Therefore, we add a buffer of 1 day to ensure we move to the next trading day if last_updated + t_plus_n falls on a non-trading day.
            action.effective_date = last_updated + t_plus_n_bdays

            ## Only generate verbose_info for non-HOLD actions (Task #4 optimization)
            if action.action != "HOLD":
                action.verbose_info = (
                    f"Analysis: {action.analysis_date} | Effective: {action.effective_date} | "
                    f"Trade: {trade_id} | DTE: {dte} | Moneyness: {moneyness_list} | "
                    f"Greeks: {greek_exposure} | Qty: {qty} | Action: {action.action} | "
                    f"Limits: DTE={dte_limit}, MN={moneyness_limit}, Greeks={greeks_limit}"
                )
            else:
                action.verbose_info = None
            position.action = action
            opinions.append(position)
        return CogActions(
            opinions=opinions, strategy_id=self.config.run_name, date=portfolio_context.date, source_cog=self.name
        )

    def get_delta_limit(self, tick_cash: float, chain_spot: float, date: pd.Timestamp, ticker: str) -> float:
        """
        Override to provide delta limits based on mean reversion logic.
        This can be used by the position manager to enforce sizing constraints.
        """

        return np.inf  # No limit by default, can be overridden by specific logic in cogs like mean reversion cog
