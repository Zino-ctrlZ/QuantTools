from typing import Dict
import math
from EventDriven.riskmanager.position.base import BaseCog
from EventDriven.configs.core import MeanReversionSizerConfigs
from typing import Optional
from EventDriven.dataclasses.states import NewPositionState, PositionAnalysisContext, CogActions
import numpy as np
from EventDriven.riskmanager.sizer._utils import default_delta_limit, delta_position_sizing
from EventDriven.riskmanager.position.cogs.limits import _LimitsMetaData
from EventDriven.types import SignalID
from trade.backtester_._multi_asset_strategy import MultiAssetStrategy
from dataclasses import dataclass
import pandas as pd
from EventDriven.configs.base import pydantic_dataclass
from EventDriven.configs.core import BaseConfigs, _CustomFrozenBaseConfigs, BaseCogConfig, StrategyLimitsEnabled
from pydantic import ConfigDict, Field
from EventDriven.dataclasses.limits import PositionLimits
from .limits import LimitsAndSizingCog
from trade.helpers.Logging import setup_logger
from EventDriven.riskmanager.actions import ROLL, Changes
from EventDriven.configs.core import VectorizedCogConfig
from .analyze_utils import get_dte_and_moneyness_from_trade_id

logger = setup_logger("EventDriven.riskmanager.position.cogs.donchian_cog")

@pydantic_dataclass(config=ConfigDict(arbitrary_types_allowed=True), )
class DonchianMomentumCogConfig(BaseCogConfig):
    """Configuration for DonchianMomentumCog."""
    
    name: str = "DonchianMomentumCog"
    sizing_lev: int = 1
    min_scale: float = 0.75
    max_scale: float = 1.50
    dte_limit_enabled: bool = True
    dte_threshold: int = 15
    

@dataclass
class _DonchianLimitsMetaData(_LimitsMetaData):
    breakout_score: float = None
    rvol: float = None
    

class DonchianMomentumCog(BaseCog):
    """
    Donchian Momentum Cog for position analysis.

    This cog evaluates the momentum of a position based on its entry price relative to
    the Donchian channel. It emits opinions on whether the position is showing positive
    or negative momentum, which can be used for risk management decisions.
    """
    default_config: DonchianMomentumCogConfig = DonchianMomentumCogConfig()
    def __init__(
            self, 
            eq_strategy: MultiAssetStrategy,
            config: Optional[DonchianMomentumCogConfig] = None
):
        if config is None:
            config: DonchianMomentumCogConfig = DonchianMomentumCogConfig()
        
        self._config: DonchianMomentumCogConfig = config
        # super().__init__(config)
        self.eq_strategy = eq_strategy
        self.position_limits: Dict[str, PositionLimits] = {}
        self.position_metadata: Dict[str, _DonchianLimitsMetaData] = {}
    
    def on_new_position(self, state: NewPositionState) -> None:
        """
        Analyze the momentum of a new position based on its entry price and the Donchian channel.

        Args:
            state: The new position state after creation.
        Returns:
            None (this cog is for analysis and opinion generation, not direct action).
        """
        order = state.order
        requet = state.request
        ticker = state.symbol
        undl_data = state.undl_at_time_data
        chain_spot = undl_data.chain_spot["close"]
        option_chain = state.at_time_data
        cash_available = requet.tick_cash
        signal_id = SignalID(order.signal_id)
        opt_price = option_chain.get_price()
        date = order["date"]
        info = self.eq_strategy.info_on_date(ticker=ticker, current_date=date)
        breakout_score = info.get("momentum_breakout_score", 0)
        rvol = info.get("momentum_rvol", 0)
        if not signal_id.strategy_slug.startswith("donchian"):
            logger.warning(f"Signal ID {signal_id} does not appear to be from a Donchian strategy. Skipping momentum analysis.")
            q = 1
            order["data"]["quantity"] = q
            breakout_score = None
            rvol = None
            delta = option_chain.delta
            scaled_limit = 1
            scaler = None


        else:
            scaler = self.get_momentum_scaler(
                breakout_score=breakout_score,
                rvol=rvol,
                min_mult=self.config.min_scale,
                max_mult=self.config.max_scale
            )

            base_limit = default_delta_limit(
                cash_available=cash_available,
                underlier_price_at_time=chain_spot,
                sizing_lev=self.config.sizing_lev,
            )

            scaled_limit = base_limit * scaler

            ## Scale the delta down to 90% to allow room for natural delta movement and reduce overtrading risk
            tgt_delta = scaled_limit * 0.9
            delta = option_chain.delta

            q = delta_position_sizing(
                cash_available=cash_available,
                option_price_at_time=opt_price,
                delta=delta,
                delta_limit=tgt_delta,
            )

            if q == 0:
                logger.warning(
                    f"Position {order['data']['trade_id']} has zero quantity after momentum scaling. "
                    f"Breakout Score: {breakout_score}, RVOL: {rvol}, Scaler: {scaler:.2f}, "
                    f"Base Delta Limit: {base_limit:.4f}, Scaled Delta Limit: {scaled_limit:.4f}, "
                    f"Target Delta (90% of Scaled Limit): {tgt_delta:.4f}, Option Delta: {delta:.4f}"
                )
                if cash_available > opt_price:
                    logger.info(
                        f"Cash available (${cash_available:.2f}) is greater than option price (${opt_price:.2f}), but quantity is zero."
                        f" Setting quantity to 1 to allow position opening and future momentum adjustments."
                    )
                    q = 1
            logger.info(
                f"Calculated delta limit: {scaled_limit:.4f}, resulting quantity: {q} lev: {self.config.sizing_lev}. Breakout Score: {breakout_score}, RVOL: {rvol}, Scaler: {scaler:.2f}. Trade ID: {order['data']['trade_id']} with option delta {delta:.4f}"
            )
        order["data"]["quantity"] = q
        metadata = _DonchianLimitsMetaData(
            trade_id=order["data"]["trade_id"],
            date=order["date"],
            signal_id=order["signal_id"],
            scalar=scaler,
            sizing_lev=self.config.sizing_lev,
            delta_per_contract=delta,
            option_price=opt_price,
            undl_price=undl_data.chain_spot["close"],
            delta_lmt=scaled_limit,
            new_quantity=q,
            breakout_score=breakout_score,
            rvol=rvol,
        )
        self.position_metadata[order["data"]["trade_id"]] = metadata

        pos_lmts = PositionLimits(
            delta=scaled_limit,
            dte=self.config.dte_threshold,
            creation_date=state.request.date,
        )
        state.limits = pos_lmts

        self._save_position_limits(order["data"]["trade_id"], order["signal_id"], pos_lmts)
        self._store_metadata(metadata)
        logger.debug(f"Stored momentum metadata for trade ID {order['data']['trade_id']}: {metadata}")
        

    def _save_position_limits(self, trade_id: str, signal_id: str, limits: PositionLimits) -> None:
        """
        Save the position limits for future reference.

        Args:
            trade_id: The unique identifier for the trade.
            signal_id: The identifier for the signal that generated the trade.
            limits: The PositionLimits object containing the limits to be saved.
        Returns:
            None
        """        
        self.position_limits[trade_id] = limits
        logger.debug(f"Saved position limits for trade ID {trade_id}: {limits}")

    def _store_metadata(self, metadata: _DonchianLimitsMetaData) -> None:
        """
        Store the metadata associated with a position for future analysis.

        Args:
            metadata: The _DonchianLimitsMetaData object containing the metadata to be stored.
        Returns:
            None
        """        
        self.position_metadata[metadata.trade_id] = metadata
        logger.debug(f"Stored metadata for trade ID {metadata.trade_id}: {metadata}")

    def get_momentum_scaler(
        self,
        *,
        breakout_score: float,
        rvol: float,
        min_mult: float = 0.75,
        max_mult: float = 1.50,
    ) -> float:
        """
        Compute bounded momentum sizing multiplier using:
            - breakout score
            - realized volatility

        Returns
        -------
        float
            Position sizing multiplier.
        """


        mult = 1.0

        # --------------------------------------------------
        # Breakout Score
        # --------------------------------------------------

        if not pd.isna(breakout_score):
            # strongest breakout regimes
            if breakout_score >= 2:
                mult *= 1.25

            elif breakout_score >= 1:
                mult *= 1.10

            # weak breakout
            elif breakout_score < 0.25:
                mult *= 0.90

        # --------------------------------------------------
        # Realized Volatility
        # --------------------------------------------------

            if not pd.isna(rvol):
                # strongest observed region
                if 0.60 <= rvol < 1.00:
                    mult *= 1.25

                elif 0.40 <= rvol < 0.60:
                    mult *= 1.10

                # compressed / weak momentum
                elif rvol < 0.15:
                    mult *= 0.85

                # ultra-extreme / unstable
                elif rvol >= 1.00:
                    mult *= 0.95

        # --------------------------------------------------
        # Final Clamp
        # --------------------------------------------------
        mult = max(min_mult, mult)
        mult = min(max_mult, mult)

        return float(mult)

    def _analyze_impl(self, context: PositionAnalysisContext) -> CogActions:
        """
        Analyze open positions for DTE-based roll triggers.

        Process:
            1. Iterate over all open positions in portfolio
            2. Calculate DTE using get_dte_and_moneyness_from_trade_id()
            3. If dte_limit_enabled and dte < dte_threshold:
               - Create ROLL opinion with reason
            4. Return aggregated CogActions

        Args:
            context: Portfolio snapshot with positions, date, and backtest parameters.

        Returns:
            CogActions: Opinions generated by this cog (roll recommendations).
        """
        opinions = []

        if not self.config.dte_limit_enabled:
            logger.debug("DTE limit disabled; no roll checks performed")
            return CogActions(date=context.date, source_cog=self.name, opinions=opinions)

        positions = context.portfolio.positions
        portfolio_meta = context.portfolio_meta
        last_updated = context.portfolio.last_updated
        t_plus_n = portfolio_meta.t_plus_n
        t_plus_n_bdays = pd.offsets.BusinessDay(max(t_plus_n, 1))
        backtest_start = portfolio_meta.start_date

        for pos_state in positions:
            try:
                # Calculate DTE using exact same method as limits cog
                dte, _ = get_dte_and_moneyness_from_trade_id(
                    trade_id=pos_state.trade_id,
                    check_date=pos_state.last_updated,
                    check_price=pos_state.current_underlier_data.chain_spot["close"],
                    start=backtest_start,
                    is_backtest=portfolio_meta.is_backtest,
                )

                qty = pos_state.quantity

                # Check roll threshold
                if dte < self.config.dte_threshold:
                    logger.info(
                        f"Position {pos_state.trade_id}: DTE {dte} below threshold {self.config.dte_threshold}. "
                        f"Recommending ROLL."
                    )

                    action = ROLL(
                        trade_id=pos_state.trade_id,
                        action=Changes(quantity_diff=0, new_quantity=qty),
                    )
                    action.reason = (
                        f"DTE {dte} below threshold {self.config.dte_threshold}. Rolling to extend duration."
                    )
                    action.analysis_date = last_updated
                    action.effective_date = last_updated + t_plus_n_bdays

                    # Create opinion for this position
                    action.verbose_info = (
                        f"Analysis: {action.analysis_date} | Effective: {action.effective_date} | "
                        f"Trade: {pos_state.trade_id} | DTE: {dte} | Threshold: {self.config.dte_threshold}"
                    )
                    pos_state.action = action
                    opinions.append(pos_state)
                else:
                    logger.debug(
                        f"Position {pos_state.trade_id}: DTE {dte} >= threshold {self.config.dte_threshold}. No action."
                    )

            except Exception as e:
                logger.warning(f"Error calculating DTE for position {pos_state.trade_id}: {e}. Skipping roll check.")
                logger.warning(f"Stack trace: ", exc_info=True)
                continue

        logger.info(
            f"VectorizedCog analysis complete. {len(opinions)} roll opinion(s) generated for {len(positions)} position(s)."
        )

        return CogActions(date=context.date, source_cog=self.name, opinions=opinions)



