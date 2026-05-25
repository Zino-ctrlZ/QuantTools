"""Mean reversion position sizing cogs for EventDriven risk management.

Implements z-score-based position sizing and fallback quantity handling for
mean-reversion strategies, plus a senior variant with additional logging.

Core Classes:
    MeanReversionSizerCog: Base z-score sizing and metadata capture cog.
    SeniorMeanReversionSizerCog: Extended wrapper with additional logging.

Core Dataclasses:
    _MRLimitsMetaData: Per-trade sizing metadata for mean-reversion logic.

Usage:
    >>> cog = MeanReversionSizerCog(eq_strategy=strategy)
    >>> cog.on_new_position(state)
"""

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
from trade.helpers.Logging import setup_logger

logger = setup_logger("EventDriven.riskmanager.position.cogs.mean_reversion")


@dataclass
class _MRLimitsMetaData(_LimitsMetaData):
    """Metadata stored for mean-reversion sizing decisions."""

    zscore: float = None
    zexcess: float = None


class MeanReversionSizerCog(BaseCog):
    """Z-score-driven sizing cog for mean-reversion strategies."""

    default_config: MeanReversionSizerConfigs = MeanReversionSizerConfigs()

    def __init__(self, eq_strategy: MultiAssetStrategy, config: Optional[MeanReversionSizerConfigs] = None):
        """Initialize the mean-reversion sizing cog.

        Args:
            eq_strategy: Strategy interface used to fetch indicator data.
            config: Optional runtime sizing configuration.
        """
        if config is None:
            config = MeanReversionSizerConfigs()
        super().__init__(config)
        self.eq_strategy = eq_strategy
        self.position_metadata = {}
        self.config: MeanReversionSizerConfigs = config

    def on_new_position(self, state: NewPositionState):
        """Size a new position using z-score-based mean-reversion scaling.

        Args:
            state: New position state to size and annotate with metadata.

        Returns:
            None. Updates are applied in place to ``state.order`` metadata.
        """
        order = state.order
        request = state.request
        ticker = state.symbol
        undl_data = state.undl_at_time_data
        option_chain = state.at_time_data
        cash_available = request.tick_cash
        opt_price = option_chain.get_price()
        date = order["date"]
        info = self.eq_strategy.info_on_date(ticker=ticker, current_date=date)
        z_raw = info["mean_reversion_indicators"]["zscore"]
        signal_id = SignalID(order["signal_id"])
        if "mean_reversion" not in signal_id.strategy_slug:
            logger.warning(
                f"MeanReversionSizerCog received a new position for strategy {signal_id.strategy_slug} which does not contain 'mean_reversion'. This cog is designed for mean reversion strategies. Proceeding with sizing but please verify if this is intentional."
            )
            return  # Return if not a mean reversion strategy.

        ## scale z-score on the remainder of the distance to the minimum z-score threshold.
        ## if z-score is below the minimum threshold, no scaling is applied
        z_excess = max(0, abs(z_raw) - self.config.min_zscore)

        ## Calculate scaler based on z-score and config parameters
        scaler_raw = 1 + self.config.beta * z_excess
        scaler = np.clip(scaler_raw, self.config.min_scale, self.config.max_scale)

        ## Update order quantity based on scaler
        limit = self.get_delta_limit(
            tick_cash=cash_available, chain_spot=undl_data.chain_spot["close"], date=date, ticker=ticker
        )

        delta = state.at_time_data.delta
        q = delta_position_sizing(
            cash_available=cash_available,
            option_price_at_time=opt_price,
            delta=delta,
            delta_limit=limit,
        )
        logger.info(
            f"Calculated delta limit: {limit}, resulting quantity: {q} lev: {self.config.sizing_lev}. Raw scaler: {scaler_raw}, applied scaler: {scaler}. Trade ID: {order['data']['trade_id']} with z-score {z_raw} and z-excess {z_excess}"
        )

        ## Update order quantity and log metadata
        order["data"]["quantity"] = q
        metadata = _MRLimitsMetaData(
            trade_id=order["data"]["trade_id"],
            date=order["date"],
            signal_id=order["signal_id"],
            scalar=scaler,
            sizing_lev=self.config.sizing_lev,
            delta_per_contract=delta,
            option_price=opt_price,
            undl_price=undl_data.chain_spot["close"],
            delta_lmt=limit,
            new_quantity=q,
            zscore=z_raw,
            zexcess=z_excess,
        )
        logger.info(f"Storing metadata for trade_id {order['data']['trade_id']}: {metadata}")
        self.position_metadata[order["data"]["trade_id"]] = metadata

        if q == 0:
            self.on_new_position_update(
                state
            )  # Check if we can adjust quantity to 1 based on cash and option price, and log the rationale behind it.

        metadata.new_quantity = order["data"][
            "quantity"
        ]  # Update metadata with final quantity after potential adjustment in on_position_update
        logger.info(
            f"Final quantity for trade_id {order['data']['trade_id']} after potential adjustment: {metadata.new_quantity}. Updated metadata: {metadata}"
        )

    def _analyze_impl(self, portfolio_context: PositionAnalysisContext) -> CogActions:
        """Return no ongoing analysis opinions for this sizing-focused cog.

        Args:
            portfolio_context: Snapshot used by analysis pipeline.

        Returns:
            Empty CogActions payload for this cycle.
        """
        return CogActions(opinions=[], date=portfolio_context.date, source_cog=self.name)

    def get_delta_limit(self, tick_cash: float, chain_spot: float, date: pd.Timestamp, ticker: str) -> float:
        """
        Calculate delta limits based on cash, spot price, and z-score for mean reversion scaling.
        Principle: The more extreme the z-score (beyond the minimum threshold), the larger the position we can take, up to a maximum scaling factor.
        Asumes that the base delta limit is calculated using the default method, and then scaled based on the z-score excess.

        Args:
            tick_cash (float): Available cash for the position.
            chain_spot (float): Spot price of the underlying at the time of order.
            date (pd.Timestamp): Current date for fetching strategy info.
            ticker (str): Ticker symbol for fetching strategy info.
        Returns:
            float: Adjusted delta limit based on mean reversion scaling.
        """

        info = self.eq_strategy.info_on_date(ticker=ticker, current_date=date)
        z_raw = info["mean_reversion_indicators"]["zscore"]

        ## scale z-score on the remainder of the distance to the minimum z-score threshold.
        ## if z-score is below the minimum threshold, no scaling is applied
        z_excess = max(0, abs(z_raw) - self.config.min_zscore)
        base_delta = default_delta_limit(
            cash_available=tick_cash,
            underlier_price_at_time=chain_spot,
            sizing_lev=self.config.sizing_lev,
        )

        ## Calculate scaler based on z-score and config parameters
        scaler_raw = 1 + self.config.beta * z_excess
        scaler = np.clip(scaler_raw, self.config.min_scale, self.config.max_scale)

        ## Update order quantity based on scaler
        limit = base_delta * scaler
        return limit

    def on_new_position_update(self, state: NewPositionState):
        """
        If q=0 for a new position, we make it 1 if the cash available and option price would allow for at least 1 contract to be bought. This is to avoid situations where the sizing logic results in 0 quantity due to the z-score being just below the threshold, but in reality, based on the cash and option price, we could still take a small position. This method checks for that scenario and adjusts the quantity accordingly while logging the rationale behind it.
        """
        if state.order["data"]["quantity"] == 0:
            logger.info(
                f"Order quantity is 0 after mean reversion sizing for trade_id {state.order['data']['trade_id']}. This means the position will not be opened based on the current z-score and cash constraints."
            )
            cash_available = state.request.tick_cash
            option_price_at_time = state.at_time_data.get_price()
            max_size_cash_can_buy = abs(math.floor(cash_available / (option_price_at_time * 100)))
            if max_size_cash_can_buy >= 1:
                logger.info(
                    f"However, based on the available cash of {cash_available} and option price of {option_price_at_time}, the strategy could afford to buy up to {max_size_cash_can_buy} contracts. Consider adjusting the beta or z-score thresholds to allow for smaller position sizes in such scenarios."
                )
                state.order["data"]["quantity"] = (
                    1  # Optionally set to 1 to allow for at least a small position, or keep at 0 to strictly follow the sizing logic.
                )
            else:
                logger.info(
                    f"Based on the available cash of {cash_available} and option price of {option_price_at_time}, even a single contract cannot be afforded. The order will remain at quantity 0."
                )
        return state


class SeniorMeanReversionSizerCog(MeanReversionSizerCog):
    """Extended mean-reversion sizing cog with extra fallback logging."""

    def __init__(self, eq_strategy: MultiAssetStrategy, config: Optional[MeanReversionSizerConfigs] = None):
        """Initialize the senior mean-reversion cog.

        Args:
            eq_strategy: Strategy interface used to fetch indicator data.
            config: Optional runtime sizing configuration.
        """
        super().__init__(eq_strategy, config)

    def on_new_position(self, state):
        """Run base sizing then apply senior-level quantity-0 diagnostics.

        Args:
            state: New position state passed through sizing workflow.

        Returns:
            Result from base ``on_new_position`` execution.
        """

        ret = super().on_new_position(state)
        if state.order["data"]["quantity"] == 0:
            logger.info(
                f"Order quantity is 0 after mean reversion sizing for trade_id {state.order['data']['trade_id']}. This means the position will not be opened based on the current z-score and cash constraints."
            )
            cash_available = state.request.tick_cash
            option_price_at_time = state.at_time_data.get_price()
            max_size_cash_can_buy = abs(math.floor(cash_available / (option_price_at_time * 100)))
            if max_size_cash_can_buy >= 1:
                logger.info(
                    f"However, based on the available cash of {cash_available} and option price of {option_price_at_time}, the strategy could afford to buy up to {max_size_cash_can_buy} contracts. Consider adjusting the beta or z-score thresholds to allow for smaller position sizes in such scenarios."
                )
                state.order["data"]["quantity"] = (
                    1  # Optionally set to 1 to allow for at least a small position, or keep at 0 to strictly follow the sizing logic.
                )
            else:
                logger.info(
                    f"Based on the available cash of {cash_available} and option price of {option_price_at_time}, even a single contract cannot be afforded. The order will remain at quantity 0."
                )
        return ret
