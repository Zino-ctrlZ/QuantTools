from EventDriven.riskmanager.position.base import BaseCog
from EventDriven.configs.core import MeanReversionSizerConfigs
from typing import Optional
from EventDriven.dataclasses.states import NewPositionState, PositionAnalysisContext, CogActions
import numpy as np
from EventDriven.riskmanager.sizer._utils import default_delta_limit, delta_position_sizing
from EventDriven.riskmanager.position.cogs.limits import _LimitsMetaData
from trade.backtester_._multi_asset_strategy import MultiAssetStrategy
from dataclasses import dataclass
import pandas as pd
from trade.helpers.Logging import setup_logger
logger = setup_logger("EventDriven.riskmanager.position.cogs.mean_reversion")


@dataclass
class _MRLimitsMetaData(_LimitsMetaData):
    zscore: float = None
    zexcess: float = None

class MeanReversionSizerCog(BaseCog):
    default_config: MeanReversionSizerConfigs = MeanReversionSizerConfigs()

    def __init__(self, eq_strategy: MultiAssetStrategy, config: Optional[MeanReversionSizerConfigs] = None):
        if config is None:
            config: MeanReversionSizerConfigs = self.default_config
        super().__init__(config)
        self.eq_strategy = eq_strategy
        self.position_metadata = {}

    def on_new_position(self, state: NewPositionState):
        order = state.order
        requet = state.request
        ticker = state.symbol
        undl_data = state.undl_at_time_data
        option_chain = state.at_time_data
        cash_available = requet.tick_cash
        opt_price = option_chain.get_price()
        date = order["date"]
        info = self.eq_strategy.info_on_date(ticker=ticker, current_date=date)
        z_raw = (info["indicators"]["zscore"])

        ## scale z-score on the remainder of the distance to the minimum z-score threshold.
        ## if z-score is below the minimum threshold, no scaling is applied
        z_excess = max(0, abs(z_raw) - self.config.min_zscore)  


        ## Calculate scaler based on z-score and config parameters
        scaler_raw = 1 + self.config.beta * z_excess
        scaler = np.clip(scaler_raw, self.config.min_scale, self.config.max_scale)

        ## Update order quantity based on scaler
        limit = self.get_delta_limit(
            tick_cash=cash_available,
            chain_spot=undl_data.chain_spot["close"],
            date=date,
            ticker=ticker
        )
        
        delta = state.at_time_data.delta
        q = delta_position_sizing(
            cash_available=cash_available,
            option_price_at_time=opt_price,
            delta=delta,
            delta_limit=limit,
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

    def _analyze_impl(self, portfolio_context: PositionAnalysisContext) -> CogActions:
        return CogActions(
            opinions=[], strategy_id=self.config.run_name, date=portfolio_context.date, source_cog=self.name
        )
    
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
        z_raw = info["indicators"]["zscore"]

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
