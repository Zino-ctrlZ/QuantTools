from typing import Dict 
from EventDriven.riskmanager.position.cogs.limits import LimitsAndSizingCog, PositionLimits
from EventDriven.configs.core import LimitsEnabledConfig, BaseSizerConfigs
from .save_utils import get_position_limit, store_position_limits
from ..cogs.vars import MEASURES_SET


class LiveCOGLimitsAndSizingCog(LimitsAndSizingCog):
    """
    A specialized LimitsAndSizingCog for live COGS (Cost of Goods Sold) management.
    This class extends the base LimitsAndSizingCog to implement live-specific logic
    for calculating and managing position limits and sizing in a live trading environment.
    """
    def __init__(self,
                 config: LimitsEnabledConfig = None,
                 sizer_configs: BaseSizerConfigs = None,
                 underlier_list: list = None):
        super().__init__(config=config, sizer_configs=sizer_configs, underlier_list=underlier_list)
        self._position_limits: Dict[str, PositionLimits] = {}

    def _save_position_limits(self, 
                              trade_id: str, 
                              signal_id: str, 
                              limits: PositionLimits) -> None:
        """
        Save the position limits for a given trade ID and signal ID.
        """
        store_position_limits(
            delta_limit=limits.delta,
            gamma_limit=limits.gamma,
            vega_limit=limits.vega,
            theta_limit=limits.theta,
            trade_id=trade_id,
            signal_id=signal_id,
            strategy_name=self.config.run_name,
            date=limits.creation_date,
        )

    def _get_position_limits(self, trade_id: str, signal_id: str) -> PositionLimits:
        """
        Retrieve the position limits for a given trade ID and signal ID.
        """
        if trade_id in self.position_limits:
            return self.position_limits[trade_id]
        
        lm = PositionLimits()
        for risk_measure in MEASURES_SET:
            date, limit_value = get_position_limit(
                trade_id=trade_id,
                strategy_name=self.config.run_name,
                signal_id=signal_id,
                risk_measure=risk_measure,
            )
            if limit_value is not None:
                setattr(lm, risk_measure, limit_value)
                lm.creation_date = date
        lm.dte = self.config.default_dte
        lm.moneyness = self.config.default_moneyness

        self.position_limits[trade_id] = lm
        return lm