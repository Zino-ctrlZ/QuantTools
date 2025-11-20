
import pandas as pd
from datetime import datetime
from EventDriven.configs.core import LimitsEnabledConfig
from EventDriven.dataclasses.states import NewPositionState
from EventDriven.types import Order
from EventDriven.exceptions import EVBacktestError
from typing import List, Optional, Union, Dict
from trade.helpers.Logging import setup_logger
from EventDriven.riskmanager.position.base import BaseCog
from EventDriven.riskmanager.sizer._sizer import DefaultSizer, BaseSizer, ZscoreRVolSizer
from EventDriven.configs.core import ZscoreSizerConfigs, DefaultSizerConfigs
from EventDriven.dataclasses.limits import PositionLimits
from EventDriven.dataclasses.states import (
    PositionAnalysisContext,
    CogActions,
)
from .analyze_utils import (
    analyze_position,
    get_dte_from_trade_id,
    get_moneyness_from_trade_id,
)
from .vars import MEASURES
logger = setup_logger('EventDriven.riskmanager.position.cogs.limits', stream_log_level="WARNING")


class LimitsAndSizingCog(BaseCog):
    """
    Class for Managing Position Limits and Sizing based on various risk metrics
    """
    default_config = LimitsEnabledConfig()

    def __init__(
        self,
        config: LimitsEnabledConfig = None,
        sizer_configs: Optional[Union[DefaultSizerConfigs, ZscoreSizerConfigs]] = None,
        underlier_list: Optional[List[str]] = None,
    ):
        self._sizer_configs: Optional[Union[DefaultSizerConfigs, ZscoreSizerConfigs]] = sizer_configs
        self.position_limits: Dict[str, PositionLimits] = {}
        self.underlier_list = underlier_list if underlier_list is not None else []
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
        assert isinstance(
            value, (DefaultSizerConfigs, ZscoreSizerConfigs)
        ), "sizer_configs must be of type DefaultSizerConfigs or ZscoreSizerConfigs"
        self._sizer_configs = value

        ## Update config delta limit type to match sizer configs
        if self.config.delta_lmt_type != value.delta_lmt_type:
            logger.info(f"Updating LimitsCog config delta_lmt_type from {self.config.delta_lmt_type} to {value.delta_lmt_type}")
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
        return new_pos_state

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
        logger.info(f"Calculated limits for position {order['data']['trade_id']}: {limits}")
        pos_lmts = PositionLimits(delta=limits, dte=self.config.default_dte, moneyness=self.config.default_moneyness)
        self.position_limits[order["data"]["trade_id"]] = pos_lmts
        new_pos_state.limits = pos_lmts
        return pos_lmts.delta

    def _update_position_quantity(self, new_position_state: NewPositionState) -> None:
        """
        Update the position quantity in the order data.
        """
        delta_lmt = self.position_limits[new_position_state.order["data"]["trade_id"]].delta
        request = new_position_state.request
        option_price = new_position_state.at_time_data.get_price()
        delta = new_position_state.at_time_data.delta
        q = self.sizer.calculate_position_size(
            current_cash=request.tick_cash, 
            option_price_at_time=option_price, 
            delta=delta,
            delta_limit=delta_lmt
        )
        order = new_position_state.order
        order_dict = order.to_dict()
        order_dict["data"]["quantity"] = q
        if q == 0:
            logger.warning(f"Calculated position size is 0 for order {order['data']['trade_id']}. Delta per contract ({delta}) exceeds limit {delta_lmt}.")
        new_position_state.order = Order.from_dict(order_dict)


    def _analyze_impl(self, portfolio_context: PositionAnalysisContext) -> CogActions:
        """
        Analyze the given portfolio state and print relevant metrics.
        """
        positions = portfolio_context.portfolio.positions
        portfolio_state = portfolio_context.portfolio
        bkt_info = portfolio_context.portfolio_meta
        opinions = []

        for position in positions:
            strat_enabled_limits = self.config.enabled_limits
            trade_id = position.trade_id
            dte_limit = self.position_limits[trade_id].dte
            moneyness_limit = self.position_limits[trade_id].moneyness
            greeks_limit = self.position_limits[trade_id].__dict__
            greeks_limit = {k: v for k, v in greeks_limit.items() if k in MEASURES}
            greek_exposure = position.current_position_data.__dict__
            greek_exposure = {k: v for k, v in greek_exposure.items() if k in MEASURES}
            qty = position.quantity
            moneyness_list = get_moneyness_from_trade_id(
                trade_id=trade_id,
                check_price=position.current_underlier_data.chain_spot["close"],
                check_date=position.last_updated,
                start=bkt_info.start_date,
            )
            dte = get_dte_from_trade_id(
                trade_id=trade_id,
                check_date=position.last_updated,
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
                t_plus_n=bkt_info.t_plus_n,
            )

            ## Update analysis_date
            action.analysis_date = portfolio_state.last_updated
            action.effective_date = portfolio_state.last_updated + pd.Timedelta(days=bkt_info.t_plus_n)
            action.verbose_info = f"""
            Analysis Date: {action.analysis_date}
            Effective Date: {action.effective_date}
            Trade ID: {trade_id}
            DTE: {dte}
            Moneyness List: {moneyness_list}
            Greek Exposure: {greek_exposure}
            Position Quantity: {qty}
            Action: {action.action}
            Limits Applied: DTE Limit - {dte_limit}, Moneyness Limit - {moneyness_limit}, Greek Limits - {greeks_limit}
            Analysis performed on {datetime.now()}
            Analysis generated by LimitsAndSizingCog
            """
            position.action = action
            opinions.append(position)
        return CogActions(opinions=opinions, strategy_id="", date=portfolio_context.date, source_cog=self.name)
