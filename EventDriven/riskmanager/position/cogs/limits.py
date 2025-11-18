
from EventDriven.configs.core import LimitsEnabledConfig
from EventDriven.dataclasses.states import NewPositionState
from EventDriven.types import Order
from EventDriven.exceptions import EVBacktestError
from typing import List, Optional, Union
from trade.helpers.Logging import setup_logger
from EventDriven.tests import BaseCog
from EventDriven.riskmanager.sizer._sizer import DefaultSizer, BaseSizer, ZscoreRVolSizer
from EventDriven.configs.core import ZscoreSizerConfigs, DefaultSizerConfigs
from EventDriven.dataclasses.limits import PositionLimits
logger = setup_logger('EventDriven.riskmanager.position.cogs.limits', stream_log_level="DEBUG")


class LimitsCog(BaseCog):
    """
    Class for Managing Position Limits based on various risk metrics
    """

    def __init__(
        self,
        config: LimitsEnabledConfig = None,
        sizer_configs: Optional[Union[DefaultSizerConfigs, ZscoreSizerConfigs]] = None,
        underlier_list: Optional[List[str]] = None,
    ):
        ## Initialize Config
        if config is None:
            config = LimitsEnabledConfig()

        ## Initialize Sizer Configs
        if sizer_configs is not None:
            self.sizer_configs = sizer_configs

        ## Initialize Sizer based on delta limit type
        if config.delta_lmt_type == "default":
            ## If None, use default configs
            if sizer_configs is None:
                self.sizer_configs: DefaultSizerConfigs = DefaultSizerConfigs()

            ## Else, ensure correct type
            else:
                assert isinstance(
                    sizer_configs, DefaultSizerConfigs
                ), "sizer_configs must be of type DefaultSizerConfigs for 'default' delta limit type"
            self.sizer: BaseSizer = DefaultSizer(**self.sizer_configs.__dict__)

        elif config.delta_lmt_type == "zscore":
            if sizer_configs is None:
                self.sizer_configs: ZscoreSizerConfigs = ZscoreSizerConfigs()

            else:
                assert isinstance(
                    sizer_configs, ZscoreSizerConfigs
                ), "sizer_configs must be of type ZscoreSizerConfigs for 'zscore' delta limit type"
            self.sizer: BaseSizer = ZscoreRVolSizer(
                **self.sizer_configs.__dict__, underlier_list=underlier_list, pm=None, rm=None
            )
        else:
            raise EVBacktestError(f"Delta limit type {config.delta_lmt_type} not recognized")

        self.position_limits: dict = {}

        super().__init__(config)

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
        pos_lmts = PositionLimits(delta=limits, dte=self.config.dte, moneyness=self.config.moneyness)
        self.position_limits[order["data"]["trade_id"]] = pos_lmts
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
            current_cash=request.tick_cash, option_price_at_time=option_price, delta=delta, delta_limit=delta_lmt
        )
        order = new_position_state.order
        order_dict = order.to_dict()
        order_dict["data"]["quantity"] = q
        new_position_state.order = Order.from_dict(order_dict)
