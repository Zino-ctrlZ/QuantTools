
from EventDriven.configs.core import LimitsEnabledConfig
from EventDriven.dataclasses.states import NewPositionState
from EventDriven.types import Order
from EventDriven.exceptions import EVBacktestError
from typing import List, Optional, Union
from trade.helpers.Logging import setup_logger
from EventDriven.tests import BaseCog
from EventDriven.riskmanager.utils import parse_option_tick, swap_ticker
from trade.helpers.helper import compare_dates
from EventDriven.riskmanager.sizer._sizer import DefaultSizer, BaseSizer, ZscoreRVolSizer
from EventDriven.configs.core import ZscoreSizerConfigs, DefaultSizerConfigs
from EventDriven.dataclasses.limits import PositionLimits
logger = setup_logger('EventDriven.riskmanager.position.cogs.limits', stream_log_level="DEBUG")


class LimitsAndSizingCog(BaseCog):
    """
    Class for Managing Position Limits and Sizing based on various risk metrics
    """

    def __init__(
        self,
        config: LimitsEnabledConfig = None,
        sizer_configs: Optional[Union[DefaultSizerConfigs, ZscoreSizerConfigs]] = None,
        underlier_list: Optional[List[str]] = None,
    ):
        self._sizer_configs: Optional[Union[DefaultSizerConfigs, ZscoreSizerConfigs]] = sizer_configs
        self.position_limits: dict = {}
        self.underlier_list = underlier_list if underlier_list is not None else []
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
        pos_lmts = PositionLimits(delta=limits, dte=self.config.dte, moneyness=self.config.moneyness)
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
            current_cash=request.tick_cash, option_price_at_time=option_price, delta=delta, delta_limit=delta_lmt
        )
        order = new_position_state.order
        order_dict = order.to_dict()
        order_dict["data"]["quantity"] = q
        new_position_state.order = Order.from_dict(order_dict)

    def adjust_for_events(
        self,
        start: str,
        date: str,
        option: str | dict,
    ):
        """
        Adjusts the option tick for events like splits or dividends.
        """
        if isinstance(option, str):
            meta = parse_option_tick(option)
        elif isinstance(option, dict):
            meta = option
        else:
            raise ValueError("Option must be a string or a dictionary.")
        split = self.splits.get(swap_ticker(meta["ticker"]), None)
        if split is None:
            return meta
        for pack in split:
            if compare_dates.is_before(start, pack[0]) and compare_dates.is_after(date, pack[0]):
                meta["strike"] /= pack[1]
        return meta
