
import pandas as pd
from dataclasses import dataclass
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
    get_dte_and_moneyness_from_trade_id,
)
from .vars import MEASURES_SET
logger = setup_logger('EventDriven.riskmanager.position.cogs.limits', stream_log_level="WARNING")

@dataclass
class _LimitsMetaData:
    trade_id: str
    date: datetime
    signal_id: str
    scalar: float
    sizing_lev: float
    delta_lmt: float
    delta: Optional[float] = None
    option_price: Optional[float] = None
    undl_price: Optional[float] = None

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
        self.position_metadata: Dict[str, _LimitsMetaData] = {}
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
        self._create_position_metadata(new_pos_state)
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
        scalar = 1 if isinstance(self.sizer, DefaultSizer) else self.sizer.scaler.get_scaler_on_date(sym=new_pos_state.symbol, date=request.date)
        metadata = _LimitsMetaData(
            trade_id=order["data"]["trade_id"],
            date=request.date,
            signal_id=order["signal_id"],
            scalar=scalar,
            sizing_lev=self.sizer_configs.sizing_lev,
            delta=delta,
            option_price=option_price,
            undl_price=undl_data.chain_spot["close"],
            delta_lmt=new_pos_state.limits.delta,
        )
        logger.info(f"Storing position metadata: {metadata}")
        self.position_metadata[order["data"]["trade_id"]] = metadata

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
        pos_lmts = PositionLimits(delta=limits, dte=self.config.default_dte, moneyness=self.config.default_moneyness, creation_date=request.date)
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

        ## Pre-compute commonly used values (Task #5 optimization)
        strat_enabled_limits = self.config.enabled_limits
        bkt_start_date = bkt_info.start_date
        t_plus_n = bkt_info.t_plus_n
        last_updated = portfolio_state.last_updated
        t_plus_n_timedelta = pd.Timedelta(days=t_plus_n)

        for position in positions:
            trade_id = position.trade_id
            scaling_qty = position.quantity if not position.current_position_data.is_qty_scaled else 1
            dte_limit = self._get_position_limits(trade_id, position.signal_id).dte
            moneyness_limit = self._get_position_limits(trade_id, position.signal_id).moneyness
            
            ## Optimized dictionary filtering (Task #3)
            ## Direct key access with MEASURES_SET for O(1) lookup
            greeks_limit = {k: v for k, v in self._get_position_limits(trade_id, position.signal_id).__dict__.items() if k in MEASURES_SET}
            greek_exposure = {k: v * scaling_qty for k, v in position.current_position_data.__dict__.items() if k in MEASURES_SET}
            
            qty = position.quantity
            
            ## Use combined function to parse trade_id only once (performance optimization)
            dte, moneyness_list = get_dte_and_moneyness_from_trade_id(
                trade_id=trade_id,
                check_date=position.last_updated,
                check_price=position.current_underlier_data.chain_spot["close"],
                start=bkt_start_date,
                is_backtest=bkt_info.is_backtest
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
            action.effective_date = last_updated + t_plus_n_timedelta
            
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
        return CogActions(opinions=opinions, strategy_id="", date=portfolio_context.date, source_cog=self.name)
