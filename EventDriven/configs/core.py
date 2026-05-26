from EventDriven.configs.base import pydantic_dataclass
from pydantic import ConfigDict
import numbers
from typing import Union, Tuple, List, Literal, Dict, Optional
from datetime import datetime, date
import pandas as pd
from abc import ABC
from pydantic import Field
from EventDriven.configs.base import (
    BaseConfigs,
    _CustomFrozenBaseConfigs,
)
from trade.optionlib.config.defaults import OPTION_TIMESERIES_START_DATE
from trade.helpers.Logging import setup_logger

logger = setup_logger("EventDriven.configs.core")


@pydantic_dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ChainConfig(BaseConfigs):
    """
    Configuration class for Option Chain related settings.
    Ultimately, it would be used to filter the chain data retrieved from the data source.
    """

    max_pct_width: numbers.Number = 0.2
    min_oi: numbers.Number = 25
    enable_delta_filter: bool = False


@pydantic_dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class OrderSchemaConfigs(BaseConfigs):
    """
    Configuration class for Order Schema related settings.
    """

    target_dte: numbers.Number = 270
    strategy: str = "vertical"
    structure_direction: str = "long"
    spread_ticks: numbers.Number = 1
    dte_tolerance: numbers.Number = 60
    min_moneyness: numbers.Number = 0.65
    max_moneyness: numbers.Number = 1
    min_total_price: numbers.Number = 0.95
    max_attempts: int = 3


@pydantic_dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class OrderPickerConfig(BaseConfigs):
    """
    Configuration class for Order Picker related settings.
    """

    start_date: Union[str, datetime, date] = pd.to_datetime(OPTION_TIMESERIES_START_DATE).date()
    end_date: Union[str, datetime, date] = datetime.now().date()

    def __post_init__(self, ctx=None):
        super().__post_init__(ctx)
        # Convert start_date and end_date to datetime if they are strings
        if isinstance(self.start_date, str):
            self.start_date = pd.to_datetime(self.start_date).date()
        if isinstance(self.end_date, str):
            self.end_date = pd.to_datetime(self.end_date).date()


@pydantic_dataclass(config=ConfigDict(arbitrary_types_allowed=True), kw_only=True)
class BaseSizerConfigs(_CustomFrozenBaseConfigs, ABC):
    """
    Base configuration class for Sizer modules.
    """

    delta_lmt_type: Literal["default", "zscore"] = "default"


@pydantic_dataclass(config=ConfigDict(arbitrary_types_allowed=True), kw_only=True)
class DefaultSizerConfigs(BaseSizerConfigs):
    """
    Default configuration class for Sizer modules.
    """

    sizing_lev: numbers.Number = 1.0


@pydantic_dataclass(config=ConfigDict(arbitrary_types_allowed=True), kw_only=True)
class ZscoreSizerConfigs(BaseSizerConfigs):
    """
    Z-score based configuration class for Sizer modules.
    """

    sizing_lev: numbers.Number = 1.0
    rvol_window: Optional[Union[numbers.Number, Tuple[numbers.Number, ...]]] = None
    rolling_window: numbers.Number = 100
    weights: Tuple[numbers.Number, numbers.Number, numbers.Number] = (0.5, 0.3, 0.2)
    vol_type: str = "mean"
    norm_const: numbers.Number = 1.0
    delta_lmt_type: Literal["default", "zscore"] = "zscore"


@pydantic_dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class UndlTimeseriesConfig(BaseConfigs):
    """
    Configuration class for underlying timeseries data.
    """

    interval: str = "1d"


@pydantic_dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class OptionPriceConfig(BaseConfigs):
    """
    Configuration class for option price data retrieval.
    """

    use_price: str = "midpoint"  # Options: "close", "bid", "ask", "midpoint"


@pydantic_dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class SkipCalcConfig(BaseConfigs):
    """
    When calculating option data for trades, skip calculations for options that meet certain criteria.
    These skips are used to determine if we should trade on a specific date or move to the next date.
    """

    window: int = 20
    skip_threshold: float = 3.0
    skip_enabled: bool = True
    abs_zscore_threshold: bool = False
    pct_zscore_threshold: bool = False
    spike_flag: bool = False
    std_window_bool: bool = False
    zero_filter: bool = True
    add_columns: List[Tuple[str, str]] = Field(
        default_factory=list,
        description="List of tuples where each tuple contains (column_name, function_name) to add additional calculated columns. Function will be fetched from ADD_COLUMNS_FACTORY.",
    )
    skip_columns: List[str] = Field(default_factory=lambda: ["Delta", "Gamma", "Vega", "Theta", "Midpoint"])


@pydantic_dataclass(config=ConfigDict(arbitrary_types_allowed=True), kw_only=True)
class BaseCogConfig(BaseConfigs):
    """
    Base configuration for any PositionAnalyzer cog.
    Each cog will usually subclass this for its specific settings.
    """

    name: Optional[str] = None
    enabled: bool = True


@pydantic_dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class StrategyLimitsEnabled(BaseConfigs):
    """
    Configuration class to hold enabled limit types for a strategy.
    Each attribute corresponds to a specific limit type.
    """

    delta: bool = True
    vega: bool = False
    gamma: bool = False
    theta: bool = False
    dte: bool = True
    moneyness: bool = True
    exercise: bool = False


@pydantic_dataclass(config=ConfigDict(arbitrary_types_allowed=True), kw_only=True)
class LimitsEnabledConfig(BaseCogConfig):
    """
    Flags to enable/disable enforcement of specific limit types for a strategy.
    """

    name: str = "LimitsEnabledCog"
    cache_actions: bool = True
    enabled: bool = True
    delta_lmt_type: Literal["default", "zscore"] = "default"
    default_dte: int = 120
    default_moneyness: float = 1.15
    enabled_limits: StrategyLimitsEnabled = Field(default_factory=StrategyLimitsEnabled)


@pydantic_dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class PositionAnalyzerConfig(BaseConfigs):
    """
    Global configuration for the PositionAnalyzer itself.
    """

    enabled: bool = True
    enabled_cogs: List[str] = Field(default_factory=list)


@pydantic_dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class PortfolioManagerConfig(BaseConfigs):
    """
    Configuration class for Backtest related settings.
    """

    weights_haircut: float = 0.0  # Haircut applied to weights
    roll_failed_orders: bool = True  # Whether signals that fail to be processed should be rolled forward


@pydantic_dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class LiquidityConfig(BaseConfigs):
    """Centralized liquidity control for both risk and execution layers."""

    level: int = 1
    max_spread_pct: float = 0.25

    def __post_init__(self, ctx=None):
        super().__post_init__(ctx)
        self.level = max(0, min(2, int(self.level)))


@pydantic_dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class BacktesterConfig(BaseConfigs):
    """
    Configuration class for Backtest related settings.
    """

    t_plus_n: int = 1
    finalize_trades: bool = True
    raise_errors: bool = False
    min_slippage_pct: float = 0.075
    max_slippage_pct: float = 0.15
    commission_per_contract_in_units: float = 0.0065
    liquidity: LiquidityConfig = Field(default_factory=LiquidityConfig)


@pydantic_dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class CashAllocatorConfig(BaseConfigs):
    """
    Threshold-based allocator for per-symbol max cash buckets.
    """

    thresholds: List[Tuple[float, float]] = Field(
        default_factory=lambda: [
            (500, 4),
            (300, 3),
            (200, 2),
            (100, 1),
            (0, 0.5),
        ],
        description="(min_alloc, bucket_value) pairs evaluated in order; first match wins.",
    )

    def alloc_for_weight(self, weight: float, cash: float) -> float:
        alloc = weight * cash
        for min_alloc, value in self.thresholds:
            if alloc >= min_alloc:
                return value
        return 0.0

    def build_max_cash_map(self, weights: Dict[str, float], cash: float) -> Dict[str, float]:
        return {sym: self.alloc_for_weight(w, cash) for sym, w in weights.items()}


@pydantic_dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class RiskManagerConfig(BaseConfigs):
    """
    Configuration class for Risk Manager related settings.
    """

    cache_orders: bool = False
    cache_position_analysis: bool = False
    cache_order_requests: bool = False


@pydantic_dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class MeanReversionSizerConfigs(BaseCogConfig):
    beta: float = 0.5
    name: str = "custom_mean_reversion_sizer"
    min_scale: float = 0.5
    max_scale: float = 2.0
    sizing_lev: int = 2
    default_dte: int = 10
    enabled_limits: StrategyLimitsEnabled = Field(
        default_factory=lambda: StrategyLimitsEnabled(delta=False, dte=True, moneyness=False)
    )
    # Minimum z-score threshold to trigger scaling adjustments
    min_zscore: float = 2.5


@pydantic_dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ScoringConfigs(BaseConfigs):
    tilt_strength: numbers.Number = 0.2
    spread_ticks: int = 2
    structure_direction: Literal["long", "short"] = "long"
    strategy: Literal["vertical", "naked"] = "vertical"
    hybrid_strategy_enabled: bool = False

    # Moneyness
    m_target: numbers.Number = 0.8
    min_moneyness: numbers.Number = 0.45
    max_moneyness: numbers.Number = 1.05
    m_sigma: numbers.Number = 0.2
    m_tilt: Literal["otm", "itm", "atm"] = "otm"

    # DTE
    target_dte: numbers.Number = 200
    dte_tolerance: numbers.Number = 100
    dte_sigma: numbers.Number = 10
    dte_tilt: Literal["flat", "short", "long"] = "short"

    # Mid price
    mid_min: numbers.Number = 0.5
    mid_max: numbers.Number = 3.0
    mid_upper_limit: numbers.Number = 5
    mid_lower_limit: numbers.Number = 0.25
    mid_sigma: numbers.Number = 0.25

    # Spread
    pct_spread_max: numbers.Number = 1.0
    target_spread_pct: numbers.Number = 0.2
    pct_spread_sigma: numbers.Number = 0.10

    # Open Interest
    oi_target: int = 1000

    # Theta burden
    theta_burden_max: numbers.Number = 0.03
    theta_burden_sigma: numbers.Number = 0.02


@pydantic_dataclass
class ExecutionHandlerConfig(BaseConfigs):
    slippage_model: Literal["randomized", "fixed", "spread_pct", "none"] = (
        "randomized"  # Whether to use randomized slippage, fixed slippage, slippage as a percentage of the spread, or no slippage. Default is randomized slippage.
    )
    pct_alpha: float = 0.25  # If using spread_pct slippage model, this is the percentage of the spread to use as slippage. For example, if pct_alpha is 0.25 and the spread is $0.20, then the slippage will be $0.05.


@pydantic_dataclass
class PnlMonitorConfig(BaseCogConfig):
    """
    Configuration dataclass for PnLMonitorCog.
    """

    name: Optional[str] = "PnLMonitorCog"
    enabled: bool = True

    @property
    def enable_stop_loss(self) -> bool:
        """
        Enable dynamic stop loss based on position characteristics or market conditions. This property can be used to determine whether to apply stop loss logic in the PnLMonitorCog. For example, you might want to enable stop loss for certain high-risk positions or during volatile market conditions, while keeping it disabled for more stable positions or in calmer markets. The actual logic for determining when to enable stop loss would depend on your specific risk management strategy and could involve factors such as volatility, position size, time to expiration, or other relevant metrics.
        """
        # Placeholder logic for dynamic stop loss enabling - this can be replaced with actual conditions based on market data or position characteristics
        return False

    @property
    def roll_profit_threshold(self) -> float:
        """
        Dynamic property to determine the maximum PnL percentage threshold for rolling positions to lock in profits.
        The idea is that if a position has achieved a certain percentage gain relative to its entry price, it may be prudent to roll the position to lock in those profits and potentially free up capital for new trades.
        This threshold can help prevent giving back gains in volatile markets while still allowing for significant profit capture.
        Note: This is a simplified example and should be tested and adjusted based on historical performance and risk tolerance.
        """
        return 1.0  # For example, this could represent a 100% gain relative to entry price as a threshold for rolling to lock in profits.

    @property
    def lock_in_profit_threshold(self) -> float:
        """
        Threshold to determine when to close a portion of the position to lock in profits. For example, if set to 0.5, it would trigger when the position has achieved a 50% gain relative to its entry price, prompting the closure of half the position to secure profits while allowing the rest to run.
        """
        return 0.5

    @property
    def stop_loss_pct(self) -> float:
        """
        Threshold to determine the stop loss percentage for exiting positions to prevent further losses. For example, if set to -0.7, it would trigger when the position has incurred a 70% loss relative to its entry price, prompting the closure of the position to limit further losses.
        """
        return -0.7

    @property
    def profit_lock_in_pct(self):
        """
        This controls the amount of profit locked to be added to cash used in next trade.
        For example, if a position is closed with a 50% gain and profit_lock_in_pct is 0.5, then an additional 25% of the entry price (which is 50% of the gain)
        would be added to the cash available for the next trade. This allows for a portion of the profits to be reinvested while still locking in gains.
        """
        return 0.25


@pydantic_dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class PnLMonitorConfigConfigurable(BaseCogConfig):
    """
    Configurable version of PnLMonitorConfig that allows dynamic adjustment of thresholds.
    """
    name: str = "PnLMonitorCog"
    enabled: bool = True
    roll_profit_threshold: float = 1.0  # Default to 100% gain threshold for rolling to lock in profits
    stop_loss_pct: float = -0.7  # Default to 70% loss threshold for exiting positions
    profit_lock_in_pct: float = 0.25  # Default to adding 25% of the gain back into cash for next trade
    lock_in_profit_threshold: float = (
        0.5  # Default to 50% gain threshold for closing half the position to lock in profits
    )
    enable_stop_loss: bool = False  # Default to having stop loss disabled, can be enabled based on position characteristics or market conditions


@pydantic_dataclass
class VectorizedCogConfig(BaseCogConfig):
    """
    Configuration dataclass for VectorizedCog.

    Simple position monitoring cog that:
    - Validates new positions by cash availability (informational)
    - Monitors open positions for DTE-based roll triggers
    """

    name: str = "VectorizedCog"
    enabled: bool = True
    dte_limit_enabled: bool = True
    dte_threshold: int = 30


@pydantic_dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class PlainSizingCogConfig(BaseCogConfig):
    """Configuration dataclass for PlainSizingCog.

    Simple sizing cog that:
    - Sizes new positions using default_delta_limit
    - Applies a one-contract affordability fallback when quantity is zero
    - Monitors open positions for DTE-based roll triggers
    - Skips checks for strategies matching excluded slug tokens
    """

    name: str = "PlainSizingCog"
    enabled: bool = True
    sizing_lev: float = 1.0
    dte_limit_enabled: bool = True
    dte_threshold: int = 30
    exclude_strategy_slug_tokens: List[str] = Field(default_factory=list)
