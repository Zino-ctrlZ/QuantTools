from pydantic.dataclasses import dataclass as pydantic_dataclass
from pydantic import ConfigDict
import numbers
from typing import Union, Tuple, List, Literal, Dict
from datetime import datetime, date
import pandas as pd
from abc import ABC
from pydantic import Field
from EventDriven.configs.base import (
    BaseConfigs,
    _CustomFrozenBaseConfigs,
    )
from EventDriven._vars import OPTION_TIMESERIES_START_DATE


@pydantic_dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ChainConfig(BaseConfigs):
    """
    Configuration class for Option Chain related settings.
    Ultimately, it would be used to filter the chain data retrieved from the data source.
    """

    max_pct_width: numbers.Number = None
    min_oi: numbers.Number = None


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
    rvol_window: Union[numbers.Number, Tuple[numbers.Number, ...]] = None
    rolling_window: numbers.Number = 100
    weights: Tuple[numbers.Number, numbers.Number, numbers.Number] = (0.5, 0.3, 0.2)
    vol_type: str = "mean"
    norm_const: numbers.Number = 1.0
    delta_lmt_type: Literal["default", "zscore"] = "zscore"

@pydantic_dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class OrderResolutionConfig(BaseConfigs):
    """
    Configuration class for Order Resolution settings.
    """
    resolve_enabled: bool = True
    otm_moneyness_width: float = 0.45
    itm_moneyness_width: float = 0.45
    max_close: float = 10.0
    max_tries: int = 20
    max_dte_tolerance: int = 90

@pydantic_dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class UndlTimeseriesConfig(BaseConfigs):
    """
    Configuration class for underlying timeseries data.
    """
    interval: str = '1d'

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
    add_columns: List[Tuple[str, str]] = Field(default_factory=list, description="List of tuples where each tuple contains (column_name, function_name) to add additional calculated columns. Function will be fetched from ADD_COLUMNS_FACTORY.")
    skip_columns: List[str] = Field(default_factory=lambda: ["Delta", "Gamma", "Vega", "Theta", "Midpoint"])

@pydantic_dataclass(config=ConfigDict(arbitrary_types_allowed=True), kw_only=True)
class BaseCogConfig(BaseConfigs):
    """
    Base configuration for any PositionAnalyzer cog.
    Each cog will usually subclass this for its specific settings.
    """

    name: str = None
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


@pydantic_dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class BacktesterConfig(BaseConfigs):
    """
    Configuration class for Backtest related settings.
    """

    t_plus_n: int = 1
    finalize_trades: bool = False
    raise_errors: bool = False


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

    max_slippage: float = 0.25
    min_slippage: float = 0.16
    cache_orders: bool = False
    cache_position_analysis: bool = False
