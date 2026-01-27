from dataclasses import dataclass
from trade.helpers.helper_types import SingletonMetaClass
from trade.optionlib.config.types import (DiscreteDivGrowthModel, DivType,)
from trade.optionlib.config.defaults import DIVIDEND_LOOKBACK_YEARS
from ._enums import (
    OptionSpotEndpointSource,
    OptionPricingModel,
    VolatilityModel,
    RealTimeFallbackOption
)
from typeguard import check_type
from typing import get_type_hints

@dataclass
class OptionDataConfig(metaclass=SingletonMetaClass):
    """Configuration for OptionDataManager."""

    option_spot_endpoint_source: OptionSpotEndpointSource = OptionSpotEndpointSource.EOD
    default_lookback_years: int = DIVIDEND_LOOKBACK_YEARS
    default_forecast_method: DiscreteDivGrowthModel = DiscreteDivGrowthModel.CONSTANT
    dividend_type: DivType = DivType.DISCRETE
    include_special_dividends: bool = False
    option_model: OptionPricingModel = OptionPricingModel.BINOMIAL
    volatility_model: VolatilityModel = VolatilityModel.MARKET
    n_steps: int = 100 
    undo_adjust: bool = True
    real_time_fallback_option: RealTimeFallbackOption = RealTimeFallbackOption.USE_LAST_AVAILABLE


    def assert_valid(self) -> None:
        """Validates all configuration values against business rules."""
        assert self.default_lookback_years > 0, "Lookback years must be positive."
        assert self.default_lookback_years <= 5, "Lookback years seems too large. Max 5."
        assert isinstance(
            self.default_forecast_method, DiscreteDivGrowthModel
        ), "Invalid forecast method. Expected DiscreteDivGrowthModel Enum."
        assert isinstance(self.dividend_type, DivType), "Invalid dividend type. Expected DivType Enum."
        assert isinstance(self.include_special_dividends, bool), "include_special_dividends must be a boolean."
        assert isinstance(
            self.option_spot_endpoint_source, OptionSpotEndpointSource
        ), "Invalid option_spot_endpoint_source. Expected OptionSpotEndpointSource Enum."
        assert isinstance(
            self.option_model, OptionPricingModel
        ), "Invalid option_model. Expected OptionPricingModel Enum."
        assert isinstance(
            self.volatility_model, VolatilityModel
        ), "Invalid volatility_model. Expected VolatilityModel Enum."

    def __post_init__(self) -> None:
        """Validates configuration after initialization."""
        self.assert_valid()

    def __setattr__(self, name, value):
        """Validates configuration after any attribute change."""
        all_hints = get_type_hints(self.__class__)
        hint = all_hints.get(name)
        if hint is not None:
            check_type(value, hint)
        super().__setattr__(name, value)
