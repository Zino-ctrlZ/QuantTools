import pandas as pd
from dataclasses import dataclass
from typing import Any, Dict, Optional
from trade.optionlib.config.types import DivType
from ._enums import OptionSpotEndpointSource, OptionPricingModel, VolatilityModel, SeriesId
from .utils.date import DATE_HINT
from typeguard import check_type
from typing import get_type_hints


@dataclass
class Result:
    """Base class for all data manager result containers."""

    model_input_keys: Optional[Dict[str, Any]] = None

    def _additional_repr_fields(self) -> Dict[str, Any]:
        """Provides additional fields for string representation. Override in subclasses."""
        return {}

    def __repr__(self) -> str:
        """Returns string representation with additional fields from subclass."""
        additional_fields = self._additional_repr_fields()
        if additional_fields:
            fields_str = ", ".join(f"{k}={v!r}" for k, v in additional_fields.items())
            return f"{self.__class__.__name__}({fields_str})"
        return f"{self.__class__.__name__}()"

    def __setattr__(self, name, value):
        """Validates inputs on attribute set."""
        all_hints = get_type_hints(self.__class__)
        hint = all_hints.get(name)
        if hint is not None:
            check_type(value, hint)
        super().__setattr__(name, value)


@dataclass
class _EquityResultsBase(Result):
    """Base class for equity-related result containers."""

    symbol: Optional[str] = None

    def __repr__(self):
        return super().__repr__()


@dataclass
class DividendsResult(_EquityResultsBase):
    """Contains dividend schedule or yield data for a date range."""
    timeseries: Optional[pd.Series] = None
    dividend_type: Optional[DivType] = None
    key: Optional[str] = None
    undo_adjust: Optional[bool] = None

    @property
    def daily_discrete_dividends(self) -> Optional[pd.Series]:
        if self.dividend_type == DivType.DISCRETE:
            return self.timeseries
        return None
    
    @daily_discrete_dividends.setter
    def daily_discrete_dividends(self, value: Optional[pd.Series]) -> None:
        self.timeseries = value

    @property
    def daily_continuous_dividends(self) -> Optional[pd.Series]:
        if self.dividend_type == DivType.CONTINUOUS:
            return self.timeseries
        return None
    
    @daily_continuous_dividends.setter
    def daily_continuous_dividends(self, value: Optional[pd.Series]) -> None:
        self.timeseries = value
        

    ## For schedule timeseries, this will be the actual schedule keys
    model_input_keys: Optional[Dict[str, Any]] = None

    def __repr__(self) -> str:
        return super().__repr__()

    def is_empty(self) -> bool:
        """Checks if dividend data is missing or empty."""
        if self.dividend_type == DivType.DISCRETE:
            return self.daily_discrete_dividends is None or self.daily_discrete_dividends.empty
        elif self.dividend_type == DivType.CONTINUOUS:
            return self.daily_continuous_dividends is None or self.daily_continuous_dividends.empty
        return True

    def _additional_repr_fields(self) -> Dict[str, Any]:
        """Provides dividend-specific fields for string representation."""
        return {
            "symbol": self.symbol,
            "dividend_type": self.dividend_type,
            "key": self.key,
            "is_empty": self.is_empty(),
            "undo_adjust": self.undo_adjust,
        }


@dataclass
class RatesResult(Result):
    """Contains risk-free rate data for a date range."""

    symbol: Optional[str] = None
    timeseries: Optional[pd.Series] = None

    @property
    def daily_risk_free_rates(self) -> Optional[pd.Series]:
        return self.timeseries
    
    @daily_risk_free_rates.setter
    def daily_risk_free_rates(self, value: Optional[pd.Series]) -> None:
        self.timeseries = value

    def is_empty(self) -> bool:
        """Checks if rate data is missing or empty."""
        return self.timeseries is None or self.timeseries.empty
    
    def _additional_repr_fields(self):
        """Provides rate-specific fields for string representation."""
        return {
            "is_empty": self.is_empty(),
        }

    def __repr__(self) -> str:
        return super().__repr__()


@dataclass
class ForwardResult(_EquityResultsBase):
    """Contains forward price data (discrete or continuous dividend model)."""
    
    timeseries: Optional[pd.Series] = None
    dividend_type: Optional[DivType] = None
    key: Optional[str] = None
    dividend_result: Optional[DividendsResult] = None
    undo_adjust: Optional[bool] = True

    ## Dividend schedule or yield model input keys
    ## Rates model input keys
    model_input_keys: Optional[Dict[str, Any]] = None

    @property
    def daily_discrete_forward(self) -> Optional[pd.Series]:
        if self.dividend_type == DivType.DISCRETE:
            return self.timeseries
        return None
    
    @daily_discrete_forward.setter
    def daily_discrete_forward(self, value: Optional[pd.Series]) -> None:
        self.timeseries = value

    @property
    def daily_continuous_forward(self) -> Optional[pd.Series]:
        if self.dividend_type == DivType.CONTINUOUS:
            return self.timeseries
        return None
    
    @daily_continuous_forward.setter
    def daily_continuous_forward(self, value: Optional[pd.Series]) -> None:
        self.timeseries = value


    def is_empty(self) -> bool:
        """Checks if forward price data is missing or empty."""
        if self.dividend_type == DivType.DISCRETE:
            return self.daily_discrete_forward is None or self.daily_discrete_forward.empty
        elif self.dividend_type == DivType.CONTINUOUS:
            return self.daily_continuous_forward is None or self.daily_continuous_forward.empty
        return True

    def _additional_repr_fields(self) -> Dict[str, Any]:
        """Provides forward-specific fields for string representation."""
        return {
            "symbol": self.symbol,
            "is_empty": self.is_empty(),
            "undo_adjust": self.undo_adjust,
            "dividend_type": self.dividend_type,
            "key": self.key,
        }

    def __repr__(self) -> str:
        return super().__repr__()


@dataclass
class SpotResult(_EquityResultsBase):
    """Contains spot price data with optional split adjustment information."""

    timeseries: Optional[pd.Series] = None
    undo_adjust: Optional[bool] = None
    key: Optional[str] = None

    ## For spot timeseries. This is nothing but an indicator of the  source of spot data.
    model_input_keys: Optional[Dict[str, Any]] = None
    
    @property
    def daily_spot(self) -> Optional[pd.Series]:
        return self.timeseries
    
    @daily_spot.setter
    def daily_spot(self, value: Optional[pd.Series]) -> None:
        self.timeseries = value

    def is_empty(self) -> bool:
        return self.daily_spot is None or self.daily_spot.empty

    def _additional_repr_fields(self) -> Dict[str, Any]:
        """Provides spot-specific fields for string representation."""
        return {
            "symbol": self.symbol,
            "key": self.key,
            "is_empty": self.is_empty(),
            "undo_adjust": self.undo_adjust,
        }

    def __repr__(self) -> str:
        return super().__repr__()


@dataclass
class _OptionResultsBase(Result):
    """Base class for option-related result containers."""

    symbol: Optional[str] = None
    strike: Optional[float] = None
    expiration: Optional[DATE_HINT] = None
    right: Optional[str] = None

    def __repr__(self) -> str:
        """Delegates to base Result repr."""
        return super().__repr__()


@dataclass
class OptionSpotResult(_OptionResultsBase):
    """Container for option spot price timeseries data."""
    
    timeseries: Optional[pd.DataFrame] = None
    key: Optional[str] = None
    endpoint_source: Optional[OptionSpotEndpointSource] = None

    ## For option spot timeseries, this will be the actual endpoint parameters
    model_input_keys: Optional[Dict[str, Any]] = None

    @property
    def daily_option_spot(self) -> Optional[pd.DataFrame]:
        return self.timeseries
    
    @daily_option_spot.setter
    def daily_option_spot(self, value: Optional[pd.DataFrame]) -> None:
        self.timeseries = value

    @property
    def close(self) -> pd.Series:
        if not self.is_empty():
            return self.daily_option_spot["close"]
        else:
            return pd.Series(name="close", index=pd.DatetimeIndex([]), dtype=float)

    @property
    def midpoint(self) -> pd.Series:
        if not self.is_empty():
            return self.daily_option_spot["midpoint"]
        else:
            return pd.Series(name="midpoint", index=pd.DatetimeIndex([]), dtype=float)

    def is_empty(self) -> bool:
        """Checks if option spot data is missing or empty."""
        return self.daily_option_spot is None or self.daily_option_spot.empty

    def _additional_repr_fields(self) -> Dict[str, Any]:
        """Provides metadata on data presence."""
        return {
            "symbol": self.symbol,
            "strike": self.strike,
            "expiration": self.expiration,
            "right": self.right,
            "key": self.key,
            "is_empty": self.is_empty(),
            "endpoint_source": self.endpoint_source,
        }

    def __repr__(self) -> str:
        """Delegates to base Result repr."""
        return super().__repr__()


@dataclass
class VolatilityResult(_OptionResultsBase):
    """Contains volatility surface data."""

    timeseries: Optional[pd.Series] = None
    key: Optional[str] = None
    endpoint_source: Optional[OptionSpotEndpointSource] = None
    market_model: Optional[OptionPricingModel] = None
    vol_model: Optional[VolatilityModel] = None
    dividend_type: Optional[DivType] = None
    model_input_keys: Optional[Dict[str, Any]] = None

    def is_empty(self) -> bool:
        """Checks if volatility data is missing or empty."""
        return self.timeseries is None or self.timeseries.empty

    def _additional_repr_fields(self) -> Dict[str, Any]:
        """Provides volatility-specific fields for string representation."""
        return {
            "symbol": self.symbol,
            "expiration": self.expiration,
            "right": self.right,
            "strike": self.strike,
            "vol_model": self.vol_model,
            "endpoint_source": self.endpoint_source,
            "market_model": self.market_model,
            "dividend_type": self.dividend_type,
            "key": self.key,
            "is_empty": self.is_empty(),
        }

    def __repr__(self) -> str:
        return super().__repr__()


@dataclass
class ModelResultPack(Result):
    """
    A container for various model result types.
    """

    ## Main Results
    spot: Optional[SpotResult] = None
    forward: Optional[ForwardResult] = None
    dividend: Optional[DividendsResult] = None
    rates: Optional[RatesResult] = None
    option_spot: Optional[OptionSpotResult] = None
    vol: Optional[VolatilityResult] = None

    ## Guiding Enums
    series_id: Optional[SeriesId] = None
    dividend_type: Optional[DivType] = None
    undo_adjust: bool = True
    endpoint_source: Optional[OptionSpotEndpointSource] = None

    ## Diagnostic Info
    time_to_load: Optional[Dict[str, float]] = None

    def _additional_repr_fields(self):
        """Provides model-specific fields for string representation."""
        return {
            "series_id": self.series_id,
            "dividend_type": self.dividend_type,
            "undo_adjust": self.undo_adjust,
            "num_empty": sum(
                1
                for result in [
                    self.spot,
                    self.forward,
                    self.dividend,
                    self.rates,
                    self.option_spot,
                    self.vol,
                ]
                if result is None or result.is_empty()
            ),
        }

    def __repr__(self) -> str:
        return super().__repr__()
