import pandas as pd
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List, Union
from trade.optionlib.config.types import DivType
from ._enums import (
    GreekType, 
    OptionSpotEndpointSource, 
    OptionPricingModel, 
    VolatilityModel, 
    SeriesId, 
    ModelPrice, 
    RealTimeFallbackOption,
    AVAILABLE_GREEKS
)
from .utils.date import DATE_HINT
from typeguard import check_type
from trade.helpers.helper import to_datetime
from typing import get_type_hints


@dataclass
class Result:
    """Base class for all data manager result containers."""

    model_input_keys: Optional[Dict[str, Any]] = None
    rt: Optional[bool] = False
    fallback_option: Optional[RealTimeFallbackOption] = None
    timeseries: Optional[Union[pd.Series, pd.DataFrame]] = None

    def __post_init__(self):
        """Simple formatting"""
        timeseries = getattr(self, "timeseries", None)
        if timeseries is not None:
            if isinstance(timeseries, (pd.Series, pd.DataFrame)):
                timeseries.index.name = "datetime"
                timeseries.index = to_datetime(timeseries.index)

    def _additional_repr_fields(self) -> Dict[str, Any]:
        """Provides additional fields for string representation. Override in subclasses."""
        return {}

    def is_empty(self) -> bool:
        """Checks if the result container has no data. Override in subclasses if needed."""
        raise NotImplementedError("is_empty method must be implemented in subclasses.")

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
    
    def __setattr__(self, name, value):
        
        ## Intercept dataframe/series, and add name attribute if missing. Only add name for series.
        ## Not ideal to do it here, but easier than finding all places where timeseries is set.
        if name == "timeseries" and value is not None:
            if isinstance(value, (pd.Series, pd.DataFrame)):
                value.index.name = "datetime"
                if isinstance(value, pd.Series):
                    if value.name is None:
                        value.name = "dividends"
        
        super().__setattr__(name, value)

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

    def __setattr__(self, name, value):
        ## Intercept dataframe/series, and add name attribute if missing. Only add name for series.
        ## Not ideal to do it here, but easier than finding all places where timeseries is set.
        if name == "timeseries" and value is not None:
            if isinstance(value, (pd.Series, pd.DataFrame)):
                value.index.name = "datetime"
                if isinstance(value, pd.Series):
                    if value.name is None:
                        value.name = "r"

        super().__setattr__(name, value)


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

    def __setattr__(self, name, value):
        ## Intercept dataframe/series, and add name attribute if missing. Only add name for series.
        ## Not ideal to do it here, but easier than finding all places where timeseries is set.
        if name == "timeseries" and value is not None:
            if isinstance(value, (pd.Series, pd.DataFrame)):
                value.index.name = "datetime"
                if isinstance(value, pd.Series):
                    if value.name is None:
                        value.name = "forward"

        super().__setattr__(name, value)


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

    def __setattr__(self, name, value):
        ## Intercept dataframe/series, and add name attribute if missing. Only add name for series.
        ## Not ideal to do it here, but easier than finding all places where timeseries is set.
        if name == "timeseries" and value is not None:
            if isinstance(value, (pd.Series, pd.DataFrame)):
                value.index.name = "datetime"
                if isinstance(value, pd.Series):
                    if value.name is None:
                        value.name = "spot" if self.undo_adjust else "spot_unadjusted"

        super().__setattr__(name, value)


@dataclass
class _OptionResultsBase(Result):
    """Base class for option-related result containers."""

    symbol: Optional[str] = None
    strike: Optional[float] = None
    expiration: Optional[DATE_HINT] = None
    right: Optional[str] = None
    model_price: Optional[ModelPrice] = ModelPrice.MIDPOINT

    def _additional_repr_fields(self) -> Dict[str, Any]:
        """Provides option-specific fields for string representation."""
        return {
            "symbol": self.symbol,
            "strike": self.strike,
            "expiration": self.expiration,
            "right": self.right,
            "model_price": self.model_price,
        }

    def __repr__(self) -> str:
        """Delegates to base Result repr."""
        return super().__repr__()
    
@dataclass
class _OptionModelResultsBase(_OptionResultsBase):
    """Base class for option model result containers."""
    endpoint_source: Optional[OptionSpotEndpointSource] = None
    market_model: Optional[OptionPricingModel] = None
    vol_model: Optional[VolatilityModel] = None
    dividend_type: Optional[DivType] = None
    undo_adjust: Optional[bool] = None

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
    def price(self) -> pd.Series:
        if self.rt:
            return self.midpoint
        
        if not self.is_empty():
            if self.model_price == ModelPrice.CLOSE:
                p = self.daily_option_spot.get("close")
            elif self.model_price == ModelPrice.MIDPOINT:
                p = self.daily_option_spot.get("midpoint")
            elif self.model_price == ModelPrice.BID:
                p = self.daily_option_spot.get("closebid")
            elif self.model_price == ModelPrice.ASK:
                p = self.daily_option_spot.get("closeask")
            elif self.model_price == ModelPrice.OPEN:
                p = self.daily_option_spot.get("open")
            else:
                p = self.daily_option_spot.get("midpoint")
        else:
            return pd.Series(name="price", index=pd.DatetimeIndex([]), dtype=float)
        
        if p is None:
            raise ValueError(f"Requested model price '{self.model_price}' not found in option spot data. Available columns: {self.daily_option_spot.columns.tolist()}")
        return p

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

    def __setattr__(self, name, value):
        ## Intercept dataframe/series, and add name attribute if missing. Only add name for series.
        ## Not ideal to do it here, but easier than finding all places where timeseries is set.
        if name == "timeseries" and value is not None:
            if isinstance(value, (pd.Series, pd.DataFrame)):
                value.index.name = "datetime"
                if isinstance(value, pd.Series):
                    if value.name is None:
                        value.name = "option_spot"

        super().__setattr__(name, value)


@dataclass
class VolatilityResult(_OptionModelResultsBase):
    """Contains volatility surface data."""

    timeseries: Optional[pd.Series] = None
    key: Optional[str] = None
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

    def __setattr__(self, name, value):
        ## Intercept dataframe/series, and add name attribute if missing. Only add name for series.
        ## Not ideal to do it here, but easier than finding all places where timeseries is set.
        if name == "timeseries" and value is not None:
            if isinstance(value, (pd.Series, pd.DataFrame)):
                value.index.name = "datetime"
                if isinstance(value, pd.Series):
                    if value.name is None:
                        value.name = "iv"

        super().__setattr__(name, value)


@dataclass
class GreekResultSet(_OptionModelResultsBase):
    key: Optional[str] = None
    timeseries: Optional[pd.DataFrame] = None

    def is_empty(self) -> bool:
        return self.timeseries is None or self.timeseries.empty
    
    def _additional_repr_fields(self) -> Dict[str, Any]:
        super_additional = super()._additional_repr_fields()
        return {
            **super_additional,
            "Available Greeks": [g for g in AVAILABLE_GREEKS if self.timeseries is not None and g in self.timeseries.columns],
            "empty": self.is_empty(),
        }

    def __repr__(self):
        return super().__repr__()

    @property
    def delta(self) -> Optional[pd.Series]:
        if self.timeseries is not None and GreekType.DELTA.value in self.timeseries.columns:
            return self.timeseries[GreekType.DELTA.value]
        return None
    
    @property
    def gamma(self) -> Optional[pd.Series]:
        if self.timeseries is not None and GreekType.GAMMA.value in self.timeseries.columns:
            return self.timeseries[GreekType.GAMMA.value]
        return None
    
    @property
    def theta(self) -> Optional[pd.Series]:
        if self.timeseries is not None and GreekType.THETA.value in self.timeseries.columns:
            return self.timeseries[GreekType.THETA.value]
        return None
    
    @property
    def vega(self) -> Optional[pd.Series]:
        if self.timeseries is not None and GreekType.VEGA.value in self.timeseries.columns:
            return self.timeseries[GreekType.VEGA.value]
        return None
    
    @property
    def rho(self) -> Optional[pd.Series]:
        if self.timeseries is not None and GreekType.RHO.value in self.timeseries.columns:
            return self.timeseries[GreekType.RHO.value]
        return None
    
    @property
    def volga(self) -> Optional[pd.Series]:
        if self.timeseries is not None and GreekType.VOLGA.value in self.timeseries.columns:
            return self.timeseries[GreekType.VOLGA.value]
        return None

    def __setattr__(self, name, value):
        ## Intercept dataframe/series, and add name attribute if missing. Only add name for series.
        ## Not ideal to do it here, but easier than finding all places where timeseries is set.
        if name == "timeseries" and value is not None:
            if isinstance(value, (pd.Series, pd.DataFrame)):
                value.index.name = "datetime"
                if isinstance(value, pd.Series):
                    if value.name is None:
                        value.name = "greeks"

        super().__setattr__(name, value)


@dataclass
class TheoreticalPriceResult(_OptionModelResultsBase):
    timeseries: Optional[pd.Series] = None

    def is_empty(self) -> bool:
        return self.timeseries is None or self.timeseries.empty

    def __repr__(self) -> str:
        return super().__repr__()

    def __setattr__(self, name, value):
        ## Intercept dataframe/series, and add name attribute if missing. Only add name for series.
        ## Not ideal to do it here, but easier than finding all places where timeseries is set.
        if name == "timeseries" and value is not None:
            if isinstance(value, (pd.Series, pd.DataFrame)):
                value.index.name = "datetime"
                if isinstance(value, pd.Series):
                    if value.name is None:
                        value.name = "theoretical_price"

        super().__setattr__(name, value)
    


@dataclass
class ScenariosResult(_OptionModelResultsBase):
    grid: Optional[pd.DataFrame] = None
    spot_scenarios: List[float] = field(default_factory=lambda: [])
    vol_scenarios: List[float] = field(default_factory=lambda: [])
    as_of: Optional[DATE_HINT] = None

    def is_empty(self) -> bool:
        return self.grid is None or self.grid.empty

    def _additional_repr_fields(self):
        return {
            "symbol": self.symbol,
            "expiration": self.expiration,
            "right": self.right,
            "strike": self.strike,
            "market_model": self.market_model,
            "dividend_type": self.dividend_type,
            "num_spot_scenarios": len(self.spot_scenarios),
            "num_vol_scenarios": len(self.vol_scenarios),
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
    greek: Optional[GreekResultSet] = None

    ## Guiding Enums
    series_id: Optional[SeriesId] = None
    dividend_type: Optional[DivType] = None
    undo_adjust: bool = True
    endpoint_source: Optional[OptionSpotEndpointSource] = None
    price: Optional[ModelPrice] = None
    rt: Optional[bool] = False
    on_date: Optional[bool] = False

    ## Diagnostic Info
    time_to_load: Optional[Dict[str, float]] = None

    def _additional_repr_fields(self):
        """Provides model-specific fields for string representation."""
        return {
            "symbol": self.spot.symbol if self.spot else None,
            "strike": self.option_spot.strike if self.option_spot else None,
            "expiration": self.option_spot.expiration if self.option_spot else None,
            "right": self.option_spot.right if self.option_spot else None,
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
                    self.greek,
                ]
                if result is None or result.is_empty()
            ),
        }

    def __repr__(self) -> str:
        return super().__repr__()
    
    def list_all_loaded(self) -> Dict[str, bool]:
        return {
            "spot": self.spot is not None and not self.spot.is_empty(),
            "forward": self.forward is not None and not self.forward.is_empty(),
            "dividend": self.dividend is not None and not self.dividend.is_empty(),
            "rates": self.rates is not None and not self.rates.is_empty(),
            "option_spot": self.option_spot is not None and not self.option_spot.is_empty(),
            "vol": self.vol is not None and not self.vol.is_empty(),
            "greek": self.greek is not None and not self.greek.is_empty(),
        }
    
    def any_loaded(self) -> bool:
        return any(
            [
                self.spot is not None and not self.spot.is_empty(),
                self.forward is not None and not self.forward.is_empty(),
                self.dividend is not None and not self.dividend.is_empty(),
                self.rates is not None and not self.rates.is_empty(),
                self.option_spot is not None and not self.option_spot.is_empty(),
                self.vol is not None and not self.vol.is_empty(),
                self.greek is not None and not self.greek.is_empty(),
            ]
        )
    
    def all_loaded(self) -> bool:
        return all(
            [
                self.spot is not None and not self.spot.is_empty(),
                self.forward is not None and not self.forward.is_empty(),
                self.dividend is not None and not self.dividend.is_empty(),
                self.rates is not None and not self.rates.is_empty(),
                self.option_spot is not None and not self.option_spot.is_empty(),
                self.vol is not None and not self.vol.is_empty(),
                self.greek is not None and not self.greek.is_empty(),
            ]
        )

    # def all_passed_loaded(self, requested: List[str]) -> bool:
    #     mapping = {
    #         "spot": self.load_spot,
    #         "forward": self.load_forward,
    #         "dividend": self.load_dividend,
    #         "rates": self.load_rates,
    #         "option_spot": self.load_option_spot,
    #         "vol": self.load_vol,
    #     }
    #     return all([mapping[req] for req in requested if req in mapping])

