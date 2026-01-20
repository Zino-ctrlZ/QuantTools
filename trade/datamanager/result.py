import pandas as pd
from dataclasses import dataclass
from typing import Any, Dict, Optional
from trade.optionlib.config.types import DivType
from trade.helpers.helper_types import validate_inputs
from ._enums import OptionSpotEndpointSource


@dataclass
class Result:
    """Base class for all data manager result containers."""

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
        super().__setattr__(name, value)
        validate_inputs(self, raise_on_fail=True)

@dataclass
class DividendsResult(Result):
    """Contains dividend schedule or yield data for a date range."""

    daily_discrete_dividends: Optional[pd.Series] = None
    daily_continuous_dividends: Optional[pd.Series] = None
    dividend_type: Optional[DivType] = None
    key: Optional[str] = None
    undo_adjust: Optional[bool] = None

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
            "dividend_type": self.dividend_type,
            "key": self.key,
            "is_empty": self.is_empty(),
            "undo_adjust": self.undo_adjust,
        }


@dataclass
class RatesResult(Result):
    """Contains risk-free rate data for a date range."""

    daily_risk_free_rates: Optional[pd.Series] = None

    def is_empty(self) -> bool:
        """Checks if rate data is missing or empty."""
        return self.daily_risk_free_rates is None or self.daily_risk_free_rates.empty

    def _additional_repr_fields(self):
        """Provides rate-specific fields for string representation."""
        return {
            "is_empty": self.is_empty(),
        }

    def __repr__(self) -> str:
        return super().__repr__()


@dataclass
class ForwardResult(Result):
    """Contains forward price data (discrete or continuous dividend model)."""

    daily_discrete_forward: Optional[pd.Series] = None
    daily_continuous_forward: Optional[pd.Series] = None
    dividend_type: Optional[DivType] = None
    key: Optional[str] = None
    dividend_result: Optional[DividendsResult] = None
    undo_adjust: Optional[bool] = True

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
            "is_empty": self.is_empty(),
            "undo_adjust": self.undo_adjust,
            "dividend_type": self.dividend_type,
            "key": self.key,
        }

    def __repr__(self) -> str:
        return super().__repr__()


@dataclass
class SpotResult(Result):
    """Contains spot price data with optional split adjustment information."""

    daily_spot: Optional[pd.Series] = None
    undo_adjust: Optional[bool] = None
    key: Optional[str] = None

    def is_empty(self) -> bool:
        return self.daily_spot is None or self.daily_spot.empty

    def _additional_repr_fields(self) -> Dict[str, Any]:
        """Provides spot-specific fields for string representation."""
        return {
            "key": self.key,
            "is_empty": self.is_empty(),
            "undo_adjust": self.undo_adjust,
        }

    def __repr__(self) -> str:
        return super().__repr__()


@dataclass
class OptionSpotResult(Result):
    """Container for option spot price timeseries data."""

    daily_option_spot: Optional[pd.DataFrame] = None
    key: Optional[str] = None
    endpoint_source: Optional[OptionSpotEndpointSource] = None

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
            "key": self.key,
            "is_empty": self.is_empty(),
            "endpoint_source": self.endpoint_source,
        }

    def __repr__(self) -> str:
        """Delegates to base Result repr."""
        return super().__repr__()
