from typing import Iterable, TypedDict, Any
from enum import Enum
from datetime import datetime
import numbers
from abc import ABC, abstractmethod
from typing import ClassVar
from weakref import WeakSet
from trade.helpers.exception import SymbolChangeError
from typing import Union, get_type_hints, Type, Dict, get_origin, get_args
from trade.helpers.Logging import setup_logger
from dataclasses import dataclass, fields
from functools import lru_cache
from typeguard import check_type
from typeguard._exceptions import TypeCheckError

logger = setup_logger(__name__)
DATE_HINT = Union[datetime, str]


@lru_cache(maxsize=None)
def _hints(cls: Type[Any]) -> Dict[str, Any]:
    return get_type_hints(cls)


def _hint_allows_numbers_number(hint: Any) -> bool:
    """Return True when a type hint includes numbers.Number."""
    if hint is numbers.Number:
        return True

    origin = get_origin(hint)
    if origin is None:
        return False

    return any(_hint_allows_numbers_number(arg) for arg in get_args(hint))


class TypeValidatedMixin:
    """Backward-compatible alias mixin for dataclass type validation.

    Prefer `DataclassTypeValidationMixin` for new code. This class remains to
    avoid breaking existing imports.
    """

    def _validate_field(self, name: str, value: Any) -> None:
        hint = _hints(type(self)).get(name)
        if hint is not None:
            # bool is a subclass of int; reject it explicitly for numeric hints.
            if isinstance(value, bool) and _hint_allows_numbers_number(hint):
                raise IncorrectTypeError(
                    f"Field '{name}' in {type(self).__name__} expected a numeric value, but got boolean {value!r}."
                )
            try:
                check_type(value, hint)
            except TypeCheckError as e:
                raise IncorrectTypeError(
                    f"Field '{name}' in {type(self).__name__} expected type {hint}, but got value {value!r} of type {type(value)}. Original error: {e}"
                ) from e

    def _validate_all_fields(self) -> None:
        for f in fields(self):
            self._validate_field(f.name, getattr(self, f.name))

    def __post_init__(self) -> None:
        self._validate_all_fields()


class DataclassTypeValidationMixin(TypeValidatedMixin):
    """Validate annotated dataclass fields after initialization.

    This mixin is designed for both stdlib dataclasses and pydantic dataclasses.
    It performs full-object validation in `__post_init__` using type annotations.

    Usage:
        - Mutable models should combine this with
          `DataclassAssignmentValidationMixin`.
        - Frozen models should use this mixin (or `FrozenTypeValidationMixin`) and
          rely on the dataclass decorator's `frozen=True` behavior.
    """


class DataclassAssignmentValidationMixin(DataclassTypeValidationMixin):
    """Add post-init assignment validation for mutable dataclasses.

    A constructor gate is used so assignment checks only run after initialization
    has completed. This avoids false positives while dataclass/pydantic populates
    fields during construction.
    """

    _validation_ready: bool = False

    def __post_init__(self, *args: Any, **kwargs: Any) -> None:
        super().__post_init__()
        object.__setattr__(self, "_validation_ready", True)

    def __setattr__(self, name: str, value: Any) -> None:
        ready = getattr(self, "_validation_ready", False)
        if ready and name in getattr(self, "__dict__", {}):
            self._validate_field(name, value)
        super().__setattr__(name, value)


class FrozenTypeValidationMixin(DataclassTypeValidationMixin):
    """Marker mixin for frozen dataclasses.

    This class intentionally adds no assignment behavior. Immutability is
    provided by `@dataclass(frozen=True)` or pydantic dataclass `frozen=True`.
    """

    pass


@dataclass
class MutableValidated(DataclassAssignmentValidationMixin):
    """Legacy mutable validated dataclass kept for compatibility."""

    pass


@dataclass(frozen=True)
class FrozenValidated(FrozenTypeValidationMixin):
    """Legacy frozen validated dataclass kept for compatibility."""

    pass


class IncorrectTypeError(Exception):
    """Custom exception for incorrect type errors in configuration validation."""

    pass


class OptionTickMetaData(TypedDict):
    ticker: str
    exp_date: str
    put_call: str
    strike: float


class PositionData(TypedDict):
    long: list[str]
    short: list[str]


class OptionModelAttributes(Enum):
    S0 = "unadjusted_S0"
    K = "K"
    exp_date = "exp"
    sigma = "sigma"
    y = "y"
    put_call = "put_call"
    r = "rf_rate"
    start = "end_date"
    spot_type = "chain_price"


class TickerMap(dict):
    invalid_tickers = {"FB": "META"}

    def __getitem__(self, key):
        if key in self.invalid_tickers:
            raise SymbolChangeError(
                f"Tick name changed from {key} to {self.invalid_tickers[key]}, access the new tick instead"
            )
        return super().__getitem__(key)


class SingletonMixin(ABC):
    """
    A mixin class to make a class a singleton by symbol.
    Still a work in progress.
    """

    _registry: ClassVar[WeakSet[type]] = WeakSet()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        SingletonMixin._registry.add(cls)

    @classmethod
    @abstractmethod
    def clear_instances(cls):
        pass

    @classmethod
    @abstractmethod
    def instances(cls):
        pass

    @classmethod
    def clear_all_instances(cls) -> None:
        for sub in list(cls._registry):
            if issubclass(sub, cls):
                try:
                    sub.clear_instances()
                except TypeError:
                    pass


class SingletonMetaClass(type):
    """
    A metaclass for singleton classes.
    It ensures that only one instance of a class is created.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


def is_iterable(obj: Any, include_str: bool = False) -> bool:
    """Check if an object is iterable, optionally excluding strings."""
    if include_str:
        return isinstance(obj, Iterable)
    else:
        return isinstance(obj, Iterable) and not isinstance(obj, (str, bytes))
