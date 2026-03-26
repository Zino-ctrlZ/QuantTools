
from typing import Iterable, TypedDict, Any
from enum import Enum
from datetime import datetime
from abc import ABC, abstractmethod
from typing import ClassVar
from weakref import WeakSet
from trade.helpers.exception import SymbolChangeError
from typing import get_origin, get_args, Union, get_type_hints, Literal, Type, Dict
import types
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


class TypeValidatedMixin:
    def _validate_field(self, name: str, value: Any) -> None:
        hint = _hints(type(self)).get(name)
        if hint is not None:
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


@dataclass
class MutableValidated(TypeValidatedMixin):
    def __setattr__(self, name: str, value: Any) -> None:
        self._validate_field(name, value)
        super().__setattr__(name, value)


@dataclass(frozen=True)
class FrozenValidated(TypeValidatedMixin):
    pass


# frozen update pattern:
# new_obj = replace(old_obj, some_field=new_value)



class IncorrectTypeError(Exception):
    """Custom exception for incorrect type errors in configuration validation."""

    pass


def validate_inputs(self: object, raise_on_fail: bool = False) -> None:
    type_hints = get_type_hints(type(self))

    for f in fields(self):
        try:
            field_name = f.name
            field_value = getattr(self, field_name)

            type_hint = type_hints.get(field_name)
            if type_hint is None:
                continue  # no annotation, skip

            origin = get_origin(type_hint)
            args = get_args(type_hint)

            # --- Handle Literal[...] ---
            if origin is Literal:
                # e.g. name: Literal["LimitsCog", "OtherCog"]
                allowed_values = args  # tuple of literals

                if field_value is None:
                    # If you want to allow None here, add it to the Literal.
                    logger.warning(f"Configuration '{field_name}' is None but expected one of {allowed_values}.")
                elif field_value not in allowed_values:
                    raise IncorrectTypeError(
                        f"Configuration '{field_name}' expected one of {allowed_values}, " f"but got {field_value!r}."
                    )
                continue

            # --- Handle Optional / Union[...] ---
            if origin in (Union, types.UnionType):
                allows_none = any(arg is type(None) for arg in args)
                if field_value is None:
                    if not allows_none:
                        logger.warning(
                            f"Configuration '{field_name}' is not set (None) and is not Optional. Please review."
                        )
                    continue

                valid_types = tuple(arg for arg in args if arg is not type(None))
                if not isinstance(field_value, valid_types):
                    raise IncorrectTypeError(
                        f"Configuration '{field_name}' expected types {valid_types}, " f"but got {type(field_value)}."
                    )
                continue

            # --- Simple (non-generic) types ---
            if origin is None:
                if field_value is None:
                    logger.warning(f"Configuration '{field_name}' is not set (None). Please review.")
                    continue

                if not isinstance(field_value, type_hint):
                    raise IncorrectTypeError(
                        f"Configuration '{field_name}' expected type {type_hint}, " f"but got {type(field_value)}."
                    )
                continue

            # --- Other generics (List, Dict, etc.) – shallow check ---
            if field_value is None:
                logger.warning(f"Configuration '{field_name}' is not set (None). Please review.")
                continue

            try:
                if not isinstance(field_value, origin):
                    raise IncorrectTypeError(
                        f"Configuration '{field_name}' expected type {origin}, " f"but got {type(field_value)}."
                    )
            except TypeError:
                logger.warning(
                    f"Could not validate field '{field_name}' with value '{field_value}' against type '{type_hint}' due to TypeError."
                )
                pass

        except Exception as e:
            logger.critical(f"Failed to validate field '{f.name}' in {self.__class__.__name__}. Error: {e}")
            if raise_on_fail:
                raise e


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