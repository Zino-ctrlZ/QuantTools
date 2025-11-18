from trade.helpers.Logging import setup_logger
from pydantic.dataclasses import dataclass as pydantic_dataclass
from typing import ClassVar, Literal
from weakref import WeakSet
from typing import get_origin, get_args, Union, get_type_hints
from EventDriven.exceptions import BacktesterIncorrectTypeError
import types
from dataclasses import fields
from EventDriven.configs.vars import get_class_config_descriptions


logger = setup_logger(__name__, stream_log_level="DEBUG")

def validate_inputs(self):
    type_hints = get_type_hints(type(self))

    for f in fields(self):
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
                raise BacktesterIncorrectTypeError(
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
                raise BacktesterIncorrectTypeError(
                    f"Configuration '{field_name}' expected types {valid_types}, " f"but got {type(field_value)}."
                )
            continue

        # --- Simple (non-generic) types ---
        if origin is None:
            if field_value is None:
                logger.warning(f"Configuration '{field_name}' is not set (None). Please review.")
                continue

            if not isinstance(field_value, type_hint):
                raise BacktesterIncorrectTypeError(
                    f"Configuration '{field_name}' expected type {type_hint}, " f"but got {type(field_value)}."
                )
            continue

        # --- Other generics (List, Dict, etc.) – shallow check ---
        if field_value is None:
            logger.warning(f"Configuration '{field_name}' is not set (None). Please review.")
            continue

        try:
            if not isinstance(field_value, origin):
                raise BacktesterIncorrectTypeError(
                    f"Configuration '{field_name}' expected type {origin}, " f"but got {type(field_value)}."
                )
        except TypeError:
            logger.warning(
                f"Could not validate field '{field_name}' with value '{field_value}' against type '{type_hint}' due to TypeError."
            )
            pass


@pydantic_dataclass
class BaseConfigs:
    """Base configuration class for all modules."""

    _registry: ClassVar[WeakSet[type]] = WeakSet()

    def __post_init__(self, ctx=None):
        pass

    def validate_inputs(self):
        """Validate configuration inputs based on type hints."""
        validate_inputs(self)


    def __setattr__(self, name, value):
        super().__setattr__(name, value)

        ## Validate inputs after setting attribute
        self.validate_inputs()

    # def __init_subclass__(cls, **kwargs):
    #     super().__init_subclass__(**kwargs)
    #     BaseConfigs._registry.add(cls)

    def display_configs(self):
        """Display the current configuration settings."""
        msg = f"""
Current Configuration Settings for {self.__class__.__name__}:
{self.__dict__}
        """
        return msg

    def describe_configs(self):
        """Describe the configuration settings with explanations."""

        # class_descriptions = get_config_for_class_description(self.__class__.__name__, None)
        class_name = self.__class__.__name__
        class_descriptions = get_class_config_descriptions(class_name)
        if not class_descriptions:
            logger.warning(f"No configuration descriptions found for {self.__class__.__name__}.")
            return

        header = f"""
Configuration Descriptions for {self.__class__.__name__}:
"""

        msg = ""
        for key, value in self.__dict__.items():
            desc = class_descriptions.get(key)
            if desc:
                msg += f"- {key}: {value}  # {desc}\n"
            else:
                logger.warning(f"No description found for config '{key}' in {self.__class__.__name__}.")
        return header + msg
    
    def display_and_describe_configs(self):
        """Display and describe the configuration settings."""
        msg1 = self.display_configs()
        msg2 = self.describe_configs()
        print(msg1)
        print(msg2)

    @classmethod
    def get_all_configs(cls):
        """Get all configuration classes and their instances."""
        configs = {}
        for config_cls in cls._registry:
            configs[config_cls.__name__] = config_cls()
        return configs
    
    @classmethod
    def get_config_instance(cls, class_name: str):
        """Get a specific configuration class instance by name."""
        for config_cls in cls._registry:
            if config_cls.__name__ == class_name:
                return config_cls()
        logger.warning(f"Configuration class '{class_name}' not found.")
        return None
    
    @classmethod
    def list_config_classes(cls):
        """List all registered configuration class names."""
        return [config_cls.__name__ for config_cls in cls._registry]
    
    @classmethod
    def is_config_registered(cls, class_name: str) -> bool:
        """Check if a configuration class is registered."""
        return any(config_cls.__name__ == class_name for config_cls in cls._registry)
    
    @classmethod
    def display_and_describe_all_configs(cls):
        """Display and describe all registered configuration classes."""
        if not cls._registry:
            print("No configuration classes registered.")
            return
        for config_cls in cls._registry:
            print(f"\n=== Configuration Class: {config_cls.__name__} ===")
            instance = config_cls()
            instance.display_and_describe_configs()


class _CustomFrozenBaseConfigs(BaseConfigs):
    """Base configuration class for all modules with frozen attributes."""

    def __setattr__(self, name, value):
        logger.warning(f"Attempting to set attribute '{name}' to '{value}' in {self.__class__.__name__}...")
        if name in self.__dict__:
            raise AttributeError(f"Cannot modify frozen attribute '{name}' in {self.__class__.__name__}.")
        super().__setattr__(name, value)
