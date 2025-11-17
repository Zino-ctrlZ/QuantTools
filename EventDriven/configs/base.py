from trade.helpers.Logging import setup_logger
from pydantic.dataclasses import dataclass as pydantic_dataclass
from typing import ClassVar
from weakref import WeakSet
from typing import get_origin, get_args, Union
from EventDriven.exceptions import BacktesterIncorrectTypeError
from EventDriven.configs.vars import get_class_config_descriptions


logger = setup_logger(__name__, stream_log_level="DEBUG")


@pydantic_dataclass
class BaseConfigs:
    """Base configuration class for all modules."""

    _registry: ClassVar[WeakSet[type]] = WeakSet()

    def __post_init__(self, ctx=None):
        pass

    def validate_inputs(self):
        for field_name, field_value in self.__dict__.items():
            if field_value is None:
                logger.warning(f"Configuration '{field_name}' is not set (None). Please review.")

            type_hint = self.__annotations__.get(field_name)
            origin = get_origin(type_hint)
            args = get_args(type_hint)

            ## No origin, simple type
            if origin is None:
                if not isinstance(field_value, type_hint) and field_value is not None:
                    raise BacktesterIncorrectTypeError(
                        f"Configuration '{field_name}' expected type {type_hint}, but got {type(field_value)}."
                    )
                continue

            ## Handle Union types
            if isinstance(origin, Union.__class__):
                valid_types = tuple(arg for arg in args if arg is not type(None))
                if not isinstance(field_value, valid_types) and field_value is not None:
                    raise BacktesterIncorrectTypeError(
                        f"Configuration '{field_name}' expected types {valid_types}, but got {type(field_value)}."
                    )
                continue

            ## Handle other generic types (e.g., List, Dict)
            if not isinstance(field_value, origin) and field_value is not None:
                raise BacktesterIncorrectTypeError(
                    f"Configuration '{field_name}' expected type {origin}, but got {type(field_value)}."
                )

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
