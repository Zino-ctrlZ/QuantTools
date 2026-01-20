from trade.helpers.Logging import setup_logger
from pydantic.dataclasses import dataclass as pydantic_dataclass
from pydantic import ConfigDict, Field
from typing import ClassVar
from trade.helpers.helper_types import validate_inputs
from weakref import WeakSet
from EventDriven.exceptions import (
    BacktestConfigAttributeError
)

from EventDriven.configs.vars import get_class_config_descriptions, get_config_class_description


logger = setup_logger(__name__, stream_log_level="WARNING")



@pydantic_dataclass(config=ConfigDict(arbitrary_types_allowed=True), kw_only=True)
class BaseConfigs:
    """Base configuration class for all modules."""

    _registry: ClassVar[WeakSet[type]] = WeakSet()
    run_name: str = Field(default="", description="A name identifier for this run/session.")
    
    def set(self, **kwargs):
        """Set multiple configuration attributes at once."""
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise BacktestConfigAttributeError(f"Configuration has no attribute named '{key}'.")
            setattr(self, key, value)

    def get(self, key: str):
        """Get a configuration attribute by name."""
        if not hasattr(self, key):
            raise BacktestConfigAttributeError(f"Configuration has no attribute named '{key}'.")
        return getattr(self, key)
    
    def __post_init__(self, ctx=None):
        pass

    def validate_inputs(self):
        """Validate configuration inputs based on type hints."""
        validate_inputs(self)


    def __setattr__(self, name, value):
        super().__setattr__(name, value)

        ## Validate inputs after setting attribute
        self.validate_inputs()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        BaseConfigs._registry.add(cls)

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
        class_desc = get_config_class_description(self.__class__.__name__)
        if class_desc:
            print(f"\n{class_desc}\n")
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
            class_desc = get_config_class_description(config_cls.__name__)
            print(f"\n{'='*80}")
            print(f"Configuration Class: {config_cls.__name__}")
            if class_desc:
                print(f"Description: {class_desc}")
            else:
                logger.warning(f"No class description found for {config_cls.__name__}")
            print('='*80)
            instance = config_cls()
            instance.display_and_describe_configs()


class _CustomFrozenBaseConfigs(BaseConfigs):
    """Base configuration class for all modules with frozen attributes."""

    def __setattr__(self, name, value):
        allow_name_changes = ["run_name"]
        if name in allow_name_changes:
            logger.warning(f"Attempting to set attribute '{name}' to '{value}' in {self.__class__.__name__}...")
            super().__setattr__(name, value)
            return
        if name in self.__dict__:
            raise AttributeError(f"Cannot modify frozen attribute '{name}' in {self.__class__.__name__}. If you need to change it within a class, create a new instance.")
        super().__setattr__(name, value)
