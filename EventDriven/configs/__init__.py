
import pkgutil
import importlib
import pathlib

# Get package path
_pkg_path = pathlib.Path(__file__).parent

# Auto-import all modules inside this package
for module in pkgutil.iter_modules([str(_pkg_path)]):
    importlib.import_module(f"{__name__}.{module.name}")

# Export key classes and functions for easy access
from .base import BaseConfigs # noqa
from .export_configs import ( # noqa
    RunConfigBundle,
    ConfigLocation,
    collect_run_configs,
    export_run_configs,
    apply_run_configs,
    validate_config_placement,
    apply_and_validate_configs,
    tag_run,
    walk_configs,
)

__all__ = [
    "BaseConfigs",
    "RunConfigBundle",
    "ConfigLocation",
    "collect_run_configs",
    "export_run_configs",
    "apply_run_configs",
    "validate_config_placement",
    "apply_and_validate_configs",
    "tag_run",
    "walk_configs",
]