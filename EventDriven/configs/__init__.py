
import pkgutil
import importlib
import pathlib

# Get package path
_pkg_path = pathlib.Path(__file__).parent

# Auto-import all modules inside this package
for module in pkgutil.iter_modules([str(_pkg_path)]):
    importlib.import_module(f"{__name__}.{module.name}")

# Export BaseConfigs for easy access
from .base import BaseConfigs # noqa

__all__ = ["BaseConfigs"]