# pylint: disable=global-statement
"""
Author: Chiemelie Nwanisobi
Date: 2025-10-01

Title: SSVI Controller Module
Manages caching, background fitting, and configuration hashing for the SSVI model.
This module provides functionalities to handle parameter caching, chain data caching,
and background fitting tasks for the SSVI model in a thread-safe manner.

Ultimately, it provides centralized management of resources and tasks related to the SSVI model. This includes:
- Caching of model parameters and option chain data to optimize performance.
- A thread pool for executing background fitting tasks with controlled concurrency.
- Utilities for hashing configuration dictionaries to detect changes.
- SSVIGlobalConfig: A singleton class to manage global configuration settings for the SSVI model.
"""

import json
import hashlib
from functools import lru_cache
from pathlib import Path
import os
from typing import Dict
from trade.helpers.Logging import setup_logger
from trade.helpers.helper import CustomCache
from trade import get_pricing_config
from trade.optionlib.config.ssvi.global_config import SSVIGlobalConfig
from trade.optionlib.vol.ssvi.threading import BackgroundFits

logger = setup_logger("optionlib.ssvi")
params_dump_path = Path(os.environ["GEN_CACHE_PATH"]) / "optionlib" / "params_dump"
chain_dump_path = Path(os.environ["GEN_CACHE_PATH"]) / "optionlib" / "chain_dumps"

PARAMS_DUMP_CACHE = CustomCache(location=params_dump_path, expire_days=300, clear_on_exit=False, fname="prod")


CHAIN_DUMP_CACHE = CustomCache(location=chain_dump_path, expire_days=300, clear_on_exit=False, fname="prod")

GLOBAL_BACKGROUND_FITS = None
GLOBAL_CONFIG = None


def set_global_config(config: SSVIGlobalConfig):
    """
    Set the global SSVI configuration.
    Args:
        config (SSVIGlobalConfig): The configuration to set.
    Raises:
        ValueError: If the provided config is not an instance of SSVIGlobalConfig.
    """
    if not isinstance(config, SSVIGlobalConfig):
        raise ValueError("Config must be an instance of SSVIGlobalConfig")

    global GLOBAL_CONFIG
    GLOBAL_CONFIG = config


def get_global_config() -> SSVIGlobalConfig:
    """
    Get the global SSVI configuration.
    Returns:
        SSVIGlobalConfig: The current global configuration.
    """
    global GLOBAL_CONFIG
    if GLOBAL_CONFIG is None:
        GLOBAL_CONFIG = SSVIGlobalConfig()  # Default configuration
    return GLOBAL_CONFIG


def hash_config(config_dict: dict) -> str:
    """
    Returns a SHA256 hash of the given configuration dictionary.
    """
    json_bytes = json.dumps(config_dict, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(json_bytes).hexdigest()


def is_latest_config(stored_hash: str) -> bool:
    """
    Check if the stored hash matches the current configuration hash.
    """
    current_hash = hash_config(get_pricing_config())
    return stored_hash == current_hash


def _get_caches() -> Dict[str, CustomCache]:
    """
    Get the parameter and chain caches.
    Returns a dictionary with 'params' and 'chain' keys.
    """
    return {"params": get_params_cache(), "chain": get_chain_cache()}


@lru_cache(maxsize=1)
def get_params_cache() -> CustomCache:
    """
    Get the parameters cache.
    """
    return PARAMS_DUMP_CACHE


@lru_cache(maxsize=1)
def get_chain_cache() -> CustomCache:
    """
    Get the chain cache.
    """
    return CHAIN_DUMP_CACHE


@lru_cache(maxsize=1)
def get_background_fits() -> BackgroundFits:
    """
    Get the global BackgroundFits instance, creating it if necessary.
    """
    global GLOBAL_BACKGROUND_FITS
    if GLOBAL_BACKGROUND_FITS is None:
        GLOBAL_BACKGROUND_FITS = BackgroundFits(max_workers=3, max_queue=1_000)
    return GLOBAL_BACKGROUND_FITS
