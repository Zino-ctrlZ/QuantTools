# pylint: disable=protected-access
from __future__ import annotations
import time
from typing import ClassVar, Optional, Tuple
from datetime import datetime
import pandas as pd
from pydantic import Field, PrivateAttr, ConfigDict
from trade.helpers.Logging import setup_logger
from trade.helpers.pydantic import loud_post_init
from trade.helpers.helper_types import SingletonMixin
from trade.optionlib.vol.ssvi.model.ssvi_model._parent_ssvi import SSVIParentModel
from trade.optionlib.vol.ssvi.model.chain import ChainOutput, MarketChainLoader
from trade.optionlib.vol.ssvi.model.param_utils import load_ssvi_params_from_cache
from trade.optionlib.vol.ssvi.model.model_utils import params_cache_key
from trade.optionlib.config.types import VolSide
from trade.optionlib.config.ssvi.controller import (
    get_background_fits,
    get_global_config,
    get_pricing_config,
    hash_config,
    get_params_cache,
)

logger = setup_logger("optionlib.ssvi.model.ssvi_model")


class EODMarketSSVIModel(SSVIParentModel, SingletonMixin):
    """
    EODMarketSSVIModel extends SSVIModel to handle end-of-day market data.
    This model is designed to work with end-of-day option chains and provides methods
    to predict implied volatility based on the SSVI model parameters.

    There's a singleton pattern to cache instances based on (symbol, valuation_date).
    Args:
        symbol (str): The underlying asset symbol.
        valuation_date (str|datetime): The date for which the option chain is evaluated.
        chain (Optional[ChainOutput]): The processed option chain data.
        chain_loader (MarketChainLoader): Loader to fetch and process market option chains.
    """

    ## Class variable to cache instances
    _instances: ClassVar[dict[Tuple[str, str], "EODMarketSSVIModel"]] = {}
    _initialized: bool = PrivateAttr(default=False)
    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    symbol: str
    valuation_date: str | datetime
    load_on_init: bool = Field(default=True, description="Whether to load chain on initialization")
    chain: Optional[ChainOutput] = Field(default=None, description="Processed option chain output")
    chain_loader: Optional[MarketChainLoader] = Field(default=None, description="Market chain loader instance")

    @classmethod
    def instances(cls) -> dict[str, "EODMarketSSVIModel"]:
        return cls._instances

    @classmethod
    def clear_instances(cls, clear_tree: bool = False):
        cls._instances.clear()
        if clear_tree:
            MarketChainLoader.clear_instances()
            EODMarketSSVIModel.clear_all_instances()

    @loud_post_init
    def model_post_init(self, _):
        if not self.load_on_init:
            return
        global_conf = get_global_config()
        if self.chain is None:
            if self.chain_loader is None:
                loader = MarketChainLoader(symbol=self.symbol, valuation_date=self.valuation_date)
            else:
                assert isinstance(
                    self.chain_loader, MarketChainLoader
                ), "chain_loader must be a MarketChainLoader instance"
                loader = self.chain_loader

            self.chain_loader = loader

            ## Load chain using loader
            ## Use GLOBAL_CONFIG settings for cache and force_calc
            self.chain = loader.build_chain(force_rebuild=global_conf.force_calc, ignore_cache=global_conf.force_calc)
        else:
            assert isinstance(self.chain, ChainOutput), "chain must be a ChainOutput instance"

            ## Cache chain immediately to get it off memory
            self.chain._cache_chain()

        super().model_post_init(_)

    def __new__(cls, symbol: str, valuation_date: str | datetime, *args, **kwargs):
        key = (symbol, pd.to_datetime(valuation_date).strftime("%Y-%m-%d"))
        if key not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[key] = instance
        else:
            logger.info("Using cached instance for %s on %s", symbol, valuation_date)
        return cls._instances[key]

    def __init__(self, *args, **data):
        # First-time init for this cached instance:
        # If __pydantic_private__ isn't set yet, it's the first real init.
        if getattr(self, "__pydantic_private__", None) is None:
            super().__init__(*args, **data)  # sets fields and creates private store
            self._initialized = True  # safe now

    def __repr__(self):
        return f"<EODMarketSSVIModel(symbol={self.symbol}, valuation_date={self.valuation_date}>"

    def load_chain(self, force_rebuild: bool = False, ignore_cache: bool = False):
        """
        Load or reload the option chain using the chain loader.
        Args:
            force_rebuild (bool): Whether to force rebuild the chain.
            ignore_cache (bool): Whether to ignore any existing cache.
        """
        if self.chain_loader is None:
            self.chain_loader = MarketChainLoader(symbol=self.symbol, valuation_date=self.valuation_date)

        self.chain = self.chain_loader.build_chain(force_rebuild=force_rebuild, ignore_cache=ignore_cache)
        ## Cache chain immediately to get it off memory
        self.chain._cache_chain()

    def load_models_from_cache(self):
        """
        Load SSVI model parameters for other option sides (call, put, otm) from cache if available.
        This method checks the PARAMS_DUMP_CACHE for each option side and loads the parameters
        into the corresponding SSVIModel instance if found and up-to-date.
        """
        for side in self.models.keys():
            cached_params = load_ssvi_params_from_cache(
                root=self.chain.root,
                valuation_date=self.valuation_date,
                div_type=self.div_type,
                vol_type=self.model,
                side=VolSide(side),
            )
            if cached_params is not None:
                logger.info("Loaded cached params for %s model on %s", side, self.chain.key)
                self.models[side].params = cached_params
            else:
                logger.info("No cached params found for %s model on %s", side, self.chain.key)

    def fit(self):
        """
        Fit the SSVI model to the option chain data.
        This method estimates the ATM variance, long-term variance, speed of mean reversion,
        skewness, kurtosis, and correlation parameters using the option chain data.
        It calculates the ATM maturities and implied volatilities, and then uses these to
        estimate the model parameters.
        Note: This method is designed to be called after the model has been initialized. It fits per right chain (call and put).
        After fitting, it saves the model parameters to the global PARAMS_DUMP_CACHE in the background.
        """
        ## Try loading other models from cache first, only if global_conf.force_calc is False
        global_conf = get_global_config()
        global_background_fits = get_background_fits()

        ## Load chain if not already loaded
        if self.chain is None:
            logger.warning("Chain not loaded for %s. Loading now...", self.chain.key)
            self.load_chain(force_rebuild=global_conf.force_calc, ignore_cache=global_conf.force_calc)

        if not global_conf.force_calc:
            self.load_models_from_cache()
            if all(model.params is not None for model in self.models.values()):
                logger.info("All models are already fitted for %s", self.chain.key)
                return

        ## If not all fitted, fit the primary model and others in background
        super().fit()

        ## Save to cache in background, only when global_conf.save_cache is True
        if global_conf.save_cache:
            global_background_fits.submit(fn=self.save_cache, key=f"{self.chain.key}_save_cache")

    def save_cache(self):
        """
        Save the model parameters to the global PARAMS_DUMP_CACHE.
        """

        def _wait_for_params(side, timeout: int = 120, interval: int = 2):
            """Wait for params for a given side until timeout and return the params or None."""
            timer = 0
            params = self.params.get(side, None)
            while params is None and timer < timeout:
                time.sleep(interval)
                timer += interval
                params = self.params.get(side, None)
                if params is None:
                    logger.warning("Parameters for %s on %s still not available. Waiting...", side, self.chain.key)
                else:
                    logger.info("Parameters for %s on %s now available.", side, self.chain.key)
                    return params
            return params

        params_cache = get_params_cache()

        ## Save chain here from now on
        self.chain._cache_chain()

        ## Cache params
        if self.params is None:
            logger.warning("Parameters for %s not available. Cannot save to cache.", self.chain.key)
            logger.debug("Current params state: %s, %s", self.params, self.__dict__)
            return

        for side, _ in self.params.items():
            params = _wait_for_params(side)
            if params is None:
                logger.error("Timeout waiting for parameters for %s on %s. Skipping cache save.", side, self.chain.key)
                continue

            params_dict = params.__dict__
            params_dict["config_hash"] = hash_config(get_pricing_config())
            key = params_cache_key(
                root=self.symbol,
                valuation_date=self.valuation_date,
                div_type=self.div_type,
                vol_type=self.model,
                side=VolSide(side),
            )
            if key not in params_cache:
                params_cache[key] = params_dict
