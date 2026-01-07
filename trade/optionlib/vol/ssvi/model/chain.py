"""
Module to handle option chain loading and processing for SSVI volatility modeling.
Provides the MarketChainLoader class to load, cache, and process option chain data.

Authors: Chiemelie Nwanisobi
Date: 2025-10-10
"""

# pylint: disable=arguments-differ
from typing import Optional, ClassVar, Dict
from datetime import datetime
import pandas as pd
from pydantic import Field, ConfigDict, BaseModel, PrivateAttr
from pydantic.dataclasses import dataclass
from trade.helpers.Logging import setup_logger
from trade.helpers.helper_types import SingletonMixin
from trade.optionlib.config.types import DivType, VolType
from trade.optionlib.config.ssvi.global_config import SSVIGlobalConfig
from trade.optionlib.vol.ssvi.model.model_utils import chain_cache_key
from trade.optionlib.vol.ssvi.chain_prep import ChainChecklist
from trade.optionlib.vol.ssvi.model.base import ChainInputModel
from trade.optionlib.config.ssvi.controller import (
    get_pricing_config,
    hash_config,
    get_global_config,
    get_chain_cache,
    is_latest_config,
)
from trade.optionlib.vol.ssvi.utils import (
    get_chain,
    format_chain,
    get_rates,
    get_forward_price_on_chain,
    get_bs_vol_on_chain,
    get_discrete_crr_vol_on_chain,
)

logger = setup_logger("optionlib.ssvi.chain")


@dataclass(config=ConfigDict(arbitrary_types_allowed=True, frozen=True))
class ChainOutput:
    """
    Dataclass to hold the output of the chain processing.
    """

    root: Optional[str] = Field(default=None, description="Root symbol of the underlying asset.")
    data_chain: Optional[pd.DataFrame] = Field(default_factory=None, description="Processed option chain DataFrame")
    source_from_cache: bool = Field(default=False, description="Indicates if the chain was sourced from cache")
    spot: float = Field(..., description="Spot price of the underlying asset")
    div_type: DivType = Field(default=get_global_config().div_type, description="Type of dividends considered")
    vol_type: VolType = Field(
        default=get_global_config().vol_type, description="Type of volatility used for calibration"
    )
    pv_div_col: str = Field(default=None, description="Column name for present value of dividends if applicable")
    div_schedule_col: str = Field(default=None, description="Column name for dividend schedule if applicable")
    fwd_col_name: str = Field(default=None, description="Column name for forward prices if applicable")
    rate_col: str = Field(default=None, description="Column name for interest rates if applicable")
    vol_col: str = Field(default="vol", description="Column name for implied volatilities")
    t_col: str = Field(default="t", description="Column name for time to maturity")
    strike_col: str = Field(default="strike", description="Column name for strike prices")
    f_log_m_col: str = Field(default="f_log_moneyness", description="Column name for log moneyness")
    fwd_m_col: str = Field(default="f_moneyness", description="Column name for forward moneyness")
    right_col: str = Field(default="right", description="Column name for option rights (call/put)")
    midpoint_col: str = Field(default="midpoint", description="Column name for option midpoints")
    valuation_date: str | datetime = Field(..., description="Valuation date for the option chain")

    def __post_init__(self):
        self.validate()

    def validate(self):
        """
        Validates the chain DataFrame to ensure all required columns are present.
        Raises ValueError if any required column is missing.
        """
        if self.source_from_cache:
            return  # Skip validation if sourced from cache
        if self.data_chain is None:
            raise ValueError("Chain DataFrame cannot be None")
        if self.data_chain.empty:
            raise ValueError("Chain DataFrame cannot be empty")
        required_columns = [
            self.strike_col,
            self.right_col,
            self.midpoint_col,
            self.pv_div_col,
            self.div_schedule_col,
            self.rate_col,
            "expiration",
            self.vol_col,
            self.t_col,
            self.fwd_col_name,
        ]

        for col in required_columns:
            if col not in self.data_chain.columns:
                raise ValueError(f"Missing required column: {col}")

    def _cache_chain(self):
        """
        Caches the chain DataFrame to optimize access to frequently used columns.

        We cache to avoid loading multiple dataframes in memory. Instead it's saved to disk.
        This is particularly useful when dealing with large datasets or when the same data is accessed multiple times.
        1. Check if the chain with the same key already exists in the cache.
        2. If not, store the current chain in the cache.
        3. If it exists, log that the chain is already cached. Location: get_chain_cache()[self.key]
        4. Access the cached chain using the key.
        5. Use lightweight accessors to retrieve specific columns from the cached chain without duplicating data.
        6. This approach minimizes memory usage and improves performance by avoiding redundant data storage.
        7. The use of properties allows for easy and intuitive access to the cached data.
        """

        if self.key not in get_chain_cache():
            if self.data_chain is None:
                raise ValueError("Chain DataFrame cannot be None when caching. Consider reloading the chain.")
            if self.data_chain.empty:
                raise ValueError("Chain DataFrame cannot be empty when caching. Consider reloading the chain.")

            ## Add config hash to the chain for versioning
            chain = self.data_chain.copy()
            chain["config_hash"] = hash_config(get_pricing_config())
            get_chain_cache()[self.key] = chain
            logger.info("Caching chain for key in ChainOutput: %s", self.key)
        else:
            logger.info("Chain with key: %s already cached.", self.key)

        if self.data_chain is not None:
            self.data_chain = None

    @property
    def key(self):
        val_date = pd.to_datetime(self.valuation_date).strftime("%Y-%m-%d")
        return chain_cache_key(self.root, val_date, self.div_type, self.vol_type)

    @property
    def chain(self):
        chain = get_chain_cache().get(self.key, None)
        if chain is None:
            chain = self.data_chain
        if chain is None:
            raise ValueError("Chain gone missing from cache. Consider reloading the chain.")
        return chain

    # Lightweight accessors (views of chain; no extra storage)
    @property
    def vol(self) -> pd.Series:
        return self.chain[self.vol_col]

    @property
    def t(self) -> pd.Series:
        return self.chain[self.t_col]

    @property
    def strike(self) -> pd.Series:
        return self.chain[self.strike_col]

    @property
    def right(self) -> pd.Series:
        return self.chain[self.right_col]

    @property
    def midpoint(self) -> pd.Series:
        return self.chain[self.midpoint_col]

    @property
    def fwd(self) -> Optional[pd.Series]:
        return None if self.fwd_col_name is None else self.chain[self.fwd_col_name]

    @property
    def rates(self) -> Optional[pd.Series]:
        return None if self.rate_col is None else self.chain[self.rate_col]

    def __repr__(self):
        return f"<ChainOutput(root={self.root}, valuation_date={self.valuation_date})>"


def _get_chain_key(symbol: str, run_date: str, global_config: SSVIGlobalConfig) -> str:
    return chain_cache_key(
        root=symbol, valuation_date=run_date, div_type=global_config.div_type, vol_type=global_config.vol_type
    )


def _calculate_vol(chain: pd.DataFrame, run_date: str, global_config: SSVIGlobalConfig) -> pd.Series:
    logger.info("Calculating vols using %s model", global_config.vol_type)
    if global_config.vol_type == VolType.BS:
        # NOTE: Consider switching to ChainChecklist.calculate_european_equiv_vol
        return get_bs_vol_on_chain(
            chain=chain,
            valuation_date=run_date,
            rate_col_name="risk_free_rate",
            forward_col_name="f",
            mid_col_name="midpoint",
        )

    if global_config.vol_type == VolType.BINOMIAL:
        return get_discrete_crr_vol_on_chain(
            chain=chain,
            valuation_date=run_date,
            n=global_config.N,
            rates_col_name="risk_free_rate",
            div_type=global_config.div_type.value,
        )

    raise ValueError(f"Invalid vol_type: {global_config.vol_type}")


def _process_fresh_chain(symbol: str, run_date: str, global_config: SSVIGlobalConfig) -> pd.DataFrame:
    logger.info(
        "Loading chain for %s on %s with key: %s", symbol, run_date, _get_chain_key(symbol, run_date, global_config)
    )
    chain = get_chain(symbol, run_date)
    chain = format_chain(chain)

    logger.info("Initial chain size: %s", chain.shape[0])

    r = get_rates(run_date)
    chain["risk_free_rate"] = r
    logger.info("Risk-free rate on %s: %s", run_date, r)

    chain = get_forward_price_on_chain(chain=chain, valuation_date=run_date, r=r, div_type=global_config.div_type)
    logger.info("After F load: %s", chain.shape[0])

    chain = ChainChecklist.remove_junk_quotes(chain)
    logger.info("After junk removal: %s", chain.shape[0])

    vol = _calculate_vol(chain, run_date, global_config)
    chain["vol"] = vol
    logger.info("After vol calculation: %s", chain.shape[0])
    return chain


def _check_cached_chain(chain_cache: dict, key: str, global_config: SSVIGlobalConfig):
    chain: pd.DataFrame = chain_cache.get(key)
    if chain is None:
        return chain, True, False  # chain, should_reload, deleted_flag

    col_available = "config_hash" in chain.columns
    update_needed = not is_latest_config(chain["config_hash"].values[0]) if col_available else True

    if update_needed and global_config.overwrite_existing:
        logger.warning("Cached chain config hash is outdated. Overwriting existing cache.")
        del chain_cache[key]
        return None, True, True

    if update_needed:
        logger.warning(
            "Cached chain config hash is outdated. Use 'overwrite_existing=True' to overwrite from get_global_config."
        )
    return chain, False, False


def _build_chain_output(
    symbol: str, run_date: str, chain: pd.DataFrame, global_config: SSVIGlobalConfig
) -> ChainOutput:
    return ChainOutput(
        root=symbol,
        data_chain=chain,
        spot=chain["spot"].iloc[0],
        div_type=global_config.div_type,
        vol_type=global_config.vol_type,
        pv_div_col="div_pv",
        fwd_col_name="f",
        rate_col="risk_free_rate",
        vol_col="vol",
        t_col="t",
        strike_col="strike",
        right_col="right",
        midpoint_col="midpoint",
        valuation_date=run_date,
        div_schedule_col="div_schedule",
    )


def _load_chain(symbol: str, run_date: str, ignore_cache: bool = False) -> ChainOutput:
    """
    Load and process the option chain for a given symbol and run date.
    Args:
        symbol (str): The underlying asset symbol.
        run_date (str): The valuation date in 'YYYY-MM-DD' format.
    Returns:
        ChainInputModel: Processed option chain model.
    """
    global_config = get_global_config()
    chain_cache = get_chain_cache()
    key = _get_chain_key(symbol, run_date, global_config)

    if key not in chain_cache or ignore_cache:
        chain = _process_fresh_chain(symbol, run_date, global_config)
        chain_cache[key] = chain
        logger.info("Processed fresh chain for %s on %s with key: %s", symbol, run_date, key)
    else:
        logger.info("Using cached chain for %s on %s with key: %s", symbol, run_date, key)
        chain, should_reload, _ = _check_cached_chain(chain_cache, key, global_config)
        if should_reload:
            # If we deleted the outdated cache entry, reload fresh (recursively)
            return _load_chain(symbol, run_date)
        if chain is None:
            # Cached entry unexpectedly missing, force reload
            if key in chain_cache:
                del chain_cache[key]
            logger.warning("Cached chain was None, reloading...")
            return _load_chain(symbol, run_date)

    return _build_chain_output(symbol, run_date, chain, global_config)


class MarketChainLoader(BaseModel, ChainInputModel, SingletonMixin):
    """
    Market model to load and process option chain data.
    """

    model_config = ConfigDict(validate_assignment=True)
    _instances: ClassVar[Dict[str, "MarketChainLoader"]] = {}
    _initialized: bool = PrivateAttr(default=False)

    symbol: str = Field(..., description="Symbol of the underlying asset")
    valuation_date: str | datetime = Field(..., description="Run date for the data")
    _chains: Optional[Dict[str, ChainOutput]] = PrivateAttr(default_factory=dict)

    ## Post init to format valuation_date
    def model_post_init(self, context):  # pylint: disable=arguments-differ
        self.valuation_date = pd.to_datetime(self.valuation_date).strftime("%Y-%m-%d")

    @property
    def run_date(self) -> datetime:
        return pd.to_datetime(self.valuation_date)

    @classmethod
    def clear_instances(cls):
        cls._instances.clear()

    @classmethod
    def instances(cls):
        return cls._instances

    def __new__(cls, symbol: str, *args, **kwargs):
        if symbol not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[symbol] = instance
        return cls._instances[symbol]

    def __init__(self, *args, **data):
        # First-time init for this cached instance:
        # If __pydantic_private__ isn't set yet, it's the first real init.
        if getattr(self, "__pydantic_private__", None) is None:
            super().__init__(*args, **data)  # sets fields and creates private store
            self._initialized = True  # safe now
            return

        # Subsequent inits for this cached instance:
        if self._initialized:
            # Already initialized, just update fields
            for key, value in data.items():
                setattr(self, key, value)

    def _force_rebuild(self) -> bool:
        """
        Determines if the chain needs to be rebuilt based on the current run date.
        And cross-referencing the GLOBAL_CONFIG settings.
        Returns:
            bool: True if the chain needs to be rebuilt, False otherwise.
        """
        global_conf = get_global_config()
        ## If run_date not in chains, we need to build
        if self.run_date not in self._chains:
            return True

        ## If GLOBAL_CONFIG has changed, we need to rebuild
        existing_chain = self._chains[self.run_date]
        if existing_chain.div_type != global_conf.div_type or existing_chain.vol_type != global_conf.vol_type:
            return True

    def build_chain(self, force_rebuild: bool = False, ignore_cache: bool = False) -> ChainOutput:
        """
        Loads and processes the option chain data.
        force_rebuild: If True, forces a rebuild of the ChainOutput object from source even if it is cached within ChainOutput self._chains.
        ignore_cache: If True, ignores any cached chain data in CHAIN_DUMP_CACHE and reloads from source.
        Returns:
            ChainOutput: Processed option chain data.
        """
        global_conf = get_global_config()
        chain_cache = get_chain_cache()
        ## Generate cache key for the chain
        chain_key = chain_cache_key(
            root=self.symbol, valuation_date=self.run_date, div_type=global_conf.div_type, vol_type=global_conf.vol_type
        )

        ## Check if key exists in cache
        key_in_cache = chain_key in chain_cache

        ## If force_rebuild is True, or if the chain needs to be rebuilt based on config changes
        if self._force_rebuild() or force_rebuild:
            logger.info(
                "Rebuilding chain for %s on %s because config changed or not cached", self.symbol, self.run_date
            )

            ## If key exists in cache, use it to create ChainOutput
            ## Doing this to avoid reloading the chain into memory again.
            if key_in_cache:
                logger.info("Using cached chain data for %s on %s to rebuild ChainOutput", self.symbol, self.run_date)
                self._chains[self.run_date] = self._create_chain_output_from_cache(chain_key)

            ## Else, load from source
            else:
                logger.info("Loading chain data from source for %s on %s", self.symbol, self.run_date)
                self._chains[self.run_date] = _load_chain(self.symbol, self.run_date, ignore_cache=ignore_cache)
            logger.info("Rebuilt chain for %s on %s", self.symbol, self.run_date)

        ## If not force build, use ChainOutput in self._chains,
        ## pegged to a single run_date, saved under a singleton instance per symbol
        else:
            logger.info("Using cached chain for %s on %s", self.symbol, self.run_date)

        ## Load ChainOutput if not already loaded for this run_date
        if self.run_date not in self._chains:
            logger.info("MarketChainLoader: Loading chain for %s on %s", self.symbol, self.run_date)
            self._chains[self.run_date] = _load_chain(self.symbol, self.run_date, ignore_cache=ignore_cache)

        return self._chains[self.run_date]

    def get_chain(self) -> ChainOutput:
        """
        Returns the processed option chain data.
        """
        if not self._chains:
            raise ValueError("Chain not built yet. Call build_chain() first.")
        return self._chains[self.run_date]

    def _create_chain_output_from_cache(self, key: str) -> ChainOutput:
        """
        Creates a ChainOutput object from the cached chain data.
        Args:
            key (str): The cache key for the chain data.
        Returns:
            ChainOutput: The ChainOutput object created from the cached data.
        """
        chain_cache = get_chain_cache()
        global_config = get_global_config()
        if key not in chain_cache:
            raise ValueError(f"No cached chain found for key: {key}")
        chain = chain_cache[key]
        return ChainOutput(
            root=self.symbol,
            data_chain=None,
            spot=chain["spot"].iloc[0],
            div_type=global_config.div_type,
            vol_type=global_config.vol_type,
            pv_div_col="div_pv",
            fwd_col_name="f",
            rate_col="risk_free_rate",
            vol_col="vol",
            t_col="t",
            strike_col="strike",
            right_col="right",
            midpoint_col="midpoint",
            valuation_date=self.run_date,
            div_schedule_col="div_schedule",
            source_from_cache=True,
        )

    @property
    def chain(self) -> Optional[ChainOutput]:
        return self.get_chain() if self._chains else None
