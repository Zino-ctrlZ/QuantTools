# pylint: disable=arguments-differ
# pylint: disable=arguments-renamed
from typing import Literal, Optional
from datetime import datetime
from pydantic import PrivateAttr, ConfigDict, BaseModel
import pandas as pd
import numpy as np
from trade.helpers.helper import assert_member_of_enum
from trade.helpers.Logging import setup_logger
from trade.helpers.pydantic import loud_post_init
from module_test.raw_code.optionlib_2.vol.ssvi.model.ssvi_model._ssvi import _SSVIModel
from module_test.raw_code.optionlib_2.vol.ssvi.model.params import SSVIModelParams
from module_test.raw_code.optionlib_2.vol.ssvi.model.chain import ChainOutput
from module_test.raw_code.optionlib_2.vol.ssvi.types import VolSide, DivType, VolType
from module_test.raw_code.optionlib_2.vol.ssvi.controller import (
    get_global_config,
    get_background_fits,
)
from module_test.raw_code.optionlib_2.vol.ssvi.model.base import BaseSSVIModel
from module_test.raw_code.optionlib_2.vol.ssvi.exceptions import ChainInputError
logger = setup_logger('optionlib.ssvi.model.ssvi_model')

class SSVIParentModel(BaseModel, BaseSSVIModel):
    """
    Parent model to manage SSVI models for different option sides (call, put, otm).
    This class initializes and manages separate SSVIModel instances for calls, puts, and OTM options.
    It provides methods to fit all models and predict implied volatilities based on the option type.
    It isn't market data aware; it relies on passed info which creates the child models.
    Attributes:
        call_model (SSVIModel): SSVI model for call options.
        put_model (SSVIModel): SSVI model for put options.
        otm_model (SSVIModel): SSVI model for OTM options.
    """

    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    ## Compulsory Inputs
    chain: ChainOutput
    valuation_date: str|datetime
    
    ## Optional Inputs/Derived inputs
    _models: Optional[dict[str, _SSVIModel]] = PrivateAttr(default_factory=dict)

    @loud_post_init
    def model_post_init(self, context):
        """
        Post-initialization to validate and initialize the parent model.
        """
        self.valuation_date = pd.to_datetime(self.valuation_date).strftime('%Y-%m-%d')
        

    @property
    def iterations(self) -> int:
        """
        Returns the number of iterations for model fitting.
        """
        return get_global_config().model_iterations
    
    @iterations.setter
    def iterations(self, value: int):
        get_global_config().model_iterations = value

    @property
    def chunk_size(self) -> int:
        """
        Returns the chunk size for processing.
        """
        return get_global_config().chunk_size

    @chunk_size.setter
    def chunk_size(self, value: int):
        get_global_config().chunk_size = value

    @property
    def params(self) -> dict[str, SSVIModelParams]:
        """
        Returns the parameters of all SSVI models as a dictionary.
        """
        if self.models is None:
            raise ValueError("Models have not been initialized.")
        return {right: model.params for right, model in self.models.items()}
    
    @property
    def model_info(self) -> dict[str, dict]:
        """
        Returns a summary of the model information for all SSVI models.
        """
        if self.models is None:
            raise ValueError("Models have not been initialized.")
        return {right: {
                    'valuation_date': model.valuation_date,
                    'right': model.right.value,
                    'params': model.params
                } for right, model in self.models.items()}
    
    @property
    def call_model(self) -> _SSVIModel:
        if self.models is None or 'call' not in self.models:
            raise ValueError("Call model has not been initialized.")
        return self._get_or_build(VolSide.CALL)

    @property
    def put_model(self) -> _SSVIModel:
        if self.models is None or 'put' not in self.models:
            raise ValueError("Put model has not been initialized.")
        return self._get_or_build(VolSide.PUT)

    @property
    def otm_model(self) -> _SSVIModel:
        if self.models is None or 'otm' not in self.models:
            raise ValueError("OTM model has not been initialized.")
        return self._get_or_build(VolSide.OTM)
    
    @property
    def div_type(self) -> DivType:
        return self.chain.div_type
    
    @property
    def model(self) -> VolType:
        return self.chain.vol_type
    
    @model.setter
    def model(self, value: VolType):
        raise ChainInputError("Model type cannot be changed for this chain.")

    @div_type.setter
    def div_type(self, value: DivType):
        raise ChainInputError("Dividend type cannot be changed for this chain.")

    def __repr__(self):
        return f"<SSVIParentModel(valuation_date={self.valuation_date}, models={list(self.models.keys())})>"


    def _get_or_build(self, side: VolSide, fit: bool=False) -> _SSVIModel:
        side = assert_member_of_enum(side, VolSide)
        key = side.value
        m = self._models.get(key)
        if m is None:
            m = _SSVIModel(
                chain=self.chain,
                valuation_date=self.valuation_date,
                right=side
            )
            if fit:
                m.fit()
            self._models[key] = m
        return m

    @property
    def models(self) -> dict[str, _SSVIModel]:
        return self._models

    
    
    def fit(self):
        """
        Fit the SSVI model to the option chain data.
        This method estimates the ATM variance, long-term variance, speed of mean reversion,
        skewness, kurtosis, and correlation parameters using the option chain data.
        It calculates the ATM maturities and implied volatilities, and then uses these to
        estimate the model parameters.

        Note: This method is designed to be called after the model has been initialized. It fits per right chain (call and put).
        """
        
        # Fit only the primary side first
        global_conf = get_global_config()
        global_background_fits = get_background_fits()
        primary = global_conf.vol_side
        self._get_or_build(primary).fit()

        if global_conf.fit_all_sides:
            # Optionally background-fit the others IF you truly need them later
            for side in (VolSide.CALL, VolSide.PUT, VolSide.OTM):
                if side is primary: 
                    continue
                global_background_fits.submit(fn=self._get_or_build(side).fit,
                                            key=f"{self.chain.key}_{side.value}")

    def predict(self,
                k: float| np.ndarray,
                exp: str| datetime| np.ndarray = None,
                strike_type: Literal['p', 'k', 'pf', 'f'] = 'f',
                right: VolSide = None):
        """
        Predict the implied volatility for a given strike and expiration.
        Args:
            k (float | np.ndarray): Strike price or array of strike prices.
            exp (str | datetime | np.ndarray): Expiration date or array of expiration dates.
            right (Literal['c', 'p', 'itm', 'otm'] | np.ndarray): Option type ('c' for call, 'p' for put, etc.).
                c & p are for call and put options, respectively.
                itm & otm are for in-the-money and out-of-the-money options, respectively.
                    itm: Will use Calls in left wing and Puts in right wing.
                    otm: Will use Calls in right wing and Puts in left wing.
            strike_type (Literal['p', 'k', 'pf', 'f']): Type of strike price ('p' for price, 'k' for strike, etc.).
                p: Percent of spot
                k: Strike price
                pf: Percent of forward
                f: Log forward moneyness
        Returns:
            np.ndarray: Predicted implied volatility for the given parameters.
        """
        global_config = get_global_config()

        ## Warning if global config differs from model config
        if global_config.div_type != self.div_type:
            logger.warning("Global div_type %s differs from model div_type %s. Using global div_type %s.", global_config.div_type, self.div_type, global_config.div_type)
        if global_config.vol_type != self.model:
            logger.warning("Global vol_type %s differs from model vol_type %s. Using global vol_type %s.", global_config.vol_type, self.model, global_config.vol_type)
        
        ## Use global config side if not provided
        if right is None:
            right = global_config.vol_side
        elif isinstance(right, str):
            right = VolSide(right.lower())

        ## Build/get the model for the requested side
        right = assert_member_of_enum(right, VolSide)
        model = self._get_or_build(right)

        ## Fit if not already fitted
        if model.params is None:
            logger.warning("Model for %s on %s not fitted yet. Fitting now...", right.value, self.chain.key)
            model.fit()
        
        ## Predict using the appropriate model
        return model.predict(k=k, exp=exp, strike_type=strike_type)
