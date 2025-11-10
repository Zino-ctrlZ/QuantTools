# pylint: disable=arguments-differ
# pylint: disable=arguments-renamed

from pydantic import BaseModel, ConfigDict, PrivateAttr
from typing import ClassVar, Optional, Literal
from datetime import datetime
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from trade.helpers.Logging import setup_logger
from trade.helpers.pydantic import loud_post_init
from trade.helpers.helper import assert_member_of_enum
from module_test.raw_code.optionlib_2.vol.ssvi.model.base import BaseSSVIModel
from module_test.raw_code.optionlib_2.vol.ssvi.model.model_utils import (
    convert_date_to_time_to_maturity,
    handle_strikes
)

from module_test.raw_code.optionlib_2.vol.ssvi.model.params import SSVIModelParams
from module_test.raw_code.optionlib_2.vol.ssvi.model.param_utils import (
    build_svi_params_obj,
    predict_vol,
)
from module_test.raw_code.optionlib_2.vol.ssvi.vol_math import (
    get_best_params,
    get_surface_params,
    get_atm_t,
    get_atm_vol,
)

from module_test.raw_code.optionlib_2.vol.ssvi.controller import get_global_config
from module_test.raw_code.optionlib_2.vol.ssvi.global_config import SSVIGlobalConfig
from module_test.raw_code.optionlib_2.vol.ssvi.model.chain import ChainOutput
from module_test.raw_code.optionlib_2.vol.ssvi.types import (
    VolSide,
    DivType,
    VolType,
)
logger = setup_logger('optionlib.ssvi.model.ssvi_model')
class _SSVIModel(BaseSSVIModel, BaseModel):
    """
    SSVI Model for Stochastic Volatility Surface.
    This class implements the SSVI model using the parameters defined in SSVIModelParams.
    It provides methods to predict implied volatility, build the model, and fit the model.

    Note: There will be no market data retrieval in this class. Technically, it is completely blind to market data.
    This model will be enforcing discrete dividends and will not support continuous dividends.

    Attributes:
        chain (ChainOutput): The option chain data.
        valuation_date (str | datetime): The valuation date for the option chain.
        right (VolSide): The side of the volatility surface to model ('call', 'put', 'otm').
        div_type (DivType): The type of dividend handling ('discrete', 'continuous').
        model (VolType): The type of volatility model ('binomial', 'normal', etc.).
        params (Optional[SSVIModelParams]): The parameters of the SSVI model after fitting.
    
    Args:
        chain (ChainOutput): The option chain data.
        valuation_date (str | datetime): The valuation date for the option chain.
        right (VolSide): The side of the volatility surface to model ('call', 'put', 'otm').
    """
    # ==============================
    # Class Variables
    # ==============================
    model_config = ConfigDict(validate_assignment=True, 
                              arbitrary_types_allowed=True,
                              frozen=True,
                              extra='forbid')
    global_config: ClassVar[SSVIGlobalConfig] = get_global_config()

    # ==============================
    # Instance Variables
    # ==============================

    ## Compulsory Inputs
    chain: ChainOutput
    valuation_date: str|datetime
    right: VolSide

    ## Private Attributes
    _atm_t:list = PrivateAttr(default_factory=list)
    _atm_iv:list = PrivateAttr(default_factory=list)
    _fwd_interp: interp1d = PrivateAttr(default=None)
    _params: SSVIModelParams = PrivateAttr(default=None)

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
    def div_type(self) -> DivType:
        return self.chain.div_type

    @ property
    def params(self) -> Optional[SSVIModelParams]:
        return self._params
    
    @property
    def fwd_interp(self) -> interp1d:
        if self._fwd_interp is None:
            raise ValueError("Forward interpolation function is not initialized. Ensure the model is initialized properly.")
        return self._fwd_interp
    
    @property
    def atm_t(self) -> list:
        return self._atm_t

    @property
    def atm_iv(self) -> list:
        return self._atm_iv

    @property
    def model(self) -> VolType:
        return self.chain.vol_type

    @property
    def dataframe_chain(self) -> pd.DataFrame:
        view = self.chain.chain
        if view is None or view.empty:
            raise ValueError("Chain cannot be None or empty")
        
        ## Seperate chain into calls, puts, and otm
        call_bool = view[self.chain.right_col].str.lower() == 'c'
        put_bool = view[self.chain.right_col].str.lower() == 'p'

        ## Spliting by right
        if self.right == VolSide.CALL:
            chain = view[call_bool].copy()
        elif self.right == VolSide.PUT:
            chain = view[put_bool].copy()
        elif self.right == VolSide.OTM:
            chain = view[((call_bool) & (view[self.chain.f_log_m_col] >= 0)) |
                          ((put_bool) & (view[self.chain.f_log_m_col] < 0))].copy()
        else:
            raise ValueError(f"Invalid right side: {self.right}. Must be 'call', 'put', or 'otm'.")

        return chain
        
    
    @model.setter
    def model(self, value: VolType):
        enum_v = assert_member_of_enum(value, VolType)
        self.chain.vol_type = enum_v

    @div_type.setter
    def div_type(self, value: DivType):
        enum_v = assert_member_of_enum(value, DivType)
        self.chain.div_type = enum_v

    def __repr__(self):
        return f"<SSVIModel(valuation_date={self.valuation_date}, right={self.right}, params={self.params})>"

    @loud_post_init
    def model_post_init(self, context): # pylint: disable=arguments-differ
        """
        Post-initialization to validate and initialize the model.
        """
        self.validate()
        self.initialize()

    def validate(self): # pylint: disable=arguments-renamed
        """
        Validate the input chain DataFrame to ensure it contains all required columns.
        Raises ValueError if any required column is missing.
        """
        if self.chain is None or self.chain.chain.empty:
            raise ValueError("Chain cannot be None or empty")

    def initialize(self):
        """
        Initialize the SSVI model by separating the option chain into calls, puts, and OTM options.
        Also prepares the ATM parameters for fitting.
        """
        
        ## Chain Now
        chain = self.dataframe_chain

        ## Get atm_t, atm_iv
        self._atm_t = get_atm_t(self.dataframe_chain, self.chain.t_col, self.chain.f_log_m_col)
        self._atm_iv = get_atm_vol(self.dataframe_chain, self.chain.f_log_m_col, self.chain.vol_col)

        ## Prepare fwd_interp
        self._fwd_interp= interp1d(
            x= chain[self.chain.t_col].values,
            y=chain[self.chain.fwd_col_name].values,)


    def fit(self):
        """
        Fit the SSVI model to the option chain data.
        This method estimates the ATM variance, long-term variance, speed of mean reversion,
        skewness, kurtosis, and correlation parameters using the option chain data.
        It calculates the ATM maturities and implied volatilities, and then uses these to
        estimate the model parameters.

        Note: This method is designed to be called after the model has been initialized. It fits per right chain (call and put).
        """
        if self.dataframe_chain is None or self.dataframe_chain.empty:
            raise ValueError("Dataframe chain is empty or not set. Ensure the model is initialized properly.")
        
        if self._params is not None:
            logger.info("Model is already fitted for %s", self.chain.key)
            return

        def inner_fit(right_chain_attr: str):
            """
            Inner function to perform the fitting process.
            This is called by the fit method.
            """
            chain: pd.DataFrame = getattr(self, right_chain_attr)
            if chain is None or chain.empty:
                raise ValueError(f"Chain for {right_chain_attr} is empty or not set.")

            atm_t = np.array(self._atm_t)
            atm_iv = np.array(self._atm_iv)
            if atm_t.size == 0 or atm_iv.size == 0:
                raise ValueError(f"No ATM maturities or volatilities found in {right_chain_attr} chain. Adjust PRICING_CONFIG['ATM_WIDTH'].")
            (var0_hat, var_inf_hat, kappa_hat), atm_loss = get_best_params(
                atm_t,
                atm_iv
            )
            eta_hat, lambda_hat, rho_hat, surface_loss = get_surface_params(
                chain[self.chain.strike_col].values,
                chain[self.chain.t_col].values,
                chain[self.chain.fwd_col_name].values,
                var0_hat,
                var_inf_hat,
                kappa_hat,
                chain[self.chain.vol_col].values,
                iterations=self.iterations,
                chunk_size=self.chunk_size
            )
            params = build_svi_params_obj(
                chain=chain,
                var0_hat=var0_hat,
                var_inf_hat=var_inf_hat,
                kappa_hat=kappa_hat,
                eta_hat=eta_hat,
                lambda_hat=lambda_hat,
                rho_hat=rho_hat,
                atm_loss=atm_loss,
                surface_loss=surface_loss
            )

            return params
        self._params = inner_fit('dataframe_chain')
    
    def predict(self,
                k: float| np.ndarray,
                exp: str| datetime| np.ndarray = None,
                strike_type: Literal['p', 'k', 'pf', 'f'] = 'f'):
        """
        Predict the implied volatility for a given strike and expiration.
        Args:
            k (float | np.ndarray): Strike price or array of strike prices.
            exp (str | datetime | np.ndarray): Expiration date or array of expiration dates.
                c & p are for call and put options, respectively.
                itm & otm are for in-the-money and out-of-the-money options, respectively.
                    itm: Will use Calls in left wing and Puts in right wing.
                    otm: Will use Calls in right wing and Puts in left wing.
            strike_type (Literal['p', 'k', 'pf', 'f']): Type of strike price ('p' for price, 'k' for strike, etc.).
                p: Percent of spot
                k: Strike price
                pf: Percent of forward
                f: Forward price
        Returns:
            np.ndarray: Predicted implied volatility for the given parameters.
        """
        if exp is None:
            exp_vals = ['3m']
        else:
            exp_vals = exp if hasattr(exp, '__iter__') and not isinstance(exp, str) else [exp]
            
        exp = np.array(exp_vals)
        dtes = np.maximum(
            np.array([convert_date_to_time_to_maturity(e, self.valuation_date) for e in exp]),
            self.chain.t.min()
        )
        fwd = np.asarray(self._fwd_interp(dtes))
        k = np.asarray(k if hasattr(k, '__iter__') else [k])

        # Broadcast k across maturities
        spot = self.dataframe_chain['spot'].iloc[0]  # dynamic access as you prefer
        K = np.vstack([
            handle_strikes(k=k, f=f, strike_type=strike_type, spot=spot)
            for f in fwd
        ])
        pretty_k = np.vstack([k for _ in fwd])

        T = np.repeat(dtes[:, None], K.shape[1], axis=1)
        F = np.repeat(fwd[:,  None], K.shape[1], axis=1)

        vols = predict_vol(k=K.ravel(), t=T.ravel(), f=F.ravel(), params=self.params)

        # Single DF build; index only if you rely on it downstream
        df = pd.DataFrame({
            'strike': pretty_k.ravel(),
            'exp':    np.repeat(dtes, K.shape[1]),
            'vol':    vols,
            'fwd':    np.repeat(fwd,  K.shape[1]),
        })

        # map back to the original exp tokens
        df['exp'] = df['exp'].map(dict(zip(dtes, exp)))
        return df.set_index(['strike','exp']).sort_index()
