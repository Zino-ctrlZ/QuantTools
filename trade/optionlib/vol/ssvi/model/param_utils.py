"""
Utility functions for handling SSVI model parameters.
This module provides functions to build SSVI model parameter objects,
predict volatilities, and load parameters from cache.

Author: Chiemelie Nwanisobi
Date: 2025-10-10
"""

from collections.abc import Iterable
from typing import Optional
from datetime import datetime
import numpy as np
import pandas as pd
from trade.helpers.helper import assert_member_of_enum
from trade.helpers.Logging import setup_logger
from trade.optionlib.vol.ssvi.model.model_utils import params_cache_key
from trade.optionlib.config.ssvi.controller import is_latest_config, get_global_config, get_params_cache
from trade.optionlib.vol.ssvi.vol_math import ssvi_implied_vol
from trade.optionlib.vol.ssvi.model.params import SSVIModelParams
from trade.optionlib.config.types import DivType, VolType, VolSide
from trade.optionlib.vol.ssvi.utils import (
    get_t_grid,
    get_fwd_grid,
    get_k_grid,
)
from trade.optionlib.vol.ssvi.model.model_utils import (
    calculate_normalized_rmse_loss,
    calculate_normalized_mae_loss,
)

logger = setup_logger("optionlib.ssvi.model.param_utils")


def build_svi_params_obj(
    chain: pd.DataFrame,
    var0_hat: float,
    var_inf_hat: float,
    kappa_hat: float,
    eta_hat: float,
    lambda_hat: float,
    rho_hat: float,
    atm_loss: float,
    surface_loss: float,
) -> SSVIModelParams:
    """
    Build an SSVIModelParams object from the given parameters.

    Args:
        chain (pd.DataFrame): The option chain DataFrame.
        var0_hat (float): Initial variance estimate at ATM.
        var_inf_hat (float): Long-term variance estimate.
        kappa_hat (float): Speed of mean reversion.
        eta_hat (float): Skewness parameter.
        lambda_hat (float): Kurtosis parameter.
        rho_hat (float): Correlation parameter.
        atm_loss (float): Loss associated with ATM volatility estimation.
        surface_loss (float): Loss associated with surface fitting.

    Returns:
        SSVIModelParams: The SSVI model parameters object.
    """
    ## Calculate normalized losses
    moneyness = chain["moneyness"].values
    market_iv = chain["vol"].values
    model_iv = ssvi_implied_vol(
        fwd=get_fwd_grid(chain),
        strike=get_k_grid(chain),
        maturity=get_t_grid(chain),
        var0=var0_hat,
        var_inf=var_inf_hat,
        kappa=kappa_hat,
        eta=eta_hat,
        lam=lambda_hat,
        rho=rho_hat,
    )

    normalized_nrmse, rw_nrmse, lw_nrmse = calculate_normalized_rmse_loss(
        market_iv=market_iv, model_iv=model_iv, moneyness=moneyness
    )
    normalized_nmae, rw_nmae, lw_nmae = calculate_normalized_mae_loss(
        market_iv=market_iv, model_iv=model_iv, moneyness=moneyness
    )
    return SSVIModelParams(
        var0_hat=var0_hat,
        var_inf_hat=var_inf_hat,
        kappa_hat=kappa_hat,
        eta_hat=eta_hat,
        lambda_hat=lambda_hat,
        rho_hat=rho_hat,
        atm_loss=atm_loss,
        surface_loss=surface_loss,
        nrmse=normalized_nrmse,
        rw_nrmse=rw_nrmse,
        lw_nrmse=lw_nrmse,
        nmae=normalized_nmae,
        rw_nmae=rw_nmae,
        lw_nmae=lw_nmae,
    )


def is_iterable(obj, *, exclude_str=True):
    if exclude_str and isinstance(obj, (str, bytes)):
        return False
    return isinstance(obj, Iterable)


def _sigmoid_func(k: np.ndarray, f: float) -> np.ndarray:
    x = np.log(k / f)
    return 1 / (1 + np.exp(4 * x))


def pick_params(call_params: SSVIModelParams, put_params: SSVIModelParams, right: str) -> SSVIModelParams:
    """
    Pick parameters based on the option type (call or put).

    Args:
        call_params (SSVIModelParams): Parameters for call options.
        put_params (SSVIModelParams): Parameters for put options.
        right (str): The option type ('c' for call, 'p' for put).

    Returns:
        SSVIModelParams: The selected parameters based on the option type.
    """
    if right.lower() == "c":
        return call_params
    elif right.lower() == "p":
        return put_params
    else:
        raise ValueError(f"Invalid option type: {right}. Expected 'c' or 'p'.")


def _predict_vol_decider(
    k: float | np.ndarray,
    t: float | np.ndarray,
    f: float | np.ndarray,
    right: str,
    call_params: SSVIModelParams,
    put_params: SSVIModelParams,
) -> float | np.ndarray:
    """
    Predict the volatility using the SSVI model parameters.
    This function selects the appropriate parameters based on the option type
    and computes the implied volatility using the SSVI formula.

    If 'right' is 'itm' or 'otm', it blends the call and put volatilities
    based on the moneyness using a sigmoid function.

    Args:
        k (float): Strike price.
        t (float): Time to maturity in years.
        f (float): Forward price.
        params (SSVIModelParams): The SSVI model parameters.

    Returns:
        float: The predicted volatility.
    """
    if right in ["c", "p"]:
        params = pick_params(call_params, put_params, right)
    elif right in ["itm", "otm"]:
        call_vols = predict_vol(k, t, f, call_params)
        put_vols = predict_vol(k, t, f, put_params)
        w = _sigmoid_func(k, f)
        if right == "itm":  ## Left: Call, Right: Put
            return w * call_vols + (1 - w) * put_vols
        else:
            return (1 - w) * call_vols + w * put_vols
    else:
        raise ValueError(f"Invalid option type: {right}. Expected 'c', 'p', 'itm', or 'otm'.")

    return ssvi_implied_vol(
        fwd=f,
        strike=k,
        maturity=t,
        var0=params.var0_hat,
        var_inf=params.var_inf_hat,
        kappa=params.kappa_hat,
        eta=params.eta_hat,
        lam=params.lambda_hat,
        rho=params.rho_hat,
    )


def predict_vol(k: float | np.ndarray, t: float | np.ndarray, f: float, params: SSVIModelParams) -> float | np.ndarray:
    """
    Predict the volatility using the SSVI model parameters.
    This function computes the implied volatility using the SSVI formula.
    """
    return ssvi_implied_vol(
        fwd=f,
        strike=k,
        maturity=t,
        var0=params.var0_hat,
        var_inf=params.var_inf_hat,
        kappa=params.kappa_hat,
        eta=params.eta_hat,
        lam=params.lambda_hat,
        rho=params.rho_hat,
    )


def load_ssvi_params_from_cache(
    root: str,
    valuation_date: str | datetime,
    div_type: DivType,
    vol_type: VolType,
    side: VolSide,
) -> Optional[SSVIModelParams]:
    """
    Load SSVI model parameters from cache if available and up-to-date.
    Args:
        root (str): The root symbol of the underlying asset.
        valuation_date (str or datetime): The date of valuation.
        div_type (DivType): Type of dividends considered.
        vol_type (VolType): Type of volatility used for calibration.
        side (VolSide): Type of side. otm, call, puts
    Returns:
        Optional[SSVIModelParams]: The cached params or None if not found/outdated.
    """
    div_type = assert_member_of_enum(div_type, DivType)
    vol_type = assert_member_of_enum(vol_type, VolType)
    side = assert_member_of_enum(side, VolSide)
    params_cache = get_params_cache()
    global_conf = get_global_config()

    key = params_cache_key(
        root=root,
        valuation_date=valuation_date,
        div_type=div_type,
        vol_type=vol_type,
        side=side,
    )

    if key in params_cache:
        params = params_cache[key]
        config_hash = params.pop("config_hash", None)
        if is_latest_config(config_hash):
            return SSVIModelParams(**params)
        if global_conf.overwrite_existing:
            logger.warning("Cached params config hash is outdated. Overwriting existing cache.")
            del params_cache[key]
        else:
            logger.warning(
                "Cached params config hash is outdated. Use "
                "'overwrite_existing=True' to overwrite from get_global_config."
            )
    return None
