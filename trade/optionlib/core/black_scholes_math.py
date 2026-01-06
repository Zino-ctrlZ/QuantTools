import numpy as np
from typing import List
from scipy.stats import norm
from ..config.defaults import DAILY_BASIS
from trade.helpers.Logging import setup_logger
logger = setup_logger('trade.optionlib.core.black_scholes_math')


def black_scholes_analytic_greeks_vectorized(F, K, T, r, sigma, option_type="c"):
    F = np.asarray(F)
    K = np.asarray(K)
    T = np.asarray(T)
    r = np.asarray(r)
    sigma = np.asarray(sigma)

    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    nd1 = norm.pdf(d1)
    df = np.exp(-r * T)

    # Handle both scalar and vector string inputs for option_type
    option_type = np.asarray(option_type)

    # Enfores option_type
    if not np.all(np.isin(option_type, ["c", "p"])):
        raise ValueError("option_type must be 'c' or 'p'")

    is_call = option_type == "c"

    delta = np.where(is_call, norm.cdf(d1), -norm.cdf(-d1))
    gamma = nd1 / (F * sigma * np.sqrt(T))
    vega = F * nd1 * np.sqrt(T)
    volga = vega * d1 * d2 / sigma
    rho = np.where(
        is_call,
        T * K * df * norm.cdf(d2),
        -T * K * df * norm.cdf(-d2)
    )
    theta = - (F * nd1 * sigma) / (2 * np.sqrt(T)) \
            - r * K * df * np.where(is_call, norm.cdf(d2), norm.cdf(-d2))

    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega / 100,
        "volga": volga / 100**2,
        "rho": rho / 100,
        "theta": theta/DAILY_BASIS
    }

def black_scholes_vectorized_base(F: np.ndarray|List[float], 
                                  K: np.ndarray|List[float],
                                  T: np.ndarray|List[float], 
                                  r: np.ndarray|List[float], 
                                  sigma: np.ndarray|List[float], 
                                  option_type: str|List[str]="c",
                                  **kwargs) -> np.ndarray:
    """
    Vectorized Black-Scholes formula for option pricing.
    F: Forward prices (array)
    K: Strike prices (array)
    T: Time to maturity (array)
    r: Risk-free rates (array)
    sigma: Volatilities (array)
    option_type: "c" for call, "p" for put (array or single string)

    Returns: Option prices (array)
    """
    # Ensure all inputs are numpy arrays for vectorized operations
    F = np.asarray(F)
    K = np.asarray(K)
    T = np.asarray(T)
    r = np.asarray(r)
    sigma = np.asarray(sigma)

    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    df = np.exp(-r * T)
    # Handle both scalar and vector string inputs for option_type
    option_type = np.asarray(option_type)

    # Enfores option_type
    if not np.all(np.isin(option_type, ["c", "p"])):
        raise ValueError("option_type must be 'c' or 'p'")
    
    call_mask = option_type == "c"
    put_mask = option_type == "p"
    price = np.zeros_like(F)
    # Calculate call prices
    price[call_mask] = df[call_mask] * (F[call_mask] * norm.cdf(d1[call_mask]) - K[call_mask] * norm.cdf(d2[call_mask]))
    # Calculate put prices
    price[put_mask] = df[put_mask] * (K[put_mask] * norm.cdf(-d2[put_mask]) - F[put_mask] * norm.cdf(-d1[put_mask]))
    return price