import numpy as np
from numpy import exp, sqrt, log, maximum
from scipy.stats import norm
from scipy.special import erf
from typing import Literal


def norm_cdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))

def phi_vectorized(s, t, gamma, H, I, r, b, sigma):
    lamb = -r + gamma * b + 0.5 * gamma * (gamma - 1) * sigma ** 2
    kappa = 2 * b / (sigma ** 2) + (2 * gamma - 1)
    sigma_sqrt_t = sigma * np.sqrt(t)
    log_s_H = np.log(s / H)
    log_I_s = np.log(I / s)
    d = -(log_s_H + (b + (gamma - 0.5) * sigma ** 2) * t) / sigma_sqrt_t

    return (
        np.exp(lamb * t)
        * s ** gamma
        * (norm_cdf(d) - (I / s) ** kappa * norm_cdf(d - 2 * log_I_s / sigma_sqrt_t))
    )



def bjerksund_stensland_2002_vectorized(
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: np.ndarray,
    sigma: np.ndarray,
    option_type: Literal["c", "p"] = "c",
    dividend_type: str = "continuous",
    dividend: float | np.ndarray = 0.0,
    **kwargs
) -> np.ndarray:
    """
    Vectorized Bjerksund-Stensland (2002) American option pricer.
    Parameters
    ----------
    S : np.ndarray
        Spot prices (array-like)
    K : np.ndarray
        Strike prices (array-like)
    T : np.ndarray
        Time to maturity (in years, array-like)
    r : np.ndarray
        Risk-free interest rates (array-like)
    sigma : np.ndarray
        Volatilities (array-like)
    option_type : 'c' or 'p'
        Option type: 'c' for call, 'p' for put
    dividend_type : 'continuous' 
        How dividends are modeled
    dividend : float or np.ndarray
        - If continuous, pass yield (float or numpy array)

    Returns
    -------
    np.ndarray
        Option prices (array-like)
    Raises
    -------
    ValueError: If dividend is a list or if dividend_type is 'discrete'.
    """
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    r = np.asarray(r, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    option_type = np.asarray(option_type, dtype=str)


    if isinstance(dividend, (int, float)):
        dividend = np.full_like(S, dividend, dtype=float)
    
    if dividend_type == 'discrete':
        raise ValueError("Discrete dividends not supported in this vectorized implementation.")
    zero = np.zeros_like(S)

    # Adjust spot for dividends
    if dividend_type == "continuous":
        q = np.asarray(dividend, dtype=float)
        S_adj = S.copy()
    else:
        raise ValueError("Only continuous dividend yield is supported in this vectorized implementation.")
    b = r - q
    sigma_sqrt_T = sigma * np.sqrt(T)

    # Mask for invalid values
    invalid = (S_adj <= 0) | (T <= 0) | (sigma <= 0)

    beta = (0.5 - b / sigma**2) + np.sqrt(((b / sigma**2 - 0.5)**2) + 2 * r / sigma**2)
    beta = np.where(np.abs(beta - 1) < 1e-6, beta + 1e-4, beta)

    # Avoid div-by-zero or negatives in B0/B_inf
    B_inf = beta / np.maximum(beta - 1, 1e-8) * K
    B0 = np.where(np.abs(r - b) > 1e-8, r / (r - b) * K, 1.5 * K)

    h = -(b * T + 2 * sigma_sqrt_T) * (B0 / np.maximum(B_inf - B0, 1e-8))
    I = B0 + (B_inf - B0) * (1 - np.exp(h))
    I = np.clip(I, 1e-6, 1e6)

    # Call value if immediate exercise
    early_ex = S_adj >= I
    call_exercise = S_adj - K

    # BS-2002 logic
    S_adj = np.clip(S_adj, 1e-6, 1e6)
    alpha = np.where(I > 0, (I - K) * I**(-beta), 0.0)

    phi_1 = phi_vectorized(S_adj, T, beta, I, I, r, b, sigma)
    phi_2 = phi_vectorized(S_adj, T, 1, I, I, r, b, sigma)
    phi_3 = phi_vectorized(S_adj, T, 1, K, I, r, b, sigma)
    phi_4 = phi_vectorized(S_adj, T, 0, I, I, r, b, sigma)
    phi_5 = phi_vectorized(S_adj, T, 0, K, I, r, b, sigma)

    call_bs2002 = (
        alpha * S_adj**beta
        - alpha * phi_1
        + phi_2
        - phi_3
        - K * phi_4
        + K * phi_5
    )

    call_price = np.where(early_ex, call_exercise, call_bs2002)
    call_price = np.where(invalid, zero, call_price)

    option_type = np.asarray(option_type)

    # Create masks for calls and puts
    if not np.all(np.isin(option_type, ['c', 'p'])):
        raise ValueError("option_type must be 'c' for call or 'p' for put.")
    call_mask = option_type == "c"
    put_mask = ~call_mask  # anything not "c" is treated as put

    # Initialize output array
    final_price = np.empty_like(call_price)

    # Assign call prices where option is call
    final_price[call_mask] = call_price[call_mask]

    # For puts: parity approximation
    european_put = K * np.exp(-r * T) - S_adj * np.exp(-q * T)
    pc_parity = (
        call_price
        - (S_adj * np.exp(-q * T) - K * np.exp(-r * T))
        + european_put
    )

    final_price[put_mask] = pc_parity[put_mask]

    # Set invalid entries to zero
    final_price = np.where(invalid, zero, final_price)

    return final_price
