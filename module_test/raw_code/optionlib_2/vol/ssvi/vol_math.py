"""
Module for SSVI volatility modeling and calibration.
Includes Black-Scholes pricing, SSVI total variance calculations,
and optimization routines for fitting SSVI parameters to market data.
Author: Chiemelie Nwanisobi
Date: 2025-10-01
"""
from typing import List, Tuple, Callable
import math
import pandas as pd
import numpy as np
from trade.helpers.pools import runProcesses

# -------------------------------------------------
# 1. Black-Scholes Call price
# -------------------------------------------------
def normal_cdf(x: float|np.ndarray) -> float|np.ndarray:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))

def bs_call_price(spot: float|np.ndarray, 
                  strike: float|np.ndarray, 
                  maturity: float|np.ndarray, 
                  rate: float|np.ndarray, 
                  vol: float|np.ndarray) -> float|np.ndarray:
    """Black-Scholes European call."""
    if vol <= 0 or maturity <= 0:
        return max(0.0, spot - strike)
    d1 = (math.log(spot / strike) + (rate + 0.5 * vol**2) * maturity) / (vol * math.sqrt(maturity))
    d2 = d1 - vol * math.sqrt(maturity)
    return (spot * normal_cdf(d1) -
            strike * math.exp(-rate * maturity) * normal_cdf(d2))

# -------------------------------------------------
# 2. SSVI helpers
# -------------------------------------------------
def atm_total_variance(t: float|np.ndarray, 
                       var0: float|np.ndarray, 
                       var_inf: float|np.ndarray, 
                       kappa: float|np.ndarray) -> float|np.ndarray:
    """
    θ(t)  = ((var0 - var_inf)*(1 - e^{-κ t})/(κ t) + var_inf) * t
    """
    return ((var0 - var_inf) * (1 - np.exp(-kappa * t))
            / (kappa * t) + var_inf) * t

def skew_phi(theta_t: float|np.ndarray, 
             eta: float|np.ndarray, 
             lam: float|np.ndarray) -> float | np.ndarray:
    """Skew function."""
    return eta * theta_t ** lam

def ssvi_total_variance(log_moneyness: float|np.ndarray, 
                        theta_t: float|np.ndarray, 
                        eta: float|np.ndarray, 
                        lam: float|np.ndarray, 
                        rho: float|np.ndarray) -> float|np.ndarray:
    phi_val = skew_phi(theta_t, eta, lam)
    term1   = rho * phi_val * log_moneyness
    term2   = np.sqrt((phi_val * log_moneyness + rho)**2 + 1 - rho**2)
    return 0.5 * theta_t * (1 + term1 + term2)

def ssvi_implied_vol(fwd: float|np.ndarray, strike: float|np.ndarray, maturity: float|np.ndarray,
                     var0: float|np.ndarray, var_inf: float|np.ndarray, kappa: float|np.ndarray,
                     eta: float|np.ndarray, lam: float|np.ndarray, rho: float|np.ndarray) -> float|np.ndarray:

    """Return implied volatility by SSVI."""
    k = np.log(strike / fwd)
    theta_t = atm_total_variance(maturity, var0, var_inf, kappa)
    total_var = ssvi_total_variance(k, theta_t, eta, lam, rho)
    return np.sqrt(total_var / maturity)

def make_candidate(bounds: List[Tuple[float, float]], iterations) -> np.ndarray:
    """
    Generate a random candidate solution within the given bounds.
    bounds: list of (low, high) for each dimension
    """
    rng = np.random.default_rng(42)
    low  = np.array([b[0] for b in bounds])
    high = np.array([b[1] for b in bounds])

    # (iterations, d) matrix of uniform random samples
    candidates = low + (high - low) * rng.random((iterations, len(bounds)))
    return candidates


def random_search_vec(objective_multi: Callable[[np.ndarray], np.ndarray],
                      bounds: List[Tuple[float, float]],
                      iterations: int = 40_000) -> Tuple[np.ndarray, float]:
    """
    Vectorised random search.
    objective_multi: accepts an (N, d) array -> returns (N,) array of losses
    bounds         : list of (low, high) for each dimension
    iterations     : how many random draws
    """

    # vectorised loss evaluation -> (iterations,)
    candidates = make_candidate(bounds, iterations)
    _losses = objective_multi(candidates)
    best_idx = np.argmin(_losses)
    return candidates[best_idx], _losses[best_idx]


def atm_loss_multi(x: np.ndarray, 
                   t: np.ndarray, 
                   iv_atm: np.ndarray) -> np.ndarray:
    """
    x : (N, 3)  – rows = [var0, var_inf, kappa]
    t, iv_atm   – market ATM maturities and vols (1-D)
    returns     – loss for each row  (shape (N,))
    """
    var0, var_inf, kappa = x[:, 0], x[:, 1], x[:, 2]
    theta_t  = atm_total_variance(t[:, None], var0, var_inf, kappa)  # broadcast
    model_iv = np.sqrt(theta_t / t[:, None])
    mse      = ((model_iv - iv_atm[:, None])**2).mean(axis=0)        # → (N,)

    # guard against NaN / huge vols
    invalid   = (np.isinf(mse)) | (np.isnan(mse))
    mse = np.where(invalid, 1e4, mse)  # penalise
    return mse

def surface_loss_multi(params_mat: np.ndarray, k_grid: np.ndarray, t_grid: np.ndarray, fwd: float|np.ndarray,
                       var0_hat: float|np.ndarray, var_inf_hat: float|np.ndarray, kappa_hat: float|np.ndarray,
                       market_iv: float|np.ndarray, weights: float|np.ndarray=None) -> float|np.ndarray:
    """
    params_mat : (N,3) rows [eta, lambda, rho]
    returns    : (N,)   weighted MSE per candidate
    """
    eta, lam, rho = params_mat.T
    M = k_grid.shape[0]

    # normalize weights -> (M,)
    if weights is None:
        weights = np.ones(M, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
        if weights.ndim != 1 or weights.shape[0] != M:
            raise ValueError(f"weights must be shape ({M},), got {weights.shape}")

    bad = (eta <= 0) | (lam <= -0.9) | (lam >= 1.0) | (np.abs(rho) >= 0.999)
    safe_eta = np.where(bad, 1.0, eta)
    safe_lam = np.where(bad, 0.0, lam)
    safe_rho = np.where(bad, 0.0, rho)

    k = np.log(k_grid / fwd)[:, None]     # (M,1)
    T = t_grid[:, None]                   # (M,1)
    theta = atm_total_variance(T, var0_hat, var_inf_hat, kappa_hat)

    total_var = ssvi_total_variance(
        k, theta, safe_eta[None, :], safe_lam[None, :], safe_rho[None, :]
    )                                      # (M,N)

    iv_model = np.sqrt(total_var / T)      # (M,N)
    invalid = (~np.isfinite(iv_model)) | (iv_model > 5)
    iv_model = np.where(invalid, 1e4, iv_model)

    sqerr = (iv_model - market_iv[:, None]) ** 2   # (M,N)

    # weighted mean over M → shape (N,)
    wmse = np.average(sqerr, axis=0, weights=weights)

    # slam bad candidates
    wmse = np.where(bad, 1e9, wmse)
    return wmse


def _loss_chunk_with_idx(idx: int,
                         params_chunk: np.ndarray,
                         k_grid: np.ndarray, t_grid: np.ndarray, fwd: float,
                         var0_hat: float, var_inf_hat: float, kappa_hat: float,
                         market_iv: float) -> tuple[int, np.ndarray]:
    # Call your original function on a chunk
    mse = surface_loss_multi(params_chunk,
                             k_grid, t_grid, fwd,
                             var0_hat, var_inf_hat, kappa_hat,
                             market_iv)
    return idx, mse  # keep index so we can reassemble in order


def surface_loss_multi_parallel(params_mat: np.ndarray,
                                k_grid: np.ndarray, t_grid: np.ndarray, fwd: float,
                                var0_hat: float, var_inf_hat: float, kappa_hat: float,
                                market_iv: float,
                                *,
                                chunk_size: int = 1024,
                                run_type: str = 'imap'):
    """
    Parallel wrapper around surface_loss_multi using runProcesses.
    params_mat: (N,3) -> returns (N,)
    No globals; constants are passed to each worker.
    """
    N = int(params_mat.shape[0])
    if N == 0:
        return np.empty((0,), dtype=float)

    # 1) Make chunks
    chunks = [params_mat[i:min(i+chunk_size, N)]
              for i in range(0, N, chunk_size)]
    idxs   = list(range(len(chunks)))
    n      = len(chunks)

    # 2) Build ordered_inputs for your runProcesses(func, [args1, args2, ...])
    ordered_inputs = [
        idxs,
        chunks,
        [k_grid]      * n,
        [t_grid]      * n,
        [fwd]         * n,
        [var0_hat]    * n,
        [var_inf_hat] * n,
        [kappa_hat]   * n,
        [market_iv]   * n,
    ]

    # 3) Fan out
    results = runProcesses(_loss_chunk_with_idx, ordered_inputs, run_type=run_type)

    # 4) Materialize depending on run_type
    if run_type == 'amap':              # async ordered
        results = results.get()
    elif run_type in ('imap', 'uimap'): # iterator / unordered
        results = list(results)

    # 5) Reassemble in original order of rows
    results.sort(key=lambda x: x[0])                 # by chunk index
    mse_chunks = [m for _, m in results]
    return np.concatenate(mse_chunks, axis=0)

def get_atm_vol(chain: pd.DataFrame,
                log_moneyness_col_name: str='log_moneyness',
                vol_col_name: str='crr_vol_discrete') -> pd.Series:
    """
    Finds the ATM volatility for a given expiration in the chain.
    Args:
        x (pd.DataFrame): The option chain DataFrame for a specific expiration.
    Returns:
        float: The ATM volatility for the given expiration.
    """
    def finder(x):
        min_l_m= abs(x[log_moneyness_col_name]).min()
        return x[x[log_moneyness_col_name].abs() == min_l_m][vol_col_name].values[0]
    return chain.groupby('expiration').apply(finder).values

def get_atm_t(chain: pd.DataFrame,
               log_moneyness_col_name: str='log_moneyness',
               t_col_name: str='t') -> pd.Series:
    """
    Finds the ATM time to expiration for a given expiration in the chain.
    
    Args:
        chain (pd.DataFrame): The option chain DataFrame for a specific expiration.
        
    Returns:
        pd.Series: The ATM time to expiration for the given expiration.
    """
    def finder(x):
        min_l_m= abs(x[log_moneyness_col_name]).min()
        return x[x[log_moneyness_col_name].abs() == min_l_m][t_col_name].values[0]
    return chain.groupby('expiration').apply(finder).values


def get_best_params(t_atm: List[float],
                    iv_atm: List[float]) -> Tuple[np.ndarray, float]:
    """
    Find the best parameters for the ATM term structure.
    Returns:
        var0_hat, var_inf_hat, kappa_hat
    """
    bounds = [(1e-4, 0.2), # var0: Min ATM Variance across DTE
              (1e-4, 0.2), # var_inf_hat: Max ATM Variance across DTE
              (0.05, 3.0)] # kappa: Speed from var0 to var_inf_hat
    def _atm_objective(X: np.ndarray) -> np.ndarray:
        return atm_loss_multi(X, np.asarray(t_atm), np.asarray(iv_atm))

    best_params, best_loss = random_search_vec(
        _atm_objective,
        bounds,
        iterations=3000,
    )
    return best_params, best_loss


def get_surface_params(
    k_grid: np.ndarray,
    t_grid: np.ndarray,
    fwd_grid: float,
    var0_hat: float,
    var_inf_hat: float,
    kappa_hat: float,
    market_iv_grid: np.ndarray,
    iterations: int = 50_000,
    chunk_size: int = None
) -> Tuple[float, float, float, float]:
    """
    Estimate the SSVI surface parameters (eta, lambda, rho) using random search.
    Args:
        k_grid (np.ndarray): The strike prices.
        t_grid (np.ndarray): The maturities.
        fwd_grid (float): The forward price.
        var0_hat (float): Estimated initial variance.
        var_inf_hat (float): Estimated long-term variance.
        kappa_hat (float): Estimated speed of mean reversion.
        market_iv_grid (np.ndarray): Market implied volatilities.
        iterations (int): Number of random search iterations.
        chunk_size (int): Size of chunks for parallel processing.
    Returns:
        Tuple[float, float, float, float]: Estimated parameters (eta, lambda, rho) and best loss.
    """
    if chunk_size is None:
        chunk_size = int(iterations / 8)

    # 1 tighter parameter bounds v1
    surf_bounds = [(0.05, 1.5),      # eta
                   (-0.8, 0.8),      # lambda
                   (-0.95, 0.95)]    # rho


    def surface_lambda(x: np.ndarray) -> np.ndarray:
        return surface_loss_multi_parallel(
            x,
            k_grid=k_grid,
            t_grid=t_grid,
            fwd=fwd_grid,
            var0_hat=var0_hat,
            var_inf_hat=var_inf_hat,
            kappa_hat=kappa_hat,
            market_iv=market_iv_grid,
            chunk_size=chunk_size,
        )

    (eta_hat, lambda_hat, rho_hat), best_loss = random_search_vec(
        surface_lambda,
        surf_bounds,
        iterations,
    )

    return eta_hat, lambda_hat, rho_hat, best_loss

