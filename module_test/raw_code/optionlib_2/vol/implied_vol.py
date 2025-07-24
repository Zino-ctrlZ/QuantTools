from typing import List, Union
from scipy.optimize import minimize, minimize_scalar
import numpy as np
from ..pricing.black_scholes import black_scholes_vectorized
from trade.helpers.Logging import setup_logger
logger = setup_logger('trade.optionlib.vol.implied_vol')


def intrinsic_check(F, K, T, r, sigma, market_price, option_type) -> None:
    """
    Check if the intrinsic value of the option is greater than the market price.
    If not, log a warning and return NaN.
    Parameters:
    - F: Forward price
    - K: Strike price
    - T: Time to maturity
    - r: Risk-free rate
    - sigma: Volatility
    - market_price: Market price of the option
    - option_type: 'c' for call, 'p' for put
    Returns:
    - None
    """
    intrinsic_value = max(F - K if option_type == 'c' else K - F, 0)

    ##TODO: Take this out of objective function to avoid repeated logging during minimization
    if intrinsic_value < market_price:
        logger.warning("Market price exceeds intrinsic value, returning NaN.")
        logger.info(f"Intrinsic Value: {intrinsic_value}, Market Price: {market_price}. Option Details: F={F}, K={K}, T={T}, r={r}, sigma={sigma}, option_type={option_type}")


def bsm_vol_est_minimization(
        F: float,
        K: float,
        T: float,
        r: float,
        market_price: float,
        option_type: str = 'c',
):
    """
    Objective function for volatility estimation using minimization.
    This function calculates the difference between the market price and the Black-Scholes price.
    
    Parameters:
    - F: Forward price
    - K: Strike price
    - T: Time to maturity
    - r: Risk-free rate
    - market_price: Market price of the option
    - option_type: 'c' for call, 'p' for put
    
    Returns:
    - Difference between market price and Black-Scholes price
    """
    intrinsic_check(F, K, T, r, 0.2, market_price, option_type)  # Check intrinsic value
    
    def objective_function(sigma):
        bs_price = black_scholes_vectorized(
            F=F, 
            K=K, 
            T=T, 
            r=r, 
            sigma=sigma, 
            option_type=option_type
        )
        return (bs_price - market_price) ** 2

    # Initial guess for volatility
    initial_guess = 0.2

    # Minimize the objective function to find the implied volatility
    result = minimize(objective_function, initial_guess, bounds=[(0.01, None)])
    
    if result.success:
        return result.x[0]  # Return the estimated volatility
    else:
        raise ValueError("Volatility estimation failed.")
    
def bsm_vol_est_brute_force(
        F: float,
        K: float,
        T: float,
        r: float,
        market_price: float,
        option_type: str = 'c',
):
    """

    Brute force method to estimate implied volatility by minimizing the difference
    between the market price and the Black-Scholes price.
    Parameters:
    - F: Forward price
    - K: Strike price
    - T: Time to maturity
    - r: Risk-free rate
    - market_price: Market price of the option
    - option_type: 'c' for call, 'p' for put
    Returns:
    - Estimated volatility
    """
    intrinsic_check(F, K, T, r, 0.2, market_price, option_type)  # Check intrinsic value
    sigmas = np.linspace(0.001, 5, 40000)  # Range of volatilities to test

    prices = black_scholes_vectorized(
        F=F, 
        K=K, 
        T=T, 
        r=r, 
        sigma=sigmas, 
        option_type=option_type
    )

    # Calculate the absolute differences between market price and calculated prices
    differences = np.abs(prices - market_price)
    # Find the index of the minimum difference
    min_index = np.argmin(differences)

    # Return the corresponding volatility
    return sigmas[min_index]  # Return the estimated volatility and corresponding price


def vector_vol_estimation(brute_callable: Union[callable, str],
                       list_input: List[tuple]) -> List[float]:
    """
    Vectorized vol estimation method to estimate implied volatility by minimizing the difference
    between the market price and the Black-Scholes price.
    
    Parameters:
    - brute_callable: Function to call for brute force estimation
    - list_input: List of inputs for the brute force estimation
        eg: [
        (S1, K1, T1, r1, market_price1, q1, option_type1),
        ]
    
    Returns:
    - Estimated volatilities as a numpy array
    """
    if len(list_input) == 0:
        return []
    estimated_vols = [brute_callable(*params) for params in list_input]

    
    return estimated_vols