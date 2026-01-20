from typing import List, Union, Literal, Callable, Optional
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from functools import lru_cache  # noqa
from ..pricing.black_scholes import black_scholes_vectorized, black_scholes_vectorized_scalar
from ..pricing.bjs2002 import bjerksund_stensland_2002_vectorized
from ..pricing.binomial import crr_binomial_pricing
from ..config.defaults import BRUTE_FORCE_MAX_ITERATIONS
from trade.helpers.Logging import setup_logger

logger = setup_logger("trade.optionlib.vol.implied_vol")


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
    df = np.exp(-r * T)
    intrinsic_value = df * max(F - K if option_type == "c" else K - F, 0)

    ##TODO: Take this out of objective function to avoid repeated logging during minimization
    if intrinsic_value < market_price:
        logger.warning("Market price exceeds intrinsic value, returning NaN.")
        logger.warning(
            f"Intrinsic Value: {intrinsic_value}, Market Price: {market_price}. Option Details: F={F}, K={K}, T={T}, r={r}, sigma={sigma}, option_type={option_type}"
        )


def bsm_vol_est_minimization(
    F: float,
    K: float,
    T: float,
    r: float,
    market_price: float,
    option_type: str = "c",
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
        bs_price = black_scholes_vectorized(F=F, K=K, T=T, r=r, sigma=sigma, option_type=option_type)
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
    option_type: str = "c",
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
    sigmas = np.linspace(0.001, 5, BRUTE_FORCE_MAX_ITERATIONS)  # Range of volatilities to test

    prices = black_scholes_vectorized_scalar(F=F, K=K, T=T, r=r, sigma=sigmas, option_type=option_type)

    # Calculate the absolute differences between market price and calculated prices
    differences = np.abs(prices - market_price)
    # Find the index of the minimum difference
    min_index = np.argmin(differences)

    # Return the corresponding volatility
    return sigmas[min_index]  # Return the estimated volatility and corresponding price


def vector_vol_estimation(
    brute_callable: Union[Callable, str], *args, list_input: Optional[List[tuple]] = None
) -> List[float]:
    """Vectorized volatility estimation using list comprehension.

    Wrapper function to replicate vectorized behavior by applying a callable to each
    set of parameters. Supports two input modes: individual parameter lists (*args)
    or pre-zipped tuples (list_input keyword argument).

    Args:
        brute_callable: Function to call for volatility estimation. Should accept
            parameters matching those in list_input or *args.
        *args: Individual parameter lists as separate arguments. Each argument should
            be a list/tuple/array of values. These will be transposed into tuples.
        list_input: Optional keyword-only. List of tuples where each tuple contains
            all parameters for one estimation call.
            Example: [(S1, K1, T1, r1, price1, q1, type1), (S2, K2, T2, ...)]

    Returns:
        List of estimated volatilities, one per parameter set.

    Raises:
        ValueError: If both list_input and *args are provided.
        ValueError: If *args elements are not lists, tuples, or arrays.

    Examples:
        >>> # Using *args (recommended for most cases)
        >>> vols = vector_vol_estimation(
        ...     bsm_vol_est_brute_force,
        ...     S_list, K_list, T_list, r_list, market_price_list, option_type_list
        ... )

        >>> # Using list_input keyword argument
        >>> vols = vector_vol_estimation(
        ...     bsm_vol_est_brute_force,
        ...     list_input=[
        ...         (100.0, 100.0, 1.0, 0.05, 10.5, 'c'),
        ...         (105.0, 100.0, 1.0, 0.05, 12.3, 'c'),
        ...     ]
        ... )

    Notes:
        - Cannot use both *args and list_input simultaneously
        - When using *args, all lists must have the same length
        - Empty inputs return empty list
    """
    ## Can either pass list_input or args, but not both
    is_list_input = list_input is not None
    is_args = len(args) > 0

    if is_list_input and is_args:
        raise ValueError("Either provide list_input (keyword-only) or *args, not both.")

    if args:
        for arg in args:
            if not isinstance(arg, (list, tuple, np.ndarray)):
                if isinstance(arg, pd.Series):
                    arg = arg.tolist()
                    continue
                raise ValueError(f"args must be a list, tuple, or numpy array. Recieved {type(arg)}.")
        list_input = list(zip(*args))  # Transpose args to create list of tuples

    if len(list_input) == 0:
        return []
    estimated_vols = [brute_callable(*params) for params in list_input]

    return estimated_vols


def vol_est_brute_force_bjs_2002(
    S: float,
    K: float,
    T: float,
    r: float,
    market_price: float,
    q: float = 0.0,
    option_type: str = "c",
):
    """

    Brute force method to estimate implied volatility by minimizing the difference
    between the market price and the Black-Scholes price.
    Parameters:
    - F: Forward price
    - K: Strike price
    - T: Time to maturity
    - r: Risk-free rate
    - q: Continuous dividend yield
    - market_price: Market price of the option
    - option_type: 'c' for call, 'p' for put
    Returns:
    - Estimated volatility
    """
    #

    sigmas = np.linspace(0.001, 5, BRUTE_FORCE_MAX_ITERATIONS)  # Range of volatilities to test
    S, K, T, r, q, option_type = map(np.asarray, (S, K, T, r, q, option_type))
    prices = bjerksund_stensland_2002_vectorized(
        S=S,
        K=K,
        T=T,
        r=r,
        sigma=sigmas,
        option_type=option_type,
        dividend_type="continuous",  # Assuming continuous dividends for this example
        dividend=q,  # No discrete dividends in this case
    )
    non_na_mask = ~np.isnan(prices) & ~np.isinf(prices)  # Filter out NaN/Inf prices
    prices = prices[non_na_mask]  # Filter prices
    sigmas = sigmas[non_na_mask]  # Filter corresponding sigmas

    # Calculate the absolute differences between market price and calculated prices
    differences = np.abs(prices - market_price)
    # Find the index of the minimum difference
    min_index = np.argmin(differences)

    # Return the corresponding volatility
    return sigmas[min_index]  # Return the estimated volatility and corresponding price


def _k(x, nd=4):
    """
    Helper function to round a number to a specified number of decimal places.
    Parameters:
    - x: Number to round
    - nd: Number of decimal places (default is 4)

    Returns:
    - Rounded number
    """
    return round(x, nd)


# @lru_cache(maxsize=2048)
def _estimate_crr_cached(
    S: float,
    K: float,
    T: float,
    r: float,
    market_price: float,
    q: float = 0.0,
    option_type: str = "c",
    N: int = 1000,
    dividend_type: Literal["continuous", "discrete"] = "continuous",
    american: bool = False,
) -> float:
    """
    Estimate implied volatility using optimization.

    Parameters:
    - S: Spot price
    - K: Strike price
    - T: Time to maturity
    - r: Risk-free interest rate
    - market_price: Market price of the option
    - q: Continuous dividend yield (default is 0.0)
    - option_type: 'c' for call, 'p' for put
    - N: Number of time steps in the binomial tree

    Returns:
    - Estimated volatility
    """

    def binomial_objective_function(sigma: float) -> float:
        calculated_price = crr_binomial_pricing(
            K=K,
            T=T,
            sigma=sigma,  # Initial guess for sigma, will be optimized
            r=r,
            N=N,
            S0=S,
            dividend_type=dividend_type,
            div_yield=q if dividend_type == "continuous" else 0.0,  # Use q for continuous dividends
            dividends=q if dividend_type == "discrete" else [],  # Use q for discrete dividends
            option_type=option_type,
            american=american,
        )
        return (calculated_price - market_price) ** 2

    result = minimize_scalar(
        binomial_objective_function,
        bounds=(0.001, 5.0),  # Reasonable bounds for volatility
        method="bounded",
    )

    return result.x if result.success else None


def estimate_crr_implied_volatility(
    S: float,
    K: float,
    T: float,
    r: float,
    market_price: float,
    q: float = 0.0,
    option_type: str = "c",
    N: int = 1000,
    dividend_type: Literal["continuous", "discrete"] = "continuous",
    american: bool = False,
) -> float:
    """
    Estimate implied volatility using optimization.

    Parameters:
    - S: Spot price
    - K: Strike price
    - T: Time to maturity
    - r: Risk-free interest rate
    - market_price: Market price of the option
    - q: Continuous dividend yield (default is 0.0)
    - option_type: 'c' for call, 'p' for put
    - N: Number of time steps in the binomial tree

    Returns:
    - Estimated volatility
    """

    S = _k(S)
    K = _k(K)
    T = _k(T, nd=6)
    r = _k(r, nd=6)
    market_price = _k(market_price)
    q = _k(q, nd=6) if dividend_type == "continuous" else q
    option_type = option_type
    N = N
    dividend_type = dividend_type
    american = american
    return _estimate_crr_cached(
        S=S,
        K=K,
        T=T,
        r=r,
        market_price=market_price,
        q=q,
        option_type=option_type,
        N=N,
        dividend_type=dividend_type,
        american=american,
    )
