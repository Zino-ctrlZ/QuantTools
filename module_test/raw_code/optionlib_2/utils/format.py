import numpy as np
import pandas as pd
from trade.helpers.Logging import setup_logger
logger = setup_logger('trade.optionlib.utils.format')

def assert_equal_length(*args):
    """
    Assert that all input lists have the same length.
    """
    lengths = [len(arg) for arg in args]
    if len(set(lengths)) != 1:
        raise ValueError("All input lists must have the same length.")
    return True

def convert_to_array(*args):
    """
    Convert input lists to numpy arrays.
    """
    return [np.asarray(arg)  for arg in args]

def option_inputs_assert(sigma, K, S0, T, r, q, market_price, flag):


    """
    Check for errors in the input parameters for the vol backout function
    Args:
        sigma (float): Volatility of the underlying asset.
        K (float): Strike price of the option.
        S0 (float): Current spot price of the underlying asset.
        T (float): Time to expiration in years.
        r (float): Risk-free interest rate.
        q (float): Dividend yield.
        market_price (float): Market price of the option.
        flag (str): 'c' for call option, 'p' for put option.
    
    """
    assert isinstance(sigma, (int, float)), f"Recieved '{type(sigma)}' for sigma. Expected 'int' or 'float'"
    assert isinstance(K, (int, float)), f"Recieved '{type(K)}' for K. Expected 'int' or 'float'"
    assert isinstance(S0, (int, float)), f"Recieved '{type(S0)}' for S0. Expected 'int' or 'float'"
    assert isinstance(r, (int, float)), f"Recieved '{type(r)}' for r. Expected 'int' or 'float'"
    assert isinstance(q, (int, float)), f"Recieved '{type(q)}' for q. Expected 'int' or 'float'"
    assert isinstance(market_price, (int, float)), f"Recieved '{type(market_price)}' for market_price. Expected 'int' or 'float'"
    assert isinstance(flag, str), f"Recieved '{type(flag)}' for flag. Expected 'str'"

    if sigma <= 0:
        raise ValueError("Volatility must be positive.")
    if K <= 0:
        raise ValueError("Strike price must be positive.")
    if S0 <= 0:
        raise ValueError("Spot price must be positive.")
    if T < 0:
        raise ValueError("Time to expiration must be positive.")
    if r < 0:
        raise ValueError("Risk-free rate must be non-negative.")
    if q < 0:
        raise ValueError("Dividend yield must be non-negative.")
    if market_price <= 0:
        raise ValueError("Market price must be positive.")
    if flag not in ['c', 'p']:
        raise ValueError("Flag must be 'c' for call or 'p' for put.")
    
    if pd.isna(sigma) or pd.isna(K) or pd.isna(S0) or pd.isna(r) or pd.isna(q) or pd.isna(market_price):
        raise ValueError("Input values cannot be NaN.")