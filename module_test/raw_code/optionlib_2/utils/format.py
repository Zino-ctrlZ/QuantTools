from datetime import datetime
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
        return False
    return True

def convert_to_array_individual(value):
    if isinstance(value, (list, np.ndarray, pd.Series)):
        return np.array(value)
    elif isinstance(value, (int, float, str, datetime, np.datetime64)):
        return np.array([value], dtype=object)  # Use dtype=object to handle mixed types
    else:
        raise ValueError(f"Unsupported type for value conversion : {type(value)}")


def convert_to_array(*args):
    """
    Convert input(s) to numpy arrays. If only one input is given, return it as a single array, not a list.
    """
    arrays = [np.asarray(arg) for arg in args]
    return arrays[0] if len(arrays) == 1 else arrays


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
    



def to_1d_array(x):
    x = np.atleast_1d(x)
    if x.ndim > 1:
        return x.flatten()
    return x

def equalize_lengths(*args):
    """
    Ensure all inputs have the same length if an arg is a list of size 1 or a single value.
    """
    max_size = max(len(arg) if isinstance(arg, (list, np.ndarray)) else 1 for arg in args)
    if max_size == 0:
        raise ValueError("All input arrays are empty.")

    return [np.full(max_size, arg) if isinstance(arg, (int, float, str, datetime)) or len(arg) == 1 else np.asarray(arg) for arg in args]
    
