"""
Utility functions for SSVI model operations.
"""
from typing import Tuple, Literal

from datetime import datetime, date
import numpy as np
import pandas as pd
from dbase.DataAPI.ThetaData import extract_numeric_value
from trade import get_pricing_config
from trade.helpers.helper import assert_member_of_enum
from module_test.raw_code.optionlib_2.config.defaults import DAILY_BASIS
from module_test.raw_code.optionlib_2.vol.ssvi.types import DivType, VolType, VolSide


def chain_cache_key(root: str,
                    valuation_date: str|datetime,
                    div_type: DivType,
                    vol_type: VolType) -> str:
    """
    Generates a unique key for caching the option chain based on root, valuation date, dividend type, and volatility type.
    
    Args:
        root (str): The root symbol of the underlying asset.
        valuation_date (str or datetime): The date of valuation.
        div_type (DivType): Type of dividends considered.
        vol_type (VolType): Type of volatility used for calibration.
        
    Returns:
        str: A unique key for caching the option chain.
    """
    div_type = assert_member_of_enum(div_type, enum_class=DivType)
    vol_type = assert_member_of_enum(vol_type, enum_class=VolType)
    val_date = pd.to_datetime(valuation_date).strftime('%Y-%m-%d')
    return f"{root}_{val_date}_{div_type.value}_{vol_type.value}"


def params_cache_key(
        root: str,
        valuation_date: str|datetime,
        div_type: DivType,
        vol_type: VolType,
        side: VolSide,
) -> str:
    """
    Generates a unique key for caching the params from SSVI fitting based on root, valuation, div type, vol type & vol side
    Args:
        root (str): The root symbol of the underlying asset.
        valuation_date (str or datetime): The date of valuation.
        div_type (DivType): Type of dividends considered.
        vol_type (VolType): Type of volatility used for calibration.
        side (VolSide): Type of side. otm, call, puts
        
    Returns:
        str: A unique key for caching the params chain.
    """
    div_type = assert_member_of_enum(div_type, enum_class=DivType)
    vol_type = assert_member_of_enum(vol_type, enum_class=VolType)
    side = assert_member_of_enum(side, enum_class=VolSide)
    val_date = pd.to_datetime(valuation_date).strftime('%Y-%m-%d')
    return f"{root}_{val_date}_{div_type.value}_{vol_type.value}_{side.value}"

def assert_k_bounds_model_range(k: list | np.ndarray,
                                f: float | np.ndarray) -> None:
    """
    Asserts that the strikes are within the bounds defined by PRICING_CONFIG.
    Args:
        k (list or np.ndarray): List of strikes.
    Raises:
        ValueError: If any strike is not within the configured bounds.
    """
    conf = get_pricing_config()
    k = np.array(k, dtype=float)
    max_m = f * conf['VOL_SURFACE_MAX_MONEYNESS_THRESHOLD']
    min_m = f * conf['VOL_SURFACE_MIN_MONEYNESS_THRESHOLD']
    within = np.all((k <= max_m) & (k >= min_m))
    if not within:
        raise ValueError(
            (
                f"Strikes {k} are out of bounds. "
                f"Must be between {min_m} and {max_m}."
            )
        )

def handle_strikes(
    k: list| np.ndarray,
    f: list| float, 
    strike_type: Literal['p', 'k', 'pf', 'f'],
    spot: float = None
) -> np.ndarray:
    """
    Convert strikes based on the specified strike type.
    Since SSVI model takes strikes values as absolute values, this function converts the strikes
    
    Args:
        k (list or np.ndarray): List of strikes.
        f (list or float): Forward price.
        strike_type (str): Type of strike ('p', 'k', 'pf', 'f').
    
    Types available:
        - 'p': Percent of spot eg: 1.0 == ATM
        - 'k': Strike to fwd_grid: if spot = 100, k=100=ATM
        - 'pf': Percent of fwd_grid/forward price eg: 1.0 == ATMF
        - 'f': Log moneyness to fwd_grid: 0.0 == ATMF
        
    Returns:
        np.ndarray: Converted strikes.
    """
    k = np.array(k, dtype=float)
    if strike_type == 'p': ## Percent of spot to fwd_grid
        if spot is None:
            raise ValueError("Spot price must be provided for 'p' strike type.")
        
        strikes= k * spot
    elif strike_type == 'k': ## Strike to fwd_grid
        strikes= k
    elif strike_type == 'pf': ## Percent of fwd_grid/forward price
        strikes= k * f
    elif strike_type == 'f': ## Log moneyness to fwd_grid
        ## Convert log moneyness to strikes
        strikes= np.exp(k) * f

    else:
        raise ValueError(f"Invalid strike type: {strike_type}")
    assert_k_bounds_model_range(strikes, f)
    return strikes



def calculate_normalized_rmse_loss(
        market_iv: np.ndarray,
        model_iv: np.ndarray,
        moneyness: np.ndarray,
) -> Tuple[float, float, float]:
    
    """
    Calculate the normalized loss between market and model implied volatilities.
    
    Args:
        market_iv (np.ndarray): Market implied volatilities.
        model_iv (np.ndarray): Model implied volatilities.
        moneyness (np.ndarray): Moneyness values.
        
    Returns:
        Tuple[float, float, float]: Normalized median loss, right wing loss, left wing loss.
    """
    
    ## Normalized Median
    median_iv = np.median(market_iv)
    normalized_median_loss = np.sqrt(np.mean((market_iv - model_iv)**2)) / median_iv

    ## Right Wing Loss (> 1.0)
    right_wing_mask = moneyness > 1.0
    median_right_wing_iv = np.median(market_iv[right_wing_mask])
    right_wing_loss = np.sqrt(np.mean(
        (market_iv[right_wing_mask] - model_iv[right_wing_mask]) **2)) / median_right_wing_iv

    ## Left Wing Loss (< 1.0)
    left_wing_mask = moneyness < 1.0
    median_left_wing_iv = np.median(market_iv[left_wing_mask])
    left_wing_loss = np.sqrt(np.mean(
        (market_iv[left_wing_mask] - model_iv[left_wing_mask])**2)) / median_left_wing_iv

    return normalized_median_loss, right_wing_loss, left_wing_loss


def calculate_normalized_mae_loss(
        market_iv: np.ndarray,
        model_iv: np.ndarray,
        moneyness: np.ndarray
) -> Tuple[float, float, float]:
    """
    Calculate the normalized mean absolute error (MAE) loss between market and model implied volatilities.
    
    Args:
        market_iv (np.ndarray): Market implied volatilities.
        model_iv (np.ndarray): Model implied volatilities.
        moneyness (np.ndarray): Moneyness values.
        
    Returns:
        Tuple[float, float, float]: Normalized median MAE loss, right wing MAE loss, left wing MAE loss.
    """
    
    ## Normalized Median
    median_iv = np.median(market_iv)
    normalized_median_mae_loss = np.mean(np.abs(market_iv - model_iv)) / median_iv

    ## Right Wing Loss (> 1.0)
    right_wing_mask = moneyness > 1.0
    median_right_wing_iv = np.median(market_iv[right_wing_mask])
    right_wing_mae_loss = np.mean(
        np.abs(market_iv[right_wing_mask] - model_iv[right_wing_mask])) / median_right_wing_iv

    ## Left Wing Loss (< 1.0)
    left_wing_mask = moneyness < 1.0
    median_left_wing_iv = np.median(market_iv[left_wing_mask])
    left_wing_mae_loss = np.mean(
        np.abs(market_iv[left_wing_mask] - model_iv[left_wing_mask])) / median_left_wing_iv

    return normalized_median_mae_loss, right_wing_mae_loss, left_wing_mae_loss



def identify_length_for_model(string, integer) -> int:
    """
    
    Identify the length of the timeframe in minutes based on the string and integer provided.
    Parameters
    
    ----------
    string : str
        The string representing the timeframe (e.g., 'm', 'd', 'w', 'y').
    integer : int
    The integer representing the number of units for the timeframe.
    Returns
    -------
    int
        The length of the timeframe in minutes.
    
    """

    TIMEFRAMES_VALUES = {'d': 1, 'w': 7, 'm': 30, 'y': DAILY_BASIS}
    assert string in TIMEFRAMES_VALUES.keys(), (
        f"Available timeframes are {list(TIMEFRAMES_VALUES.keys())}, "
        f'received "{string}"'
    )
    return integer * TIMEFRAMES_VALUES[string]

def convert_date_to_time_to_maturity(dt: str,
                                     valuation_date: str = None) -> float:
    """
    Converts a date to time to maturity in years.
    
    Args:
        dt (datetime): The date to convert.
            example: '3m', '2025-08-08', 1
        
    Returns:
        float: Time to maturity in years.
    """

    ## If dt is a string, check if it is a date or a duration
    if isinstance(dt, (str, pd.Timestamp, datetime, date)):
        try:
            # Try to parse as a date first
            dt = pd.to_datetime(dt)
            assert valuation_date is not None, "valuation_date must be provided if dt is a date string"
            valuation_date = pd.to_datetime(valuation_date)
            dt = (dt - valuation_date).days
        except ValueError as e:
            # If it fails, assume it's a duration
            dt = identify_length_for_model(*extract_numeric_value(dt))
            if dt is None:
                raise ValueError(f"Invalid date or duration format: {dt}") from e
    elif isinstance(dt, (float,int)):
        # If dt is a number, assume it's a duration in days
        dt = float(dt)
    elif isinstance(dt, pd.Timedelta):
        # If dt is a timedelta, convert it to days
        dt = dt.days

    else:
        raise ValueError(
            "Unsupported type for dt: "
            f"{type(dt)}. Expected str, int, float, datetime, or pd.Timedelta."
        )

    assert_dt_within_range(dt)
    return dt/DAILY_BASIS

def assert_dt_within_range(dt: float):
    """
    Asserts that the time to maturity is within the range defined by PRICING_CONFIG (from trade.__init__).
    
    Args:
        dt (float): The time to maturity in years.
        
    Raises:
        ValueError: If dt is not within the configured range.
    """
    config = get_pricing_config()
    min_dte = config['VOL_SURFACE_MIN_DTE_THRESHOLD']
    max_dte = config['VOL_SURFACE_MAX_DTE_THRESHOLD']
    if not (min_dte <= dt <= max_dte):
        raise ValueError(
            (
                f"Time to maturity {dt} is out of bounds. "
                f"Must be between {min_dte} and {max_dte}."
            )
        )


