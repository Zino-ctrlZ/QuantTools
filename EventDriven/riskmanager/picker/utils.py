import pandas as pd
import numpy as np
from typing import Union
from trade.helpers.Logging import setup_logger

logger = setup_logger("EventDriven.riskmanager.picker.utils")


def _verify_delta_in_chain(chain: pd.DataFrame) -> pd.DataFrame:
    """
    Verify if the 'delta' column exists in the option chain DataFrame. If it does not exist, add it with NaN values.
    This ensures that downstream functions that expect a 'delta' column can operate without errors.
    Args:
        chain (pd.DataFrame): The option chain DataFrame to verify.
    Returns:
        pd.DataFrame: The option chain DataFrame with a 'delta' column (either existing or newly added with NaN values).
    """
    if "delta" not in chain.columns:
        logger.warning("Delta column not found in option chain. Adding 'delta' column with NaN values.")
        ## Opting for np.inf instead of NaN to avoid potential issues with NaN values in calculations later on. This way, if delta is not available, it will be treated as infinitely far from any reasonable delta limit, effectively excluding those contracts if delta filtering is applied.
        chain["delta"] = np.inf
    return chain


def _delta_lmt(f: float) -> Union[float, np.float64]:
    """
    Utility function to convert a delta limit percentage (e.g., 0.10 for 10%) into an absolute delta limit value.
    This is based on the assumption that the delta of an option ranges from -1 to 1, where -1 represents a deep ITM put, 0 represents an ATM option, and 1 represents a deep ITM call.
    Args:
        f (float): The delta limit percentage (e.g., 0.10 for 10%).
    Returns:
        Union[float, np.float64]: The absolute delta limit value corresponding to the provided percentage.
    """
    if f is None:
        return np.inf  # If no delta limit percentage is provided, we set it to infinity to not filter based on delta.

    if isinstance(f, str):
        try:
            f = float(f)
        except ValueError as e:
            raise ValueError(f"Invalid delta limit percentage string: {f}. Error: {e}") from e

    if not isinstance(f, (float, int)):
        raise ValueError(f"Delta limit percentage must be a float or int. Received type {type(f)} with value: {f}")

    if np.isnan(f):
        return np.inf  # If no delta limit percentage is provided, we set it to infinity to not filter based on delta.

    return np.float64(f)


def _build_common_pair_mask(
    spread_mid: pd.Series,
    spread_bid: pd.Series,
    spread_ask: pd.Series,
    spread_pct_ratio: pd.Series,
    spread_oi: pd.Series,
    spread_delta: pd.Series,
    min_total_price: float,
    max_total_price: float,
    max_pct_width: float,
    min_oi: int,
    delta_lmt: Union[float, np.float64],
) -> pd.Series:
    """Build a shared mask used by pairers for spread contract filtering."""
    mid_mask = spread_mid.between(min_total_price, max_total_price)
    spread_oi_mask = spread_oi >= min_oi
    pct_width_mask = spread_pct_ratio <= max_pct_width
    spread_bid_mask = spread_bid > 0
    spread_ask_mask = spread_ask > 0
    delta_mask = spread_delta.abs() <= abs(delta_lmt)
    return mid_mask & spread_oi_mask & pct_width_mask & spread_bid_mask & spread_ask_mask & delta_mask
