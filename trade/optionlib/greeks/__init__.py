from datetime import datetime
from typing import List, Union
from .numerical.black_scholes import vectorized_market_greeks_numerical
from .analytical.black_scholes import _ptched_bsm_for_analytical
from ..assets.forward import time_distance_helper
from trade.helpers.Logging import setup_logger
logger = setup_logger('trade.optionlib.greeks.__init__')

def vectorized_market_greeks_bsm(
        ticks: List[str],
        S: List[float],
        K: List[float],
        valuation_dates: List[datetime],
        end_dates: List[datetime],
        r: List[float],
        sigma: List[float],
        option_type: str|List[str] = "c",
        div_type='discrete',
        greek_style: str = 'analytic'
) -> List[dict]:
    """
    Vectorized calculation of Greeks for market options.
    ticks: List of ticker symbols
    S: List of spot prices (current prices of the underlying assets)
    K: List of strike prices
    valuation_dates: List of valuation dates (dates for which the option is priced)
    end_dates: List of end dates (expiration dates of the options)
    r: List of risk-free rates (annualized)
    sigma: List of volatilities (annualized)
    option_type: "c" for call, "p" for put (single string or list of strings)
    div_type: Type of dividend ('discrete' or 'continuous')
    greek_style: 'analytic' or 'numerical' for Greek calculation style
    
    Returns: Dictionary of Greeks for each option
    """
    # Ensure option_type is a list
    if isinstance(option_type, str):
        option_type = [option_type] * len(ticks)
    elif len(option_type) != len(ticks):
        raise ValueError("option_type must be a single string or a list of strings with the same length as ticks.")
    
    # Convert valuation_dates and end_dates to Timedelta
    T = [time_distance_helper(end_dates[i], valuation_dates[i]) for i in range(len(end_dates))]

    # Calculate the Greeks using the specified style
    if greek_style == 'analytic':
        greeks = _ptched_bsm_for_analytical(
            ticks=ticks,
            S=S,
            K=K,
            valuation_dates=valuation_dates,
            end_dates=end_dates,
            r=r,
            sigma=sigma,
            option_type=option_type,
            div_type=div_type
        )
    elif greek_style == 'numerical':
        greeks=vectorized_market_greeks_numerical(
            ticks=ticks,
            S=S,
            K=K,
            valuation_dates=valuation_dates,
            end_dates=end_dates,
            r=r,
            sigma=sigma,
            option_type=option_type,
            div_type=div_type
        )
    else:
        raise ValueError(f"Unknown Greek calculation style '{greek_style}'. Use 'analytic' or 'numerical'.")
    greeks = dict(sorted(greeks.items(), key=lambda item: item[0]))
    return greeks

