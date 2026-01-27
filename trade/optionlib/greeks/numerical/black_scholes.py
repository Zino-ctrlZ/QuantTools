from datetime import datetime
import numpy as np
from typing import List, Union
from ...core.black_scholes_math import (
    black_scholes_vectorized_base
)
from ...assets.forward import (
    vectorized_forward_continuous,
    vectorized_forward_discrete,
    time_distance_helper,
    vectorized_market_forward_calc
)
from ..numerical.finite_diff import FiniteGreeksEstimator
from trade.helpers.Logging import setup_logger
logger = setup_logger('trade.optionlib.greeks.numerical.black_scholes')



## For numerical Greeks, we can use the patched BSM model to calculate Greeks
## This is because scenario bumps are applied to T & S which also affect the forward price.
def _ptched_bsm_for_numerical(
    K: float,
    T: List[float],
    r: float,
    sigma: float,
    S: float,
    dividend_type: str = 'discrete',
    div_amount: Union[float, List[float]] = 1.0,
    option_type: str = "c",
    **kwargs
):
    
    """
    Patched Black-Scholes model for numerical Greeks.
    This model allows for scenario bumps on T & S which affect the forward price.
    """
    if dividend_type == 'continuous' :
        F = vectorized_forward_continuous(
            S=S,
            r=r,
            q_factor=div_amount,  # Assuming div_amount is the continuous yield rate
            T=T
        )
    elif dividend_type == 'discrete':
        F = vectorized_forward_discrete(
            S=S,
            r=r,
            T=T,
            pv_divs=div_amount  # Assuming div_amount is the present value of discrete dividends
        )
    else:
        raise ValueError(f"Unsupported dividend type '{dividend_type}'. Use 'discrete' or 'continuous'.")
    return black_scholes_vectorized_base(
        F=F, 
        K=K, 
        T=T,
        r=r, 
        sigma=sigma, 
        option_type=option_type
    )



def vectorized_market_greeks_numerical(
        ticks: List[str],
        S: List[float],
        K: List[float],
        valuation_dates: List[datetime],
        end_dates: List[datetime],
        r: List[float],
        sigma: List[float],
        option_type: str|List[str] = "c",
        div_type='discrete',
        div_amount=None
) -> List[dict]:
    """
    Vectorized calculation of Greeks for market options using analytical method.
    """

    ## For analytical greeks, bumps are applied and recalculation is needed.
    ## Therefore, we need to ensure that either div_amount or ticks are provided.
    
    if div_amount is None and ticks is None:
        raise ValueError("div_amount must be provided if ticks are not provided.")

    F, div_amount = vectorized_market_forward_calc(
        ticks=ticks,
        S=S,
        valuation_dates=valuation_dates,
        end_dates=end_dates,
        r=r,
        div_type=div_type,
        return_div=True
    )

    return vectorized_black_scholes_greeks(
        F=F,
        S=S,
        K=K,
        valuation_dates=valuation_dates,
        end_dates=end_dates,
        r=r,
        sigma=sigma,
        option_type=option_type,
        div_type=div_type,
        div_amount=div_amount
    )
    
def vectorized_black_scholes_greeks(
        F: List[str],
        S: List[float],
        K: List[float],
        valuation_dates: List[datetime],
        end_dates: List[datetime],
        r: List[float],
        sigma: List[float],
        option_type: str|List[str] = "c",
        div_type='discrete',
        div_amount=None) -> dict:
    """
    Vectorized Black-Scholes Greeks calculation.
    F: Forward prices (array)
    S: Spot prices (array)
    K: Strike prices (array)
    valuation_dates: List of valuation dates (dates for which the option is priced)
    end_dates: List of end dates (expiration dates of the options)
    r: Risk-free rates (annualized, array)
    sigma: Volatilities (annualized, array)
    option_type: "c" for call, "p" for put (single string or list of strings)
    div_type: Type of dividend ('discrete' or 'continuous')
    div_amount: Dividend amount (single float or list of floats, ignored for continuous dividends)
        if discrete expecting present value of discrete dividends, else if continuous expecting continuous yield rate.
    Returns: Greeks (dictionary)
    """

    T = [time_distance_helper(end_dates[i], valuation_dates[i]) for i in range(len(end_dates))]
    finite_estimator = FiniteGreeksEstimator(
        price_func=_ptched_bsm_for_numerical,
        base_params={
            'F': np.asarray(F),
            'K': np.asarray(K),
            'T': np.asarray(T),
            'r': np.asarray(r),
            'sigma': np.asarray(sigma),
            'q': 0.0,  # Assuming no continuous dividend yield for simplicity
            'S': np.asarray(S),  # Including spot price for delta calculation
            'option_type': option_type,
            'div_type': div_type,
            'div_amount': div_amount  # Placeholder, will be ignored in the patched function
        },
        dx_thresh=0.00001,
        method='central'  # Use backward method for finite differences
    )
    greeks = finite_estimator.all_first_order()
    greeks.update(finite_estimator.all_second_order())
    greeks = dict(sorted(greeks.items(), key=lambda item: item[0]))
    return greeks
