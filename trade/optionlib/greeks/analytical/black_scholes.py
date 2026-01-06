from datetime import datetime
import numpy as np
from typing import List, Union
from scipy.stats import norm
from ...config.defaults import DAILY_BASIS
from ...core.black_scholes_math import (
    black_scholes_analytic_greeks_vectorized,
)
from ...assets.forward import (
    time_distance_helper,
    vectorized_market_forward_calc
)
from trade.helpers.Logging import setup_logger
logger = setup_logger('trade.optionlib.greeks.analytical.black_scholes')


## To seperate the F calculation from Market Greeks Funciton
def _ptched_bsm_for_analytical(
    ticks: List[str],
    S: List[float],
    K: List[float],
    valuation_dates: List[datetime],
    end_dates: List[datetime],
    r: List[float],
    sigma: List[float],
    option_type: str|List[str] = "c",
    div_type='discrete',
    **kwargs
):
    
    F = vectorized_market_forward_calc(
        ticks=ticks,
        S=S,
        valuation_dates=valuation_dates,
        end_dates=end_dates,
        r=r,
        div_type=div_type
    )
    T = [time_distance_helper(end_dates[i], valuation_dates[i]) for i in range(len(end_dates))]
    greeks = black_scholes_analytic_greeks_vectorized(
        F=F, 
        K=K, 
        T=T, 
        r=r, 
        sigma=sigma, 
        option_type=option_type
    )
    greeks = dict(sorted(greeks.items(), key=lambda item: item[0]))
    return greeks