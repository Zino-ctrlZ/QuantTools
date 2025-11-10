from typing import List, Union
from ...utils.format import (
    convert_to_array,
)
from .finite_diff import FiniteGreeksEstimator
from ...pricing.bjs2002 import bjerksund_stensland_2002_vectorized
from trade.helpers.Logging import setup_logger
logger = setup_logger('trade.optionlib.greeks.numerical.bjs2002')

def bjs2002_numerical_greeks(
    K: float,
    T: List[float],
    r: float,
    sigma: float,
    S: float,
    div_yield: Union[float, List[float]] = 1.0,
    option_type: str = "c",
    **kwargs
):
    
    option_type = list(map(lambda x: x.lower(), option_type)) 
    K, T, r, sigma, S, div_yield, option_type = map(
        convert_to_array, (K, T, r, sigma, S, div_yield, option_type)
    )
     # Ensure option_type is lowercase
    finite_estimator = FiniteGreeksEstimator(
        price_func = bjerksund_stensland_2002_vectorized,
        base_params = {
            'K': K,
            'T': T,
            'r': r,
            'sigma': sigma,
            'S': S,
            'div_amount': div_yield,
            'option_type': option_type,
            'q': None
        },
        dx_thresh = 0.00001,
        method = 'backward',
    )
    greeks = finite_estimator.all_first_order()
    greeks.update(finite_estimator.all_second_order())
    greeks = dict(sorted(greeks.items(), key=lambda item: item[0]))
    del finite_estimator
    return greeks