from typing import Callable, Dict, Union
import numpy as np
from ...config.defaults import DAILY_BASIS
from trade.helpers.Logging import setup_logger
logger = setup_logger('trade.optionlib.greeks.numerical.finite_diff')



# ----------------------
# Vectorized Finite Differences (First Order)
# ----------------------
def finite_diff_first_order_vec(x: str, price_func, params: dict, dx_thresh=0.001, method="forward"):
    if x == 'T':
        dx = 1 / DAILY_BASIS
    else:
        dx = (params[x]) * dx_thresh  # Ensure dx is float

    def to_float_copy(val):
        if isinstance(val, (int, float, np.integer, np.floating)):
            return float(val)
        # elif isinstance(val, np.ndarray):
        #     try:
        #         return val.astype(np.float64)
        #     except ValueError:
        #         return val
        return val

    # Convert all values to float-compatible copies
    p0 = {k: to_float_copy(v) for k, v in params.items()}
    p1 = {k: to_float_copy(v) for k, v in params.items()}
    p2 = {k: to_float_copy(v) for k, v in params.items()}

    if method == "forward":
        p1[x] = p1[x] + dx
        return (price_func(**p1) - price_func(**p0)) / dx
    elif method == "backward":
        p1[x] = p1[x] - dx
        return (price_func(**p0) - price_func(**p1)) / dx
    elif method == "central":
        p1[x] = p0[x] + dx
        p2[x] = p0[x] - dx
        return (price_func(**p1) - price_func(**p2)) / (2 * dx)

    else:
        raise ValueError("Unknown method. Expected central, forward or backward")


# ----------------------
# Vectorized Finite Differences (Second Order)
# ----------------------
def finite_diff_second_order_vec(x: str, price_func, params: dict, dx_thresh=0.001, method="central"):
    if x == 'T':
        dx = 1 / DAILY_BASIS
    else:
        # dx = float(params[x]) * dx_thresh  # Ensure dx is float # NOSONAR
        dx = (params[x]) * dx_thresh  # Ensure dx is float

    def to_float_copy(val):
        if isinstance(val, (int, float, np.integer, np.floating)):
            return float(val)
        # elif isinstance(val, np.ndarray):
        #     try:
        #         return val.astype(np.float64)
        #     except ValueError:
        #         return val
        return val

    # Convert all values to float-compatible copies
    p0 = {k: to_float_copy(v) for k, v in params.items()}
    p1 = {k: to_float_copy(v) for k, v in params.items()}
    p2 = {k: to_float_copy(v) for k, v in params.items()}


    if method == "central":
        p1[x] = p0[x] + dx
        p2[x] = p0[x] - dx
        return (price_func(**p1) - 2 * price_func(**p0) + price_func(**p2)) / dx**2

    elif method == "forward":
        p1[x] = p1[x] + dx
        p2[x] = p2[x] + 2 * dx
        return (price_func(**p2) - 2 * price_func(**p1) + price_func(**p0)) / dx**2
    elif method == "backward":
        p1[x] = p1[x] - dx
        p2[x] = p2[x] - 2 * dx
        return (price_func(**p0) - 2 * price_func(**p1) + price_func(**p2)) / dx**2
    else:
        raise ValueError("Unknown method. Expected central, forward or backward")


class FiniteGreeksEstimator:
    def __init__(self,
                 price_func: Callable,
                 base_params: Dict[str, Union[float, int]],
                 dx_thresh: float = 0.001,
                 method: str = 'central'):
        """
        Estimate Greeks using finite difference methods.

        Parameters:
        - price_func: Callable pricing function accepting kwargs.
        - base_params: Dictionary with keys like 'S', 'K', 'T', 'r', 'sigma', 'q'.
        - dx_thresh: Relative step size.
        - method: 'forward', 'backward', or 'central'.
        """
        self.price_func = price_func
        self.params = base_params.copy()
        self.dx_thresh = dx_thresh
        self.method = method.lower()
        self._validate_params()

    def _validate_params(self):
        required = {'S', 'K', 'T', 'r', 'sigma', 'q'}
        missing = required - set(self.params.keys())
        if missing:
            raise ValueError(f"Missing required keys: {missing}")
        if self.method not in {'forward', 'backward', 'central'}:
            raise ValueError(f"Invalid method '{self.method}'. Must be 'forward', 'backward', or 'central'.")

    def _step(self, x: str) -> float:
        val = self.params[x]
        return max(self.dx_thresh * abs(val), 1e-6)

    def first_order(self, x: str) -> float:
        return finite_diff_first_order_vec(x, self.price_func, self.params, dx_thresh=self.dx_thresh, method=self.method)

    def second_order(self, x: str) -> float:
        return finite_diff_second_order_vec(x, self.price_func, self.params, dx_thresh=self.dx_thresh, method=self.method)

    def all_first_order(self) -> Dict[str, float]:
        
        return {
            'delta': self.first_order('S'),
            'vega': self.first_order('sigma') * 0.01,
            'theta': -self.first_order('T')/DAILY_BASIS,
            'rho': self.first_order('r')*0.01
        }

    def all_second_order(self) -> Dict[str, float]:
        return {
            'gamma': self.second_order('S'),
            'volga': self.second_order('sigma') * 0.0001,
        }
