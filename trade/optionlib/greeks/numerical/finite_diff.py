"""General finite-difference primitives and greek-estimation wrapper.

Comment density: orchestration

General ``finite_diff_*_vec`` helpers take an absolute step ``dx`` and do not
read bump config. ``FiniteGreeksEstimator`` resolves per-factor bumps via
``resolve_greek_bump`` (multiplicative spot, additive vol/rate, dedicated theta)
then calls those primitives.
"""

from typing import Any, Callable, Dict, Union

import numpy as np

from ...config.defaults import DAILY_BASIS, resolve_greek_bump
from trade.helpers.Logging import setup_logger

logger = setup_logger('trade.optionlib.greeks.numerical.finite_diff')

_MIN_ABS_BUMP = 1e-12


def _to_float_copy(val: Any) -> Any:
    """Copy scalars to float; leave non-numeric values unchanged."""
    if isinstance(val, (int, float, np.integer, np.floating)):
        return float(val)
    return val


def _floor_abs_bump(dx: Any) -> Any:
    """Ensure bump magnitude is at least ``_MIN_ABS_BUMP`` (scalar or array)."""
    arr = np.asarray(dx, dtype=float)
    floored = np.maximum(np.abs(arr), _MIN_ABS_BUMP)
    if floored.shape == ():
        return float(floored)
    return floored


# ----------------------
# Vectorized Finite Differences (First Order) — absolute dx
# ----------------------
def finite_diff_first_order_vec(
    x: str,
    price_func: Callable,
    params: dict,
    dx: Any,
    method: str = "forward",
) -> Any:
    """First-order finite difference of ``price_func`` w.r.t. factor ``x``.

    Args:
        x: Parameter key to bump.
        price_func: Callable accepting ``params`` as kwargs.
        params: Base parameter dict.
        dx: Absolute step size (scalar or array matching ``params[x]``).
        method: ``forward``, ``backward``, or ``central``.

    Returns:
        Approximate ∂V/∂x.
    """
    dx = _floor_abs_bump(dx)

    p0 = {k: _to_float_copy(v) for k, v in params.items()}
    p1 = {k: _to_float_copy(v) for k, v in params.items()}
    p2 = {k: _to_float_copy(v) for k, v in params.items()}

    if method == "forward":
        p1[x] = p1[x] + dx
        return (price_func(**p1) - price_func(**p0)) / dx
    if method == "backward":
        p1[x] = p1[x] - dx
        return (price_func(**p0) - price_func(**p1)) / dx
    if method == "central":
        p1[x] = p0[x] + dx
        p2[x] = p0[x] - dx
        return (price_func(**p1) - price_func(**p2)) / (2 * dx)
    raise ValueError("Unknown method. Expected central, forward or backward")


# ----------------------
# Vectorized Finite Differences (Second Order) — absolute dx
# ----------------------
def finite_diff_second_order_vec(
    x: str,
    price_func: Callable,
    params: dict,
    dx: Any,
    method: str = "central",
) -> Any:
    """Second-order finite difference of ``price_func`` w.r.t. factor ``x``.

    Args:
        x: Parameter key to bump.
        price_func: Callable accepting ``params`` as kwargs.
        params: Base parameter dict.
        dx: Absolute step size.
        method: ``forward``, ``backward``, or ``central``.

    Returns:
        Approximate ∂²V/∂x².
    """
    dx = _floor_abs_bump(dx)

    p0 = {k: _to_float_copy(v) for k, v in params.items()}
    p1 = {k: _to_float_copy(v) for k, v in params.items()}
    p2 = {k: _to_float_copy(v) for k, v in params.items()}

    if method == "central":
        p1[x] = p0[x] + dx
        p2[x] = p0[x] - dx
        return (price_func(**p1) - 2 * price_func(**p0) + price_func(**p2)) / dx**2
    if method == "forward":
        p1[x] = p1[x] + dx
        p2[x] = p2[x] + 2 * dx
        return (price_func(**p2) - 2 * price_func(**p1) + price_func(**p0)) / dx**2
    if method == "backward":
        p1[x] = p1[x] - dx
        p2[x] = p2[x] - 2 * dx
        return (price_func(**p0) - 2 * price_func(**p1) + price_func(**p2)) / dx**2
    raise ValueError("Unknown method. Expected central, forward or backward")


def finite_diff_mixed_second_order_vec(
    x: str,
    y: str,
    price_func: Callable,
    params: dict,
    dx: Any,
    dy: Any,
) -> Any:
    """Mixed second-order finite difference ∂²V/(∂x ∂y) with absolute steps.

    Args:
        x: First parameter key.
        y: Second parameter key.
        price_func: Callable accepting ``params`` as kwargs.
        params: Base parameter dict.
        dx: Absolute step for ``x``.
        dy: Absolute step for ``y``.

    Returns:
        Approximate ∂²V/(∂x ∂y).
    """
    ## Floor separately so vectorized S/sigma arrays stay valid
    dx = _floor_abs_bump(dx)
    dy = _floor_abs_bump(dy)

    p_pp = params.copy()
    p_pm = params.copy()
    p_mp = params.copy()
    p_mm = params.copy()

    p_pp[x] = np.asarray(p_pp[x], dtype=float) + dx
    p_pp[y] = np.asarray(p_pp[y], dtype=float) + dy

    p_pm[x] = np.asarray(p_pm[x], dtype=float) + dx
    p_pm[y] = np.asarray(p_pm[y], dtype=float) - dy

    p_mp[x] = np.asarray(p_mp[x], dtype=float) - dx
    p_mp[y] = np.asarray(p_mp[y], dtype=float) + dy

    p_mm[x] = np.asarray(p_mm[x], dtype=float) - dx
    p_mm[y] = np.asarray(p_mm[y], dtype=float) - dy

    return (
        price_func(**p_pp)
        - price_func(**p_pm)
        - price_func(**p_mp)
        + price_func(**p_mm)
    ) / (4 * dx * dy)


class FiniteGreeksEstimator:
    """Estimate option greeks via finite differences with config bump policy.

    Resolves absolute steps per factor through ``resolve_greek_bump``, then
    delegates to the general FD primitives.
    """

    def __init__(
        self,
        price_func: Callable,
        base_params: Dict[str, Union[float, int]],
        method: str = "central",
    ):
        """
        Args:
            price_func: Callable pricing function accepting kwargs.
            base_params: Dictionary with keys like 'S', 'K', 'T', 'r', 'sigma', 'q'.
            method: 'forward', 'backward', or 'central'.
        """
        self.price_func = price_func
        self.params = base_params.copy()
        self.method = method.lower()
        self._validate_params()

    def _validate_params(self) -> None:
        """Require pricing keys and a known difference method."""
        required = {'S', 'K', 'T', 'r', 'sigma', 'q'}
        missing = required - set(self.params.keys())
        if missing:
            raise ValueError(f"Missing required keys: {missing}")
        if self.method not in {'forward', 'backward', 'central'}:
            raise ValueError(
                f"Invalid method '{self.method}'. Must be 'forward', 'backward', or 'central'."
            )

    def _bump(self, factor: str) -> Any:
        """Absolute FD step for ``factor`` from greek bump policy."""
        return resolve_greek_bump(factor, self.params[factor])

    def first_order(self, x: str) -> Any:
        """First-order greek w.r.t. factor ``x``."""
        return finite_diff_first_order_vec(
            x, self.price_func, self.params, dx=self._bump(x), method=self.method
        )

    def second_order(self, x: str) -> Any:
        """Second-order greek w.r.t. factor ``x``."""
        return finite_diff_second_order_vec(
            x, self.price_func, self.params, dx=self._bump(x), method=self.method
        )

    def mixed_second_order(self, x: str, y: str) -> Any:
        """Mixed second-order greek w.r.t. factors ``x`` and ``y``."""
        return finite_diff_mixed_second_order_vec(
            x,
            y,
            self.price_func,
            self.params,
            dx=self._bump(x),
            dy=self._bump(y),
        )

    def vanna(self) -> Any:
        """Cross derivative ∂²V/(∂S ∂σ) before Price-Sensitivity Unit scaling."""
        return self.mixed_second_order("S", "sigma")

    def all_first_order(self) -> Dict[str, Any]:
        """First-order greeks in Price-Sensitivity Units."""
        return {
            'delta': self.first_order('S'),
            'vega': self.first_order('sigma') * 0.01,
            'theta': -self.first_order('T') / DAILY_BASIS,
            'rho': self.first_order('r') * 0.01,
        }

    def all_second_order(self) -> Dict[str, Any]:
        """Second-order greeks in Price-Sensitivity Units."""
        ## Match analytic / binomial: volga /100**2, vanna /100
        return {
            'gamma': self.second_order('S'),
            'volga': self.second_order('sigma') * (0.01 ** 2),
            'vanna': self.vanna() * 0.01,
        }
