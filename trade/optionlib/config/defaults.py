"""Optionlib shared configuration defaults loaded from YAML.

Exposes module-level constants for non-bump defaults, plus getters that read the
live ``config`` mapping (via ``get_config_value``). Greek bump sizes are
getter-only (no module snapshots); use ``resolve_greek_bump`` for policy.
"""

from typing import Any, Dict, Mapping, Union

from trade.helpers.Logging import setup_logger
from dbase.DataAPI.ThetaData.utils import THETA_FIRST_ALLOW_DATE
from . import load_config
from .ssvi.controller import get_global_config
from .types import GreekBumpMode, GreekFactor

logger = setup_logger('trade.optionlib.config.defaults')

config = load_config()
SSVI_GLOBAL_CONFIG = get_global_config()

## Normalize caller spellings (params keys, tree attrs) onto GreekFactor
_GREEK_FACTOR_ALIASES: Dict[str, GreekFactor] = {
    "s": GreekFactor.SPOT,
    "s0": GreekFactor.SPOT,
    "spot": GreekFactor.SPOT,
    "sigma": GreekFactor.VOL,
    "vol": GreekFactor.VOL,
    "iv": GreekFactor.VOL,
    "volatility": GreekFactor.VOL,
    "r": GreekFactor.RATE,
    "rate": GreekFactor.RATE,
    "rf": GreekFactor.RATE,
    "t": GreekFactor.TIME,
    "tau": GreekFactor.TIME,
    "time": GreekFactor.TIME,
    "expiry": GreekFactor.TIME,
}

_BUMP_MODE_BY_FACTOR: Mapping[GreekFactor, GreekBumpMode] = {
    GreekFactor.SPOT: GreekBumpMode.MULTIPLICATIVE,
    GreekFactor.VOL: GreekBumpMode.ADDITIVE,
    GreekFactor.RATE: GreekBumpMode.ADDITIVE,
    GreekFactor.TIME: GreekBumpMode.THETA,
}


def get_config_value(key: str) -> Any:
    """Return a value from the loaded optionlib YAML config by key.

    Args:
        key: Exact config key (e.g. ``MULTIPLICATIVE_BUMP_SIZE``, ``DAILY_BASIS``).

    Returns:
        The value stored under ``key`` in ``config``.

    Raises:
        TypeError: If ``config`` is not a mapping.
        KeyError: If ``key`` is absent; message lists known keys.
    """
    if not isinstance(config, dict):
        raise TypeError(
            f"optionlib config is not a dict (got {type(config).__name__}); "
            f"cannot look up key {key!r}"
        )
    if key not in config:
        known = ", ".join(sorted(str(k) for k in config.keys())) or "(none)"
        raise KeyError(
            f"optionlib config missing required key {key!r}. "
            f"Known keys: {known}"
        )
    return config[key]


def get_daily_basis() -> float:
    """Return day-count basis used for year fractions (config ``DAILY_BASIS``)."""
    return float(get_config_value("DAILY_BASIS"))


def get_dividend_lookback_years() -> int:
    """Return dividend forecast lookback years (``DIVIDEND_FORECAST_LOOKBACK_YEARS``)."""
    return int(get_config_value("DIVIDEND_FORECAST_LOOKBACK_YEARS"))


def get_dividend_lookforward_years() -> int:
    """Return dividend forecast lookforward years (``DIVIDEND_FORECAST_LOOKFORWARD_YEARS``)."""
    return int(get_config_value("DIVIDEND_FORECAST_LOOKFORWARD_YEARS"))


def get_dividend_forecast_method() -> str:
    """Return dividend forecast method name (config ``DIVIDEND_FORECAST_METHOD``)."""
    return str(get_config_value("DIVIDEND_FORECAST_METHOD"))


def get_option_timeseries_start_date() -> str:
    """Return option timeseries start date from config (``OPTION_TIMESERIES_START_DATE``)."""
    return str(get_config_value("OPTION_TIMESERIES_START_DATE"))


def get_vol_est_upper_bound() -> float:
    """Return implied-vol estimation upper bound (config ``VOL_EST_UPPER_BOUND``)."""
    return float(get_config_value("VOL_EST_UPPER_BOUND"))


def get_vol_est_lower_bound() -> float:
    """Return implied-vol estimation lower bound (config ``VOL_EST_LOWER_BOUND``)."""
    return float(get_config_value("VOL_EST_LOWER_BOUND"))


def get_n_precision_greeks() -> int:
    """Return binomial tree steps for precision greeks (config ``N_PRECISION_GREEKS``)."""
    return int(get_config_value("N_PRECISION_GREEKS"))


def get_multiplicative_bump_size() -> float:
    """Return relative bump fraction for spot (config ``MULTIPLICATIVE_BUMP_SIZE``).

    Used as ``dx = |S| * get_multiplicative_bump_size()``.
    """
    return float(get_config_value("MULTIPLICATIVE_BUMP_SIZE"))


def get_additive_bump_size() -> float:
    """Return absolute bump for vol/rate (config ``ADDITIVE_BUMP_SIZE``).

    Used as ``dx = get_additive_bump_size()`` on decimal ``sigma`` / ``r``.
    """
    return float(get_config_value("ADDITIVE_BUMP_SIZE"))


def get_theta_bump_size() -> float:
    """Return absolute time bump in years (config ``THETA_BUMP_SIZE``)."""
    return float(get_config_value("THETA_BUMP_SIZE"))


def get_brute_force_max_iterations() -> int:
    """Return max iterations for brute-force IV search (``BRUTE_FORCE_MAX_ITERATIONS``)."""
    return int(get_config_value("BRUTE_FORCE_MAX_ITERATIONS"))


def normalize_greek_factor(factor: Union[str, GreekFactor]) -> GreekFactor:
    """Map a caller factor name or enum onto canonical ``GreekFactor``.

    Args:
        factor: ``GreekFactor`` or alias string (e.g. ``S``, ``spot``, ``sigma``, ``vol``).

    Returns:
        Canonical ``GreekFactor``.

    Raises:
        ValueError: If ``factor`` is not a known alias or enum member.
    """
    if isinstance(factor, GreekFactor):
        return factor
    key = str(factor).strip()
    ## Exact enum values ("S", "sigma", …) first, then case-folded aliases
    try:
        return GreekFactor(key)
    except ValueError:
        pass
    alias = _GREEK_FACTOR_ALIASES.get(key.lower())
    if alias is None:
        known = ", ".join(sorted({*GreekFactor._value2member_map_, *_GREEK_FACTOR_ALIASES}))
        raise ValueError(
            f"unknown greek factor {factor!r}; expected one of: {known}"
        )
    return alias


def resolve_greek_bump(factor: Union[str, GreekFactor], value: Any) -> Any:
    """Resolve the absolute greek finite-difference step for a pricing factor.

    Normalizes ``factor`` via ``normalize_greek_factor``, then applies the bump
    mode for that factor (multiplicative spot, additive vol/rate, dedicated theta).

    Args:
        factor: Canonical name, alias, or ``GreekFactor``.
        value: Current factor level (scalar or array-like for spot).

    Returns:
        Absolute step ``dx`` in the same shape as ``value`` for multiplicative
        bumps, otherwise a float.

    Raises:
        ValueError: If ``factor`` cannot be normalized to a known greek factor.
    """
    canonical = normalize_greek_factor(factor)
    mode = _BUMP_MODE_BY_FACTOR[canonical]
    if mode is GreekBumpMode.MULTIPLICATIVE:
        return abs(value) * get_multiplicative_bump_size()
    if mode is GreekBumpMode.ADDITIVE:
        return get_additive_bump_size()
    if mode is GreekBumpMode.THETA:
        return get_theta_bump_size()
    raise ValueError(f"no bump size getter for mode {mode!r} (factor {canonical!r})")


## Import-time snapshots for non-bump call sites / default args
DAILY_BASIS = get_daily_basis()
DIVIDEND_LOOKBACK_YEARS = get_dividend_lookback_years()
DIVIDEND_LOOKFORWARD_YEARS = get_dividend_lookforward_years()
DIVIDEND_FORECAST_METHOD = get_dividend_forecast_method()
## Live code uses Theta allow-date, not the YAML OPTION_TIMESERIES_START_DATE value
OPTION_TIMESERIES_START_DATE = THETA_FIRST_ALLOW_DATE
VOL_EST_UPPER_BOUND = get_vol_est_upper_bound()
VOL_EST_LOWER_BOUND = get_vol_est_lower_bound()
N_PRECISION_GREEKS = get_n_precision_greeks()
BRUTE_FORCE_MAX_ITERATIONS = get_brute_force_max_iterations()
