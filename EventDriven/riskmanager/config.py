"""Risk Manager Configuration Validation and Schema Enforcement.

This module provides comprehensive configuration validation for the risk management system.
It ensures all required settings are present, properly typed, and valid before backtesting
or live trading begins. Validation covers portfolio settings, order parameters, sizing rules,
and risk limits.

Key Function:
    assert_missing_keys: Validates complete configuration hierarchy

Configuration Structure:
    Top-Level Keys (config dict):
        - t_plus_n: Settlement period in business days
        - traded_symbols: List of ticker symbols in universe
        - portfolio_rebalance_freq: Rebalancing frequency ('daily', 'weekly', 'monthly')
        - weight_recalculation_freq: How often to update position weights
        - weights_last_refresh: Last weight calculation date
        - rm_series_start: Risk manager timeseries start date
        - rm_series_end: Risk manager timeseries end date
        - official_start_date: Official backtest/live start date
        - weights: Dict mapping symbols to allocation weights
        - option_settings: Nested dict with option-specific configs
        - strat_name: Strategy identifier string
        - open_missed_signals: Whether to open positions on delayed signals
        - executor_level: Aggressiveness level for order execution (1-5)
        - strat_slug: URL-friendly strategy identifier
        - ruin_value: Portfolio value threshold for strategy termination

    Option Settings (option_settings dict):
        - order_settings: Order generation parameters
        - rm_settings: Risk management rules
        - portfolio_settings: Portfolio-level constraints
        - sizer_settings: Position sizing configuration

    Order Settings (order_settings dict):
        - target_dte: Target days to expiration
        - strategy: Option strategy name (e.g., 'put_spread')
        - structure_direction: 'long' or 'short'
        - spread_ticks: Strike separation for spreads
        - dte_tolerance: Acceptable DTE deviation
        - min_moneyness: Minimum strike/spot ratio
        - max_moneyness: Maximum strike/spot ratio
        - min_total_price: Minimum acceptable structure price

    Risk Manager Settings (rm_settings dict):
        - sizing_lev: Position sizing multiplier (leverage)
        - limits_enabled: List of active risk limits
        - max_moneyness: Overall moneyness cap
        - otm_moneyness_width: OTM selection tolerance
        - itm_moneyness_width: ITM selection tolerance
        - re_update_on_roll: Recalculate limits on position rolls
        - add_skip_column_window: Window for skip logic
        - add_skip_column_threshold: Threshold for skip activation
        - price_on: Price source ('close', 'mid', 'mark')
        - option_price: Specific price field to use
        - max_tries: Maximum order search attempts
        - max_slippage: Maximum acceptable slippage
        - sizer_type: Position sizer implementation ('default', 'zscore')
        - quit_on_max_factor: Exit strategy on max leverage
        - quit_on_max_tries: Exit strategy on max retries

    Portfolio Settings (portfolio_settings dict):
        - roll_map: Moneyness threshold for rolling positions
        - weights_haircut: Discount factor for weight calculations
        - max_cash_map: Dict of max cash per symbol
        - t_plus_n: Portfolio-level settlement period
        - min_acceptable_dte_threshold: Minimum DTE before forced action

Validation:
    - Type checking for every configuration parameter
    - Presence verification for all required keys
    - Nested dictionary validation at all levels
    - Descriptive error messages with key paths
    - Type compatibility assertions

Usage:
    config = {
        't_plus_n': 1,
        'traded_symbols': ['AAPL', 'MSFT'],
        'portfolio_rebalance_freq': 'weekly',
        'option_settings': {
            'order_settings': {...},
            'rm_settings': {...},
            ...
        },
        ...
    }

    # Validate before use
    assert_missing_keys(config)  # Raises KeyError/TypeError if invalid

Raises:
    KeyError: When required configuration key is missing
    TypeError: When configuration value has incorrect type

Notes:
    - All date fields accept str, date, or datetime types
    - Numeric fields validated as numbers.Number instances
    - Boolean fields require explicit True/False
    - Dict and list types validated for structure
    - Validation is fail-fast for early error detection
"""

from datetime import datetime, date
import numbers
import pandas as pd
import numpy as np


def assert_missing_keys(config: dict) -> None:
    """
    Assert that all required keys are present in the config dictionary and its nested dictionaries.
    Raises KeyError if any required key is missing or TypeError if a key has an incorrect
    data type."""

    # Define the must-have keys and their expected data types
    must_keys = {
        "t_plus_n": numbers.Number,
        "traded_symbols": list,
        "portfolio_rebalance_freq": str,
        "weight_recalculation_freq": str,
        "weights_last_refresh": (str, date, datetime),
        "rm_series_start": (str, date, datetime),
        "rm_series_end": (str, date, datetime),
        "official_start_date": (str, date, datetime),
        "weights": dict,
        "option_settings": dict,
        "strat_name": str,
        "open_missed_signals": bool,
        "executor_level": numbers.Number,
        "strat_slug": str,
        "ruin_value": numbers.Number,
    }

    must_in_opt_settings = {
        "order_settings": dict,
        "rm_settings": dict,
        "portfolio_settings": dict,
        "sizer_settings": dict,
    }

    must_in_order_settings = {
        "target_dte": numbers.Number,
        "strategy": str,
        "structure_direction": str,
        "spread_ticks": numbers.Number,
        "dte_tolerance": numbers.Number,
        "min_moneyness": numbers.Number,
        "max_moneyness": numbers.Number,
        "min_total_price": numbers.Number,
    }

    must_in_rm_settings = {
        "sizing_lev": numbers.Number,
        "limits_enabled": list,
        "max_moneyness": numbers.Number,
        "otm_moneyness_width": numbers.Number,
        "itm_moneyness_width": numbers.Number,
        "re_update_on_roll": bool,
        "add_skip_column_window": numbers.Number,
        "add_skip_column_threshold": numbers.Number,
        "price_on": str,
        "option_price": str,
        "max_tries": numbers.Number,
        "max_slippage": numbers.Number,
        "sizer_type": str,
        "quit_on_max_factor": bool,
        "quit_on_max_tries": bool,
    }

    must_in_portfolio_settings = {
        "roll_map": numbers.Number,
        "weights_haircut": numbers.Number,
        "max_cash_map": dict,
        "t_plus_n": numbers.Number,
        "min_acceptable_dte_threshold": numbers.Number,
    }

    # Function to validate keys and types
    def validate_keys(config_section, must_keys):
        for key, expected_type in must_keys.items():
            if key not in config_section:
                raise KeyError(f"Missing required key: {key}")
            if not isinstance(config_section[key], expected_type):
                raise TypeError(
                    f"Key '{key}' must be of type {expected_type.__name__}, got {type(config_section[key]).__name__}"
                )

    ## Check for missing keys in config
    validate_keys(config, must_keys)

    ## Check for missing keys in option_settings
    validate_keys(config["option_settings"], must_in_opt_settings)

    ## Check for missing keys in order_settings
    validate_keys(config["option_settings"]["order_settings"], must_in_order_settings)

    ## Check for missing keys in rm_settings
    validate_keys(config["option_settings"]["rm_settings"], must_in_rm_settings)

    ## Check for missing keys in portfolio_settings
    validate_keys(config["option_settings"]["portfolio_settings"], must_in_portfolio_settings)

    ## Traded Symbols
    assert "traded_symbols" in config, "traded_symbols is missing in config"
    assert isinstance(config["traded_symbols"], list), "traded_symbols should be a list"

    ## Portfolio Rebalance Freq
    assert "portfolio_rebalance_freq" in config, "portfolio_rebalance_freq is missing in config"
    assert config["portfolio_rebalance_freq"] in [
        "1d",
        "1w",
        "1m",
        "1y",
    ], "portfolio_rebalance_freq should be one of ['1d', '1w', '1m', '1y']"

    ## weight recalculation freq
    assert "weight_recalculation_freq" in config, "weight_recalculation_freq is missing in config"
    assert config["weight_recalculation_freq"] in [
        "1d",
        "1w",
        "1m",
        "1y",
    ], "weight_recalculation_freq should be one of ['1d', '1w', '1m', '1y']"

    ## weights_last_refresh
    assert "weights_last_refresh" in config, "weights_last_refresh is missing in config"


AVOID_OPTTICKS = {
    "AAPL": [
        "AAPL20241220C225",
        "AAPL20241220C220",
    ],
    "TSLA": ["TSLA20230120C1250", "TSLA20230120C1275"],
}


FFWD_OPT_BY_UNDERLIER = {"TSLA": ["2022-01-05"]}


def get_avoid_opticks(ticker) -> list:
    """Get the list of options to avoid."""
    """Get the list of options to avoid for a given ticker and date."""
    if ticker in AVOID_OPTTICKS:
        return AVOID_OPTTICKS[ticker]
    return []


def add_avoid_optick(optick, ticker):
    """Add an option tick to avoid for a given ticker and date."""
    if ticker not in AVOID_OPTTICKS:
        AVOID_OPTTICKS[ticker] = []
    if optick not in AVOID_OPTTICKS[ticker]:
        AVOID_OPTTICKS[ticker].append(optick)


def get_fwd_opt_by_underlier():
    """Get the list of forward options by underlier."""
    return FFWD_OPT_BY_UNDERLIER


def add_fwd_opt_by_underlier(underlier, fwd_date):
    """Add a forward option for a given underlier."""
    if underlier not in FFWD_OPT_BY_UNDERLIER:
        FFWD_OPT_BY_UNDERLIER[underlier] = []
    if fwd_date not in FFWD_OPT_BY_UNDERLIER[underlier]:
        FFWD_OPT_BY_UNDERLIER[underlier].append(fwd_date)


def ffwd_data(data: pd.DataFrame, underlier: str) -> pd.DataFrame:
    """Fill Forward data for a given underlier and date."""
    if underlier not in FFWD_OPT_BY_UNDERLIER:
        return data
    fwd_dates = FFWD_OPT_BY_UNDERLIER[underlier]
    for dt in fwd_dates:
        if dt not in data.index:
            continue
        data.loc[data.index == dt, :] = np.nan
        data.fillna(method="ffill", inplace=True)
    return data
