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
        't_plus_n': numbers.Number,
        'traded_symbols': list,
        'portfolio_rebalance_freq': str,
        'weight_recalculation_freq': str,
        'weights_last_refresh': (str, date, datetime),
        'rm_series_start': (str, date, datetime),
        'rm_series_end': (str, date, datetime),
        'official_start_date': (str, date, datetime),
        'weights': dict,
        'option_settings': dict,
        'strat_name': str,
        'open_missed_signals': bool,
        'executor_level': numbers.Number,
        'strat_slug': str,
    }

    must_in_opt_settings = {
        'order_settings': dict,
        'rm_settings': dict,
        'portfolio_settings': dict,
        'sizer_settings': dict,
    }

    must_in_order_settings = {
        'target_dte': numbers.Number,
        'strategy': str,
        'structure_direction': str,
        'spread_ticks': numbers.Number,
        'dte_tolerance': numbers.Number,
        'min_moneyness': numbers.Number,
        'max_moneyness': numbers.Number,
        'min_total_price': numbers.Number,
    }

    must_in_rm_settings = {
        'sizing_lev': numbers.Number,
        'limits_enabled': list,
        'max_moneyness': numbers.Number,
        'otm_moneyness_width': numbers.Number,
        'itm_moneyness_width': numbers.Number,
        're_update_on_roll': bool,
        'add_skip_column_window': numbers.Number,
        'add_skip_column_threshold': numbers.Number,
        'price_on': str,
        'option_price': str,
        'max_tries': numbers.Number,
        'max_slippage': numbers.Number,
        'sizer_type': str,
        'quit_on_max_factor': bool,
        'quit_on_max_tries': bool,
    }

    must_in_portfolio_settings = {
        'roll_map': numbers.Number,
        'weights_haircut': numbers.Number,
        'max_cash_map': dict,
        't_plus_n': numbers.Number,
        'min_acceptable_dte_threshold': numbers.Number,
    }

    # Function to validate keys and types
    def validate_keys(config_section, must_keys):
        for key, expected_type in must_keys.items():
            if key not in config_section:
                raise KeyError(f"Missing required key: {key}")
            if not isinstance(config_section[key], expected_type):
                raise TypeError(f"Key '{key}' must be of type {expected_type.__name__}, got {type(config_section[key]).__name__}")


    ## Check for missing keys in config
    validate_keys(config, must_keys)

    ## Check for missing keys in option_settings
    validate_keys(config['option_settings'], must_in_opt_settings)

    ## Check for missing keys in order_settings
    validate_keys(config['option_settings']['order_settings'], must_in_order_settings)

    ## Check for missing keys in rm_settings
    validate_keys(config['option_settings']['rm_settings'], must_in_rm_settings)

    ## Check for missing keys in portfolio_settings
    validate_keys(config['option_settings']['portfolio_settings'], must_in_portfolio_settings)

    ## Traded Symbols
    assert 'traded_symbols' in config, "traded_symbols is missing in config"
    assert isinstance(config['traded_symbols'], list), "traded_symbols should be a list"

    ## Portfolio Rebalance Freq
    assert 'portfolio_rebalance_freq' in config, "portfolio_rebalance_freq is missing in config"
    assert config['portfolio_rebalance_freq'] in ['1d', '1w', '1m', '1y'], "portfolio_rebalance_freq should be one of ['1d', '1w', '1m', '1y']"

    ## weight recalculation freq
    assert 'weight_recalculation_freq' in config, "weight_recalculation_freq is missing in config"
    assert config['weight_recalculation_freq'] in ['1d', '1w', '1m', '1y'], "weight_recalculation_freq should be one of ['1d', '1w', '1m', '1y']"

    ## weights_last_refresh
    assert 'weights_last_refresh' in config, "weights_last_refresh is missing in config"


AVOID_OPTTICKS = {
    'AAPL': ['AAPL20241220C225', 'AAPL20241220C220', ],
    'TSLA': ['TSLA20230120C1250', 'TSLA20230120C1275']
    }


FFWD_OPT_BY_UNDERLIER = {
    'TSLA': ['2022-01-05']
}


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

def ffwd_data(data:pd.DataFrame, underlier:str) -> pd.DataFrame:
    """Fill Forward data for a given underlier and date."""
    if underlier not in FFWD_OPT_BY_UNDERLIER:
        return data
    fwd_dates = FFWD_OPT_BY_UNDERLIER[underlier]
    for dt in fwd_dates:
        if dt not in data.index:
            continue
        data.loc[data.index == dt,:] = np.nan
        data.fillna(method='ffill', inplace=True)
    return data
    