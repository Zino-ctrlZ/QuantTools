
CONFIG_DEFINITIONS = {
    # 'Config class: { config_name}': 'Description of what this config class is for and how to use it.'
    "ChainConfig": {
        "max_pct_width": "Maximum abs spread/mid price percentage width for an option to be included in the option chain.",
        "min_oi": "Minimum open interest required for an option to be included in the option chain.",
    },
    "OrderSchemaConfigs": {
        "target_dte": "Target days to expiration for the options in the order schema.",
        "strategy": "The options strategy to be used (e.g., vertical, iron condor).",
        "structure_direction": "Direction of the structure, either long or short.",
        "spread_ticks": "Number of strike price ticks between legs of the spread.",
        "dte_tolerance": "Allowed deviation in days to expiration from the target DTE.",
        "min_moneyness": "Minimum moneyness level for selecting options.",
        "max_moneyness": "Maximum moneyness level for selecting options.",
        "min_total_price": "Minimum total price for the option structure.",
    },
    "OrderPicker": {
        "start_date": "The start date for selecting orders.",
        "end_date": "The end date for selecting orders.",
    },
    "DefaultSizerConfigs": {
        "sizing_lev": "Leverage level to be used for sizing positions.",
    },
    "ZscoreSizerConfigs": {
        "sizing_lev": "Leverage level to be used for sizing positions.",
        "rvol_window": "Rolling volatility window size for calculating relative volatility.",
        "rolling_window": "Rolling window size for z-score calculation.",
        "weights": "Weights to be applied in the z-score calculation.",
        "vol_type": "Type of volatility measure to be used (e.g., mean, weighted).",
        "norm_const": "Normalization constant for z-score calculation.",
    },
    "OrderResolutionConfig": {
        "resolve_enabled": "Flag to enable or disable order resolution.",
        "max_tries": "Maximum number of attempts to resolve an order schema.",
        "otm_moneyness_width": "This should be a float representing the OTM moneyness width max for ATM against OTM.",
        "itm_moneyness_width": "This should be a float representing the ITM moneyness width max for ATM against ITM.",
        "max_close": "Maximum close price allowed for the order.",
        "max_dte_tolerance": "Minimum days to expiration tolerance allowed for the order.",
    },
}

def get_class_config_descriptions(class_name: str) -> dict:
    """
    Retrieve the configuration descriptions for a given class name.

    params:
    class_name: str: The name of the configuration class.

    returns:
    dict: A dictionary of configuration names and their descriptions.
    """
    return CONFIG_DEFINITIONS.get(class_name, {})

def get_variable_in_class_config_description(class_name: str, config_name: str) -> str:
    """
    Retrieve the description for a specific configuration of a given class.

    params:
    class_name: str: The name of the configuration class.
    config_name: str: The name of the specific configuration.

    returns:
    str: The description of the configuration.
    """
    
    return get_class_config_descriptions(class_name).get(config_name, None)
