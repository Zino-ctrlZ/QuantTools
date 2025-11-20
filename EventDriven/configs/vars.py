
# Class-level descriptions for each config class
CONFIG_CLASS_DESCRIPTIONS = {
    "BaseConfigs": "Base configuration class providing common functionality and run tracking for all configuration types.",
    "_CustomFrozenBaseConfigs": "Frozen configuration base class that prevents modification of attributes after initialization, except for run_name.",
    "ChainConfig": "Configuration for filtering and selecting options from the option chain based on spread width and open interest.",
    "OrderSchemaConfigs": "Configuration defining the structure and parameters for options orders including strategy type, DTE, and moneyness.",
    "OrderPickerConfig": "Configuration for the order picker component that selects optimal orders from available option chains.",
    "BaseSizerConfigs": "Base configuration for position sizing modules, defining the type of delta limit calculation.",
    "DefaultSizerConfigs": "Standard position sizing configuration using fixed leverage without volatility adjustments.",
    "ZscoreSizerConfigs": "Advanced position sizing configuration using volatility-adjusted z-score based limits for dynamic risk management.",
    "OrderResolutionConfig": "Configuration for automatic order resolution when initial order schemas fail to find suitable options.",
    "UndlTimeseriesConfig": "Configuration for underlying asset timeseries data retrieval and interval settings.",
    "OptionPriceConfig": "Configuration specifying which price type (bid, ask, midpoint, close) to use for option valuation.",
    "SkipCalcConfig": "Configuration for anomaly detection in option data, determining when to skip calculations due to data quality issues.",
    "BaseCogConfig": "Base configuration for position analyzer cog components that perform specific analysis tasks.",
    "StrategyLimitsEnabled": "Configuration flags controlling which types of risk limits (delta, gamma, vega, theta, DTE, moneyness) are enforced.",
    "LimitsEnabledConfig": "Configuration for the limits enforcement cog, managing risk thresholds and position rolling triggers.",
    "PositionAnalyzerConfig": "Configuration for the position analyzer orchestrating multiple cogs for comprehensive position analysis.",
    "PortfolioManagerConfig": "Configuration for portfolio management including settlement delays and trade execution timing.",
    "RiskManagerConfig": "Configuration for the risk manager controlling slippage limits and order caching behavior.",
}

CONFIG_DEFINITIONS = {
    # 'Config class: { config_name}': 'Description of what this config class is for and how to use it.'
    "BaseConfigs": {
        "run_name": "A name identifier for this run/session, used to tag and track configuration across backtest runs.",
    },
    "_CustomFrozenBaseConfigs": {
        "run_name": "A name identifier for this run/session, used to tag and track configuration across backtest runs.",
    },
    "ChainConfig": {
        "run_name": "A name identifier for this run/session, used to tag and track configuration across backtest runs.",
        "max_pct_width": "Maximum abs spread/mid price percentage width for an option to be included in the option chain.",
        "min_oi": "Minimum open interest required for an option to be included in the option chain.",
    },
    "OrderSchemaConfigs": {
        "run_name": "A name identifier for this run/session, used to tag and track configuration across backtest runs.",
        "target_dte": "Target days to expiration for the options in the order schema.",
        "strategy": "The options strategy to be used (e.g., vertical, iron condor).",
        "structure_direction": "Direction of the structure, either long or short.",
        "spread_ticks": "Number of strike price ticks between legs of the spread.",
        "dte_tolerance": "Allowed deviation in days to expiration from the target DTE.",
        "min_moneyness": "Minimum moneyness level for selecting options.",
        "max_moneyness": "Maximum moneyness level for selecting options.",
        "min_total_price": "Minimum total price for the option structure.",
    },
    "OrderPickerConfig": {
        "run_name": "A name identifier for this run/session, used to tag and track configuration across backtest runs.",
        "start_date": "The start date for selecting orders from option chain data.",
        "end_date": "The end date for selecting orders from option chain data.",
    },
    "BaseSizerConfigs": {
        "run_name": "A name identifier for this run/session, used to tag and track configuration across backtest runs.",
        "delta_lmt_type": "Type of delta limit calculation to use: 'default' uses fixed limits, 'zscore' uses volatility-adjusted limits.",
    },
    "DefaultSizerConfigs": {
        "run_name": "A name identifier for this run/session, used to tag and track configuration across backtest runs.",
        "sizing_lev": "Leverage level to be used for sizing positions (multiplier on equity equivalent size).",
        "delta_lmt_type": "Type of delta limit calculation to use: 'default' uses fixed limits, 'zscore' uses volatility-adjusted limits.",
    },
    "ZscoreSizerConfigs": {
        "run_name": "A name identifier for this run/session, used to tag and track configuration across backtest runs.",
        "sizing_lev": "Leverage level to be used for sizing positions (multiplier on equity equivalent size).",
        "rvol_window": "Rolling volatility window size(s) for calculating relative volatility. Can be a single number or tuple of windows.",
        "rolling_window": "Rolling window size for z-score calculation (default 100 days).",
        "weights": "Weights tuple (w1, w2, w3) applied in the z-score calculation for combining different volatility measures.",
        "vol_type": "Type of volatility measure to be used (e.g., 'mean', 'weighted').",
        "norm_const": "Normalization constant for z-score calculation to scale the volatility adjustment.",
        "delta_lmt_type": "Type of delta limit calculation to use: 'default' uses fixed limits, 'zscore' uses volatility-adjusted limits.",
    },
    "OrderResolutionConfig": {
        "run_name": "A name identifier for this run/session, used to tag and track configuration across backtest runs.",
        "resolve_enabled": "Flag to enable or disable automatic order resolution when initial order schema fails.",
        "otm_moneyness_width": "Maximum OTM moneyness width for ATM vs OTM option selection (default 0.45).",
        "itm_moneyness_width": "Maximum ITM moneyness width for ATM vs ITM option selection (default 0.45).",
        "max_close": "Maximum close price allowed for the order structure (default 10.0).",
        "max_tries": "Maximum number of attempts to resolve an order schema before giving up (default 20).",
        "max_dte_tolerance": "Maximum days to expiration tolerance allowed for the order (default 90).",
    },
    "UndlTimeseriesConfig": {
        "run_name": "A name identifier for this run/session, used to tag and track configuration across backtest runs.",
        "interval": "Time interval for underlying price data (e.g., '1d' for daily, '1h' for hourly).",
    },
    "OptionPriceConfig": {
        "run_name": "A name identifier for this run/session, used to tag and track configuration across backtest runs.",
        "use_price": "Price type to use for option pricing: 'midpoint', 'close', 'bid', or 'ask' (default 'midpoint').",
    },
    "SkipCalcConfig": {
        "run_name": "A name identifier for this run/session, used to tag and track configuration across backtest runs.",
        "window": "Rolling window size for skip calculation statistics (default 20).",
        "skip_threshold": "Z-score threshold for determining when to skip a calculation date (default 3.0).",
        "skip_enabled": "Flag to enable or disable skip calculation logic.",
        "abs_zscore_threshold": "Use absolute z-score threshold for skip detection.",
        "pct_zscore_threshold": "Use percentage-based z-score threshold for skip detection.",
        "spike_flag": "Flag to enable spike detection in price data.",
        "std_window_bool": "Use standard deviation window for anomaly detection.",
        "zero_filter": "Filter out zero values when calculating skip conditions (default True).",
        "add_columns": "List of (column_name, function_name) tuples to add additional calculated columns using ADD_COLUMNS_FACTORY.",
        "skip_columns": "List of column names to apply skip calculation logic to (default ['Delta', 'Gamma', 'Vega', 'Theta', 'Midpoint']).",
    },
    "BaseCogConfig": {
        "run_name": "A name identifier for this run/session, used to tag and track configuration across backtest runs.",
        "name": "Name identifier for the cog component.",
        "enabled": "Flag to enable or disable this cog in the position analyzer.",
    },
    "StrategyLimitsEnabled": {
        "run_name": "A name identifier for this run/session, used to tag and track configuration across backtest runs.",
        "delta": "Enable delta-based risk limits for the strategy (default True).",
        "vega": "Enable vega-based risk limits for the strategy (default False).",
        "gamma": "Enable gamma-based risk limits for the strategy (default False).",
        "theta": "Enable theta-based risk limits for the strategy (default False).",
        "dte": "Enable DTE-based position rolling limits (default True).",
        "moneyness": "Enable moneyness-based position rolling limits (default True).",
        "exercise": "Enable automatic exercise logic for expiring positions (default False).",
    },
    "LimitsEnabledConfig": {
        "run_name": "A name identifier for this run/session, used to tag and track configuration across backtest runs.",
        "name": "Name identifier for the limits cog (default 'LimitsEnabledCog').",
        "cache_actions": "Cache analyzed actions for performance optimization (default True).",
        "enabled": "Flag to enable or disable the limits enforcement cog (default True).",
        "delta_lmt_type": "Type of delta limit calculation: 'default' for fixed limits or 'zscore' for volatility-adjusted limits.",
        "default_dte": "Default days to expiration threshold for rolling positions (default 120).",
        "default_moneyness": "Default moneyness threshold for rolling positions (default 1.15).",
        "enabled_limits": "StrategyLimitsEnabled instance specifying which limit types are active.",
    },
    "PositionAnalyzerConfig": {
        "run_name": "A name identifier for this run/session, used to tag and track configuration across backtest runs.",
        "enabled": "Flag to enable or disable the position analyzer (default True).",
        "enabled_cogs": "List of cog names that are enabled for position analysis.",
    },
    "PortfolioManagerConfig": {
        "run_name": "A name identifier for this run/session, used to tag and track configuration across backtest runs.",
        "t_plus_n": "Settlement delay for orders in business days (T+N, default 1).",
    },
    "RiskManagerConfig": {
        "run_name": "A name identifier for this run/session, used to tag and track configuration across backtest runs.",
        "max_slippage": "Maximum allowable slippage percentage for trade execution (default 0.25).",
        "min_slippage": "Minimum allowable slippage percentage for trade execution (default 0.16).",
        "cache_orders": "Flag to enable caching of generated orders for reuse (default False).",
    },
}

def get_config_class_description(class_name: str) -> str:
    """
    Retrieve the class-level description for a given configuration class.

    params:
    class_name: str: The name of the configuration class.

    returns:
    str: The description of what the configuration class does.
    """
    return CONFIG_CLASS_DESCRIPTIONS.get(class_name, "")

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
