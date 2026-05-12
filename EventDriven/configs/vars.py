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
    "UndlTimeseriesConfig": "Configuration for underlying asset timeseries data retrieval and interval settings.",
    "OptionPriceConfig": "Configuration specifying which price type (bid, ask, midpoint, close) to use for option valuation.",
    "SkipCalcConfig": "Configuration for anomaly detection in option data, determining when to skip calculations due to data quality issues.",
    "BaseCogConfig": "Base configuration for position analyzer cog components that perform specific analysis tasks.",
    "StrategyLimitsEnabled": "Configuration flags controlling which types of risk limits (delta, gamma, vega, theta, DTE, moneyness) are enforced.",
    "LimitsEnabledConfig": "Configuration for the limits enforcement cog, managing risk thresholds and position rolling triggers.",
    "PositionAnalyzerConfig": "Configuration for the position analyzer orchestrating multiple cogs for comprehensive position analysis.",
    "PortfolioManagerConfig": "Configuration for portfolio management including weights haircut adjustments.",
    "BacktesterConfig": "Configuration for backtest execution including settlement delays and trade finalization.",
    "RiskManagerConfig": "Configuration for the risk manager controlling order and analysis caching behavior.",
    "CashAllocatorConfig": "Threshold-based cash bucket allocator for symbols.",
    "LiquidityConfig": "Centralized liquidity control for both risk and execution layers, managing spread and liquidity level.",
    "MeanReversionSizerConfigs": "Custom mean reversion position sizer using z-score scaling to adjust sizing dynamically around a target DTE.",
    "ScoringConfigs": "Configuration for scoring and selecting options based on moneyness, DTE, mid price, spread, and theta burden targets.",
    "ExecutionHandlerConfig": "Configuration for the execution handler controlling slippage model and spread percentage parameters.",
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
        "weights_haircut": "Haircut applied to position weights for conservative allocation (default 0.0).",
        "roll_failed_orders": "Whether signals that fail to be processed should be rolled forward to the next available date (default True).",
    },
    "BacktesterConfig": {
        "run_name": "A name identifier for this run/session, used to tag and track configuration across backtest runs.",
        "t_plus_n": "Settlement delay for orders in business days (T+N, default 1).",
        "finalize_trades": "Flag to enable finalization of trades at end of backtest (default False).",
        "raise_errors": "Flag to raise errors during backtest execution instead of logging them (default False).",
        "min_slippage_pct": "Minimum slippage percentage applied to trade execution (default 0.075).",
        "max_slippage_pct": "Maximum slippage percentage applied to trade execution (default 0.15).",
        "commission_per_contract_in_units": "Commission charged per contract in dollar units (default 0.0065).",
        "liquidity": "LiquidityConfig instance controlling spread and liquidity level for backtest execution.",
    },
    "RiskManagerConfig": {
        "run_name": "A name identifier for this run/session, used to tag and track configuration across backtest runs.",
        "cache_orders": "Flag to enable caching of generated orders for reuse (default False).",
        "cache_position_analysis": "Flag to enable caching of position analysis results for performance optimization (default False).",
        "cache_order_requests": "Flag to enable caching of order requests to avoid redundant order generation (default False).",
    },
    "CashAllocatorConfig": {
        "run_name": "A name identifier for this run/session, used to tag and track configuration across backtest runs.",
        "thresholds": "(min_alloc, bucket_value) pairs; first pair whose min_alloc is satisfied sets the bucket. Cash is supplied at runtime.",
    },
    "LiquidityConfig": {
        "run_name": "A name identifier for this run/session, used to tag and track configuration across backtest runs.",
        "level": "Liquidity enforcement level (0=none, 1=standard, 2=strict). Clamped to [0, 2] on init.",
        "max_spread_pct": "Maximum allowable spread as a percentage of mid price (default 0.25).",
    },
    "MeanReversionSizerConfigs": {
        "run_name": "A name identifier for this run/session, used to tag and track configuration across backtest runs.",
        "name": "Name identifier for this sizer cog (default 'custom_mean_reversion_sizer').",
        "beta": "Mean reversion speed parameter controlling how aggressively sizing reverts to the mean (default 0.5).",
        "min_scale": "Minimum scaling factor applied to base sizing level (default 0.5).",
        "max_scale": "Maximum scaling factor applied to base sizing level (default 2.0).",
        "sizing_lev": "Base leverage level for position sizing (default 2).",
        "default_dte": "Default days to expiration used in scaling calculations (default 10).",
        "enabled_limits": "StrategyLimitsEnabled instance controlling which limit types are active for this sizer.",
        "min_zscore": "Minimum z-score threshold required to trigger scaling adjustments (default 2.5).",
    },
    "ScoringConfigs": {
        "run_name": "A name identifier for this run/session, used to tag and track configuration across backtest runs.",
        "tilt_strength": "Strength of the directional tilt applied during scoring (default 0.2).",
        "spread_ticks": "Number of strike ticks between spread legs (default 2).",
        "structure_direction": "Direction of the structure: 'long' or 'short' (default 'long').",
        "strategy": "Options strategy type: 'vertical' or 'naked' (default 'vertical').",
        "hybrid_strategy_enabled": "Enable hybrid strategy combining multiple structure types (default False).",
        "m_target": "Target moneyness for option selection (default 0.8).",
        "min_moneyness": "Minimum moneyness allowed for selected options (default 0.45).",
        "max_moneyness": "Maximum moneyness allowed for selected options (default 1.05).",
        "m_sigma": "Moneyness scoring sigma for gaussian weighting (default 0.2).",
        "m_tilt": "Moneyness tilt preference: 'otm', 'itm', or 'atm' (default 'otm').",
        "target_dte": "Target days to expiration for scoring (default 200).",
        "dte_tolerance": "Allowed DTE deviation from target (default 100).",
        "dte_sigma": "DTE scoring sigma for gaussian weighting (default 10).",
        "dte_tilt": "DTE tilt preference: 'flat', 'short', or 'long' (default 'short').",
        "mid_min": "Minimum acceptable mid price (default 0.5).",
        "mid_max": "Maximum acceptable mid price (default 3.0).",
        "mid_upper_limit": "Hard upper limit on mid price (default 5).",
        "mid_lower_limit": "Hard lower limit on mid price (default 0.25).",
        "mid_sigma": "Mid price scoring sigma for gaussian weighting (default 0.25).",
        "pct_spread_max": "Maximum allowable spread as a percentage of mid price (default 1.0).",
        "target_spread_pct": "Target spread percentage for scoring (default 0.2).",
        "pct_spread_sigma": "Spread scoring sigma for gaussian weighting (default 0.10).",
        "oi_target": "Target open interest for scoring (default 1000).",
        "theta_burden_max": "Maximum theta burden as a fraction of position value (default 0.03).",
        "theta_burden_sigma": "Theta burden scoring sigma for gaussian weighting (default 0.02).",
    },
    "ExecutionHandlerConfig": {
        "run_name": "A name identifier for this run/session, used to tag and track configuration across backtest runs.",
        "slippage_model": "Slippage model to apply: 'randomized', 'fixed', 'spread_pct', or 'none' (default 'randomized').",
        "pct_alpha": "Fraction of the spread used as slippage when using spread_pct model (default 0.25).",
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
