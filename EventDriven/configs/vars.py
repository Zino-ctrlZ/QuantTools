# Class-level descriptions for each config class
CONFIG_CLASS_DESCRIPTIONS = {
    "BaseConfigs": "Base configuration class providing common functionality and run tracking for all configuration types.",
    "_CustomFrozenBaseConfigs": "Frozen configuration base class that prevents modification of attributes after initialization, except for run_name.",
    "ChainConfig": "Controls which option quotes are allowed into candidate chains before order construction.",
    "OrderSchemaConfigs": "Defines the target option structure (strategy, direction, DTE, moneyness, and pricing bounds) used when building an order.",
    "OrderPickerConfig": "Defines the date window the order picker is allowed to search when selecting trades.",
    "BaseSizerConfigs": "Base sizing controls shared by sizer implementations, including delta-limit mode selection.",
    "DefaultSizerConfigs": "Simple leverage-based sizing with fixed-style delta limit behavior.",
    "ZscoreSizerConfigs": "Volatility-aware sizing using z-score style normalization and weighted windows.",
    "UndlTimeseriesConfig": "Controls underlying price timeseries retrieval granularity.",
    "OptionPriceConfig": "Selects which option price field is treated as the canonical execution/valuation price.",
    "SkipCalcConfig": "Controls data-quality skip logic that suppresses trading or calculations on anomalous option datapoints.",
    "BaseCogConfig": "Shared base settings for all PositionAnalyzer cogs (name + enabled flag).",
    "StrategyLimitsEnabled": "Feature flags for which risk checks are actively enforced by limit-aware cogs.",
    "LimitsEnabledConfig": "Configures the LimitsEnabledCog that applies delta/DTE/moneyness style constraints.",
    "PositionAnalyzerConfig": "Top-level switchboard for PositionAnalyzer and its enabled cogs.",
    "PortfolioManagerConfig": "Controls portfolio-level behavior such as weight haircuting and failed-order roll behavior.",
    "BacktesterConfig": "Top-level runtime configuration for settlement delay, slippage bounds, commissions, and liquidity policy.",
    "RiskManagerConfig": "Controls cache behavior for generated orders, analyses, and request objects.",
    "CashAllocatorConfig": "Maps weighted capital to per-symbol cash buckets using threshold rules.",
    "LiquidityConfig": "Defines multi-level liquidity enforcement shared by risk sizing and execution-time spread gates.",
    "MeanReversionSizerConfigs": "Custom mean-reversion sizing controls for scaling position size between min/max bounds.",
    "ScoringConfigs": "Scoring hyperparameters used to rank candidate option structures across spread, moneyness, DTE, and theta burden.",
    "ExecutionHandlerConfig": "Execution slippage model selection and spread-alpha settings used when fills are simulated.",
    "PnlMonitorConfig": "Non-configurable PnL monitor defaults provided through computed properties.",
    "PnLMonitorConfigConfigurable": "Explicit threshold-based PnL monitor config where each trigger is directly user-configurable.",
    "VectorizedCogConfig": "Config for a lightweight cog that monitors DTE thresholds using vectorized-friendly checks.",
    "PlainSizingCogConfig": "Config for a simple sizing cog with fallback one-lot behavior and optional strategy-token exclusions.",
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
        "max_pct_width": "Hard filter: keeps only contracts whose relative spread width is <= this value.",
        "min_oi": "Hard filter: keeps only contracts whose open interest is >= this value.",
        "enable_delta_filter": "If True, also applies delta-based filtering in chain preselection; if False, no delta pre-filter is applied.",
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
        "max_attempts": "Maximum retry attempts when trying to construct a valid order from available chain candidates.",
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
        "finalize_trades": "If True, finalization logic runs at the end of backtest to close/settle remaining trades (default True).",
        "raise_errors": "Flag to raise errors during backtest execution instead of logging them (default False).",
        "min_slippage_pct": "Minimum slippage percentage applied to trade execution (default 0.075).",
        "max_slippage_pct": "Maximum slippage percentage applied to trade execution (default 0.15).",
        "commission_per_contract_in_units": "Commission charged per contract in dollar units (default 0.0065).",
        "liquidity": "LiquidityConfig object used by both RiskManager and ExecutionHandler for quantity haircuting and spread-based drop/reschedule behavior.",
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
        "level": "Liquidity enforcement level, clamped to [0, 2]: 0 disables liquidity adjustments; 1 enables quantity haircuting in RiskManager; 2 also enables execution-time spread gate with silent drop+next-business-day reschedule.",
        "max_spread_pct": "Execution spread threshold used by level-2 gate: orders are dropped/rescheduled when computed spread_pct exceeds this value.",
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
    "PnlMonitorConfig": {
        "run_name": "A name identifier for this run/session, used to tag and track configuration across backtest runs.",
        "name": "Cog name used to register/identify the PnL monitor component (default 'PnLMonitorCog').",
        "enabled": "If False, the PnL monitor cog is skipped entirely.",
        "enable_stop_loss": "Computed property in this class; currently always returns False unless logic is changed in code.",
        "roll_profit_threshold": "Computed property in this class; threshold at which profitable positions are considered for rolling.",
        "lock_in_profit_threshold": "Computed property in this class; threshold to partially close and lock in profits.",
        "stop_loss_pct": "Computed property in this class; loss threshold that triggers exit behavior.",
        "profit_lock_in_pct": "Computed property in this class; fraction of realized gains fed into next-trade cash.",
    },
    "PnLMonitorConfigConfigurable": {
        "run_name": "A name identifier for this run/session, used to tag and track configuration across backtest runs.",
        "roll_profit_threshold": "Directly configurable threshold for rolling a winning position.",
        "stop_loss_pct": "Directly configurable threshold for loss-cut exits.",
        "profit_lock_in_pct": "Directly configurable fraction of gains to retain for future allocation.",
        "lock_in_profit_threshold": "Directly configurable threshold for partial close to secure gains.",
        "enable_stop_loss": "Direct switch for enabling stop-loss behavior in this configurable variant.",
    },
    "VectorizedCogConfig": {
        "run_name": "A name identifier for this run/session, used to tag and track configuration across backtest runs.",
        "name": "Cog name used to register/identify this vectorized monitoring cog.",
        "enabled": "If False, vectorized DTE checks are skipped.",
        "dte_limit_enabled": "If True, DTE threshold checks are evaluated for open positions.",
        "dte_threshold": "DTE value below/at which the cog marks positions for DTE-driven handling.",
    },
    "PlainSizingCogConfig": {
        "run_name": "A name identifier for this run/session, used to tag and track configuration across backtest runs.",
        "name": "Cog name used to register/identify the plain sizing cog.",
        "enabled": "If False, plain sizing adjustments are not applied.",
        "sizing_lev": "Base size multiplier applied by this simple sizing cog.",
        "dte_limit_enabled": "If True, this cog also enforces DTE-based checks for monitored positions.",
        "dte_threshold": "DTE cutoff used when dte_limit_enabled is True.",
        "exclude_strategy_slug_tokens": "Strategy slug tokens that bypass this cog; any match skips plain sizing logic for that strategy.",
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
