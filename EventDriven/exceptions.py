class BacktesterIncorrectTypeError(Exception):
    pass

class EVBacktestError(Exception):
    """Custom exception for Backtest-related errors."""
    pass

class UnaccessiblePropertyError(Exception):
    """Custom exception for unaccessible property errors."""
    pass

class BacktestConfigAttributeError(Exception):
    """Custom exception for missing Backtest configuration attributes or invalid attribute access."""
    pass

class BacktestNotImplementedError(Exception):
    """Exception raised for unimplemented backtest features."""

    pass