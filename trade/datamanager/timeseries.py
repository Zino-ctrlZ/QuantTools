"""Unified timeseries interface for all DataManager classes.

This module provides TimeseriesDataManager and TimeseriesAdapter classes that create
a consistent, simplified API across all specialized data managers (spot, vol, greeks,
etc.). Each manager's specific method names are mapped to standardized names while
preserving original docstrings and type signatures.

Key Features:
    - Standardized API: rt(), get_at_time(), get_timeseries() across all managers
    - Preserves original docstrings and signatures for IDE support
    - Property-based access to underlying managers via adapters
    - Pass-through to underlying manager attributes when needed
    - Single entry point for all market data retrieval

Typical Usage:
    >>> from trade.datamanager.timeseries import TimeseriesDataManager
    >>>
    >>> # Initialize for a symbol
    >>> ts = TimeseriesDataManager("AAPL")
    >>>
    >>> # Spot data with consistent interface
    >>> spot_rt = ts.spot.rt()
    >>> spot_series = ts.spot.get_timeseries(
    ...     start_date="2025-01-01",
    ...     end_date="2025-01-31"
    ... )
    >>>
    >>> # Options data (pass strike/expiration/right explicitly)
    >>> vol = ts.vol.get_timeseries(
    ...     start_date="2025-01-01",
    ...     end_date="2025-01-31",
    ...     strike=150.0,
    ...     expiration="2025-06-20",
    ...     right="call"
    ... )
    >>>
    >>> # Greeks with same consistent interface
    >>> greeks = ts.greeks.rt(
    ...     strike=150.0,
    ...     expiration="2025-06-20",
    ...     right="call"
    ... )
    >>>
    >>> # Access underlying manager if needed
    >>> underlying_vol_mgr = ts.vol._manager

Architecture:
    TimeseriesDataManager acts as a facade, exposing properties (spot, vol, greeks, etc.)
    that return TimeseriesAdapter instances. Each adapter wraps the underlying DataManager
    and provides the standardized method names that delegate to the actual methods.
"""

from typing import Any, Optional
import inspect
from trade.datamanager.result import Result
from trade.helpers.Logging import setup_logger
from .spot import SpotDataManager
from .vol import VolDataManager
from .dividend import DividendDataManager
from .forward import ForwardDataManager
from .option_spot import OptionSpotDataManager
from .greeks import GreekDataManager
from .rates import RatesDataManager
from trade.datamanager.utils.logging import get_logging_level, register_to_factor_list

logger = setup_logger("trade.datamanager.timeseries", stream_log_level=get_logging_level())
register_to_factor_list("trade.datamanager.timeseries")


class TimeseriesAdapter:
    """Adapter that provides a consistent interface for any DataManager.

    Maps standardized method names (rt, get_at_time, get_timeseries) to
    the actual underlying DataManager methods while preserving original
    docstrings, signatures, and type hints.
    """

    def __init__(
        self,
        manager: Any,
        rt_method: Optional[str] = "rt",
        get_at_time_method: Optional[str] = None,
        get_timeseries_method: Optional[str] = None,
    ):
        """Initialize adapter with method name mappings.

        Args:
            manager: The underlying DataManager instance
            rt_method: Name of the real-time method (default: "rt")
            get_at_time_method: Name of the get-at-time method (e.g., "get_at_time", "get_at_time_implied_volatility")
            get_timeseries_method: Name of the timeseries method (e.g., "get_spot_timeseries", "get_implied_volatility_timeseries")
        """
        self._manager = manager
        self._rt_method = rt_method
        self._get_at_time_method = get_at_time_method
        self._get_timeseries_method = get_timeseries_method

        # Create wrapper methods with copied metadata
        self._create_wrapper_method("rt", rt_method)
        self._create_wrapper_method("get_at_time", get_at_time_method)
        self._create_wrapper_method("get_timeseries", get_timeseries_method)

    def _create_wrapper_method(self, wrapper_name: str, underlying_method_name: Optional[str]):
        """Create a wrapper method that copies docstring and signature from underlying method.

        Args:
            wrapper_name: Name of the wrapper method on this adapter
            underlying_method_name: Name of the actual method on the underlying manager
        """
        if not underlying_method_name or not hasattr(self._manager, underlying_method_name):
            return

        underlying_method = getattr(self._manager, underlying_method_name)

        # Create wrapper function that calls the underlying method
        def wrapper(*args, **kwargs):
            return underlying_method(*args, **kwargs)

        # Copy metadata for proper introspection
        wrapper.__name__ = wrapper_name
        wrapper.__doc__ = underlying_method.__doc__
        wrapper.__wrapped__ = underlying_method  # Allows ? to find original source

        # Copy module and qualname for correct file location display
        if hasattr(underlying_method, "__module__"):
            wrapper.__module__ = underlying_method.__module__
        if hasattr(underlying_method, "__qualname__"):
            # Keep the wrapper name but use the module path
            wrapper.__qualname__ = f"{underlying_method.__qualname__.rsplit('.', 1)[0]}.{wrapper_name}"

        # Copy signature
        try:
            wrapper.__signature__ = inspect.signature(underlying_method)
        except (ValueError, TypeError):
            pass

        # Copy annotations if available
        if hasattr(underlying_method, "__annotations__"):
            wrapper.__annotations__ = underlying_method.__annotations__.copy()

        # Set as instance attribute (overrides class method)
        setattr(self, wrapper_name, wrapper)

    def rt(self, *args, **kwargs) -> Result:
        """Call the underlying manager's real-time method."""
        if self._rt_method and hasattr(self._manager, self._rt_method):
            method = getattr(self._manager, self._rt_method)
            return method(*args, **kwargs)
        raise NotImplementedError(f"{self._manager.__class__.__name__} does not support rt()")

    def get_at_time(self, *args, **kwargs) -> Result:
        """Call the underlying manager's get-at-time method."""
        if self._get_at_time_method and hasattr(self._manager, self._get_at_time_method):
            method = getattr(self._manager, self._get_at_time_method)
            return method(*args, **kwargs)
        raise NotImplementedError(f"{self._manager.__class__.__name__} does not support get_at_time()")

    def get_timeseries(self, *args, **kwargs) -> Result:
        """Call the underlying manager's timeseries method."""
        if self._get_timeseries_method and hasattr(self._manager, self._get_timeseries_method):
            method = getattr(self._manager, self._get_timeseries_method)
            return method(*args, **kwargs)
        raise NotImplementedError(f"{self._manager.__class__.__name__} does not support get_timeseries()")

    def __getattr__(self, name: str):
        """Pass through any other attribute access to the underlying manager."""
        return getattr(self._manager, name)


class TimeseriesDataManager:
    """Unified interface for all data managers with consistent method naming.

    Each data manager is wrapped with a TimeseriesAdapter that maps standardized
    method names (rt, get_at_time, get_timeseries) to the actual underlying methods.

    Examples:
        >>> # Basic spot data access
        >>> ts = TimeseriesDataManager("AAPL")
        >>> spot_result = ts.spot.rt()
        >>> spot_series = ts.spot.get_timeseries(start_date="2025-01-01", end_date="2025-01-31")

        >>> # Options data - pass parameters explicitly
        >>> ts = TimeseriesDataManager("AAPL")
        >>> vol_result = ts.vol.rt(strike=150.0, expiration="2025-06-20", right="call")
        >>> greeks = ts.greeks.get_timeseries(
        ...     start_date="2025-01-01", end_date="2025-01-31",
        ...     strike=150.0, expiration="2025-06-20", right="call"
        ... )
    """

    def __init__(self, symbol: str):
        """Initialize unified timeseries data manager.

        Args:
            symbol: Ticker symbol (e.g., "AAPL")
        """
        self.symbol = symbol

        # Initialize underlying managers
        self._spot_manager = SpotDataManager(symbol=symbol)
        self._vol_manager = VolDataManager(symbol=symbol)
        self._dividend_manager = DividendDataManager(symbol=symbol)
        self._forward_manager = ForwardDataManager(symbol=symbol)
        self._option_spot_manager = OptionSpotDataManager(symbol=symbol)
        self._greeks_manager = GreekDataManager(symbol=symbol)
        self._rates_manager = RatesDataManager()

    @property
    def spot(self) -> TimeseriesAdapter:
        """Access spot price data with standardized interface.

        Methods:
            - rt(): Get real-time spot price
            - get_at_time(date): Get spot price at specific date
            - get_timeseries(start_date, end_date, undo_adjust=True): Get spot price series
        """
        return TimeseriesAdapter(
            manager=self._spot_manager,
            rt_method="rt",
            get_at_time_method="get_at_time",
            get_timeseries_method="get_spot_timeseries",
        )

    @property
    def vol(self) -> TimeseriesAdapter:
        """Access implied volatility data with standardized interface.

        Methods:
            - rt(strike, expiration, right, ...): Get real-time implied volatility
            - get_at_time(date, strike, expiration, right, ...): Get implied vol at specific date
            - get_timeseries(start_date, end_date, strike, expiration, right, ...): Get implied vol series
        """
        return TimeseriesAdapter(
            manager=self._vol_manager,
            rt_method="rt",
            get_at_time_method="get_at_time_implied_volatility",
            get_timeseries_method="get_implied_volatility_timeseries",
        )

    @property
    def greeks(self) -> TimeseriesAdapter:
        """Access option greeks data with standardized interface.

        Requires: strike, expiration, right parameters set in constructor

        Methods:
            - rt(strike, expiration, right, ...): Get real-time option greeks
            - get_at_time(date, strike, expiration, right, ...): Get greeks at specific date
            - get_timeseries(start_date, end_date, strike, expiration, right, ...): Get greeks series
        """
        return TimeseriesAdapter(
            manager=self._greeks_manager,
            rt_method="rt",
            get_at_time_method="get_at_time_greeks",
            get_timeseries_method="get_greeks_timeseries",
        )

    @property
    def forward(self) -> TimeseriesAdapter:
        """Access forward price data with standardized interface.

        Methods:
            - rt(maturity_date): Get real-time forward price
            - get_timeseries(start_date, end_date, maturity_date): Get forward price series
        """
        return TimeseriesAdapter(
            manager=self._forward_manager,
            rt_method="rt",
            get_at_time_method="get_forward",  # Forward doesn't have get_at_time
            get_timeseries_method="get_forward_timeseries",
        )

    @property
    def dividend(self) -> TimeseriesAdapter:
        """Access dividend data with standardized interface.

        Methods:
            - rt(maturity_date): Get real-time dividend schedule
            - get_timeseries(start_date, end_date, maturity_date): Get dividend series
        """
        return TimeseriesAdapter(
            manager=self._dividend_manager,
            rt_method="rt",
            get_at_time_method="get_schedule",  # Dividend doesn't have get_at_time
            get_timeseries_method="get_schedule_timeseries",
        )

    @property
    def rates(self) -> TimeseriesAdapter:
        """Access risk-free rate data with standardized interface.

        Methods:
            - rt(): Get real-time risk-free rate
            - get_timeseries(start_date, end_date): Get rate series
        """
        return TimeseriesAdapter(
            manager=self._rates_manager,
            rt_method="rt",
            get_at_time_method="get_rate",  # Rates doesn't have get_at_time
            get_timeseries_method="get_risk_free_rate_timeseries",
        )

    @property
    def option_spot(self) -> TimeseriesAdapter:
        """Access option market price data with standardized interface.

        Requires: strike, expiration, right parameters set in constructor

        Methods:
            - rt(strike, expiration, right, ...): Get real-time option market price
            - get_at_time(date, strike, expiration, right, ...): Get option price at specific date
            - get_timeseries(start_date, end_date, strike, expiration, right, ...): Get option price series
        """
        return TimeseriesAdapter(
            manager=self._option_spot_manager,
            rt_method="rt",
            get_at_time_method="get_option_spot_at_time",
            get_timeseries_method="get_option_spot_timeseries",
        )
