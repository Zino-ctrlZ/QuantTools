"""QuantTools DataManager - Comprehensive market data infrastructure for options trading.

This module provides a complete suite of data managers for retrieving, caching, and
computing market data and derived metrics for quantitative options trading and backtesting.

Key Components
--------------

**Market Data Managers:**
    - SpotDataManager: Underlying equity spot prices with split adjustment
    - RatesDataManager: Risk-free interest rates from Treasury bills
    - DividendDataManager: Dividend schedules (discrete/continuous models)
    - OptionSpotDataManager: Option contract market prices from ThetaData
    - MarketTimeseries: Central data repository with lazy loading

**Derived Metrics Managers:**
    - ForwardDataManager: Forward price computation using cost-of-carry models
    - VolDataManager: Implied volatility calculation (BSM, Binomial)
    - GreekDataManager: Option sensitivities (delta, gamma, vega, theta, rho)

**Unified Interfaces:**
    - TimeseriesDataManager: Consistent API across all managers (rt, get_at_time, get_timeseries)
    - loaders.load_full_option_data(): One-call comprehensive data loading

**Utilities:**
    - BaseDataManager: Abstract base with caching, configuration, logging
    - Result classes: Type-safe containers (SpotResult, VolatilityResult, GreekResultSet, etc.)
    - CacheSpec: Cache configuration with expiration control
    - Theoretical pricing: get_option_theoretical_price(), calculate_scenarios()

Design Features
---------------

- **Singleton pattern** per symbol for efficient resource management
- **Multi-tier caching** (memory + disk) with intelligent expiration
- **Automatic data loading** from multiple sources (ThetaData, OpenBB, YFinance)
- **Split adjustment handling** for accurate backtesting
- **Type-safe results** with full metadata preservation
- **Consistent API** across all managers via adapter pattern

Quick Start Examples
--------------------

**Individual Manager Usage:**
    >>> from trade.datamanager import SpotDataManager, VolDataManager, GreekDataManager
    >>>
    >>> # Load spot prices
    >>> spot_mgr = SpotDataManager("AAPL")
    >>> spot_result = spot_mgr.get_spot_timeseries(
    ...     start_date="2025-01-01",
    ...     end_date="2025-01-31",
    ...     undo_adjust=True
    ... )
    >>> prices = spot_result.daily_spot
    >>>
    >>> # Compute implied volatilities
    >>> vol_mgr = VolDataManager("AAPL")
    >>> vol_result = vol_mgr.get_implied_volatility_timeseries(
    ...     start_date="2025-01-01",
    ...     end_date="2025-01-31",
    ...     strike=150.0,
    ...     expiration="2025-06-20",
    ...     right="call"
    ... )
    >>> ivs = vol_result.timeseries

**Unified Interface (Recommended):**
    >>> from trade.datamanager import TimeseriesDataManager
    >>>
    >>> # Single entry point for all data types
    >>> ts = TimeseriesDataManager("AAPL")
    >>>
    >>> # Consistent interface across managers
    >>> spot = ts.spot.get_timeseries(start_date="2025-01-01", end_date="2025-01-31")
    >>> vol = ts.vol.get_timeseries(
    ...     start_date="2025-01-01",
    ...     end_date="2025-01-31",
    ...     strike=150.0,
    ...     expiration="2025-06-20",
    ...     right="call"
    ... )
    >>> greeks = ts.greeks.rt(strike=150.0, expiration="2025-06-20", right="call")

**One-Call Data Loading:**
    >>> from trade.datamanager.loaders import load_full_option_data
    >>> from trade.datamanager._enums import DivType
    >>>
    >>> # Load all option data (spot, forward, dividend, vol, greeks, rates)
    >>> pack = load_full_option_data(
    ...     symbol="AAPL",
    ...     strike=150.0,
    ...     expiration="2025-06-20",
    ...     right="call",
    ...     start_date="2025-01-01",
    ...     end_date="2025-01-31",
    ...     dividend_type=DivType.DISCRETE
    ... )
    >>>
    >>> # Access all components
    >>> spot = pack.spot.timeseries
    >>> vol = pack.vol.timeseries
    >>> greeks = pack.greek.timeseries

**Scenario Analysis:**
    >>> from trade.datamanager import calculate_scenarios
    >>>
    >>> # Run stress tests on option position
    >>> scenarios = calculate_scenarios(
    ...     symbol="AAPL",
    ...     as_of="2025-01-15",
    ...     strike=150.0,
    ...     expiration="2025-06-20",
    ...     right="call",
    ...     spot_scenarios=[0.95, 1.0, 1.05],  # ±5% spot moves
    ...     vol_scenarios=[-0.05, 0.0, 0.05],   # ±5 vol points
    ...     return_pnl=True
    ... )
    >>> print(scenarios.grid)  # pd.DataFrame with spot × vol grid

Configuration
-------------

Global settings via OptionDataConfig singleton:
    >>> from trade.datamanager.config import OptionDataConfig
    >>> from trade.datamanager._enums import OptionPricingModel, DivType
    >>>
    >>> config = OptionDataConfig()
    >>> config.option_model = OptionPricingModel.BSM
    >>> config.dividend_type = DivType.DISCRETE
    >>> config.n_steps = 200  # Binomial tree steps

Important Notes
---------------

**Date Conversion:**
    Always use `to_datetime()` from `trade.helpers.helper` for date conversions.
    Never use `datetime.strptime()` or `pd.to_datetime()` directly.

**Split Adjustment:**
    Use `undo_adjust=True` for backtesting to get split-adjusted prices.
    Ensure consistent `undo_adjust` across all managers in same calculation.

**Caching:**
    Managers use singleton pattern per symbol - avoid creating duplicate instances.
    Cache is automatically managed with configurable expiration.

**Real-time Data:**
    Use `.rt()` methods for real-time/latest data.
    Configure fallback behavior via `OptionDataConfig.real_time_fallback_option`.

See Also
--------

For comprehensive documentation, see:
    - trade/datamanager/README.md: Complete module guide
    - Individual manager docstrings: Detailed API documentation
    - trade/datamanager/loaders.py: Convenience loader functions
    - trade/datamanager/timeseries.py: Unified interface documentation

Module Structure
----------------

    datamanager/
    ├── Market Data Layer
    │   ├── spot.py              - SpotDataManager
    │   ├── rates.py             - RatesDataManager
    │   ├── dividend.py          - DividendDataManager
    │   ├── option_spot.py       - OptionSpotDataManager
    │   └── market_data.py       - MarketTimeseries
    │
    ├── Derived Metrics Layer
    │   ├── forward.py           - ForwardDataManager
    │   ├── vol.py               - VolDataManager
    │   ├── greeks.py            - GreekDataManager
    │   └── theo.py              - Theoretical pricing functions
    │
    ├── Unified Interfaces
    │   ├── timeseries.py        - TimeseriesDataManager
    │   └── loaders.py           - load_full_option_data()
    │
    ├── Core Infrastructure
    │   ├── base.py              - BaseDataManager, CacheSpec
    │   ├── config.py            - OptionDataConfig
    │   ├── result.py            - Result dataclasses
    │   ├── _enums.py            - Enumerations
    │   └── vars.py              - Global instances
    │
    └── utils/                   - Helper utilities
"""

from .dividend import DividendDataManager
from .forward import ForwardDataManager
from .rates import RatesDataManager
from .option_spot import OptionSpotDataManager
from .spot import SpotDataManager
from .base import BaseDataManager, CacheSpec
from .vol import VolDataManager
from .greeks import GreekDataManager
from .result import Result, SpotResult, ForwardResult, DividendsResult, RatesResult, OptionSpotResult
from .timeseries import TimeseriesDataManager
from .market_data import MarketTimeseries
from .theo import get_option_theoretical_price, calculate_scenarios
from .utils.model import assert_synchronized_model

__all__ = [
    "DividendDataManager",
    "ForwardDataManager",
    "RatesDataManager",
    "OptionSpotDataManager",
    "SpotDataManager",
    "BaseDataManager",
    "Result",
    "SpotResult",
    "ForwardResult",
    "DividendsResult",
    "RatesResult",
    "OptionSpotResult",
    "MarketTimeseries",
    "TimeseriesDataManager",
    "CacheSpec",
    "VolDataManager",
    "GreekDataManager",
    "assert_synchronized_model",
    "get_option_theoretical_price",
    "calculate_scenarios",
]
