# QuantTools DataManager Module

Comprehensive data infrastructure for quantitative options trading and backtesting.

## Documentation

| Document | Description |
|----------|-------------|
| [README.md](README.md) | This guide — API, managers, caching |
| [UPGRADE_PLAN.md](UPGRADE_PLAN.md) | Certification rollout (completed) |
| [FUTURE_CHANGES.md](FUTURE_CHANGES.md) | Backlog and deferred work |

Package root: [../README.md](../README.md) (index into this folder).

## Table of Contents

- [Overview](#overview)
- [Certification](#certification)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Core Data Managers](#core-data-managers)
- [Derived Metrics Managers](#derived-metrics-managers)
- [Unified Timeseries Interface](#unified-timeseries-interface)
- [Convenience Loaders](#convenience-loaders)
- [Configuration](#configuration)
- [Caching System](#caching-system)
- [Best Practices](#best-practices)

---

## Overview

The DataManager module provides a complete data infrastructure for options trading with:

- **Historical and real-time market data** from multiple sources (ThetaData, OpenBB, YFinance)
- **Intelligent multi-tier caching** (memory + disk with expiration)
- **Derived metrics calculation** (forwards, volatilities, greeks, theoretical prices)
- **Type-safe result containers** with full metadata
- **Singleton pattern** per symbol for efficient resource management
- **Consistent API** across all managers

### Design Principles

- Automatic data loading from multiple sources
- Split adjustment handling for accurate backtesting
- Dividend schedule construction (discrete/continuous)
- Forward price computation with carry models
- Implied volatility calculation (BSM, Binomial)
- Greek calculation with multiple pricing models
- Theoretical pricing and scenario analysis

---

## Certification

Timeseries returns from certified managers pass through **L1 / L2 / L3** structural
certification at the return boundary (`certify_manager_result`). Cache stores
post-sanitize, **pre-certify** data; L3 repairs are not persisted.

| Level | Behavior |
|-------|----------|
| **L1** | Log issues; return data as-is |
| **L2** | Raise `DataNotCertifiedException` on unexplained violations (default via `OptionDataConfig`) |
| **L3** | Fix (dedupe, ffill, B-day reindex); log fixes; write to `result.timeseries` |

**Key behaviors:**

- **NA forensics** run in `certify_manager_result` before certification (not on cached re-read).
- **Option artifacts** — vendor-confirmed absent dates (`checked_missing_dates`) exempt NaNs at cert time; always resolved from vendor at call time, not stored on `Result`.
- **Pre-market today** — excluded from the certification B-day grid via `is_available_on_date`.
- **Date sync** — `_sync_date` (options) and `_sync_equity_date` (equity EOD) clamp fetch windows so pre-market requests do not pull unavailable today rows.
- **Point-in-time** — `utils/point_in_time.resolve_value_at_date` (10 BDay lookback, L1 on fetch) wired to `get_at_time` / `rt()` paths.

**Logging:**

| Logger | File pattern | Content |
|--------|--------------|---------|
| `trade.datamanager.certification.report` | `logs/trade.datamanager.certification.report*.log` | Plain-text certification audit reports |
| `trade.datamanager.certification.pipeline` / `respond` / `fixers` | respective `*_test.log` | Structural warnings, L3 fix actions |
| `trade.datamanager.utils.model_na` | `logs/trade.datamanager.utils.model_na*.log` | NaN forensics on pre-cert payloads |

**Smoke test:** `trade/datamanager/tests/run_certification_deep_test.py` (25 cases).

See [UPGRADE_PLAN.md](UPGRADE_PLAN.md) for architecture and [FUTURE_CHANGES.md](FUTURE_CHANGES.md) for backlog.

---

## Quick Start

### Basic Usage

```python
from trade.datamanager import SpotDataManager, VolDataManager, GreekDataManager
from trade.datamanager._enums import DivType, OptionPricingModel

# Load spot prices
spot_mgr = SpotDataManager("AAPL")
spot_result = spot_mgr.get_spot_timeseries(
    start_date="2025-01-01",
    end_date="2025-01-31",
    undo_adjust=True  # Split-adjusted prices
)
prices = spot_result.daily_spot

# Get implied volatilities
vol_mgr = VolDataManager("AAPL")
vol_result = vol_mgr.get_implied_volatility_timeseries(
    start_date="2025-01-01",
    end_date="2025-01-31",
    strike=150.0,
    expiration="2025-06-20",
    right="call",
    dividend_type=DivType.DISCRETE
)
ivs = vol_result.timeseries

# Compute option greeks
greek_mgr = GreekDataManager("AAPL")
greek_result = greek_mgr.get_greeks_timeseries(
    start_date="2025-01-01",
    end_date="2025-01-31",
    strike=150.0,
    expiration="2025-06-20",
    right="call"
)
greeks_df = greek_result.timeseries  # DataFrame with delta, gamma, vega, etc.
```

### Using the Unified Interface

```python
from trade.datamanager.timeseries import TimeseriesDataManager

# Single entry point for all data types
ts = TimeseriesDataManager("AAPL")

# Consistent interface across all managers
spot = ts.spot.get_timeseries(start_date="2025-01-01", end_date="2025-01-31")
vol = ts.vol.get_timeseries(
    start_date="2025-01-01",
    end_date="2025-01-31",
    strike=150.0,
    expiration="2025-06-20",
    right="call"
)
greeks = ts.greeks.rt(strike=150.0, expiration="2025-06-20", right="call")
```

### One-Call Data Loading

```python
from trade.datamanager.loaders import load_full_option_data
from trade.datamanager._enums import DivType

# Load all option data (spot, forward, dividend, vol, greeks, rates) in one call
pack = load_full_option_data(
    symbol="AAPL",
    strike=150.0,
    expiration="2025-06-20",
    right="call",
    start_date="2025-01-01",
    end_date="2025-01-31",
    dividend_type=DivType.DISCRETE
)

# Access individual components
spot = pack.spot.timeseries
vol = pack.vol.timeseries
greeks = pack.greek.timeseries
```

---

## Architecture

### Component Hierarchy

```
BaseDataManager (ABC)
│
├── Market Data Layer
│   ├── SpotDataManager          - Underlying equity prices
│   ├── RatesDataManager          - Risk-free interest rates
│   ├── DividendDataManager       - Dividend schedules
│   ├── OptionSpotDataManager     - Option contract prices
│   └── MarketTimeseries          - Central data repository
│
└── Derived Metrics Layer
    ├── ForwardDataManager        - Forward price computation
    ├── VolDataManager            - Implied volatility calculation
    ├── GreekDataManager          - Option sensitivities
    └── TheoDataFunctions         - Theoretical pricing
```

### Key Features by Layer

**BaseDataManager** provides:
- Cache management (CustomCache)
- Key construction (namespaced, artifact-based)
- Configuration (OptionDataConfig singleton)
- Logger setup

**Market Data Layer** handles:
- Data retrieval from external sources
- Split adjustment handling
- Corporate action processing
- Historical and real-time access

**Derived Metrics Layer** computes:
- Forward prices using cost-of-carry models
- Implied volatilities via model inversion
- Option greeks using analytical/numerical methods
- Theoretical prices and scenario analysis

---

## Core Data Managers

### SpotDataManager

Manages underlying equity spot prices with split adjustment support.

**Features:**
- Singleton pattern (per symbol)
- 45-day cache expiration
- Split-adjusted (chain_spot) and unadjusted (spot) prices
- Data source: MarketTimeseries (OpenBB/YFinance)

**Key Methods:**

```python
# Historical timeseries
result = spot_mgr.get_spot_timeseries(
    start_date="2025-01-01",
    end_date="2025-01-31",
    undo_adjust=True  # True for split-adjusted, False for raw
)
prices = result.daily_spot  # pd.Series with DatetimeIndex

# Single date
result = spot_mgr.get_at_time(date="2025-01-15")

# Real-time
result = spot_mgr.rt()
```

### RatesDataManager

Manages risk-free interest rates from US Treasury bills (^IRX).

**Features:**
- Singleton pattern (global, no symbol)
- 30-day cache expiration
- Data source: YFinance (13-week T-Bill)
- Automatic forward fill for missing dates

**Key Methods:**

```python
# Historical rates
result = rates_mgr.get_risk_free_rate_timeseries(
    start_date="2025-01-01",
    end_date="2025-01-31"
)
rates = result.daily_risk_free_rates  # pd.Series (annualized)

# Real-time
rate = rates_mgr.rt()
```

### DividendDataManager

Manages dividend data with schedule construction for option pricing.

**Features:**
- Singleton pattern (per symbol)
- 60-day cache expiration + temp cache
- Supports discrete (schedule) and continuous (yield) models
- Handles split adjustments
- Smart partial caching with merging

**Key Methods:**

```python
# Get dividend schedules for option pricing
result = div_mgr.get_schedule_timeseries(
    start_date="2025-01-01",
    end_date="2025-01-31",
    maturity_date="2025-06-20",
    dividend_type=DivType.DISCRETE,
    undo_adjust=True
)
schedules = result.daily_discrete_dividends  # pd.Series of Schedule objects

# Real-time dividend schedule
result = div_mgr.rt(maturity_date="2025-06-20")
```

### ForwardDataManager

Computes forward prices using cost-of-carry models.

**Features:**
- Singleton pattern (per symbol)
- 30-day cache expiration
- Dependencies: SpotDataManager, RatesDataManager, DividendDataManager
- Discrete model: F = S × exp(r×T) - PV(dividends)
- Continuous model: F = S × exp((r-q)×T)

**Key Methods:**

```python
# Compute forward prices
result = fwd_mgr.get_forward_timeseries(
    start_date="2025-01-01",
    end_date="2025-01-31",
    maturity_date="2025-06-20",
    dividend_type=DivType.DISCRETE,
    use_chain_spot=True
)
forwards = result.daily_discrete_forward  # pd.Series

# Real-time forward
result = fwd_mgr.rt(maturity_date="2025-06-20")
```

### OptionSpotDataManager

Retrieves option contract market prices from ThetaData API.

**Features:**
- Not singleton (per symbol)
- 7-day cache expiration
- Data source: ThetaData (EOD or Quote endpoint)
- Returns OHLC data

**Key Methods:**

```python
# Historical option prices
result = opt_mgr.get_option_spot_timeseries(
    start_date="2025-01-01",
    end_date="2025-01-31",
    strike=150.0,
    expiration="2025-06-20",
    right="call",
    endpoint_source=OptionSpotEndpointSource.EOD
)
ohlc = result.daily_option_spot  # pd.DataFrame [open, high, low, close]

# Real-time option price
result = opt_mgr.rt(strike=150.0, expiration="2025-06-20", right="call")
```

### MarketTimeseries

Central market data repository with lazy loading and caching.

**Features:**
- Singleton pattern (global instance)
- Multi-tier caching (memory + disk)
- Data sources: OpenBB, ThetaData, YFinance
- Loads all data for a symbol on first request
- Thread-safe access
- Point-in-time snapshots

**Usage:**

```python
from trade.datamanager.vars import get_times_series

ts = get_times_series()
ts.load("AAPL")  # Lazy load on first access

# Get point-in-time data
data = ts.get_at_index("AAPL", "2025-01-15")
spot = data.spot  # pd.Series with OHLCV
```

---

## Derived Metrics Managers

### VolDataManager

Computes implied volatilities from option market prices.

**Features:**
- Singleton pattern (per symbol)
- 7-day cache expiration
- Multiple models: BSM, CRR binomial, European equivalent
- Supports American and European exercise
- Automatic data loading

**Key Methods:**

```python
# Compute implied volatilities
result = vol_mgr.get_implied_volatility_timeseries(
    start_date="2025-01-01",
    end_date="2025-01-31",
    strike=150.0,
    expiration="2025-06-20",
    right="call",
    market_model=OptionPricingModel.BSM,
    american=False,
    dividend_type=DivType.DISCRETE
)
ivs = result.timeseries  # pd.Series of implied vols

# Real-time IV
iv = vol_mgr.rt(strike=150.0, expiration="2025-06-20", right="call")
```

**Pricing Models:**
- `BSM`: Black-Scholes-Merton (fast, European only)
- `BINOMIAL`: Cox-Ross-Rubinstein tree (supports American)
- `EURO_EQIV`: European equivalent IV

### GreekDataManager

Computes option sensitivities (delta, gamma, vega, theta, rho).

**Features:**
- Singleton pattern (per symbol)
- 7-day cache expiration
- Models: BSM (analytical), Binomial (numerical)
- Selectable greeks to reduce computation
- Returns DataFrame with all greeks

**Key Methods:**

```python
# Compute option greeks
result = greek_mgr.get_greeks_timeseries(
    start_date="2025-01-01",
    end_date="2025-01-31",
    strike=150.0,
    expiration="2025-06-20",
    right="call",
    greeks_to_compute=[GreekType.DELTA, GreekType.GAMMA, GreekType.VEGA]
)
greeks = result.timeseries  # pd.DataFrame with [delta, gamma, vega, ...]

# Real-time greeks
result = greek_mgr.rt(strike=150.0, expiration="2025-06-20", right="call")
delta = result.timeseries["delta"].iloc[0]
```

**Available Greeks:**
- `DELTA`: Rate of change of option price with respect to spot
- `GAMMA`: Rate of change of delta with respect to spot
- `VEGA`: Sensitivity to volatility changes
- `THETA`: Time decay
- `RHO`: Sensitivity to interest rate changes
- `VOLGA`: Vomma (sensitivity of vega to volatility)
- `VANNA`: Sensitivity of delta to volatility

### Theoretical Pricing Functions

Module-level functions for option pricing and scenario analysis.

```python
from trade.datamanager.theo import get_option_theoretical_price, calculate_scenarios

# Theoretical pricing
theo_result = get_option_theoretical_price(
    symbol="AAPL",
    start_date="2025-01-01",
    end_date="2025-01-31",
    strike=150.0,
    expiration="2025-06-20",
    right="call",
    market_model=OptionPricingModel.BSM,
    dividend_type=DivType.DISCRETE
)
prices = theo_result.timeseries

# Scenario analysis (stress testing)
scenarios = calculate_scenarios(
    symbol="AAPL",
    as_of="2025-01-15",
    strike=150.0,
    expiration="2025-06-20",
    right="call",
    spot_scenarios=[0.95, 1.0, 1.05],  # -5%, 0%, +5% spot moves
    vol_scenarios=[-0.05, 0.0, 0.05],   # -5, 0, +5 vol points
    return_pnl=True
)
grid = scenarios.grid  # pd.DataFrame with spot × vol grid
```

---

## Unified Timeseries Interface

The `TimeseriesDataManager` provides a consistent API across all data types.

### Features

- **Standardized methods**: `rt()`, `get_at_time()`, `get_timeseries()`
- **Preserves original signatures**: Full docstrings and type hints
- **Property-based access**: `ts.spot`, `ts.vol`, `ts.greeks`, etc.
- **Pass-through to underlying**: Access specialized methods when needed

### Usage

```python
from trade.datamanager.timeseries import TimeseriesDataManager

ts = TimeseriesDataManager("AAPL")

# Spot data - simple interface
spot_rt = ts.spot.rt()
spot_hist = ts.spot.get_timeseries(start_date="2025-01-01", end_date="2025-01-31")

# Options data - pass parameters explicitly
vol = ts.vol.get_timeseries(
    start_date="2025-01-01",
    end_date="2025-01-31",
    strike=150.0,
    expiration="2025-06-20",
    right="call"
)

greeks = ts.greeks.rt(strike=150.0, expiration="2025-06-20", right="call")

# Access underlying manager if needed
vol_mgr = ts.vol._manager
```

### Available Properties

| Property | Manager | Methods Available |
|----------|---------|-------------------|
| `.spot` | SpotDataManager | rt, get_at_time, get_timeseries |
| `.vol` | VolDataManager | rt, get_at_time, get_timeseries |
| `.greeks` | GreekDataManager | rt, get_at_time, get_timeseries |
| `.forward` | ForwardDataManager | rt, get_timeseries |
| `.dividend` | DividendDataManager | rt, get_timeseries |
| `.rates` | RatesDataManager | rt, get_timeseries |
| `.option_spot` | OptionSpotDataManager | rt, get_at_time, get_timeseries |

---

## Convenience Loaders

### load_full_option_data()

One-call function to load complete option data packages.

```python
from trade.datamanager.loaders import load_full_option_data
from trade.datamanager._enums import DivType

# Load all required data in one call
pack = load_full_option_data(
    symbol="AAPL",
    strike=150.0,
    expiration="2025-06-20",
    right="call",
    start_date="2025-01-01",
    end_date="2025-01-31",
    dividend_type=DivType.DISCRETE
)

# Access all components
spot = pack.spot.timeseries         # Spot prices
forward = pack.forward.timeseries   # Forward prices
dividend = pack.dividend.timeseries # Dividend schedules
rates = pack.rates.timeseries       # Risk-free rates
option = pack.option_spot.timeseries # Market option prices
vol = pack.vol.timeseries           # Implied volatilities
greeks = pack.greek.timeseries      # Option greeks (DataFrame)
```

**Modes:**
- **Timeseries**: Provide `start_date` and `end_date`
- **Single date**: Provide `as_of`
- **Real-time**: Set `rt=True`

---

## Configuration

### OptionDataConfig (Singleton)

Global configuration for all data managers.

```python
from trade.datamanager.config import OptionDataConfig

config = OptionDataConfig()

# Modify settings
config.option_model = OptionPricingModel.BSM
config.dividend_type = DivType.DISCRETE
config.n_steps = 200  # Binomial tree steps
config.undo_adjust = True  # Use split-adjusted prices
```

**Key Settings:**

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `option_spot_endpoint_source` | OptionSpotEndpointSource | EOD | Data source for option prices |
| `dividend_type` | DivType | DISCRETE | Dividend model (DISCRETE/CONTINUOUS) |
| `option_model` | OptionPricingModel | BSM | Pricing model |
| `volatility_model` | VolatilityModel | MARKET | Vol source (MARKET/MODEL_DYNAMIC) |
| `n_steps` | int | 100 | Binomial tree steps |
| `undo_adjust` | bool | True | Use split-adjusted prices |
| `model_price` | ModelPrice | MIDPOINT | Price type (MIDPOINT/BID/ASK/etc.) |
| `real_time_fallback_option` | RealTimeFallbackOption | USE_LAST_AVAILABLE | Fallback for missing RT data |

### Key Enumerations

**DivType** (from optionlib):
- `DISCRETE`: Schedule-based dividends
- `CONTINUOUS`: Yield-based dividends

**OptionPricingModel**:
- `BSM`: Black-Scholes-Merton (fast, European)
- `BINOMIAL`: CRR tree (slower, American)
- `EURO_EQIV`: European equivalent

**VolatilityModel**:
- `MARKET`: Implied from market prices
- `MODEL_DYNAMIC`: Computed from model

**GreekType**:
- `DELTA`, `GAMMA`, `VEGA`, `THETA`, `RHO`, `VOLGA`, `VANNA`

---

## Caching System

### Three-Tier Architecture

1. **Memory Cache**: Fastest access, per-manager instance, cleared on exit
2. **Disk Cache**: Persistent across sessions, configurable expiration
3. **Partial Merging**: Detects missing dates, fetches only gaps, merges with existing

### Cache Configuration

Each manager defines cache behavior via `CacheSpec`:

```python
from trade.datamanager.base import CacheSpec

CACHE_SPEC = CacheSpec(
    base_dir=DM_GEN_PATH,          # Cache directory
    cache_fname="spot_data_manager",  # Cache filename
    default_expire_days=45,         # Full cache expiration
    clear_on_exit=False             # Auto-clear on exit
)
```

### Cache Keys

Constructed using `construct_cache_key()` utility:

**Components:**
- Symbol (e.g., "AAPL")
- Artifact type (SPOT, IV, GREEKS, etc.)
- Series ID (HIST, AT_TIME, SNAPSHOT)
- Interval (EOD, INTRADAY, NA)
- Optional namespace
- Additional metadata (strike, expiration, model, etc.)

**Example Keys:**
```
AAPL__hist__eod__spot__undo_True
AAPL__hist__eod__iv__K150.0_exp20250620_rc_model_bsm
```

---

## Best Practices

### ✅ DO

- **Use singleton managers** - They cache internally, avoid duplicate instances
- **Use `to_datetime()`** for all date conversions (from `trade.helpers.helper`)
- **Provide `DividendsResult`** to `ForwardDataManager` to avoid re-fetching
- **Use `undo_adjust=True`** for backtesting (split-adjusted prices)
- **Specify `greeks_to_compute`** to reduce computation time
- **Use `model_price=MIDPOINT`** for fair value calculations
- **Check `result.is_empty()`** before using data
- **Check `result.is_certified`** when relying on certification guarantees (distinct from `CertificationResult.success`)
- **Use `rt()` methods** for real-time data
- **Let managers handle data loading** automatically

### ❌ DON'T

- **Don't create multiple instances** of same symbol manager
- **Don't use `datetime.strptime()`** or `pd.to_datetime()` directly
- **Don't mix `undo_adjust=True/False`** in same calculation
- **Don't ignore `dividend_type`** when comparing prices
- **Don't call `_private_methods()`** directly
- **Don't modify `OptionDataConfig`** after initialization
- **Don't assume cache is warm** - check for empty results
- **Don't use BSM model** for American options pricing

### Common Issues

| Issue | Solution |
|-------|----------|
| "Data not available" | Check date range with `is_available_on_date()` |
| "Cache miss" | Normal on first run, subsequent runs hit cache |
| "IV solver failed" | Option may be deep ITM/OTM or have bad data |
| "Mismatched undo_adjust" | Ensure consistent split adjustment across managers |

### Date Handling (CRITICAL)

**Always use** `to_datetime` from `trade.helpers.helper`:

```python
from trade.helpers.helper import to_datetime

# Handles strings, datetime objects, and iterables
date_obj = to_datetime("2025-01-15")
dates = to_datetime(["2025-01-15", "2025-01-16"])
```

**Never use:**
- `datetime.strptime()`
- `pd.to_datetime()` directly

---

## Complete Example: Option Backtesting Workflow

```python
from trade.datamanager import (
    SpotDataManager, VolDataManager, GreekDataManager
)
from trade.datamanager.timeseries import TimeseriesDataManager
from trade.datamanager.loaders import load_full_option_data
from trade.datamanager._enums import DivType, OptionPricingModel, GreekType

# Parameters
symbol = "AAPL"
start, end = "2025-01-01", "2025-01-31"
strike, expiration, right = 150.0, "2025-06-20", "call"

# Method 1: Using individual managers (granular control)
spot_mgr = SpotDataManager(symbol)
vol_mgr = VolDataManager(symbol)
greek_mgr = GreekDataManager(symbol)

spot_result = spot_mgr.get_spot_timeseries(start, end, undo_adjust=True)
vol_result = vol_mgr.get_implied_volatility_timeseries(
    start, end, strike, expiration, right,
    market_model=OptionPricingModel.BSM,
    dividend_type=DivType.DISCRETE
)
greek_result = greek_mgr.get_greeks_timeseries(
    start, end, strike, expiration, right,
    greeks_to_compute=[GreekType.DELTA, GreekType.GAMMA, GreekType.VEGA]
)

# Method 2: Using unified interface (simplified)
ts = TimeseriesDataManager(symbol)
spot_result = ts.spot.get_timeseries(start, end)
vol_result = ts.vol.get_timeseries(start, end, strike, expiration, right)
greek_result = ts.greeks.get_timeseries(start, end, strike, expiration, right)

# Method 3: One-call loader (most convenient)
pack = load_full_option_data(
    symbol=symbol,
    strike=strike,
    expiration=expiration,
    right=right,
    start_date=start,
    end_date=end,
    dividend_type=DivType.DISCRETE
)

# Access results
spots = pack.spot.timeseries
vols = pack.vol.timeseries
greeks_df = pack.greek.timeseries  # DataFrame with delta, gamma, vega, etc.

# Run scenario analysis
from trade.datamanager.theo import calculate_scenarios

scenarios = calculate_scenarios(
    symbol=symbol,
    as_of="2025-01-15",
    strike=strike,
    expiration=expiration,
    right=right,
    spot_scenarios=[0.95, 1.0, 1.05],  # ±5% spot moves
    vol_scenarios=[-0.05, 0.0, 0.05],   # ±5 vol points
    return_pnl=True
)

print(scenarios.grid)
```

---

## Module Structure

```
trade/datamanager/
├── README.md                # Index into docs/
├── docs/
│   ├── README.md            # This guide
│   ├── UPGRADE_PLAN.md      # Certification rollout
│   └── FUTURE_CHANGES.md    # Backlog
├── __init__.py              # Public API exports
├── base.py                  # BaseDataManager, CacheSpec
├── config.py                # OptionDataConfig singleton
├── result.py                # Result dataclasses
├── _enums.py                # Enumerations
│
├── certification/           # L1/L2/L3 pipeline, integration, reports
├── spot.py                  # SpotDataManager
├── rates.py                 # RatesDataManager
├── dividend.py              # DividendDataManager
├── forward.py               # ForwardDataManager
├── option_spot.py           # OptionSpotDataManager
├── market_data.py           # MarketTimeseries
│
├── vol.py                   # VolDataManager
├── greeks.py                # GreekDataManager
├── theo.py                  # Theoretical pricing functions
│
├── timeseries.py            # TimeseriesDataManager, TimeseriesAdapter
├── loaders.py               # load_full_option_data()
├── requests.py              # LoadRequest (gap-aware validation)
│
├── utils/
│   ├── model.py             # Model data loading
│   ├── vol_helpers.py       # Volatility calculation
│   ├── greeks_helpers.py    # Greek calculation
│   ├── date.py              # Date sync, is_available_on_date
│   ├── point_in_time.py     # resolve_value_at_date
│   ├── cache.py             # Cache utilities
│   ├── data_structure.py    # Sanitize + validation
│   ├── logging.py           # Logger names (incl. certification.report)
│   └── enums_utils.py       # Cache key construction
│
├── tests/
│   └── run_certification_deep_test.py
│
└── market_data_helpers/
    └── spot.py              # Spot price loading helpers
```

---

## Contributing

To add a new manager:

1. Inherit from `BaseDataManager`
2. Define `CACHE_NAME` (unique string)
3. Define `CACHE_SPEC` (CacheSpec instance)
4. Define `DEFAULT_SERIES_ID` (SeriesId enum)
5. Implement `__init__` with singleton pattern if needed
6. Add methods returning Result subclass
7. Use `self.cache.get()` / `self.cache.set()` for caching
8. Use `construct_cache_key()` for key generation
9. Wire `certify_manager_result` at the return boundary for new timeseries getters (see `certification/integration.py`)

**Example skeleton:**

```python
from trade.datamanager.base import BaseDataManager, CacheSpec
from trade.datamanager._enums import SeriesId

class MyDataManager(BaseDataManager):
    CACHE_NAME: str = "my_data_manager"
    CACHE_SPEC: CacheSpec = CacheSpec(cache_fname=CACHE_NAME)
    DEFAULT_SERIES_ID: SeriesId = SeriesId.HIST
    
    def __init__(self, symbol: str):
        super().__init__(symbol=symbol)
        self.symbol = symbol
    
    def get_my_data(self, start, end) -> MyResult:
        key = construct_cache_key(...)
        cached = self.cache.get(key)
        if cached:
            return cached
        
        # Fetch data
        result = MyResult(...)
        self.cache.set(key, result)
        return result
```

---

## License

See main project LICENSE file.

## Support

For issues and questions, please refer to the main QuantTools repository.
