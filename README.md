# QuantTools

**QuantTools** is a comprehensive Python framework for quantitative trading research, backtesting, and portfolio management with a focus on options trading strategies. Built around an event-driven architecture, it enables realistic simulation of trading workflows including signal generation, risk management, order execution, and performance analysis.

## What Problems Does This Solve?

QuantTools addresses key challenges in algorithmic trading development:

1. **Realistic Backtesting**: Traditional vectorized backtests ignore market microstructure, execution timing, and position management complexities. QuantTools uses an event-driven engine that processes trades chronologically with T+N settlement, slippage modeling, and realistic order flow.

2. **Options-First Design**: Most backtesting frameworks focus on equities. QuantTools provides native support for options strategies with Greek-based risk management, option chain analysis, position rolling, and exercise handling.

3. **Modular Risk Management**: Separates risk logic from strategy logic, allowing position limits, Greek constraints, and sizing rules to be configured independently via "cogs" (pluggable risk modules).

4. **Data Infrastructure**: Includes caching mechanisms, market data management, and timeseries handling optimized for backtesting large option universes across multi-year periods.

5. **Production-Ready Patterns**: Code structure mirrors production trading systems with clear separation between data handlers, strategy signals, portfolio state, execution simulation, and risk controls.

---

## Features

- **Event-Driven Backtesting Engine** (`EventDriven/`)
  - Queue-based event processing (MarketEvent, SignalEvent, OrderEvent, FillEvent)
  - Realistic T+N settlement delays and slippage modeling
  - Support for corporate actions (dividends, splits, assignments)
  - Position-level P&L tracking and attribution

- **Portfolio Management** (`EventDriven/new_portfolio.py`)
  - Real-time position tracking for options and equities
  - Holdings valuation with mark-to-market updates
  - Trade ledger with complete audit trail
  - Cash allocation and margin management

- **Risk Management** (`EventDriven/riskmanager/`)
  - Greek-based limits (Delta, Gamma, Vega, Theta)
  - Position analyzer with pluggable "cogs" for custom rules
  - Intelligent order picker for option chain selection
  - Dynamic position sizing (fixed, z-score-based, etc.)

- **Data Handling** (`EventDriven/data.py`, `trade/datamanager/`)
  - Historic trade data handler with bar-by-bar iteration
  - Disk-based caching with configurable TTL (`CustomCache`)
  - Integration with ThetaData, yfinance, OpenBB
  - Market timeseries management with holiday/trading day awareness

- **Options Pricing Library** (`trade/optionlib/`)
  - Black-Scholes-Merton pricing and Greeks
  - Implied volatility calculations
  - Dividend schedule handling (discrete and continuous)
  - Support for European and American options

- **Performance Analytics** (`EventDriven/performance.py`)
  - Sharpe ratio calculation
  - Drawdown analysis
  - Returns attribution
  - Trade-level statistics

- **Configuration System** (`EventDriven/configs/`)
  - Centralized configuration management via dataclasses
  - Frozen configs for immutability guarantees
  - Validation and type checking
  - Export/import configuration dictionaries

---

## Installation

### Requirements

- **Python**: 3.10+ recommended (uses modern type hints)
- **Key Dependencies**: numpy, pandas, matplotlib, yfinance, QuantLib, diskcache
- **Optional**: ThetaData API access for high-quality options data

### Editable Install (Development)

```bash
# Clone the repository
git clone https://github.com/Zino-ctrlZ/QuantTools.git
cd QuantTools

# Install in editable mode with dependencies
pip install -e .

# Verify installation
python -c "from EventDriven.backtest import OptionSignalBacktest; print('Success!')"
```

### Environment Setup

Create a `.env` file in the repo root:

```env
# Required paths
WORK_DIR=/path/to/QuantTools
DBASE_DIR=/path/to/your/database/module  # if using ThetaData

# Optional: data sources
THETA_USERNAME=your_username
THETA_PASSWORD=your_password

# Cache configuration
GEN_CACHE_PATH=/path/to/cache/directory
```

---

## Repository Structure

```
QuantTools/
├── EventDriven/              # Core backtesting engine
│   ├── backtest.py           # Main OptionSignalBacktest class
│   ├── new_portfolio.py      # Portfolio state management
│   ├── strategy.py           # Strategy base classes
│   ├── data.py               # Data handlers (HistoricTradeDataHandler)
│   ├── event.py              # Event types (MarketEvent, SignalEvent, etc.)
│   ├── eventScheduler.py     # Event queue coordinator
│   ├── execution.py          # Execution handlers with slippage
│   ├── trade.py              # Trade object definitions
│   ├── tradeLedger.py        # Trade ledger for audit trail
│   ├── performance.py        # Performance metrics (Sharpe, drawdowns)
│   ├── helpers.py            # Utility functions
│   ├── types.py              # Type definitions and enums
│   ├── exceptions.py         # Custom exceptions
│   ├── riskmanager/          # Risk management system
│   │   ├── new_base.py       # RiskManager orchestrator
│   │   ├── order_picker.py   # Option chain search logic
│   │   ├── cogs/             # Pluggable risk rules
│   │   └── market_data.py    # Market data caching
│   ├── configs/              # Configuration system
│   │   ├── core.py           # Config dataclasses
│   │   ├── base.py           # Base config functionality
│   │   └── export_configs.py # Config serialization
│   ├── dataclasses/          # Structured data types
│   │   ├── orders.py         # Order request structures
│   │   └── states.py         # Position/portfolio states
│   ├── demos/                # Example scripts
│   │   └── demoRun.py        # Full backtest example
│   └── notebooks/            # Jupyter exploration notebooks
│
├── trade/                    # Trading utilities library
│   ├── helpers/              # Helper functions
│   │   ├── helper.py         # Core utilities, CustomCache
│   │   ├── Logging.py        # Logging setup
│   │   └── Configuration.py  # Config management
│   ├── optionlib/            # Options pricing library
│   │   ├── pricing/          # Pricing models (BSM, etc.)
│   │   ├── greeks/           # Greek calculations
│   │   ├── vol/              # Volatility models
│   │   └── assets/           # Dividend handling
│   ├── assets/               # Asset classes (Stock, Option)
│   ├── datamanager/          # Data management (WIP on CHIDI-JAN04 branch)
│   ├── backtester_/          # Legacy backtester (being replaced)
│   └── models/               # Statistical/ML models
│
├── module_test/              # Module-level test scripts
├── setup.py                  # Package installation
├── setup.cfg                 # Package metadata
├── ruff.toml                 # Linting configuration (Ruff)
├── pricingConfig.json        # Option pricing defaults
└── logs/                     # Runtime logs
```

### Key Module Responsibilities

- **EventDriven/**: Complete event-driven backtesting framework. Import from here for backtests.
- **trade/helpers/**: Cross-cutting utilities (caching, logging, date helpers, config management).
- **trade/optionlib/**: Options pricing and Greeks. Used by RiskManager for valuation.
- **trade/assets/**: Asset class definitions (Stock, Option). Provides data access methods.
- **EventDriven/riskmanager/**: All risk logic (limits, sizing, order selection). Plugs into Portfolio.
- **EventDriven/configs/**: Configuration dataclasses. Pass these to control backtest behavior.

---

## Core Concepts

### 1. Event-Driven Backtesting Flow

QuantTools processes trades chronologically through an event queue:

```
1. MarketEvent → 2. SignalEvent → 3. OrderEvent → 4. FillEvent
        ↓                ↓               ↓              ↓
   DataHandler    Strategy    RiskManager/Portfolio  Executor
```

**Event Types** (see `EventDriven/event.py`):
- **MarketEvent**: New bar available, triggers strategy to check for signals
- **SignalEvent**: Strategy wants to open/close a position
- **OrderEvent**: Portfolio approved an order, sends to executor
- **FillEvent**: Order filled, updates portfolio holdings
- **RollEvent**: Option position needs rolling (exercise, expiry)
- **ExerciseEvent**: Option exercised early (American options)

**Workflow**:
1. `HistoricTradeDataHandler` iterates through dates, emitting `MarketEvent`
2. `Strategy.calculate_signals()` analyzes current bar, emits `SignalEvent`
3. `Portfolio.analyze_signal()` checks cash/limits, requests order from `RiskManager`
4. `RiskManager` searches option chains, returns `OrderRequest`
5. Portfolio converts to `OrderEvent`, puts in queue
6. `SimulatedExecutionHandler` simulates fill with slippage, emits `FillEvent`
7. `Portfolio.update_fill()` updates positions and cash

### 2. Data Access and Caching

**CustomCache** (`trade/helpers/helper.py`):
- Disk-backed cache using `diskcache` library
- Configurable expiration (default 7 days)
- Used for market data, option chains, pricing calculations
- Example:
  ```python
  from trade.helpers.helper import CustomCache
  
  cache = CustomCache(location='/tmp/mycache', fname='test', expire_days=30)
  cache['AAPL_2024-01-01'] = {'close': 185.92}
  value = cache.get('AAPL_2024-01-01', default=None)
  ```

**HistoricTradeDataHandler** (`EventDriven/data.py`):
- Loads trade data from DataFrame
- Iterates bar-by-bar simulating real-time data feed
- Handles multi-symbol backtests
- Emits `MarketEvent` for each new bar

**Market Timeseries** (`EventDriven/riskmanager/market_data.py`):
- Manages historical price data for underlyings
- Caches OHLCV data per symbol
- Provides dividend yield history
- Handles trading day calendars

### 3. Portfolio and Risk Management

**OptionSignalPortfolio** (`EventDriven/new_portfolio.py`):
- Maintains current positions (long/short options and equities)
- Tracks cash, holdings value, unrealized P&L
- Generates orders from signals (via RiskManager)
- Updates state on fills
- Manages position lifecycle (open, roll, close, exercise)

**RiskManager** (`EventDriven/riskmanager/new_base.py`):
- **OrderPicker**: Searches option chains for contracts matching criteria (delta, strike, expiry)
- **PositionAnalyzer**: Runs "cogs" (modular rules) on current positions to recommend actions
- **LimitsAndSizing**: Enforces Greek limits, position size constraints
- **Market Data**: Caches option chains, spot prices, dividends

**Cogs** (`EventDriven/riskmanager/cogs/`):
Pluggable modules that analyze positions and return recommendations:
- `LimitsAndSizingCog`: Check if position violates Greek/size limits
- `PositionSignalsCog`: Re-evaluate signals for open positions
- `StrategyRollCog`: Determine if positions should be rolled
- Custom cogs: Extend `BaseCog` for your own logic

### 4. Configuration System

Configurations use frozen dataclasses (`EventDriven/configs/core.py`):

```python
from EventDriven.configs.core import BacktesterConfig, RiskManagerConfig

# Configure backtest behavior
backtest_config = BacktesterConfig(
    t_plus_n=1,                    # T+1 settlement
    max_slippage_pct=0.0015,       # 0.15% slippage
    slippage_enabled=True,
    logger_override_level="INFO"
)

# Configure risk controls
risk_config = RiskManagerConfig(
    delta_limit=0.30,              # Max 30% delta exposure per position
    gamma_limit=0.10,
    vega_limit=0.50,
    use_portfolio_level_limits=True
)
```

Pass configs to classes:
```python
backtest = OptionSignalBacktest(trades_df, config=backtest_config)
risk_manager = RiskManager(bars, events, config=risk_config)
```

### 5. Logging Conventions

Logging uses Python's standard `logging` module configured via `trade/helpers/Logging.py`:

```python
from trade.helpers.Logging import setup_logger

logger = setup_logger('MyModule', log_level='DEBUG')
logger.info("Backtest started")
logger.warning("Position limit exceeded")
logger.error("Failed to retrieve option chain", exc_info=True)
```

Logs are written to:
- Console (stdout/stderr)
- `logs/__main__.log.<date>` (daily rotation)

**Linting**: Uses `ruff` (configured in `ruff.toml`):
```bash
ruff check .                # Check for issues
ruff check --fix .          # Auto-fix issues
ruff format .               # Format code
```

---

## Quickstart

### Minimal Backtest Example

This example creates a simple backtest with dummy trade data:

```python
import pandas as pd
from EventDriven.backtest import OptionSignalBacktest
from EventDriven.configs.core import BacktesterConfig

# Create sample trade data
trades = pd.DataFrame({
    'Ticker': ['AAPL', 'AAPL', 'MSFT'],
    'EntryTime': ['2024-01-02', '2024-01-10', '2024-01-05'],
    'ExitTime': ['2024-01-15', '2024-01-25', '2024-01-20'],
    'Size': [10, -5, 8],                    # Positive = long, negative = short
    'EntryPrice': [150.0, 155.0, 380.0],
    'ExitPrice': [158.0, 152.0, 390.0],
    'Type': ['CALL', 'PUT', 'CALL'],
    'Strike': [145, 160, 375],
    'Expiry': ['2024-02-16', '2024-02-16', '2024-02-16']
})

# Configure backtest
config = BacktesterConfig(
    t_plus_n=1,                # T+1 settlement
    max_slippage_pct=0.001,    # 0.1% slippage
    slippage_enabled=True
)

# Run backtest
backtest = OptionSignalBacktest(
    trades=trades,
    initial_capital=100000,
    config=config
)

# Execute
import asyncio
asyncio.run(backtest.run())

# Access results
print(f"Final Portfolio Value: ${backtest.portfolio.current_holdings['total']:,.2f}")
print(f"Total Return: {backtest.portfolio.current_holdings['total'] / 100000 - 1:.2%}")

# View trade ledger
ledger = backtest.portfolio.ledger
print("\nTrade Ledger:")
print(ledger[['date', 'ticker', 'action', 'quantity', 'price', 'cost']])

# Performance metrics
from EventDriven.performance import create_sharpe_ratio, create_drawdowns

equity_curve = backtest.portfolio.all_holdings
returns = equity_curve['total'].pct_change().dropna()
sharpe = create_sharpe_ratio(returns, periods=252)
print(f"\nSharpe Ratio: {sharpe:.3f}")
```

**Expected Output Shape**:
- `portfolio.ledger`: DataFrame with columns `['date', 'ticker', 'action', 'quantity', 'price', 'cost', 'commission']`
- `portfolio.all_holdings`: DataFrame with datetime index, columns for each symbol + 'cash', 'commission', 'total'
- `portfolio.current_positions`: Dict mapping position_id → PositionState objects
- Logs printed to console showing event processing

---

## How-To Examples

### 1. Run a Backtest with Custom Risk Limits

Enforce Greek limits and position sizing constraints:

```python
from EventDriven.backtest import OptionSignalBacktest
from EventDriven.configs.core import BacktesterConfig, RiskManagerConfig

# Configure tight risk controls
risk_config = RiskManagerConfig(
    delta_limit=0.25,              # Max 25% delta per position
    gamma_limit=0.08,              # Limit gamma exposure
    vega_limit=0.40,               # Limit vega exposure
    max_position_size=20,          # Max 20 contracts per position
    use_portfolio_level_limits=True
)

backtest_config = BacktesterConfig(
    t_plus_n=1,
    risk_manager_config=risk_config
)

# Run backtest - orders violating limits will be rejected
backtest = OptionSignalBacktest(
    trades=your_trades_df,
    initial_capital=100000,
    config=backtest_config
)
await backtest.run()

# Check rejected orders
print("Rejected Orders:", backtest.portfolio.ledger[backtest.portfolio.ledger['action'] == 'REJECTED'])
```

**Assumptions**: Requires `your_trades_df` with columns: `['Ticker', 'EntryTime', 'ExitTime', 'Size', 'EntryPrice', 'ExitPrice', 'Type', 'Strike', 'Expiry']`

### 2. Using CustomCache for Data Management

Cache expensive computations (e.g., option chains, pricing data):

```python
from trade.helpers.helper import CustomCache
import pandas as pd

# Initialize cache with 30-day expiration
cache = CustomCache(
    location='/tmp/options_cache',
    fname='aapl_chains',
    expire_days=30,
    clear_on_exit=False  # Persist across runs
)

# Cache option chain data
ticker = 'AAPL'
date = '2024-01-02'
key = f"{ticker}_{date}_chain"

if key in cache:
    chain = cache[key]
    print("Loaded from cache")
else:
    # Expensive operation: fetch option chain
    chain = fetch_option_chain(ticker, date)  # Your data source
    cache[key] = chain
    print("Cached for future use")

# Use the chain
print(chain[['strike', 'call_bid', 'call_ask', 'iv']])
```

**Notes**: 
- Cache automatically cleans up expired entries
- Set `clear_on_exit=True` for temporary caches (testing)
- Cache location persists; same `fname` retrieves same cache across sessions

### 3. Constructing and Updating a Portfolio Manually

Directly interact with Portfolio for custom workflows:

```python
from EventDriven.new_portfolio import OptionSignalPortfolio
from EventDriven.data import HistoricTradeDataHandler
from EventDriven.eventScheduler import EventScheduler
from EventDriven.riskmanager.new_base import RiskManager
from EventDriven.event import FillEvent
from EventDriven.types import FillDirection

# Setup components
bars = HistoricTradeDataHandler(trades_df, symbol_list=['AAPL'])
events = EventScheduler()
risk_mgr = RiskManager(bars, events)

portfolio = OptionSignalPortfolio(
    bars=bars,
    eventScheduler=events,
    risk_manager=risk_mgr,
    initial_capital=50000
)

# Manually create a fill (simulating order execution)
fill = FillEvent(
    timeindex=pd.Timestamp('2024-01-02'),
    symbol='AAPL',
    exchange='NASDAQ',
    quantity=10,
    direction=FillDirection.BUY,
    fill_cost=1500.0,  # Total cost including commission
    commission=1.50,
    option_type='CALL',
    strike=150,
    expiry='2024-02-16'
)

# Update portfolio with fill
portfolio.update_fill(fill)

# Check positions
print("Current Positions:", portfolio.current_positions)
print("Current Cash:", portfolio.current_holdings['cash'])
```

**Use Case**: Manual portfolio construction for sensitivity analysis or custom order sequences.

### 4. Adding a Custom Strategy

Create a strategy that generates signals based on your logic:

```python
from EventDriven.strategy import Strategy
from EventDriven.event import SignalEvent
from EventDriven.types import SignalTypes
from EventDriven.helpers import generate_signal_id

class MomentumStrategy(Strategy):
    """
    Simple momentum strategy: buy when 20-day MA > 50-day MA
    """
    
    def __init__(self, bars, events, short_window=20, long_window=50):
        self.bars = bars
        self.events = events
        self.symbol_list = bars.symbol_list
        self.short_window = short_window
        self.long_window = long_window
        self.positions = {s: False for s in self.symbol_list}
    
    def calculate_signals(self, event):
        if event.type != 'MARKET':
            return
        
        for symbol in self.symbol_list:
            bars = self.bars.get_latest_bars(symbol, N=self.long_window)
            if len(bars) < self.long_window:
                continue
            
            # Calculate moving averages
            closes = [bar['close'] for bar in bars]
            short_ma = sum(closes[-self.short_window:]) / self.short_window
            long_ma = sum(closes) / self.long_window
            
            # Generate signal
            if short_ma > long_ma and not self.positions[symbol]:
                # Buy signal
                signal = SignalEvent(
                    ticker=symbol,
                    datetime=bars[-1]['datetime'],
                    signal_id=generate_signal_id(),
                    signal_type=SignalTypes.LONG,
                    suggested_quantity=1
                )
                self.events.put(signal)
                self.positions[symbol] = True
            
            elif short_ma < long_ma and self.positions[symbol]:
                # Sell signal
                signal = SignalEvent(
                    ticker=symbol,
                    datetime=bars[-1]['datetime'],
                    signal_id=generate_signal_id(),
                    signal_type=SignalTypes.EXIT,
                    suggested_quantity=-1
                )
                self.events.put(signal)
                self.positions[symbol] = False

# Use in backtest
from EventDriven.backtest import OptionSignalBacktest

# Replace default strategy with custom one
backtest = OptionSignalBacktest(trades_df, initial_capital=100000)
backtest.strategy = MomentumStrategy(backtest.bars, backtest.eventScheduler)
await backtest.run()
```

**Note**: Custom strategies must inherit from `Strategy` and implement `calculate_signals(event)`.

### 5. Analyzing Position Greeks and Limits

Use RiskManager to evaluate Greeks for current positions:

```python
from EventDriven.riskmanager.new_base import RiskManager
from EventDriven.configs.core import RiskManagerConfig

# Configure Greek limits
config = RiskManagerConfig(
    delta_limit=0.30,
    gamma_limit=0.10,
    vega_limit=0.50,
    theta_limit=-0.05  # Negative = losing value over time
)

risk_mgr = RiskManager(bars, events, config=config)

# Analyze a specific position
position_state = portfolio.current_positions['AAPL_CALL_150_2024-02-16']

# Load market data for analysis
analysis_date = pd.Timestamp('2024-01-15')
risk_mgr._load_market_data_for_date(analysis_date)

# Calculate Greeks (done automatically in analyze_position)
from EventDriven.dataclasses.states import PositionAnalysisContext

context = PositionAnalysisContext(
    position=position_state,
    analysis_date=analysis_date,
    portfolio_state=portfolio.get_portfolio_state(),
    market_data=risk_mgr.timeseries
)

# Run analysis (checks limits, signals, rolls)
recommendations = risk_mgr.analyze_position(context)

print("Position Analysis:")
print(f"Delta: {position_state.greeks.delta:.3f} (Limit: {config.delta_limit})")
print(f"Gamma: {position_state.greeks.gamma:.3f} (Limit: {config.gamma_limit})")
print(f"Vega: {position_state.greeks.vega:.3f} (Limit: {config.vega_limit})")
print(f"Recommendations: {recommendations}")
```

**Use Case**: Real-time position monitoring, pre-trade risk checks, limit breach alerts.

### 6. Handling Dividends in Option Pricing

QuantTools supports both discrete and continuous dividend models:

```python
from trade.optionlib.assets.dividend import (
    get_vectorized_dividend_schedule,
    get_vectorized_continuous_dividends
)
from trade.optionlib.config.types import DiscreteDivGrowthModel

# Discrete dividends (actual ex-dates and amounts)
div_schedule = get_vectorized_dividend_schedule(
    tickers=['AAPL'],
    start_dates=['2024-01-01'],
    end_dates=['2024-12-31'],
    method=DiscreteDivGrowthModel.CONSTANT_AVG.value,
    lookback_years=2
)

print("Discrete Dividend Schedule:")
for entry in div_schedule[0].schedule:
    print(f"  {entry.date}: ${entry.amount:.2f}")

# Continuous dividend yield (for European options)
continuous_yield = get_vectorized_continuous_dividends(
    tickers=['AAPL'],
    start_dates=['2024-01-01'],
    end_dates=['2024-12-31']
)

print(f"\nAnnualized Dividend Yield: {continuous_yield[0]:.2%}")

# Use in option pricing
from trade.optionlib.pricing.bsm import black_scholes_merton_price

price = black_scholes_merton_price(
    S=150.0,           # Spot price
    K=145.0,           # Strike
    T=0.5,             # Time to expiry (years)
    r=0.05,            # Risk-free rate
    sigma=0.25,        # Implied volatility
    q=continuous_yield[0],  # Dividend yield
    option_type='call'
)
print(f"Option Price: ${price:.2f}")
```

**Assumptions**: Requires dividend history data (fetched from yfinance or database).

### 7. Performance Tips and Common Pitfalls

**Tip 1: Cache Option Chains Aggressively**
```python
# BAD: Fetching chains every iteration
for date in dates:
    chain = fetch_option_chain(ticker, date)  # Network call every time
    
# GOOD: Cache chains with CustomCache
cache = CustomCache(fname='option_chains', expire_days=7)
for date in dates:
    key = f"{ticker}_{date}"
    if key not in cache:
        cache[key] = fetch_option_chain(ticker, date)
    chain = cache[key]
```

**Tip 2: Use T+N Settlement Delays**
```python
# BAD: Assumes instant settlement
config = BacktesterConfig(t_plus_n=0)  # Unrealistic

# GOOD: T+1 for options/equities
config = BacktesterConfig(t_plus_n=1)  # Matches real settlement
```

**Tip 3: Handle Missing Data Gracefully**
```python
# BAD: Crash on missing bar
bars = self.bars.get_latest_bars(symbol, N=50)
close = bars[-1]['close']  # IndexError if bars empty

# GOOD: Check before accessing
bars = self.bars.get_latest_bars(symbol, N=50)
if bars is None or len(bars) < 50:
    return  # Skip this symbol
close = bars[-1]['close']
```

**Pitfall 1: Forgetting to Update Market Data**
When using RiskManager standalone, call `_load_market_data_for_date()` before analysis:
```python
risk_mgr._load_market_data_for_date(analysis_date)
recommendations = risk_mgr.analyze_position(context)
```

**Pitfall 2: Mutating Shared State**
Portfolio state is mutable. Use `deepcopy` if needed:
```python
from copy import deepcopy

current_state = deepcopy(portfolio.current_positions)
# Modify current_state without affecting portfolio
```

**Pitfall 3: Mixing Async and Sync Code**
`OptionSignalBacktest.run()` is async. Always use `await` or `asyncio.run()`:
```python
# BAD
backtest.run()  # Returns coroutine, doesn't execute

# GOOD
import asyncio
asyncio.run(backtest.run())

# Or in Jupyter
await backtest.run()
```

---

## Extended Examples: trade/ Module Deep Dive

### 8. Option Pricing with Black-Scholes

Use the vectorized Black-Scholes implementation for fast option pricing:

```python
from trade.optionlib.pricing.black_scholes import (
    black_scholes_vectorized,
    black_scholes_with_carry_div
)
from trade.optionlib.assets.forward import EquityForward
import numpy as np

# Single option pricing
spot = 150.0
strike = 155.0
time_to_expiry = 0.25  # 3 months
risk_free_rate = 0.05
volatility = 0.30
dividend_yield = 0.02

# Calculate forward price
forward = EquityForward(
    S=spot,
    T=time_to_expiry,
    r=risk_free_rate,
    q=dividend_yield
)
F = forward.price()

# Price call and put
call_price = black_scholes_vectorized(
    F=F,
    K=strike,
    T=time_to_expiry,
    r=risk_free_rate,
    sigma=volatility,
    option_type='c'
)

put_price = black_scholes_vectorized(
    F=F,
    K=strike,
    T=time_to_expiry,
    r=risk_free_rate,
    sigma=volatility,
    option_type='p'
)

print(f"Call Price: ${call_price[0]:.2f}")
print(f"Put Price: ${put_price[0]:.2f}")

# Vectorized pricing across multiple strikes
strikes = np.array([145, 150, 155, 160, 165])
call_prices = black_scholes_vectorized(
    F=F,
    K=strikes,
    T=time_to_expiry,
    r=risk_free_rate,
    sigma=volatility,
    option_type='c'
)

import pandas as pd
chain = pd.DataFrame({
    'Strike': strikes,
    'Call': call_prices,
    'Intrinsic': np.maximum(spot - strikes, 0),
    'Time_Value': call_prices - np.maximum(spot - strikes, 0)
})
print("\nOption Chain:")
print(chain)
```

**Output**: Vectorized pricing is ~10x faster than looping for large chains.

### 9. Calculating Greeks (Delta, Gamma, Vega, Theta)

Greeks can be calculated analytically or numerically:

```python
from trade.optionlib.greeks import vectorized_market_greeks_bsm
from trade.optionlib.greeks.numerical.finite_diff import FiniteGreeksEstimator

# Analytical Greeks (fast, exact for BSM)
greeks_result = vectorized_market_greeks_bsm(
    S=150.0,
    K=155.0,
    T=0.25,
    r=0.05,
    sigma=0.30,
    q=0.02,
    option_type='c'
)

print("Analytical Greeks (BSM):")
print(f"  Delta: {greeks_result.delta[0]:.4f}")
print(f"  Gamma: {greeks_result.gamma[0]:.4f}")
print(f"  Vega: {greeks_result.vega[0]:.4f}")
print(f"  Theta: {greeks_result.theta[0]:.4f}")
print(f"  Rho: {greeks_result.rho[0]:.4f}")

# Numerical Greeks (for American options or complex models)
from trade.optionlib.pricing.bjs2002 import bjerksund_stensland_2002_vectorized

estimator = FiniteGreeksEstimator(
    price_function=bjerksund_stensland_2002_vectorized,
    params={
        'S': 150.0,
        'K': 155.0,
        'T': 0.25,
        'r': 0.05,
        'sigma': 0.30,
        'q': 0.02,
        'option_type': 'c'
    }
)

# Calculate first-order Greeks
delta = estimator.delta()
vega = estimator.vega()
theta = estimator.theta()

# Calculate second-order Greeks
gamma = estimator.gamma()
vomma = estimator.vomma()  # d²V/dσ²
vanna = estimator.vanna()  # d²V/dS/dσ

print("\nNumerical Greeks (American Option):")
print(f"  Delta: {delta:.4f}")
print(f"  Gamma: {gamma:.4f}")
print(f"  Vega: {vega:.4f}")
print(f"  Vomma: {vomma:.6f}")
print(f"  Vanna: {vanna:.6f}")
```

**Use Case**: Risk management, hedging, Greeks-based trading strategies.

### 10. Implied Volatility Calculation

Extract implied volatility from market prices:

```python
from trade.optionlib.vol.implied_vol import (
    bsm_vol_est_minimization,
    bsm_vol_est_brute_force,
    vectorized_iv_calculation
)

# Single IV calculation
market_price = 5.50
spot = 150.0
strike = 155.0
time_to_expiry = 0.25
risk_free_rate = 0.05
dividend_yield = 0.02

# Calculate forward
from trade.optionlib.assets.forward import EquityForward
forward = EquityForward(S=spot, T=time_to_expiry, r=risk_free_rate, q=dividend_yield)
F = forward.price()

# Method 1: Optimization-based (fast, accurate)
iv_optimized = bsm_vol_est_minimization(
    F=F,
    K=strike,
    T=time_to_expiry,
    r=risk_free_rate,
    market_price=market_price,
    option_type='c'
)
print(f"Implied Volatility (Optimized): {iv_optimized:.2%}")

# Method 2: Brute force (slower but robust)
iv_brute = bsm_vol_est_brute_force(
    F=F,
    K=strike,
    T=time_to_expiry,
    r=risk_free_rate,
    market_price=market_price,
    option_type='c'
)
print(f"Implied Volatility (Brute Force): {iv_brute:.2%}")

# Vectorized IV calculation for entire chain
import numpy as np
strikes = np.array([145, 150, 155, 160, 165])
market_prices = np.array([8.50, 5.80, 3.90, 2.50, 1.60])

ivs = vectorized_iv_calculation(
    S=spot,
    K=strikes,
    T=time_to_expiry,
    r=risk_free_rate,
    q=dividend_yield,
    market_prices=market_prices,
    option_type='c'
)

# Plot volatility smile
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(strikes, ivs * 100, marker='o')
plt.xlabel('Strike Price')
plt.ylabel('Implied Volatility (%)')
plt.title('Volatility Smile')
plt.grid(True)
plt.show()
```

**Note**: If market price is below intrinsic value, IV calculation will fail (returns NaN).

### 11. Using Decorators for Performance Monitoring

QuantTools includes powerful decorators for logging, timing, and profiling:

```python
from trade.helpers.decorators import timeit, log_time, log_error, cProfiler

# @timeit: Track execution time and save metadata
@timeit
def expensive_computation(n):
    """Simulate expensive computation"""
    result = sum(i**2 for i in range(n))
    return result

# Run multiple times
for _ in range(10):
    expensive_computation(1000000)

# Timing data is automatically saved to .cache/timeit_log.csv
# Columns: date, timestamp, func_name, module, duration, args, kwargs

# @log_time: Log execution time with custom logger
from trade.helpers.Logging import setup_logger
logger = setup_logger('MyModule')

@log_time(logger=logger)
def fetch_market_data(ticker, start, end):
    """Simulated data fetch"""
    import time
    time.sleep(0.5)
    return f"Data for {ticker} from {start} to {end}"

result = fetch_market_data('AAPL', '2024-01-01', '2024-12-31')
# Logs: "fetch_market_data took 0.5012s"

# @log_error: Automatic error logging with stack traces
@log_error(logger=logger, raise_exception=True)
def risky_calculation(x, y):
    return x / y  # Will crash if y=0

try:
    risky_calculation(10, 0)
except ZeroDivisionError:
    print("Error was logged automatically")

# @cProfiler: Detailed profiling
@cProfiler
def complex_backtest():
    # Your backtest code here
    pass

# Generates profile.prof file for analysis with snakeviz or pstats
```

**Best Practice**: Use `@timeit` in production to track performance regressions.

### 12. Context Manager for Time Window Management

The `Context` manager handles time windows for data fetching:

```python
from trade.helpers.Context import Context

# Example 1: Fetch 1 year of daily data ending today
with Context(timewidth='1', timeframe='year', print_context=True):
    # Inside this context, all data fetches use these settings
    from trade.assets.Stock import Stock
    
    stock = Stock('AAPL')
    data = stock.spot(ts=True)  # Uses context settings
    print(f"Fetched {len(data)} days of data")

# Example 2: Fetch specific date range
with Context(start_date='2024-01-01', end_date='2024-12-31'):
    stock = Stock('MSFT')
    data = stock.spot(ts=True)
    print(f"Date range: {data.index.min()} to {data.index.max()}")

# Example 3: Intraday data
with Context(
    start_date='2024-01-02 09:30',
    end_date='2024-01-02 16:00',
    timeframe='minute',
    timewidth='1'
):
    stock = Stock('GOOGL')
    intraday = stock.spot(ts=True, interval='1m')
    print(f"Fetched {len(intraday)} minute bars")

# Context automatically:
# - Validates dates (no weekends/holidays)
# - Adjusts to last business day if needed
# - Sets default times (9:30 AM start, 4:00 PM end)
# - Configures global settings accessed by Stock/Option classes
```

**Use Case**: Consistent time handling across multiple data sources.

### 13. Legacy Vectorized Backtester (PTBacktester)

The `trade.backtester_` module provides a fast vectorized backtester:

```python
from trade.backtester_.backtester_ import PTBacktester
from trade.backtester_.data import PTDataset
from trade.backtester_._strategy import StrategyBase
import pandas as pd
import yfinance as yf

# Define a simple moving average strategy
class SimpleMAStrategy(StrategyBase):
    # Strategy parameters (can be optimized)
    fast_ma = 20
    slow_ma = 50
    
    def init(self):
        """Initialize indicators"""
        self.fast_sma = self.I(lambda: self.data.Close.rolling(self.fast_ma).mean())
        self.slow_sma = self.I(lambda: self.data.Close.rolling(self.slow_ma).mean())
    
    def next(self):
        """Trading logic called on each bar"""
        if self.fast_sma[-1] > self.slow_sma[-1] and self.fast_sma[-2] <= self.slow_sma[-2]:
            # Golden cross: buy
            if not self.position:
                self.buy()
        
        elif self.fast_sma[-1] < self.slow_sma[-1] and self.fast_sma[-2] >= self.slow_sma[-2]:
            # Death cross: sell
            if self.position:
                self.sell()

# Fetch data
tickers = ['AAPL', 'MSFT', 'GOOGL']
datasets = []
for ticker in tickers:
    df = yf.download(ticker, start='2023-01-01', end='2024-12-31', interval='1d', progress=False)
    df.columns = [col.capitalize() for col in df.columns]  # Backtester expects capitalized columns
    datasets.append(PTDataset(ticker, df))

# Run backtest
backtest = PTBacktester(
    datalist=datasets,
    strategy=SimpleMAStrategy,
    cash=100000,
    commission=0.001  # 0.1% commission
)

# Execute
stats = backtest.run()

# View results
print("\nBacktest Results:")
print(stats)

# Access detailed metrics
print(f"\nTotal Return: {stats['Return [%]'].mean():.2f}%")
print(f"Sharpe Ratio: {stats['Sharpe Ratio'].mean():.2f}")
print(f"Max Drawdown: {stats['Max. Drawdown [%]'].mean():.2f}%")

# Get trade history
trades = backtest.__trades()
print(f"\nTotal Trades: {len(trades)}")
print(trades[['Ticker', 'EntryTime', 'ExitTime', 'Size', 'ReturnPct']].head())

# Optimize parameters
from trade.backtester_.utils.utils import optimize

optimized_stats = optimize(
    backtest=backtest,
    params={
        'fast_ma': range(10, 30, 5),
        'slow_ma': range(40, 80, 10)
    },
    maximize='Sharpe Ratio',
    max_tries=50
)

print(f"\nOptimal Parameters:")
print(f"  Fast MA: {optimized_stats['fast_ma']}")
print(f"  Slow MA: {optimized_stats['slow_ma']}")
print(f"  Best Sharpe: {optimized_stats['Sharpe Ratio']:.2f}")
```

**Performance**: PTBacktester is ~100x faster than event-driven for simple strategies but less realistic (no slippage, T+N delays).

### 14. Volatility Surface Modeling

Build and query volatility surfaces for options:

```python
from trade.models.VolSurface import (
    SurfaceBuilder,
    SurfaceManager,
    DumasModelBuilder
)
import pandas as pd

# Sample option chain data (from market)
chain_data = pd.DataFrame({
    'strike': [140, 145, 150, 155, 160, 165, 170],
    'dte': [30, 30, 30, 30, 30, 30, 30],
    'implied_vol': [0.28, 0.26, 0.24, 0.25, 0.27, 0.29, 0.31],
    'right': ['C', 'C', 'C', 'C', 'C', 'C', 'C']
})

spot_price = 150.0

# Build SVI (Stochastic Volatility Inspired) surface
surface_builder = SurfaceBuilder(
    chain_data=chain_data,
    spot=spot_price,
    model_type='SVI'
)

# Fit the model
surface_builder.fit()

# Query implied volatility for arbitrary strikes/DTE
query_strikes = [147, 152, 158]
query_dte = 30

predicted_ivs = surface_builder.predict(
    strikes=query_strikes,
    dte=query_dte
)

print("Predicted Implied Volatilities:")
for strike, iv in zip(query_strikes, predicted_ivs):
    print(f"  Strike {strike}: {iv:.2%}")

# Dumas model (rolling regression for IV surface)
dumas_model = DumasModelBuilder(
    chain_data=chain_data,
    spot=spot_price,
    rolling_window=7  # Days
)

dumas_model.fit()

# Predict IV for out-of-sample strikes
oos_strikes = [142, 168]
oos_predictions = dumas_model.predict(strikes=oos_strikes, dte=30)

print("\nDumas Model Predictions:")
for strike, iv in zip(oos_strikes, oos_predictions):
    print(f"  Strike {strike}: {iv:.2%}")

# Visualize surface
surface_builder.plot_surface(
    strike_range=(130, 170),
    dte_range=(7, 90)
)
```

**Use Case**: Interpolate IVs for strikes not quoted, extrapolate to different DTEs, detect arbitrage opportunities.

### 15. Custom Technical Indicators

Extend pandas-ta with custom indicators:

```python
from trade.helpers.custom_ta import (
    atr_trailing_stop,
    wilders_average,
    hull_average
)
import pandas as pd
import yfinance as yf

# Fetch data
df = yf.download('AAPL', start='2024-01-01', end='2024-12-31', progress=False)

# ATR Trailing Stop (for stop-loss calculation)
atr_stop = atr_trailing_stop(
    df=df,
    period=14,
    multiplier=2.0,
    trend_col='Close'  # Column to base trend on
)

df['ATR_Stop_Long'] = atr_stop['long_stop']
df['ATR_Stop_Short'] = atr_stop['short_stop']
df['Trend'] = atr_stop['trend']  # 1 = uptrend, -1 = downtrend

print("ATR Trailing Stop:")
print(df[['Close', 'ATR_Stop_Long', 'Trend']].tail())

# Wilder's smoothing (for RSI, ATR)
wilders = wilders_average(df['Close'], length=14)
print(f"\nWilder's Average (14): {wilders.iloc[-1]:.2f}")

# Hull Moving Average (reduced lag)
hull = hull_average(df['Close'], length=20)
df['HMA_20'] = hull

# Plot
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Close'], label='Close', alpha=0.7)
plt.plot(df.index, df['HMA_20'], label='HMA(20)', linewidth=2)
plt.plot(df.index, df['ATR_Stop_Long'], label='ATR Stop (Long)', linestyle='--')
plt.legend()
plt.title('Custom Indicators')
plt.show()
```

**Integration**: Use these in `StrategyBase` or `Strategy` classes for signals.

### 16. Thread Pool for Parallel Processing

Speed up data fetching and computations:

```python
from trade.helpers.threads import runThreads

# Example: Fetch option chains for multiple underlyings
def fetch_option_chain(ticker, date):
    """Fetch option chain from data source"""
    # Your data fetching logic
    print(f"Fetching chain for {ticker} on {date}")
    import time
    time.sleep(0.5)  # Simulate network call
    return {'ticker': ticker, 'date': date, 'contracts': 100}

# Sequential (slow)
import time
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
date = '2024-01-15'

start = time.time()
results_seq = [fetch_option_chain(t, date) for t in tickers]
print(f"Sequential: {time.time() - start:.2f}s")

# Parallel with runThreads (fast)
inputs = [[ticker, date] for ticker in tickers]

start = time.time()
results_parallel = runThreads(
    func=fetch_option_chain,
    OrderedInputs=inputs,
    max_workers=5,
    return_results=True
)
print(f"Parallel: {time.time() - start:.2f}s")

# Example 2: Parallel option pricing
from trade.optionlib.pricing.black_scholes import black_scholes_vectorized
from trade.optionlib.assets.forward import EquityForward

def price_option_with_greeks(strike, spot, T, r, sigma, q):
    F = EquityForward(S=spot, T=T, r=r, q=q).price()
    price = black_scholes_vectorized(F=F, K=strike, T=T, r=r, sigma=sigma, option_type='c')
    return {'strike': strike, 'price': price[0]}

strikes = range(100, 200, 5)
inputs = [[k, 150.0, 0.25, 0.05, 0.30, 0.02] for k in strikes]

prices = runThreads(
    func=price_option_with_greeks,
    OrderedInputs=inputs,
    max_workers=10,
    return_results=True
)

print("\nOption Prices:")
for result in prices[:5]:
    print(f"  Strike {result['strike']}: ${result['price']:.2f}")
```

**Performance**: ~5x speedup for I/O-bound tasks (data fetching), ~2x for CPU-bound (if GIL-limited).

### 17. Working with Configuration Files

Centralized configuration management:

```python
from trade.helpers.Configuration import ConfigProxy, initialize_configuration

# Initialize global config
initialize_configuration()

# Access config (singleton pattern)
config = ConfigProxy()

# Read current settings
print(f"Current timeframe: {config.timeframe}")
print(f"Start date: {config.start_date}")
print(f"End date: {config.end_date}")

# Modify settings (affects all subsequent data fetches)
config.timeframe = 'hour'
config.timewidth = '4'

# Context manager auto-configures and restores
from trade.helpers.Context import Context, clear_context

with Context(timewidth='1', timeframe='day', start_date='2024-01-01'):
    print(f"Inside context: {config.timeframe}")  # 'day'

print(f"Outside context: {config.timeframe}")  # Back to 'hour'

# Clear all context settings
clear_context()
print(f"After clear: {config.timeframe}")  # Default value
```

**Use Case**: Consistent settings across modules without passing parameters everywhere.

---

## Samples

Additional sample code and notebooks demonstrating specific workflows:

- **Full Backtest Workflow**: See `EventDriven/demos/demoRun.py` for a complete example using real data
- **Risk Manager Deep Dive**: Explore `EventDriven/riskmanager/` submodules for advanced risk controls
- **Option Pricing Examples**: See `trade/optionlib/notebooks/` for pricing and Greek calculations
- **Portfolio Construction**: Check `EventDriven/notebooks/` for portfolio analysis examples
- **Data Management**: See `trade/datamanager/notebooks/create.ipynb` for DataManager patterns (WIP on CHIDI-JAN04 branch)

---

## Troubleshooting

### Common Import Errors

**Error**: `ModuleNotFoundError: No module named 'EventDriven'`
- **Fix**: Install package in editable mode: `pip install -e .` from repo root
- **Check**: Ensure you're in a venv/conda env with dependencies installed

**Error**: `ModuleNotFoundError: No module named 'dbase'`
- **Context**: Some modules import from `dbase` (separate database package for ThetaData)
- **Fix Option 1**: Install `dbase` package if you have it
- **Fix Option 2**: Comment out ThetaData imports if not using that data source
- **Fix Option 3**: Use yfinance data source instead (no external DB needed)

**Error**: `ImportError: cannot import name 'query_database' from 'dbase.database.SQLHelpers'`
- **Fix**: Set `DBASE_DIR` in `.env` or remove ThetaData-specific code

### Missing Configuration

**Error**: `KeyError: 'WORK_DIR'` or `KeyError: 'GEN_CACHE_PATH'`
- **Fix**: Create `.env` file with required paths:
  ```env
  WORK_DIR=/path/to/QuantTools
  GEN_CACHE_PATH=/path/to/cache
  ```

**Error**: `FileNotFoundError: [Errno 2] No such file or directory: 'pricingConfig.json'`
- **Context**: Option pricing looks for config in repo root
- **Fix**: Ensure `pricingConfig.json` exists or specify config explicitly
- **Default Values**: Config is optional; defaults will be used if missing

### Path Issues

**Error**: `FileNotFoundError` when accessing logs or cache directories
- **Fix**: Use absolute paths or ensure working directory is repo root
- **Check**: `os.getcwd()` should return `.../QuantTools`
- **Workaround**: Set `WORK_DIR` environment variable explicitly

### Running Tests

**Unit Tests**:
```bash
# Run all tests (if pytest configured)
pytest

# Run specific test file
python -m pytest tests/test_portfolio.py -v
```

**Module Tests** (in `module_test/`):
```bash
# Test specific module
python module_test/test_backtest.py

# Test with verbose logging
python module_test/test_backtest.py --log-level DEBUG
```

**Notebook Tests**:
- Open notebooks in `EventDriven/notebooks/` or `EventDriven/demos/`
- Run cells sequentially
- Check for import errors or missing data files

### Performance Issues

**Slow Backtests**:
- Enable caching for market data
- Reduce option chain search space (fewer strikes, expirations)
- Use `slippage_enabled=False` for faster runs (testing only)
- Profile with `cProfile` (see `EventDriven/demos/demoRun.py` for example)

**Memory Issues**:
- Clear cache periodically: `cache.clear()`
- Reduce backtest date range
- Limit number of symbols
- Use `del` to remove large objects after use

### Logging Issues

**Too Much Log Output**:
```python
from EventDriven.configs.core import BacktesterConfig

config = BacktesterConfig(logger_override_level='WARNING')  # Reduce verbosity
```

**Logs Not Appearing**:
- Check console output first (stdout/stderr)
- Verify `logs/` directory exists
- Check log level: `logger.setLevel(logging.DEBUG)`

---

## Contributing

### Code Style

QuantTools uses **Ruff** for linting and formatting (configured in `ruff.toml`):

```bash
# Check for issues
ruff check .

# Auto-fix fixable issues
ruff check --fix .

# Format code
ruff format .
```

**Key Rules**:
- Line length: 120 characters
- Import organization: Auto-sorted
- Ignored rules: `E501` (line length, handled by formatter), `I001` (import order), `E722` (bare except), `B009` (getattr with constant)

### Type Hints and Docstrings

- **Type hints required** for all public functions and methods
- **Docstrings required** for classes and public methods (Google or NumPy style)
- Use `from typing import` for compatibility with Python 3.10+

Example:
```python
from typing import Optional
import pandas as pd

def calculate_returns(prices: pd.Series, periods: int = 252) -> pd.Series:
    """
    Calculate period returns from price series.
    
    Args:
        prices: Time series of prices
        periods: Number of periods for annualization (default: 252 for daily)
    
    Returns:
        Series of period returns
    
    Raises:
        ValueError: If prices series is empty
    """
    if prices.empty:
        raise ValueError("Price series cannot be empty")
    return prices.pct_change(periods=periods)
```

### Branch Workflow

1. **Create Feature Branch**: `git checkout -b feature/your-feature-name` or `git checkout -b YOURNAME-MMDD-feature-description`
2. **Make Changes**: Follow code style, add tests
3. **Commit**: Use descriptive commit messages
4. **Test**: Run relevant tests before pushing
5. **Push**: `git push origin your-branch-name`
6. **Pull Request**: Submit PR to `main` with description

**Branch Naming Examples**:
- `feature/add-iron-condor-strategy`
- `fix/portfolio-cash-calculation`
- `CHIDI-JAN04-DATAMANAGER-REBUILD` (for larger initiatives)

### Testing Guidelines

- Add tests for new features in `module_test/` or `tests/`
- Test edge cases (empty data, missing fields, extreme values)
- Use fixtures for common test data
- Document test assumptions in docstrings

---

## License

MIT License - see LICENSE file for details

## Authors

- **Chidi** - Core architecture, event-driven engine, options pricing
- **Zino** - Risk management, portfolio systems, data infrastructure

## Acknowledgments

- Event-driven architecture inspired by QuantStart tutorials
- Option pricing models based on industry-standard formulas (Black-Scholes-Merton, etc.)
- Built with support from the quantitative finance community

---

**Questions or Issues?** Check existing issues on GitHub or open a new one with a minimal reproducible example.

