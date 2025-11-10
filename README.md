# QuantTools

QuantTools is a Python library designed to facilitate programmatic trading algorithms. It provides tools for event-driven backtesting, portfolio management, risk management, and data handling, enabling traders and developers to build and test trading strategies efficiently.

## Features
- Event-driven backtesting framework
- Portfolio management tools
- Risk management utilities
- Data handling and caching mechanisms
- Support for options trading and signal analysis

## Installation

To install QuantTools, clone the repository and use `pip install -e` for an editable installation:

```bash
# Clone the repository
git clone https://github.com/Zino-ctrlZ/QuantTools.git

# Navigate to the repository
cd QuantTools

# Install the package in editable mode
pip install -e .
```

## Usage

```python
from QuantTools.EventDriven.backtest import Backtest
from QuantTools.EventDriven.portfolio import Portfolio

# Initialize backtest and portfolio
backtest = Backtest(...)
portfolio = Portfolio(...)
```

