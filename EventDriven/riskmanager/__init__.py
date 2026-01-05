"""EventDriven Risk Manager Module.

This module provides the core risk management infrastructure for options trading strategies
within the EventDriven backtesting framework. It orchestrates position sizing, Greek-based
risk limits, order generation, and market data management.

Core Components:
    RiskManager: Main orchestrator for risk management, position analysis, and order execution
    OrderPicker: Intelligent order selection from available option chains based on strategy criteria

Key Features:
    - Greek-based risk management (delta, gamma, vega, theta limits)
    - Dynamic position sizing with configurable strategies (fixed, delta-neutral, volatility-adjusted)
    - Automated position rolling based on DTE and moneyness thresholds
    - Market data caching and efficient timeseries management
    - Corporate action handling (splits, dividends, adjustments)
    - Multi-strategy portfolio coordination

Usage:
    from EventDriven.riskmanager import RiskManager

    rm = RiskManager(
        symbol_list=['AAPL', 'MSFT', 'GOOGL'],
        bkt_start='2024-01-01',
        bkt_end='2024-12-31',
        initial_capital=100000
    )

See Also:
    - new_base.RiskManager: Main risk manager class documentation
    - picker.OrderPicker: Order selection and filtering logic
    - position.PositionAnalyzer: Position analysis and action recommendation
    - sizer: Position sizing strategies and implementations
"""

from .new_base import RiskManager, OrderPicker

__all__ = ["RiskManager", "OrderPicker"]
