"""Position Sizing Strategies Package.

This package provides various position sizing implementations for options portfolios,
including fixed sizing, delta-neutral sizing, and volatility-adjusted sizing.
Each sizer calculates position quantities and Greek limits based on capital
allocation, risk parameters, and market conditions.

Available Sizers:
    BaseSizer: Abstract base class for all sizing strategies
        - Defines common interface for all sizers
        - Implements shared functionality
        - Enforces contract for subclasses
        - Must be subclassed, not used directly

    DefaultSizer: Fixed delta limit sizing strategy
        - Static risk allocation
        - Delta limit = (cash * leverage) / (price * 100)
        - Simple and predictable
        - Best for stable market conditions
        - No volatility adjustment

    ZscoreRVolSizer: Volatility-adjusted sizing strategy
        - Dynamic risk scaling based on realized vol
        - Increases size in low-vol regimes
        - Decreases size in high-vol regimes
        - Z-score normalization of volatility
        - Adapts to changing market conditions

Sizing Philosophy:
    Capital Allocation:
        - Sizes based on available cash per position/signal/ticker
        - Respects portfolio leverage constraints
        - Accounts for existing positions
        - Prevents over-allocation

    Risk Management:
        - Greek limits (especially delta) enforce risk
        - Limits scale with market conditions (ZscoreRVolSizer)
        - Position size inversely related to option price
        - Quantity calculated to stay within limits

    Flexibility:
        - Multiple cash allocation rules
        - Configurable leverage levels
        - Custom sizers via subclassing
        - Runtime sizer selection

Cash Use Rules:
    USE_CASH_EVERY_NEW_POSITION_ID:
        - Each unique position gets fresh capital
        - Most granular cash management
        - Position-specific limits
        - Recommended for live trading

    USE_CASH_EVERY_NEW_TICK:
        - Cash allocated per ticker symbol
        - Shared across positions in same underlying
        - Ticker-level capital tracking

    USE_CASH_EVERY_NEW_SIGNAL_ID:
        - Cash allocated per strategy signal
        - Multiple positions share signal allocation
        - Signal-level capital management

Sizer Selection Guide:
    DefaultSizer - Use when:
        - Market volatility is stable
        - Simple risk management preferred
        - Consistent sizing desired
        - Computational efficiency critical

    ZscoreRVolSizer - Use when:
        - Market volatility varies significantly
        - Dynamic risk adjustment desired
        - Historical data available
        - Adapting to regimes important

Common Interface:
    All sizers implement:
        - update_delta_limit(): Calculate position delta limit
        - calculate_position_size(): Determine contract quantity
        - pre_analyze_task(): Daily setup before analysis
        - post_analyze_task(): Daily cleanup after analysis

Configuration:
    Sizer-Specific Configs:
        - BaseSizerConfigs: Base configuration
        - DefaultSizerConfig: Fixed sizing params
        - ZscoreRVolSizerConfig: Vol-adjusted params

    Common Parameters:
        - cash_rule: How to allocate capital
        - sizing_lev: Leverage multiplier
        - min_quantity: Minimum contracts
        - max_quantity: Maximum contracts

Usage:
    from EventDriven.riskmanager.sizer import (
        DefaultSizer,
        ZscoreRVolSizer,
        BaseSizer
    )

    # Use fixed sizing
    sizer = DefaultSizer(
        config=default_config,
        cash_available=10000,
        sizing_lev=2.0
    )

    # Use volatility-adjusted sizing
    sizer = ZscoreRVolSizer(
        config=zscore_config,
        cash_available=10000,
        sizing_lev=2.0,
        rvol_window=30
    )

    # Calculate delta limit
    delta_limit = sizer.update_delta_limit(
        underlier_price=150.0,
        date=pd.Timestamp('2024-01-15')
    )

    # Calculate position size
    quantity = sizer.calculate_position_size(
        delta_limit=delta_limit,
        structure_delta=0.35
    )

Extending:
    Create custom sizers by subclassing BaseSizer:
        1. Inherit from BaseSizer
        2. Implement update_delta_limit()
        3. Implement calculate_position_size()
        4. Optional: Override pre/post_analyze_task()

    Example:
        class CustomSizer(BaseSizer):
            def update_delta_limit(self, ...):
                # Custom limit logic
                return limit

            def calculate_position_size(self, ...):
                # Custom quantity logic
                return quantity

Integration:
    - LimitsAndSizingCog uses sizers for all sizing
    - RiskManager configuration selects sizer
    - PositionAnalyzer queries sizer for limits
    - All position decisions use sizer calculations

Performance:
    - DefaultSizer: O(1) calculation time
    - ZscoreRVolSizer: O(n) for window-based stats
    - Caching reduces repeated calculations
    - Pre-computed lookups where possible

Notes:
    - All quantities are integers (contracts)
    - Leverage can be fractional (e.g., 1.5x)
    - Delta limits in absolute terms
    - Thread-safe implementations

See Also:
    - _sizer.py: Full sizer implementations
    - _utils.py: Sizing calculation utilities
    - ../position/cogs/limits.py: Limit enforcement
"""

from ._sizer import ZscoreRVolSizer, BaseSizer, DefaultSizer

__all__ = ["ZscoreRVolSizer", "BaseSizer", "DefaultSizer"]
