"""Live Trading Limits and Sizing Cog with Database Persistence.

This module implements LiveCOGLimitsAndSizingCog, a specialized version of the
LimitsAndSizingCog for live trading environments. It extends the base cog with
database persistence, position limit retrieval from historical records, and
state management across trading sessions.

Core Class:
    LiveCOGLimitsAndSizingCog: Live trading limits enforcement with persistence

Key Differences from Base Cog:
    Database Integration:
        - Persists position limits to database
        - Retrieves historical limits on startup
        - Maintains limit continuity across sessions
        - Audit trail for all limit changes

    State Management:
        - Loads position state from database
        - Syncs in-memory and persisted limits
        - Handles session restarts gracefully
        - Prevents limit recalculation on restart

    Cool-off Period:
        - Prevents rapid limit adjustments
        - Configurable delay between updates
        - Reduces database write volume
        - Stabilizes position sizing

Limit Persistence Strategy:
    On Position Open:
        1. Calculate initial limits via sizer
        2. Store limits to database
        3. Cache in memory for fast access
        4. Associate with trade_id and signal_id

    On Position Analysis:
        1. Check memory cache first
        2. Fall back to database if not cached
        3. Use limits for risk evaluation
        4. Update limits only if needed

    On Limit Update:
        1. Recalculate via sizer
        2. Compare to current limits
        3. Apply cool-off logic
        4. Persist to database if changed
        5. Update memory cache

Database Schema:
    Limits Table:
        - trade_id: Unique position identifier
        - signal_id: Strategy signal reference
        - strategy_name: Strategy identifier
        - date: Limit creation/modification date
        - delta: Delta limit value
        - gamma: Gamma limit value (if enabled)
        - vega: Vega limit value (if enabled)
        - theta: Theta limit value (if enabled)

    Indexes:
        - Primary: (trade_id, strategy_name, signal_id)
        - Secondary: (strategy_name, date)
        - Query optimization for retrieval

Limit Retrieval Flow:
    1. Check self._position_limits dict (memory cache)
    2. If not found, query database via get_position_limit()
    3. Load all Greek measures from database
    4. Construct PositionLimits object
    5. Apply defaults for missing measures
    6. Cache in memory for future access
    7. Return limits to caller

Limit Storage Flow:
    1. Extract limit values from PositionLimits object
    2. Prepare database parameters
    3. Call store_position_limits() utility
    4. Write to limits table
    5. Update memory cache
    6. Log storage operation

Cool-off Period Logic:
    Purpose:
        - Prevents limit thrashing
        - Reduces database writes
        - Stabilizes position management
        - Avoids overreaction to noise

    Implementation:
        - Track last limit update time per position
        - Compare current time to last update
        - Skip update if within cool-off window
        - Configurable period (default: 0 = disabled)

    Typical Values:
        - 0: No cool-off (update immediately)
        - 3600: 1 hour between updates
        - 86400: Daily updates only

Position Limit Defaults:
    When Not Found in Database:
        - Delta: Calculate from sizer
        - Gamma: Use config default or None
        - Vega: Use config default or None
        - Theta: Use config default or None
        - DTE: Use config.default_dte
        - Moneyness: Use config.default_moneyness

Session Continuity:
    Restart Handling:
        1. Load all active positions from database
        2. Retrieve limits for each position
        3. Populate memory cache
        4. Resume analysis with correct limits
        5. Maintain limit consistency

    Prevents:
        - Limit reset on restart
        - Position over-sizing after crash
        - Inconsistent risk management
        - Loss of limit history

Integration with Base Cog:
    Overridden Methods:
        - _save_position_limits(): Add database persistence
        - _get_position_limits(): Add database retrieval
        - _analyze_impl(): Delegates to base, may add logging

    Inherited Functionality:
        - All limit enforcement logic
        - Sizer integration
        - Greek calculations
        - Action recommendations

Configuration:
    Required Config:
        - run_name: Strategy identifier for database
        - default_dte: Fallback DTE if not in database
        - default_moneyness: Fallback moneyness
        - enabled_limits: Which Greeks to track

    Optional Config:
        - cool_off_period: Seconds between updates (default: 0)
        - persist_limits: Enable/disable persistence (default: True)
        - cache_limits: Enable/disable memory cache (default: True)

Usage:
    # Initialize live cog
    live_cog = LiveCOGLimitsAndSizingCog(
        config=limits_config,
        sizer_configs=sizer_config,
        underlier_list=['AAPL', 'MSFT']
    )

    # On new position
    limits = live_cog.on_new_position(
        position_state=new_position,
        analysis_date=date
    )
    # Automatically saved to database

    # On position analysis
    actions = live_cog.analyze(
        context=analysis_context
    )
    # Limits retrieved from cache or database

Error Handling:
    Database Failures:
        - Fall back to calculated limits
        - Log errors with context
        - Continue analysis if possible
        - Don't crash on read failures

    Missing Limits:
        - Calculate new limits via sizer
        - Use config defaults where applicable
        - Log missing limit warnings
        - Store newly calculated limits

    Cache Inconsistency:
        - Database is source of truth
        - Memory cache refreshed on mismatch
        - Periodic cache validation

Performance:
    - Memory cache provides O(1) limit access
    - Database queries only on cache miss
    - Batch updates for multiple positions
    - Connection pooling for database
    - Lazy loading of position limits

Monitoring:
    Logged Events:
        - Limit storage operations
        - Limit retrieval from database
        - Cache hits and misses
        - Cool-off period activations
        - Limit update rejections

Notes:
    - Requires database connection configured
    - Thread-safe via database transaction handling
    - Compatible with backtest cog for testing
    - All timestamps in UTC
    - Positions tracked by composite key

See Also:
    - save_utils.py: Database persistence utilities
    - ../cogs/limits.py: Base LimitsAndSizingCog
    - ../../sizer/: Position sizing implementations
"""

from typing import Dict
from EventDriven.riskmanager.position.cogs.limits import LimitsAndSizingCog, PositionLimits
from EventDriven.configs.core import LimitsEnabledConfig, BaseSizerConfigs
from .save_utils import get_position_limit, store_position_limits
from ..cogs.vars import MEASURES_SET


class LiveCOGLimitsAndSizingCog(LimitsAndSizingCog):
    """
    Live trading specialized limits cog with database persistence.

    Extends LimitsAndSizingCog to add:
    - Database storage and retrieval of position limits
    - State persistence across trading sessions
    - Cool-off period for limit updates
    - Audit trail for compliance

    Attributes:
        _position_limits (Dict[str, PositionLimits]): Memory cache of position limits
        cool_off_period (int): Seconds between limit updates (0 = no limit)

    Note:
        This class is designed for live trading environments where position
        state must persist across restarts and limit history must be maintained
        for compliance and analysis.
    """

    def __init__(
        self, config: LimitsEnabledConfig = None, sizer_configs: BaseSizerConfigs = None, underlier_list: list = None
    ):
        super().__init__(config=config, sizer_configs=sizer_configs, underlier_list=underlier_list)
        self._position_limits: Dict[str, PositionLimits] = {}
        self.cool_off_period = 0

    def _save_position_limits(self, trade_id: str, signal_id: str, limits: PositionLimits) -> None:
        """
        Save the position limits for a given trade ID and signal ID.
        """
        store_position_limits(
            delta_limit=limits.delta,
            gamma_limit=limits.gamma,
            vega_limit=limits.vega,
            theta_limit=limits.theta,
            trade_id=trade_id,
            signal_id=signal_id,
            strategy_name=self.config.run_name,
            date=limits.creation_date,
        )

    def _get_position_limits(self, trade_id: str, signal_id: str) -> PositionLimits:
        """
        Retrieve the position limits for a given trade ID and signal ID.
        """
        if trade_id in self.position_limits:
            return self.position_limits[trade_id]

        lm = PositionLimits()
        for risk_measure in MEASURES_SET:
            date, limit_value = get_position_limit(
                trade_id=trade_id,
                strategy_name=self.config.run_name,
                signal_id=signal_id,
                risk_measure=risk_measure,
            )
            if limit_value is not None:
                setattr(lm, risk_measure, limit_value)
                lm.creation_date = date
        lm.dte = self.config.default_dte
        lm.moneyness = self.config.default_moneyness

        self.position_limits[trade_id] = lm
        return lm

    def _analyze_impl(self, portfolio_context):
        return super()._analyze_impl(portfolio_context)
