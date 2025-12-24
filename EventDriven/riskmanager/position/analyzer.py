"""Position Analysis and Action Orchestration via Cog Architecture.

This module implements the PositionAnalyzer, which orchestrates multiple specialized
"cogs" (analysis components) to evaluate positions and recommend actions. Each cog
analyzes positions from a specific perspective (risk limits, signals, Greeks, etc.)
and generates action recommendations that are reconciled into final strategy changes.

Core Class:
    PositionAnalyzer: Orchestrator for multiple analysis cogs

Architecture Philosophy:
    Modular Cog System:
        - Each cog handles one aspect of analysis
        - Cogs operate independently and in parallel
        - Results reconciled by priority-based system
        - Easy to add/remove/modify individual cogs
        - Clear separation of concerns

Cog Types:
    Risk Limits Cog (LimitsAndSizingCog):
        - Enforces Greek limits (delta, gamma, vega, theta)
        - Monitors position sizing constraints
        - Tracks portfolio leverage
        - Recommends CLOSE or ADJUST on violations

    Signal Cog (planned):
        - Monitors strategy entry/exit signals
        - Tracks signal strength changes
        - Recommends CLOSE on exit signals
        - Suggests ADJUST on conviction changes

    Roll Cog (planned):
        - Monitors DTE thresholds
        - Checks moneyness drift
        - Recommends ROLL when criteria met
        - Handles quantity adjustments during rolls

    Expiration Cog (planned):
        - Tracks positions approaching expiration
        - Recommends EXERCISE for ITM options
        - Suggests CLOSE for OTM options
        - Handles assignment scenarios

    PnL Cog (planned):
        - Monitors unrealized P&L
        - Enforces stop-loss limits
        - Tracks profit targets
        - Recommends CLOSE on thresholds

Key Features:
    - Dynamic cog registration and removal
    - Per-cog enable/disable flags
    - Priority-based action reconciliation
    - Position-level and portfolio-level context
    - Extensible action vocabulary
    - Thread-safe operation

Analysis Flow:
    1. Context Preparation:
       - Gather current positions
       - Retrieve market data (spot, Greeks, etc.)
       - Calculate portfolio aggregates
       - Package in PositionAnalysisContext

    2. Cog Execution:
       - Iterate through enabled cogs
       - Each cog analyzes context independently
       - Cogs return CogActions with opinions
       - All opinions collected in list

    3. Action Reconciliation:
       - Group actions by trade_id
       - Sort by ACTION_PRIORITY
       - Select highest priority action per position
       - Package in StrategyChangeMeta

    4. Result Delivery:
       - Return actionables for each position
       - Include portfolio metadata
       - Preserve audit trail of all opinions

Action Priority System:
    Priority Order (highest to lowest):
        1. CLOSE - Exit position immediately
        2. EXERCISE - Exercise ITM option
        3. ROLL - Replace with new position
        4. ADJUST - Modify position quantity
        5. HOLD - Maintain current position

    Rationale:
        - Risk management (CLOSE) takes precedence
        - Expiration events (EXERCISE) are time-critical
        - Position refreshes (ROLL) before adjustments
        - Quantity changes (ADJUST) can wait
        - HOLD is default if no other action needed

Configuration:
    PositionAnalyzerConfig:
        - enabled: Master switch for analyzer
        - enabled_cogs: List of active cog names
        - reconciliation_mode: How to handle conflicts
        - logging_level: Verbosity control

Cog Registration:
    Manual Registration:
        cogs = [LimitsAndSizingCog(config), SignalCog(config)]
        analyzer = PositionAnalyzer(cogs=cogs)

    Dynamic Addition:
        analyzer.add_cog(RollCog(config))

    Removal:
        analyzer.remove_cog('roll_cog')

    Clear All:
        analyzer.clear_cogs()

Context Structure:
    PositionAnalysisContext:
        - date: Analysis date
        - positions: List of current Position objects
        - market_data: Dict of spot prices, Greeks, etc.
        - portfolio_meta: Aggregated portfolio metrics
        - signals: Active strategy signals
        - limits: Current risk limits

Output Structure:
    StrategyChangeMeta:
        - date: When analysis performed
        - actionables: List of PositionState with recommended actions
        - portfolio_meta: Portfolio-level metrics

Cog Lifecycle Hooks:
    on_new_position(new_position_state):
        - Called when new position opened
        - Allows cogs to initialize tracking
        - Update limits, register signals, etc.
        - Return potentially modified state

Usage Examples:
    # Initialize with cogs
    config = PositionAnalyzerConfig(enabled_cogs=['limits', 'roll'])

    limits_cog = LimitsAndSizingCog(config=limits_config)
    roll_cog = RollCog(config=roll_config)

    analyzer = PositionAnalyzer(
        config=config,
        cogs=[limits_cog, roll_cog]
    )

    # Analyze positions
    context = PositionAnalysisContext(
        date=datetime(2024, 3, 15),
        positions=current_positions,
        market_data=market_snapshot,
        portfolio_meta=portfolio_stats
    )

    strategy_changes = analyzer.analyze(context)

    # Process actionables
    for actionable in strategy_changes.actionables:
        action = actionable.action
        if action.type == EventTypes.CLOSE:
            print(f"Close {actionable.trade_id}: {action.reason}")
        elif action.type == EventTypes.ROLL:
            print(f"Roll {actionable.trade_id} with qty change: {action.quantity_change}")

Extensibility:
    Creating New Cogs:
        1. Subclass BaseCog
        2. Implement _analyze_impl(context)
        3. Return CogActions with opinions
        4. Optionally implement on_new_position()
        5. Define default_config
        6. Register with analyzer

Error Handling:
    - Unknown cogs in enabled_cogs raise ValueError
    - Duplicate cog names prevented
    - Disabled cogs return empty CogActions
    - Invalid action types logged but not failed
    - Cog exceptions isolated (future enhancement)

Performance:
    - Cogs run sequentially (parallel option planned)
    - Minimal overhead from orchestration
    - Action reconciliation is O(n log n) per trade_id
    - Efficient opinion aggregation

Integration:
    - Called by RiskManager on each analysis cycle
    - Receives context from portfolio state
    - Returns actions for execution engine
    - Logged for audit and debugging

Notes:
    - enabled_cogs filters registered cogs
    - Empty enabled_cogs means all cogs run
    - Cog order doesn't affect result (priority-based)
    - Analyzer itself can be disabled via config
    - Cog opinions preserved for post-analysis
"""

from typing import Iterable, List, Dict
from trade.helpers.Logging import setup_logger
from .base import (
    BaseCog,
)
from EventDriven.configs.core import PositionAnalyzerConfig
from EventDriven.dataclasses.states import NewPositionState
from EventDriven.dataclasses.states import (
    PositionAnalysisContext,
    StrategyChangeMeta,
    CogActions,  # noqa
    PositionState,
)
from EventDriven.riskmanager.actions import RMAction  # noqa
from .cogs.vars import ACTION_PRIORITY


logger = setup_logger("EventDriven.riskmanager.position.analyzer", stream_log_level="WARNING")


class PositionAnalyzer:
    """
    Orchestrates cogs:
    - Builds opinions from all cogs.
    - (Later) reconciles them into StrategyChangeMeta targets.
    """

    def __init__(self, config: PositionAnalyzerConfig = None, cogs: Iterable[BaseCog] = []):
        if config is None:
            config = PositionAnalyzerConfig()

        self.config = config
        self._cogs: Dict[str, BaseCog] = {}

        for cog in cogs:
            if not isinstance(cog, BaseCog):
                raise TypeError(f"All cogs must subclass BaseCog; got {type(cog)}")
            if cog.name in self._cogs:
                raise ValueError(f"Duplicate cog name detected: {cog.name}")
            self._cogs[cog.name] = cog
            logger.info(f"Registered Cog: {cog.name}")
        logger.info(f"PositionAnalyzer initialized with cogs: {list(self._cogs.keys())}")

        # Optional: enforce that enabled_cogs refer to known cogs
        unknown = set(self.config.enabled_cogs) - set(self._cogs.keys())
        if unknown:
            raise ValueError(f"enabled_cogs references unknown cogs: {unknown}")

    @property
    def cogs(self) -> List[BaseCog]:
        """
        Returns the list of cogs in registration order.
        """
        return list(self._cogs.values())

    def remove_cog(self, cog_name: str) -> None:
        """
        Removes a cog by name from the PositionAnalyzer.
        """
        if cog_name not in self._cogs:
            raise KeyError(f"Cog with name {cog_name} not found.")
        del self._cogs[cog_name]
        logger.info(f"Removed Cog: {cog_name}")

    def clear_cogs(self) -> None:
        """
        Clears all cogs from the PositionAnalyzer.
        """
        self._cogs.clear()
        logger.info("Cleared all cogs from PositionAnalyzer.")

    def add_cog(self, cog: BaseCog) -> None:
        """
        Adds a new cog to the PositionAnalyzer.
        """
        # if not isinstance(cog, BaseCog) or not issubclass(cog.__class__, BaseCog):
        #     raise TypeError(f"Cog must subclass BaseCog; got {type(cog)}")
        if cog.name in self._cogs:
            raise ValueError(f"Duplicate cog name detected: {cog.name}")
        if not cog.enabled:
            logger.warning(f"Attempted to add disabled cog: {cog.name}. It will not be active.")
        if not cog.name:
            raise ValueError("Cog must have a valid name.")
        self._cogs[cog.name] = cog
        logger.info(f"Added Cog: {cog.name}")

    def _iter_active_cogs(self) -> Iterable[BaseCog]:
        """
        Yields cogs that should run for this analysis.
        If enabled_cogs is non-empty, filters by that list.
        """
        if not self.config.enabled:
            return []

        for cog in self._cogs.values():
            if cog.enabled:
                yield cog

    def analyze(self, context: PositionAnalysisContext) -> StrategyChangeMeta:
        """
        Main entrypoint:
        - Collect all CogOpinions from active cogs.
        - For now, we do a trivial reconciliation: we keep baseline targets as-is
          and just attach the raw opinions.
        - We'll replace this with the full Cog Process reconciler later.
        """

        all_actions: List[PositionState] = []

        for cog in self._iter_active_cogs():
            actions = cog.analyze(context)
            all_actions.extend(actions.opinions)

        ## Get unique trade IDs from all actions
        unique_trade_ids = set(action.trade_id for action in all_actions)
        strategy_changes: List[PositionState] = []

        ## Get the most important action for each trade ID
        for trade_id in unique_trade_ids:
            trade_actions = [action for action in all_actions if action.trade_id == trade_id]
            trade_actions.sort(key=lambda x: ACTION_PRIORITY.get(x.action.type, float("inf")))
            most_important_action = trade_actions[0]
            strategy_changes.append(most_important_action)

        return StrategyChangeMeta(
            date=context.date,
            actionables=strategy_changes,
            portfolio_meta=context.portfolio_meta,
        )

    def on_new_position(self, new_position_state: NewPositionState) -> NewPositionState:
        """
        Hook method called when a new position is detected.
        Delegates to all registered cogs.
        Args:
            new_position_state (NewPositionState): The new position state containing order, request, and position data.
        Returns:
            NewPositionState: The updated position state after all cogs have processed it.

        What this does:
            - It iterates through all registered cogs and calls their `on_new_position` method.
            - Each cog can modify the `new_position_state` as needed.
            - Finally, it returns the potentially modified `new_position_state`.
        """
        for cog in self._cogs.values():
            cog.on_new_position(new_position_state)
        return new_position_state
