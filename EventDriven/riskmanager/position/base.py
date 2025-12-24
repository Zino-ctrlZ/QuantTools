"""Base Cog Architecture for Position Analysis.

This module defines the abstract base class and core data structures for the cog-based
position analysis system. All analysis components (cogs) inherit from BaseCog and
implement standardized interfaces for position evaluation and action recommendation.

Core Components:
    BaseCog: Abstract base class for all analysis cogs
    PositionKey: Unique identifier for tracked positions
    StrategyChangeMeta: Final action recommendations container

Cog Architecture Principles:
    Standardization:
        - All cogs implement same interface (_analyze_impl)
        - Consistent input (PositionAnalysisContext)
        - Consistent output (CogActions with opinions)
        - Uniform configuration via BaseCogConfig subclasses

    Modularity:
        - Each cog handles one analysis aspect
        - Cogs operate independently
        - Easy to add, remove, or modify individual cogs
        - No inter-cog dependencies

    Flexibility:
        - Cogs can be enabled/disabled individually
        - Configurable parameters per cog
        - Custom cogs created by subclassing BaseCog
        - Runtime cog registration

BaseCog Responsibilities:
    Configuration Management:
        - Holds cog-specific config (BaseCogConfig subclass)
        - Validates config on initialization
        - Exposes config properties for runtime access
        - Enforces config type safety

    Analysis Template:
        - Checks enabled flag before analysis
        - Calls subclass _analyze_impl() method
        - Validates returned CogActions structure
        - Tags opinions with source_cog name

    Lifecycle Hooks:
        - on_new_position(): Called when position opened
        - Allows cog-specific initialization
        - Position tracking setup
        - Limit registration, etc.

PositionKey Structure:
    Immutable identifier for logical positions:
        - strategy_id: Strategy type identifier
        - trade_id: Unique trade identifier
        - signal_id: Originating signal ID

    Usage:
        - Keys for tracking position state
        - Grouping related positions
        - Mapping positions to metadata

StrategyChangeMeta Structure:
    Final action recommendations:
        - key: PositionKey identifying position
        - strategy_changes: List of CogActions

    Contains:
        - All cog opinions for position
        - Reconciled final action
        - Reasoning from each cog

Config System:
    BaseCogConfig Attributes:
        - name: Unique cog identifier
        - enabled: Master switch for cog
        - Additional cog-specific parameters

    Subclass Pattern:
        class MyCogConfig(BaseCogConfig):
            threshold: float = 0.05
            max_limit: float = 1000.0

        my_cog = MyCog(config=MyCogConfig(threshold=0.1))

Subclass Implementation Pattern:
    class RiskLimitsCog(BaseCog):
        default_config = RiskLimitsConfig()

        def _analyze_impl(self, context: PositionAnalysisContext) -> CogActions:
            opinions = []
            for position in context.positions:
                if self._check_limits(position):
                    action = CLOSE(position.trade_id)
                    action.reason = "Delta limit exceeded"
                    opinions.append(PositionState(
                        trade_id=position.trade_id,
                        action=action
                    ))
            return CogActions(
                date=context.date,
                source_cog=self.name,
                opinions=opinions
            )

        def on_new_position(self, order: Order, request: OrderRequest) -> None:
            # Register position limits
            self._setup_limits(order.trade_id)

Validation and Enforcement:
    __init_subclass__ Hook:
        - Validates default_config exists
        - Ensures default_config is BaseCogConfig instance
        - Raises errors on invalid subclass definitions
        - Enforces architectural contracts

    Runtime Validation:
        - Config type checking on assignment
        - CogActions source_cog verification
        - Opinion structure validation

CogActions Structure:
    Output from cog analysis:
        - date: Analysis date
        - source_cog: Name of cog generating opinions
        - opinions: List of PositionState objects

    Each PositionState contains:
        - trade_id: Position identifier
        - action: RMAction (HOLD, CLOSE, ROLL, etc.)
        - Additional metadata

Lifecycle Methods:
    analyze(context):
        - Public method called by PositionAnalyzer
        - Checks enabled flag
        - Calls _analyze_impl() if enabled
        - Validates and returns CogActions
        - Template method pattern

    _analyze_impl(context):
        - Abstract method for subclasses
        - Core analysis logic goes here
        - Returns CogActions with opinions
        - Must be implemented by all cogs

    on_new_position(order, request):
        - Abstract method for subclasses
        - Called when new position detected
        - Setup tracking, limits, etc.
        - Optional implementation

Properties:
    name:
        - Returns cog name from config
        - Used for logging and identification
        - Must be unique within analyzer

    enabled:
        - Returns enabled flag from config
        - Controls whether cog runs
        - Can be changed at runtime

    config:
        - Getter/setter for cog configuration
        - Type-validated on assignment
        - Allows runtime reconfiguration

Usage Example:
    # Define custom cog
    class SignalCog(BaseCog):
        default_config = SignalCogConfig()

        def _analyze_impl(self, context):
            # Implement signal-based analysis
            ...
            return CogActions(...)

        def on_new_position(self, order, request):
            # Track signal for this position
            self.position_signals[order.trade_id] = request.signal_id

    # Instantiate and use
    signal_cog = SignalCog(config=SignalCogConfig(name='signal'))
    analyzer.add_cog(signal_cog)

Error Handling:
    - Missing default_config raises AttributeError at class definition
    - Wrong config type raises TypeError at instantiation
    - NotImplementedError if _analyze_impl() not overridden
    - ValueError if source_cog mismatch detected

Design Benefits:
    - Type-safe configuration management
    - Consistent interface across all cogs
    - Easy to test individual cogs
    - Clear separation of concerns
    - Extensible without modifying existing code
    - Self-documenting via config classes

Integration:
    - BaseCog used by all position analysis components
    - PositionAnalyzer orchestrates multiple BaseCog instances
    - RiskManager provides context and consumes actions
    - Config classes loaded from YAML/JSON files

Notes:
    - Frozen PositionKey prevents accidental modification
    - BaseCog is abstract - cannot be instantiated directly
    - Subclasses must set default_config class attribute
    - Config validation happens at multiple levels
    - Template method pattern ensures consistent behavior
"""

from dataclasses import dataclass
from typing import List
from EventDriven.configs.core import BaseCogConfig
from EventDriven.types import Order
from EventDriven.dataclasses.orders import OrderRequest
from EventDriven.dataclasses.states import (
    PositionAnalysisContext,
    CogActions,
)


### RUNTIME DATACLASSES
@dataclass(frozen=True)
class PositionKey:
    """
    Uniquely identifies a logical position we are managing.
    Typically: (strategy, underlying).
    """

    strategy_id: str
    trade_id: str
    signal_id: str


@dataclass
class StrategyChangeMeta:
    """
    Final instruction for how a given (strategy, underlying) should be adjusted.
    This is what RiskManager will consume.
    """

    key: PositionKey
    ## Actions proposed by various cogs.
    strategy_changes: List[CogActions]


# ======================================================================
# BaseCog: all cogs must subclass this
# ======================================================================


class BaseCog:
    """
    Base class for all PositionAnalyzer cogs.

    Responsibilities:
    - Holds a config (all knobs live there).
    - Exposes a public 'analyze' method with fixed signature.
    - Ensures all outputs are well-formed CogActions instances and tagged
      with the cog's name.
    """

    default_config_class = BaseCogConfig
    default_config_class_attr_name = "default_config"

    def __init__(self, config: BaseCogConfig):
        assert isinstance(config, BaseCogConfig), "config must be an instance of BaseCogConfig or its subclass"
        self._config = config

    def __init_subclass__(cls):
        super().__init_subclass__()

        ## Enforce that subclasses have a '_config' attribute of type BaseCogConfig
        if not hasattr(cls, cls.default_config_class_attr_name):
            raise AttributeError(
                f"Subclass {cls.__name__} must have a '{cls.default_config_class_attr_name}' attribute."
            )

        if not isinstance(cls.default_config, BaseCogConfig):
            raise TypeError(
                f"Subclass {cls.__name__}.{cls.default_config_class_attr_name} must be an instance of BaseCogConfig or its subclass. Received type: {type(cls.default_config_class)}"
            )

    @property
    def config(self) -> BaseCogConfig:
        return self._config

    @config.setter
    def config(self, new_config: BaseCogConfig):
        if not isinstance(new_config, BaseCogConfig):
            raise TypeError("new_config must be an instance of BaseCogConfig or its subclass")
        self._config = new_config

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def analyze(self, context: PositionAnalysisContext) -> CogActions:
        """
        Template method: checks enabled flag, calls the subclass implementation,
        and guarantees that all returned opinions are tagged with source_cog.
        """
        if not self.enabled:
            return CogActions(date=context.date, source_cog=self.name, opinions=[])

        cog_actions = self._analyze_impl(context) or []

        # Ensure schema invariants (minimal for now; can be extended later).
        if cog_actions.source_cog and cog_actions.source_cog != self.name:
            raise ValueError(f"CogActions source_cog {cog_actions.source_cog} does not match cog name {self.name}")

        return cog_actions

    # ---- to be implemented by subclasses ----
    def _analyze_impl(self, context: PositionAnalysisContext) -> CogActions:
        """
        Subclasses implement this. Must return a CogActions object.
        """
        raise NotImplementedError("Subclasses must implement _analyze_impl().")

    def on_new_position(self, order: Order, request: OrderRequest) -> None:
        """
        Hook method called when a new position is detected.
        Subclasses can override this to perform any initialization or logging.
        """
        raise NotImplementedError("Subclasses must implement on_new_position().")
