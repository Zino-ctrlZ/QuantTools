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
            raise AttributeError(f"Subclass {cls.__name__} must have a '{cls.default_config_class_attr_name}' attribute.")

        
        if not isinstance(cls.default_config, BaseCogConfig):
            raise TypeError(
                f"Subclass {cls.__name__}.{cls.default_config_class_attr_name} must be an instance of BaseCogConfig or its subclass. Received type: {type(cls.default_config_class)}"
            )
        

    @property
    def config(self) -> BaseCogConfig:
        return self._config


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

