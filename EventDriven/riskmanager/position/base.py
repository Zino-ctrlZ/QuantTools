from dataclasses import dataclass
from typing import List
from datetime import datetime
from EventDriven.configs.base import BaseConfigs
from EventDriven.types import EventTypes, Order
from EventDriven.dataclasses.orders import OrderRequest
from EventDriven.dataclasses.states import PortfolioState




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
class PositionAnalysisContext:
    """
    Read-only snapshot of everything a cog needs to reason about positions.
    This is passed into every cog on each analysis run.
    """

    timestamp: datetime
    portfolio: PortfolioState


@dataclass
class CogOpinion:
    """
    A single cog's 'vote' about what a given (strategy, underlying) target should be.

    This is the core schema all cogs must adhere to.
    """

    key: PositionKey
    event_action: EventTypes
    position_change: float  # positive = increase position, negative = decrease position

    # Provenance & diagnostics
    reason_code: str = ""
    message: str = ""
    source_cog: str = ""  # filled in by BaseCog so we always know who said it

    # For reconciliation: higher priority may override lower, or be used differently.
    priority: int = 0
    weight: float = 1.0


@dataclass
class StrategyChangeMeta:
    """
    Final instruction for how a given (strategy, underlying) should be adjusted.
    This is what RiskManager will consume.
    """

    key: PositionKey
    ## Actions proposed by various cogs.
    strategy_changes: List[CogOpinion]


# ======================================================================
# BaseCog: all cogs must subclass this
# ======================================================================


class BaseCog:
    """
    Base class for all PositionAnalyzer cogs.

    Responsibilities:
    - Holds a config (all knobs live there).
    - Exposes a public 'analyze' method with fixed signature.
    - Ensures all outputs are well-formed CogOpinion instances and tagged
      with the cog's name.
    """

    def __init__(self, config: BaseConfigs):
        self.config = config

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def analyze(self, context: PositionAnalysisContext) -> List[CogOpinion]:
        """
        Template method: checks enabled flag, calls the subclass implementation,
        and guarantees that all returned opinions are tagged with source_cog.
        """
        if not self.enabled:
            return []

        opinions = self._analyze_impl(context) or []

        # Ensure schema invariants (minimal for now; can be extended later).
        tagged: List[CogOpinion] = []
        for op in opinions:
            if op.source_cog and op.source_cog != self.name:
                # You *can* override this behavior if you ever need a "proxy" cog,
                # but by default we enforce self.name for clarity.
                pass
            else:
                op.source_cog = self.name
            tagged.append(op)

        return tagged

    # ---- to be implemented by subclasses ----
    def _analyze_impl(self, context: PositionAnalysisContext) -> List[CogOpinion]:
        """
        Subclasses implement this. Must return a list of CogOpinion objects.
        """
        raise NotImplementedError("Subclasses must implement _analyze_impl().")

    def on_new_position(self, order: Order, request: OrderRequest) -> None:
        """
        Hook method called when a new position is detected.
        Subclasses can override this to perform any initialization or logging.
        """
        raise NotImplementedError("Subclasses must implement on_new_position().")

