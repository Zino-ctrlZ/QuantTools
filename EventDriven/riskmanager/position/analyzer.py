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
    CogActions, # noqa
    PositionState,
)
from EventDriven.riskmanager.actions import RMAction # noqa
from .cogs.vars import ACTION_PRIORITY


logger = setup_logger('EventDriven.riskmanager.position.analyzer', stream_log_level="WARNING")
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
            trade_actions.sort(key=lambda x: ACTION_PRIORITY.get(x.action.type, float('inf')))
            most_important_action = trade_actions[0]
            strategy_changes.append(most_important_action)

        return StrategyChangeMeta(
            date=context.date,
            actionables=strategy_changes,
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
