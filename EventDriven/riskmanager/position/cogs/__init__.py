"""Position Analysis Cogs Collection.

This package contains individual cog implementations for position analysis.
Each cog focuses on a specific aspect of position management and risk control.

Available Cogs:
    LimitsAndSizingCog (limits.py):
        - Greek-based risk limit enforcement
        - Dynamic position sizing
        - Delta, gamma, vega, theta monitoring
        - Capital allocation management

    VectorizedCog (vectorized.py):
        - DTE-based roll trigger detection
        - New position cash validation (informational)
        - Minimal, stateless position monitoring

    PlainSizingCog (plain_sizing.py):
        - default_delta_limit based quantity sizing
        - Quantity fallback to 1 when affordable and computed size is 0
        - DTE-based roll trigger detection
        - Strategy-slug exclusion checks for both sizing and analysis

Utilities:
    analyze_utils.py:
        - DTE calculation from position IDs
        - Moneyness computation with corporate action adjustments
        - Position parsing and metadata extraction
        - Split and dividend handling

    vars.py:
        - Shared constants and enumerations
        - Greek measure definitions (MEASURES)
        - Action priority mappings
        - Actionable event types

Design Pattern:
    All cogs implement the BaseCog interface:
    - _analyze_impl(): Core analysis logic
    - on_new_position(): Position initialization hook
    - Standardized configuration via BaseCogConfig subclasses
    - Opinion-based recommendation system

Usage:
    from EventDriven.riskmanager.position.cogs import PlainSizingCog, VectorizedCog
    from EventDriven.riskmanager.position.cogs.analyze_utils import (
        get_dte_and_moneyness_from_trade_id
    )

See Also:
    - ../base.py: BaseCog abstract class definition
    - ../analyzer.py: Cog orchestration and reconciliation
"""

from EventDriven.riskmanager.position.cogs.vectorized import VectorizedCog
from EventDriven.riskmanager.position.cogs.plain_sizing import PlainSizingCog
from EventDriven.configs.core import VectorizedCogConfig, PlainSizingCogConfig

__all__ = ["VectorizedCog", "VectorizedCogConfig", "PlainSizingCog", "PlainSizingCogConfig"]
