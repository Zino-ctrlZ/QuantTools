"""Liquidity policy utilities shared across risk and execution layers."""

from typing import Optional
from EventDriven.configs.core import LiquidityConfig


class LiquidityPolicy:
    """Encapsulates liquidity-level decisions from a shared LiquidityConfig."""

    def __init__(self, config: Optional[LiquidityConfig] = None):
        self.config = config or LiquidityConfig()

    @property
    def level(self) -> int:
        return self.config.level

    @property
    def max_spread_pct(self) -> float:
        return self.config.max_spread_pct

    def enabled(self, required_level: int) -> bool:
        return self.level >= required_level

    def should_drop_for_spread(self, spread_pct: Optional[float]) -> bool:
        if spread_pct is None:
            return False
        return self.enabled(2) and spread_pct > self.max_spread_pct
