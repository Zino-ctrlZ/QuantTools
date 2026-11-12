"""Result container for FeatureDataManager returns."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import pandas as pd

from trade.datamanager.result import Result


@dataclass
class FeatureResult(Result):
    """Named feature Series from ``FeatureDataManager``.

    Attributes:
        series: Mapping of feature name → Series (already sliced if an index
            was requested).
        ticker: Asset ticker when hybrid/asset features were requested.
    """

    series: Dict[str, pd.Series] = field(default_factory=dict)
    ticker: Optional[str] = None

    def is_empty(self) -> bool:
        """Return True when no feature Series were produced."""
        return not self.series

    def _additional_repr_fields(self) -> Dict[str, object]:
        """Include ticker and feature count in ``repr``."""
        return {"ticker": self.ticker, "n_features": len(self.series)}
