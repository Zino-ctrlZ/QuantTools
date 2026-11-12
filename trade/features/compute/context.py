"""Shared input bag for feature compute functions.

Every market / asset / hybrid feature receives one ``FeatureContext`` and pulls
what it needs. No per-scope handler signatures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional

import pandas as pd


@dataclass
class FeatureContext:
    """Inputs available to one feature compute call.

    Attributes:
        closes: Market symbol key → close Series (``SPY``, ``VIX``, …).
        ivs: IV key → decimal IV Series (``FXI_IV``, ``EEM_IV``, …).
        ohlcv: Optional normalized asset OHLCV frame (lowercase columns).
    """

    closes: Mapping[str, pd.Series] = field(default_factory=dict)
    ivs: Mapping[str, pd.Series] = field(default_factory=dict)
    ohlcv: Optional[pd.DataFrame] = None

    def close(self, symbol: str) -> pd.Series:
        """Return a market close Series by symbol key."""
        try:
            return self.closes[symbol]
        except KeyError as exc:
            raise KeyError(f"FeatureContext missing close for {symbol!r}") from exc

    def iv(self, key: str) -> pd.Series:
        """Return an IV Series by logical key."""
        try:
            return self.ivs[key]
        except KeyError as exc:
            raise KeyError(f"FeatureContext missing IV for {key!r}") from exc

    def asset_close(self) -> pd.Series:
        """Return the asset close column."""
        if self.ohlcv is None or "close" not in self.ohlcv.columns:
            raise ValueError("FeatureContext.ohlcv with a close column is required")
        return self.ohlcv["close"].astype(float)

    def asset_col(self, name: str) -> pd.Series:
        """Return one asset OHLCV column by lowercase name."""
        if self.ohlcv is None:
            raise ValueError("FeatureContext.ohlcv is required")
        if name not in self.ohlcv.columns:
            raise ValueError(f"FeatureContext.ohlcv missing column {name!r}")
        return self.ohlcv[name].astype(float)
