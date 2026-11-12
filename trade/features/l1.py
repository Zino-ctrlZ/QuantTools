"""Process-local L1 Series cache in front of CustomCache L2.

A plain dict: same key string as L2. No LRU / market-vs-asset policy.
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd


class FeatureL1Cache:
    """In-process map of feature cache key → Series."""

    def __init__(self) -> None:
        """Create an empty L1 map."""
        self._store: Dict[str, pd.Series] = {}

    def get(self, key: str) -> Optional[pd.Series]:
        """Return a cached Series for ``key``, or None on miss."""
        return self._store.get(key)

    def set(self, key: str, series: pd.Series) -> None:
        """Store a defensive copy of ``series`` under ``key``."""
        if not isinstance(series, pd.Series):
            raise TypeError(f"L1 expects pd.Series, got {type(series)}")
        self._store[key] = series.copy()

    def clear(self) -> None:
        """Drop all L1 entries."""
        self._store.clear()

    def __len__(self) -> int:
        """Return number of cached Series."""
        return len(self._store)
