"""Cache key helpers for FeatureDataManager L1/L2 entries.

Each durable feature is one ``pd.Series`` addressed by symbol (ticker or
``__MARKET__``), feature name, and ``formula_id``. Caller-supplied OHLCV uses
the same shape with ``bars="custom"`` so it cannot collide with vendor-backed
hist keys; those custom keys are L1-only by manager policy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from trade.datamanager._enums import ArtifactType, Interval, SeriesId

if TYPE_CHECKING:
    from trade.datamanager.base import BaseDataManager

## Synthetic symbol for market-only features shared across tickers.
MARKET_SYMBOL: str = "__MARKET__"

## Extra key part for caller-supplied OHLCV (L1 only; never written to L2).
CUSTOM_BARS_TAG: str = "custom"


def feature_cache_key(
    manager: "BaseDataManager",
    *,
    symbol: str,
    feature: str,
    formula_id: str,
    interval: Interval = Interval.EOD,
    custom_bars: bool = False,
) -> str:
    """Build a namespaced CustomCache key for one feature Series.

    Args:
        manager: Manager providing ``make_key`` (usually ``FeatureDataManager``).
        symbol: Ticker or ``MARKET_SYMBOL`` for market-scope features.
        feature: Catalog feature name (e.g. ``spy_ret_20d``).
        formula_id: Stable formula identity from ``Feature``.
        interval: Bar interval; features are EOD-only today.
        custom_bars: When True, tag the key as caller OHLCV (L1-only path).

    Returns:
        Deterministic cache key string.
    """
    extra = {}
    if custom_bars:
        extra["bars"] = CUSTOM_BARS_TAG
    return manager.make_key(
        symbol=symbol,
        interval=interval,
        artifact_type=ArtifactType.FEATURE,
        series_id=SeriesId.HIST,
        feature=feature,
        formula_id=formula_id,
        **extra,
    )
