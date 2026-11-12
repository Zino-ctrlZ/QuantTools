"""Normalize asset OHLCV frames for feature compute."""

from __future__ import annotations

import pandas as pd

from trade.helpers.helper import to_datetime


def normalize_ohlcv(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Lowercase columns and coerce a datetime index; keep caller ``close``.

    MarketTimeseries-style frames already expose split-adjusted ``close`` (and
    ``unadjusted_close`` when present). This helper does not overwrite ``close``
    from ``adj_close`` / ``adjusted_close``.
    """
    frame = ohlcv.copy()
    frame.columns = [str(c).lower().replace(" ", "_") for c in frame.columns]
    frame.index = to_datetime(frame.index)
    if "close" not in frame.columns:
        raise ValueError("ohlcv must include a close column")
    return frame
