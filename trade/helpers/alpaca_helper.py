"""Alpaca market-data helpers for equity timeseries.

Thin wrapper over alpaca-py's ``StockHistoricalDataClient`` that returns daily
bars in the same shape as yfinance ``Ticker.history`` so callers can swap
vendors without reshaping. Credentials come from the environment, never code.

Core Functions:
    alpaca_stock_history: Daily OHLCV bars for one symbol in yfinance shape.

Risk/Assumptions:
    - Prices are raw (``Adjustment.RAW``); ``Adj Close`` mirrors ``Close`` and
      ``Dividends`` / ``Stock Splits`` are always 0.0.
    - The default IEX feed carries IEX-only prints and volume, which can differ
      from the consolidated (SIP) close and volume.
    - ``end`` is exclusive for daily bars (Alpaca stamps them at midnight NY).

Usage:
    >>> from trade.helpers.alpaca_helper import alpaca_stock_history
    >>> alpaca_stock_history("TSLA", start="2026-09-23", end="2026-09-25")  # doctest: +SKIP
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional, Union

import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from trade.helpers.helper import to_datetime

YF_COLS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
    "Dividends",
    "Stock Splits",
]


def alpaca_stock_history(
    tick: str,
    start: Union[str, datetime],
    end: Optional[Union[str, datetime]] = None,
    *,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    feed: DataFeed = DataFeed.IEX,
) -> pd.DataFrame:
    """Daily equity bars from Alpaca in yfinance ``Ticker.history`` shape.

    Args:
        tick: Equity symbol (e.g. ``"TSLA"``).
        start: Inclusive start (``YYYY-MM-DD`` or datetime).
        end: Exclusive end (Alpaca convention). ``None`` means through now.
        api_key: Alpaca key. Defaults to ``EMEFIELE_ALPACA_KEY`` from the environment.
        api_secret: Alpaca secret. Defaults to ``EMEFIELE_ALPACA_SECRET`` from the environment.
        feed: Market-data feed. Use ``DataFeed.SIP`` if the plan allows it.

    Returns:
        DataFrame indexed by tz-aware ``Date`` (America/New_York) with columns
        Open, High, Low, Close, Adj Close, Volume, Dividends, Stock Splits.
        Empty frame with those columns when Alpaca returns no bars.

    Raises:
        KeyError: If no key/secret is passed and the environment variables are unset.

    Examples:
        >>> df = alpaca_stock_history("TSLA", "2026-09-23", "2026-09-25")  # doctest: +SKIP
        >>> list(df.columns) == YF_COLS  # doctest: +SKIP
        True
    """
    key = api_key or os.environ["EMEFIELE_ALPACA_KEY"]
    secret = api_secret or os.environ["EMEFIELE_ALPACA_SECRET"]
    client = StockHistoricalDataClient(key, secret)

    bars = client.get_stock_bars(
        StockBarsRequest(
            symbol_or_symbols=tick,
            timeframe=TimeFrame.Day,
            start=to_datetime(start),
            end=to_datetime(end) if end is not None else None,
            adjustment=Adjustment.RAW,
            feed=feed,
        )
    )
    raw = bars.df
    if raw is None or raw.empty:
        return pd.DataFrame(columns=YF_COLS)

    ## Alpaca returns a (symbol, timestamp) MultiIndex even for one symbol.
    if isinstance(raw.index, pd.MultiIndex):
        raw = raw.xs(tick, level="symbol")

    ## Bar timestamps are UTC; NY conversion lands daily bars on the session date at midnight.
    idx = pd.DatetimeIndex(to_datetime(raw.index, ny_utc=True))
    idx.name = "Date"

    out = pd.DataFrame(
        {
            "Open": raw["open"].to_numpy(),
            "High": raw["high"].to_numpy(),
            "Low": raw["low"].to_numpy(),
            "Close": raw["close"].to_numpy(),
            "Adj Close": raw["close"].to_numpy(),
            "Volume": raw["volume"].to_numpy(),
            "Dividends": 0.0,
            "Stock Splits": 0.0,
        },
        index=idx,
    )
    return out[YF_COLS]
