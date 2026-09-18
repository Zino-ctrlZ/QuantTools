"""Append ThetaData quote 472 misses to a CSV under QuantTools ``.cache``.

Range quote fetches omit per-date 472s so the rest of the window can finish.
This file records those misses (contract ids, session date, request URL) for
offline review. Writes are locked so multithreaded range fetches can append
safely. Failures to write must not abort the fetch.

Core Functions:
    quote_472_csv_path: Resolve ``GEN_CACHE_PATH/thetadata/quote_472.csv``.
    append_quote_472_row: Append one 472 row.

Usage:
    >>> from trade.datamanager.utils.quote_472_log import append_quote_472_row
    >>> append_quote_472_row(
    ...     symbol="AMD",
    ...     expiration="20251017",
    ...     strike="220.00",
    ...     right="C",
    ...     missing_date="2024-02-21",
    ...     url="http://localhost:25503/v3/option/history/quote?date=20240221",
    ... )
"""

from __future__ import annotations

import csv
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from trade.helpers.Logging import setup_logger

logger = setup_logger("trade.datamanager.utils.quote_472_log", stream_log_level="WARNING")

QUOTE_472_CSV_COLUMNS = [
    "logged_at",
    "symbol",
    "expiration",
    "strike",
    "right",
    "missing_date",
    "url",
]
_WRITE_LOCK = threading.Lock()


def quote_472_csv_path() -> Path:
    """Return the quote-472 CSV path under ``GEN_CACHE_PATH``.

    Returns:
        ``<GEN_CACHE_PATH>/thetadata/quote_472.csv``.
    """
    ## Import at call time so tests can monkeypatch trade.GEN_CACHE_PATH.
    from trade import GEN_CACHE_PATH

    return Path(GEN_CACHE_PATH) / "thetadata" / "quote_472.csv"


def append_quote_472_row(
    *,
    symbol: Optional[str],
    expiration: Optional[str],
    strike: Optional[str],
    right: Optional[str],
    missing_date: Optional[str],
    url: str,
) -> Optional[Path]:
    """Append one quote 472 row. Never raise into the ThetaData worker.

    Args:
        symbol: Underlying ticker.
        expiration: Expiration as sent to ThetaData (usually ``YYYYMMDD``).
        strike: Strike as sent to ThetaData (usually ``%.2f``).
        right: Option right.
        missing_date: Session date as ``YYYY-MM-DD``.
        url: Full history/quote request URL for that session.

    Returns:
        Path written, or None when the write failed.
    """
    path = quote_472_csv_path()
    row = {
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol or "",
        "expiration": expiration or "",
        "strike": strike or "",
        "right": right or "",
        "missing_date": missing_date or "",
        "url": url,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _WRITE_LOCK:
            new_file = (not path.exists()) or path.stat().st_size == 0
            with path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=QUOTE_472_CSV_COLUMNS)
                if new_file:
                    writer.writeheader()
                writer.writerow(row)
        return path
    except Exception as exc:
        logger.warning("Could not append quote 472 csv to %s: %s", path, exc)
        return None
