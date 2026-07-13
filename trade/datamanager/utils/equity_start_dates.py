"""Cached per-symbol equity listing start dates for datamanager date sync.

Resolves the first available EOD observation for a ticker by probing vendor
timeseries, then persists results in ``equity_start_dates.yaml`` beside the
datamanager package. An in-process dict mirrors the YAML so repeat lookups
avoid disk I/O.

Core Functions:
    get_start_date: Return cached or probe-and-store first trade date for a symbol.

Usage:
    >>> from trade.datamanager.utils.equity_start_dates import get_start_date
    >>> get_start_date("AAPL")
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import yaml

from trade.helpers.Logging import setup_logger
from trade.helpers.helper import YFinanceEmptyData, ny_now, retrieve_timeseries, to_datetime
from trade.datamanager.utils.logging import get_logging_level, UTILS_LOGGER_NAME

logger = setup_logger(UTILS_LOGGER_NAME, stream_log_level=get_logging_level())

## Wide probe window; first returned bar is treated as listing/IPO proxy.
EQUITY_PROBE_START_DATE = "1800-01-01"

_DATAMANAGER_DIR = Path(__file__).resolve().parent.parent
_YAML_PATH = _DATAMANAGER_DIR / "equity_start_dates.yaml"

## In-memory mirror of equity_start_dates.yaml; survives for process lifetime.
_START_DATE_CACHE: Dict[str, str] = {}


def _normalize_symbol(symbol: str) -> str:
    """Return an uppercase ticker key for cache and YAML storage.

    Args:
        symbol: Raw ticker string.

    Returns:
        Uppercase ticker suitable for dict/YAML keys.
    """
    return symbol.strip().upper()


def _merge_yaml_into_cache() -> None:
    """Load entries from ``equity_start_dates.yaml`` into the in-memory cache.

    Merges without clearing existing cache entries so runtime discoveries in the
    same process are preserved. Re-reads the file on each call so manual YAML
    edits are visible on the next cache miss for a symbol.
    """
    if not _YAML_PATH.exists():
        return

    with _YAML_PATH.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)

    if not loaded:
        return

    if not isinstance(loaded, dict):
        logger.warning("Expected mapping in %s; ignoring contents.", _YAML_PATH)
        return

    for raw_symbol, raw_date in loaded.items():
        symbol = _normalize_symbol(str(raw_symbol))
        _START_DATE_CACHE[symbol] = to_datetime(raw_date).strftime("%Y-%m-%d")


def _persist_cache_to_yaml() -> None:
    """Write the in-memory cache to ``equity_start_dates.yaml``."""
    payload = dict(sorted(_START_DATE_CACHE.items()))
    header = (
        "# Per-symbol first observed EOD date (IPO / listing proxy).\n"
        "# Keys are uppercase tickers; values are YYYY-MM-DD strings.\n"
        "# Populated on demand via get_start_date; safe to edit manually.\n"
    )
    with _YAML_PATH.open("w", encoding="utf-8") as handle:
        handle.write(header)
        yaml.safe_dump(payload, handle, default_flow_style=False, sort_keys=True)


def _probe_first_trade_date(symbol: str) -> Optional[str]:
    """Query vendor EOD history and return the first observed calendar date.

    Args:
        symbol: Uppercase equity ticker.

    Returns:
        First trade date as ``YYYY-MM-DD``, or ``None`` when no history exists.
    """
    end_str = ny_now().strftime("%Y-%m-%d")
    try:
        frame = retrieve_timeseries(
            tick=symbol,
            start=EQUITY_PROBE_START_DATE,
            end=end_str,
        )
    except YFinanceEmptyData:
        logger.warning(
            "No equity history for %s between %s and %s; skipping start-date floor.",
            symbol,
            EQUITY_PROBE_START_DATE,
            end_str,
        )
        return None

    if frame is None or frame.empty:
        logger.warning(
            "Empty equity history for %s between %s and %s; skipping start-date floor.",
            symbol,
            EQUITY_PROBE_START_DATE,
            end_str,
        )
        return None

    first_ts = to_datetime(frame.index.min())
    return first_ts.strftime("%Y-%m-%d")


def get_start_date(symbol: str) -> Optional[str]:
    """Return the first known EOD date for an equity ticker.

    Lookup order: in-memory cache, YAML file, then vendor timeseries probe from
    ``EQUITY_PROBE_START_DATE`` through today. New discoveries are written to YAML
    and merged into the in-memory cache.

    Args:
        symbol: Equity ticker (case-insensitive).

    Returns:
        First trade date as ``YYYY-MM-DD``, or ``None`` when history is unavailable.

    Examples:
        >>> get_start_date("AAPL")  # doctest: +SKIP
    """
    symbol_key = _normalize_symbol(symbol)

    if symbol_key in _START_DATE_CACHE:
        return _START_DATE_CACHE[symbol_key]

    ## YAML may have been edited on disk since the last in-process lookup.
    _merge_yaml_into_cache()
    if symbol_key in _START_DATE_CACHE:
        return _START_DATE_CACHE[symbol_key]

    first_date = _probe_first_trade_date(symbol_key)
    if first_date is None:
        return None

    _START_DATE_CACHE[symbol_key] = first_date
    _persist_cache_to_yaml()
    logger.info("Resolved equity start date for %s: %s", symbol_key, first_date)
    return first_date
