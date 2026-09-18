"""Asset sector classification for datamanager.

Resolves GICS-style sectors from a curated YAML override file, then Yahoo
Finance ticker info when the symbol is missing. ETFs typically lack a Yahoo
sector, so symbols such as QQQ are mapped in YAML.

Comment density: orchestration

Core Classes:
    Sector: Yahoo Finance sector labels exposed as an enum.

Core Functions:
    get_asset_sector: YAML override first, then yfinance ``info.sector`` (persisted).

Processing Flow:
    1. Normalize the ticker to uppercase.
    2. Load ``asset_sectors.yaml`` and return a matching Sector if present.
    3. Otherwise call ``yf.Ticker(asset).info.get("sector")`` and map to Sector.
    4. Persist that Yahoo result to YAML and the in-memory cache so later lookups skip yfinance.
    5. Missing Yahoo/YAML sector (``None`` or blank) maps to ``Sector.UNKNOWN``.

Usage:
    >>> from trade.datamanager.category import get_asset_sector, Sector
    >>> get_asset_sector("QQQ")
    <Sector.TECHNOLOGY: 'Technology'>
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Dict, Optional

import yaml
import yfinance as yf

from trade.helpers.Logging import setup_logger
from trade.datamanager.utils.logging import get_logging_level

logger = setup_logger("trade.datamanager.category", stream_log_level=get_logging_level())

_DATAMANAGER_DIR = Path(__file__).resolve().parent
_YAML_PATH = _DATAMANAGER_DIR / "asset_sectors.yaml"

## In-memory mirror of asset_sectors.yaml; process-lifetime, re-merged on miss.
_SECTOR_OVERRIDE_CACHE: Dict[str, str] = {}


class Sector(str, Enum):
    """Yahoo Finance equity sector labels.

    Values match ``yf.Ticker(...).info["sector"]`` strings so YAML overrides and
    vendor lookups share one mapping. ``UNKNOWN`` is the fallback when Yahoo
    (or YAML) has no sector.
    """

    BASIC_MATERIALS = "Basic Materials"
    COMMUNICATION_SERVICES = "Communication Services"
    CONSUMER_CYCLICAL = "Consumer Cyclical"
    CONSUMER_DEFENSIVE = "Consumer Defensive"
    ENERGY = "Energy"
    FINANCIAL_SERVICES = "Financial Services"
    HEALTHCARE = "Healthcare"
    INDUSTRIALS = "Industrials"
    REAL_ESTATE = "Real Estate"
    TECHNOLOGY = "Technology"
    UTILITIES = "Utilities"
    UNKNOWN = "Unknown"

    @classmethod
    def from_label(cls, label: Optional[str]) -> "Sector":
        """Return the enum member whose value matches ``label``.

        Matching is case-insensitive and strips surrounding whitespace.
        ``None``, blank, and ``"None"`` map to ``UNKNOWN``.

        Args:
            label: Sector string from YAML or Yahoo Finance.

        Returns:
            Matching ``Sector`` member.

        Raises:
            ValueError: If ``label`` is a non-empty string that is not a known sector.
        """
        ## Yahoo returns None for many ETFs/indexes; YAML may store null the same way.
        if label is None:
            return cls.UNKNOWN
        normalized = str(label).strip().casefold()
        if not normalized or normalized in {"none", "nan", "null"}:
            return cls.UNKNOWN
        for member in cls:
            if member.value.casefold() == normalized:
                return member
        known = ", ".join(member.value for member in cls)
        raise ValueError(f"Unknown sector {label!r}. Expected one of: {known}.")


def _normalize_symbol(symbol: str) -> str:
    """Return an uppercase ticker key for YAML lookup.

    Args:
        symbol: Raw ticker string.

    Returns:
        Uppercase ticker suitable for dict/YAML keys.
    """
    return symbol.strip().upper()


def _merge_yaml_into_cache() -> None:
    """Load entries from ``asset_sectors.yaml`` into the in-memory cache.

    Re-reads the file on each call so manual YAML edits are visible on the next
    cache miss for a symbol. Merges without clearing existing entries.
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

    for raw_symbol, raw_sector in loaded.items():
        symbol = _normalize_symbol(str(raw_symbol))
        ## YAML null becomes UNKNOWN via from_label; keep the raw token as-is.
        if raw_sector is None:
            _SECTOR_OVERRIDE_CACHE[symbol] = Sector.UNKNOWN.value
        else:
            _SECTOR_OVERRIDE_CACHE[symbol] = str(raw_sector).strip()


def _persist_cache_to_yaml() -> None:
    """Write the in-memory cache to ``asset_sectors.yaml``.

    Overwrites the file with sorted ticker keys so yfinance discoveries survive
    process restart. Callers must merge disk into cache first so manual YAML
    edits are not dropped.
    """
    payload = dict(sorted(_SECTOR_OVERRIDE_CACHE.items()))
    header = (
        "# Curated ticker -> sector overrides (Yahoo Finance sector labels).\n"
        "# Keys are uppercase tickers. Used before yfinance info.sector lookup.\n"
        "# Missing symbols are filled from yfinance on first lookup and saved here.\n"
        "# ETFs usually have no Yahoo sector; map those here or they persist as Unknown.\n"
    )
    with _YAML_PATH.open("w", encoding="utf-8") as handle:
        handle.write(header)
        yaml.safe_dump(payload, handle, default_flow_style=False, sort_keys=True)


def _store_sector_override(symbol_key: str, sector: Sector) -> None:
    """Cache ``sector`` for ``symbol_key`` and persist the mapping to YAML.

    Args:
        symbol_key: Uppercase ticker.
        sector: Resolved sector to remember for later YAML lookups.
    """
    _SECTOR_OVERRIDE_CACHE[symbol_key] = sector.value
    # ponytail: whole-file rewrite; lock or merge-on-write if concurrent processes collide
    _persist_cache_to_yaml()
    logger.info("Cached sector for %s from yfinance: %s", symbol_key, sector.value)


def _lookup_yaml_sector(symbol_key: str) -> Optional[str]:
    """Return the YAML sector label for ``symbol_key``, if present.

    Args:
        symbol_key: Uppercase ticker.

    Returns:
        Raw sector string from YAML, or ``None`` when the ticker is absent.
    """
    if symbol_key in _SECTOR_OVERRIDE_CACHE:
        return _SECTOR_OVERRIDE_CACHE[symbol_key]

    ## YAML may have been edited on disk since the last in-process lookup.
    _merge_yaml_into_cache()
    return _SECTOR_OVERRIDE_CACHE.get(symbol_key)


def _lookup_yfinance_sector(symbol_key: str) -> Optional[str]:
    """Return Yahoo Finance ``info.sector`` for ``symbol_key``.

    Args:
        symbol_key: Uppercase ticker.

    Returns:
        Sector string from Yahoo, or ``None`` when the field is missing.
    """
    info = yf.Ticker(symbol_key).info or {}
    sector = info.get("sector")
    if sector is None:
        return None
    sector_str = str(sector).strip()
    if not sector_str:
        return None
    return sector_str


def get_asset_sector(asset: str) -> Sector:
    """Return the sector for ``asset``.

    Lookup order: curated ``asset_sectors.yaml`` override, then Yahoo Finance
    ticker info. Yahoo misses are written back to YAML so later calls use the
    file. Missing Yahoo/YAML sector maps to ``Sector.UNKNOWN``.

    Args:
        asset: Equity or ETF ticker (case-insensitive).

    Returns:
        Matching ``Sector`` enum member, or ``Sector.UNKNOWN`` when the vendor
        sector is ``None``.

    Raises:
        ValueError: If Yahoo/YAML returns a non-empty label that is not a known sector.

    Examples:
        >>> get_asset_sector("QQQ")
        <Sector.TECHNOLOGY: 'Technology'>
    """
    symbol_key = _normalize_symbol(asset)

    yaml_sector = _lookup_yaml_sector(symbol_key)
    if yaml_sector is not None:
        return Sector.from_label(yaml_sector)

    ## YAML miss: Yahoo is the remaining source. Persist so later processes
    ## (and the rest of this process) hit YAML instead of repeating the vendor call.
    sector = Sector.from_label(_lookup_yfinance_sector(symbol_key))
    _store_sector_override(symbol_key, sector)
    return sector
