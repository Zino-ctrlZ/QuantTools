"""Import every feature family so ``@feature`` registrations populate the registry.

Call ``load_families()`` once at package import (or explicitly in tests).
After imports, ``validate_registry`` fails fast if any feature is incomplete.

The loaded flag lives on ``specs`` (not this module) so autoreload of
``families.py`` does not forget a prior successful load while
``FEATURE_SPECS`` is still populated.
"""

from __future__ import annotations

import importlib
import sys
from typing import Tuple

from trade.features.compute import specs as _specs

## Leaf family modules that run ``@feature`` at import time.
_FAMILY_MODULES: Tuple[str, ...] = (
    "trade.features.compute.market.china",
    "trade.features.compute.market.dollar",
    "trade.features.compute.market.fred_iv",
    "trade.features.compute.market.ratios",
    "trade.features.compute.market.spy",
    "trade.features.compute.market.vix",
    "trade.features.compute.asset.confirm",
    "trade.features.compute.asset.downside",
    "trade.features.compute.asset.gaps",
    "trade.features.compute.asset.momentum",
    "trade.features.compute.asset.path",
    "trade.features.compute.asset.range_",
    "trade.features.compute.asset.regime",
    "trade.features.compute.asset.trend",
    "trade.features.compute.asset.vol",
    "trade.features.compute.asset.volume",
    "trade.features.compute.hybrid.iv_rvol",
)


def _import_or_reload_families() -> None:
    """Import family packages; reload leaf modules when specs were wiped."""
    ## Package imports pull leaf modules the first time.
    from trade.features.compute import asset as _asset  # noqa: F401
    from trade.features.compute import hybrid as _hybrid  # noqa: F401
    from trade.features.compute import market as _market  # noqa: F401

    ## If ``specs`` reloaded empty, cached family modules never re-ran
    ## ``@feature`` — force leaf reloads so registrations rebuild.
    if _specs.FEATURE_SPECS:
        return
    for mod_name in _FAMILY_MODULES:
        mod = sys.modules.get(mod_name)
        if mod is None:
            importlib.import_module(mod_name)
        else:
            importlib.reload(mod)


def load_families() -> None:
    """Import market / asset / hybrid family modules and validate (idempotent).

    Skips work when ``specs.FAMILIES_LOADED`` is set and ``FEATURE_SPECS`` is
    non-empty. If autoreload wiped ``FEATURE_SPECS`` but left the flag true,
    clears the flag and re-imports so registrations rebuild.
    """
    ## Specs wiped under us (rare: specs.py itself reloaded) — force rebuild.
    if _specs.FAMILIES_LOADED and not _specs.FEATURE_SPECS:
        _specs.FAMILIES_LOADED = False
    if _specs.FAMILIES_LOADED:
        return

    _import_or_reload_families()
    from trade.features.compute.validate import validate_registry

    validate_registry()
    _specs.FAMILIES_LOADED = True
