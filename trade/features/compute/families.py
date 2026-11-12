"""Import every feature family so ``@feature`` registrations populate the registry.

Call ``load_families()`` once at package import (or explicitly in tests).
After imports, ``validate_registry`` fails fast if any feature is incomplete.
"""

from __future__ import annotations

_LOADED = False


def load_families() -> None:
    """Import market / asset / hybrid family modules and validate (idempotent)."""
    global _LOADED
    if _LOADED:
        return
    from trade.features.compute import asset as _asset  # noqa: F401
    from trade.features.compute import hybrid as _hybrid  # noqa: F401
    from trade.features.compute import market as _market  # noqa: F401
    from trade.features.compute.validate import validate_registry

    validate_registry()
    _LOADED = True
