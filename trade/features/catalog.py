"""Public catalog helpers over the ``@feature`` registry.

The source of truth is family modules under ``compute/market|asset|hybrid``.
This module only adds lookup helpers used by the manager and callers.
"""

from __future__ import annotations

import warnings
from typing import Iterable, List, Optional

from trade.features.compute.definitions import (
    ASSET_IV_RVOL_FEATURES,
    ASSET_SETUP_FEATURES,
    FEATURE_SPECS,
    Feature,
    MARKET_SETUP_FEATURES,
)
from trade.features.compute.registry import get_feature, list_features, list_features_by_theme
from trade.helpers.Logging import setup_logger

logger = setup_logger(__name__)

## Re-exports.
__all__ = [
    "FEATURE_SPECS",
    "Feature",
    "get_spec",
    "list_features",
    "list_features_by_theme",
    "resolve_features",
]


def get_spec(name: str) -> Feature:
    """Return the registry row for ``name``."""
    return get_feature(name)


def _warn_if_deprecated(name: str, spec: Feature) -> None:
    """Emit a deprecation warning when a soft-retired feature is requested."""
    if not spec.deprecated:
        return
    message = (
        f"Feature {name!r} is deprecated (theme=deprecated) and will be removed; "
        "prefer an active replacement or drop it from the request"
    )
    warnings.warn(message, DeprecationWarning, stacklevel=3)
    logger.warning("deprecated feature requested name=%s", name)


def resolve_features(
    features: Optional[Iterable[str]] = None,
    *,
    include_hybrid: bool = False,
    include_asset: bool = False,
) -> List[str]:
    """Validate names, or default to market (+ hybrid/asset when requested).

    Soft-retired (``deprecated``) names are omitted from the default set. An
    explicit list may still include them; each triggers a ``DeprecationWarning``.

    Args:
        features: Explicit names, or ``None`` for the default set.
        include_hybrid: When ``features is None``, also include hybrid IV−rvol names.
        include_asset: When ``features is None``, also include asset-local names.

    Returns:
        Ordered unique feature names.

    Raises:
        ValueError: Empty explicit list.
        KeyError: Unknown name.
    """
    if features is None:
        names = list(MARKET_SETUP_FEATURES)
        if include_hybrid:
            names.extend(ASSET_IV_RVOL_FEATURES)
        if include_asset:
            names.extend(ASSET_SETUP_FEATURES)
        ## Defense in depth: setup tuples already skip deprecated, but filter again.
        return [name for name in names if not get_spec(name).deprecated]

    names = list(features)
    if not names:
        raise ValueError("features must be non-empty, or omit it for the default set")
    out: List[str] = []
    seen = set()
    for name in names:
        if name in seen:
            continue
        spec = get_spec(name)
        _warn_if_deprecated(name, spec)
        out.append(name)
        seen.add(name)
    return out
