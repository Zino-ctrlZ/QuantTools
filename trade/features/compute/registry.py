"""Feature registry: ``@feature`` binds name + metadata + compute callable.

Import family modules (via ``trade.features.compute.families``) so every
``@feature`` function registers into ``FEATURE_SPECS`` before callers look up
names. The durable dict and ``Feature`` type live in ``specs`` so autoreload
of this module cannot wipe registrations.

Comment density: orchestration.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple, Union

from trade.features.compute.specs import (
    BASE_ASSET_FEATURE_SET,
    BASE_MARKET_FEATURE_SET,
    FEATURE_SPECS,
    ComputeFn,
    Feature,
)
from trade.features.themes import FeatureTheme, coerce_theme, map_features_to_themes


def feature(
    *,
    scope: str,
    formula_id: str,
    inputs: Tuple[str, ...],
    theme: Optional[Union[FeatureTheme, str]] = None,
    deprecated: Optional[bool] = None,
    **params: int,
) -> Callable[[ComputeFn], ComputeFn]:
    """Register ``fn`` under ``fn.__name__`` in ``FEATURE_SPECS``.

    Re-registration of the same name replaces the prior row so Jupyter
    autoreload of a family module can refresh the callable without wiping
    the durable ``FEATURE_SPECS`` dict.

    Args:
        scope: ``market`` | ``hybrid`` | ``asset``.
        formula_id: Cache-bust identity for this formula.
        inputs: Declared input keys for dependency loading.
        theme: Optional ``FeatureTheme`` or string value; default from
            ``FEATURE_THEMES`` via ``map_features_to_themes`` (raises if missing).
        deprecated: Optional soft-retire flag. When omitted, derived from
            theme ``deprecated``. When set, must match that theme membership.
        **params: Optional int knobs stored on the ``Feature`` record.

    Returns:
        Decorator that registers the function and returns it unchanged.

    Raises:
        KeyError: Feature name is not listed in ``FEATURE_THEMES``.
        ValueError: ``deprecated`` disagrees with theme.
    """

    def deco(fn: ComputeFn) -> ComputeFn:
        name = fn.__name__
        ## Theme must come from FEATURE_THEMES (including unmapped / deprecated).
        table_theme = map_features_to_themes(name)
        resolved = coerce_theme(theme) if theme is not None else table_theme
        if resolved is not table_theme:
            raise ValueError(
                f"Feature {name!r}: theme={resolved.value!r} disagrees with "
                f"FEATURE_THEMES theme={table_theme.value!r}"
            )
        theme_says_deprecated = resolved is FeatureTheme.DEPRECATED
        if deprecated is None:
            deprecated_flag = theme_says_deprecated
        else:
            deprecated_flag = bool(deprecated)
            if deprecated_flag != theme_says_deprecated:
                raise ValueError(
                    f"Feature {name!r}: deprecated={deprecated_flag} disagrees with "
                    f"FEATURE_THEMES theme={resolved.value!r} "
                    "(move the name to FEATURE_THEMES['deprecated'] to retire it)"
                )
        ## Mutate the durable dict in specs — never rebind FEATURE_SPECS here.
        FEATURE_SPECS[name] = Feature(
            name=name,
            compute=fn,
            inputs=tuple(inputs),
            formula_id=formula_id,
            scope=scope,
            theme=resolved,
            deprecated=deprecated_flag,
            params=dict(params),
        )
        return fn

    return deco


def get_feature(name: str) -> Feature:
    """Return the registered ``Feature`` for ``name``.

    Raises:
        KeyError: If ``name`` is not registered.
    """
    try:
        return FEATURE_SPECS[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown feature {name!r}. Known count={len(FEATURE_SPECS)}."
        ) from exc


def list_features(scope: str = "base_market ") -> List[str]:
    """List registered feature names, optionally filtered by scope or base pack.

    Includes deprecated names for ``all`` / scope filters so callers can discover
    what still exists. Base packs are curated shortlists.
    """
    if scope == "base_asset":
        return sorted(BASE_ASSET_FEATURE_SET)
    if scope == "base_market":
        return sorted(BASE_MARKET_FEATURE_SET)
    if scope == "base_all":
        return sorted(BASE_ASSET_FEATURE_SET + BASE_MARKET_FEATURE_SET)
    if scope == "all":
        return sorted(FEATURE_SPECS)
    return sorted(name for name, spec in FEATURE_SPECS.items() if spec.scope == scope)


def list_features_by_theme(theme: Union[FeatureTheme, str]) -> List[str]:
    """List registered feature names for one research theme."""
    wanted = coerce_theme(theme)
    return sorted(name for name, spec in FEATURE_SPECS.items() if spec.theme == wanted)


def features_for_scope(scope: str) -> Tuple[str, ...]:
    """Stable-order active names for one scope (registration order).

    Soft-retired (deprecated) names are omitted so default catalog tuples stay
    clean. Explicit requests still resolve via ``FEATURE_SPECS``.
    """
    return tuple(
        name
        for name, spec in FEATURE_SPECS.items()
        if spec.scope == scope and not spec.deprecated
    )


def symbols_for_features(features: List[str]) -> List[str]:
    """Market close symbols required by ``features`` (excludes close/ohlcv/IV)."""
    from trade.features.compute.symbols import FRED_IV_SERIES, MARKET_SYMBOLS

    needed = []
    seen = set()
    for name in features:
        spec = FEATURE_SPECS.get(name)
        if spec is None:
            continue
        for key in spec.inputs:
            if key in {"close", "ohlcv"} or key in FRED_IV_SERIES:
                continue
            if key in MARKET_SYMBOLS and key not in seen:
                needed.append(key)
                seen.add(key)
    return needed


def iv_keys_for_features(features: List[str]) -> List[str]:
    """FRED IV keys required by ``features``."""
    from trade.features.compute.symbols import FRED_IV_SERIES

    needed = []
    seen = set()
    for name in features:
        spec = FEATURE_SPECS.get(name)
        if spec is None:
            continue
        for key in spec.inputs:
            if key in FRED_IV_SERIES and key not in seen:
                needed.append(key)
                seen.add(key)
    return needed
