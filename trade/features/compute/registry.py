"""Feature registry: ``@feature`` binds name + metadata + compute callable.

Import family modules (via ``trade.features.compute.families``) so every
``@feature`` function registers into ``FEATURE_SPECS`` before callers look up
names.

Comment density: orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Tuple, Union

import pandas as pd

from trade.features.compute.context import FeatureContext
from trade.features.themes import FeatureTheme, coerce_theme, map_features_to_themes

ComputeFn = Callable[[FeatureContext], pd.Series]


@dataclass(frozen=True, slots=True)
class Feature:
    """One named feature: metadata plus the compute callable.

    Attributes:
        name: Public feature name (function name / cache key / caller API).
        compute: ``(FeatureContext) -> Series`` implementing the formula.
        inputs: Declared inputs (``SPY``, ``close``, ``FXI_IV``, ``ohlcv``, …).
        formula_id: Cache-bust label; bump when the math changes.
        scope: ``market`` | ``hybrid`` | ``asset``.
        theme: Research theme from ``FEATURE_THEMES`` (explicit ``unmapped`` only).
        params: Optional knobs recorded for introspection (horizon, lookback, …).
    """

    name: str
    compute: ComputeFn
    inputs: Tuple[str, ...]
    formula_id: str
    scope: str
    theme: FeatureTheme
    params: Mapping[str, int] = field(default_factory=dict)


## Populated by ``@feature`` when family modules are imported.
FEATURE_SPECS: Dict[str, Feature] = {}

## Alias so older imports keep working during the cutover.
FeatureSpec = Feature


def feature(
    *,
    scope: str,
    formula_id: str,
    inputs: Tuple[str, ...],
    theme: Optional[Union[FeatureTheme, str]] = None,
    **params: int,
) -> Callable[[ComputeFn], ComputeFn]:
    """Register ``fn`` under ``fn.__name__`` in ``FEATURE_SPECS``.

    Args:
        scope: ``market`` | ``hybrid`` | ``asset``.
        formula_id: Cache-bust identity for this formula.
        inputs: Declared input keys for dependency loading.
        theme: Optional ``FeatureTheme`` or string value; default from
            ``FEATURE_THEMES`` via ``map_features_to_themes`` (raises if missing).
        **params: Optional int knobs stored on the ``Feature`` record.

    Returns:
        Decorator that registers the function and returns it unchanged.

    Raises:
        KeyError: Feature name is not listed in ``FEATURE_THEMES``.
        ValueError: Duplicate registration for the same name.
    """

    def deco(fn: ComputeFn) -> ComputeFn:
        name = fn.__name__
        ## Theme must come from FEATURE_THEMES (including explicit unmapped).
        table_theme = map_features_to_themes(name)
        resolved = coerce_theme(theme) if theme is not None else table_theme
        if resolved is not table_theme:
            raise ValueError(
                f"Feature {name!r}: theme={resolved.value!r} disagrees with "
                f"FEATURE_THEMES theme={table_theme.value!r}"
            )
        spec = Feature(
            name=name,
            compute=fn,
            inputs=tuple(inputs),
            formula_id=formula_id,
            scope=scope,
            theme=resolved,
            params=dict(params),
        )
        if name in FEATURE_SPECS:
            raise ValueError(f"Duplicate feature registration for {name!r}")
        FEATURE_SPECS[name] = spec
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


def list_features(scope: str = "all") -> List[str]:
    """List registered feature names, optionally filtered by scope."""
    if scope == "all":
        return sorted(FEATURE_SPECS)
    return sorted(name for name, spec in FEATURE_SPECS.items() if spec.scope == scope)


def list_features_by_theme(theme: Union[FeatureTheme, str]) -> List[str]:
    """List registered feature names for one research theme."""
    wanted = coerce_theme(theme)
    return sorted(name for name, spec in FEATURE_SPECS.items() if spec.theme == wanted)


def features_for_scope(scope: str) -> Tuple[str, ...]:
    """Stable-order names for one scope (registration order)."""
    return tuple(name for name, spec in FEATURE_SPECS.items() if spec.scope == scope)


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
