"""Load-time checks that every registered feature is fully wired.

Runs once after family modules import. Fails fast on missing theme membership,
unresolvable inputs, or scope/input mismatches.

Comment density: orchestration.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, List, Optional, Sequence, Set

from trade.features.compute.registry import FEATURE_SPECS, Feature
from trade.features.compute.symbols import FRED_IV_SERIES, MARKET_SYMBOLS
from trade.features.themes import FEATURE_THEMES, FeatureTheme, map_features_to_themes
from trade.helpers.Logging import setup_logger

logger = setup_logger(__name__)

## Inputs that mean asset bars (caller-supplied ohlcv), not market symbols.
_ASSET_INPUTS = frozenset({"close", "ohlcv"})
_VALID_SCOPES = frozenset({"market", "asset", "hybrid"})


def _known_input_keys() -> Set[str]:
    """Return the closed set of legal ``@feature(inputs=...)`` tokens."""
    return set(MARKET_SYMBOLS) | set(FRED_IV_SERIES) | set(_ASSET_INPUTS)


def validate_feature(spec: Feature) -> List[str]:
    """Return human-readable problems for one registered feature (empty if ok).

    Args:
        spec: Registry row to check.

    Returns:
        Problem strings; empty list means the feature is properly set up.
    """
    problems: List[str] = []
    name = spec.name

    if not spec.formula_id or not str(spec.formula_id).strip():
        problems.append(f"{name}: formula_id must be non-empty")
    if spec.scope not in _VALID_SCOPES:
        problems.append(
            f"{name}: scope={spec.scope!r} must be one of {sorted(_VALID_SCOPES)}"
        )
    if not callable(spec.compute):
        problems.append(f"{name}: compute must be callable")
    if not spec.inputs:
        problems.append(f"{name}: inputs must be non-empty")

    known = _known_input_keys()
    for key in spec.inputs:
        if key not in known:
            problems.append(
                f"{name}: unknown input {key!r} "
                "(expected MARKET_SYMBOLS, FRED_IV_SERIES, or close/ohlcv)"
            )

    asset_inputs = _ASSET_INPUTS.intersection(spec.inputs)
    if spec.scope == "market" and asset_inputs:
        problems.append(
            f"{name}: market scope cannot declare asset inputs {sorted(asset_inputs)}"
        )
    if spec.scope in {"asset", "hybrid"} and not asset_inputs:
        problems.append(
            f"{name}: {spec.scope} scope requires close and/or ohlcv in inputs"
        )

    try:
        table_theme = map_features_to_themes(name)
    except KeyError as exc:
        problems.append(str(exc))
        return problems

    if spec.theme is not table_theme:
        problems.append(
            f"{name}: registry theme={spec.theme.value!r} disagrees with "
            f"FEATURE_THEMES theme={table_theme.value!r}"
        )
    return problems


def validate_theme_table() -> List[str]:
    """Return problems in ``FEATURE_THEMES`` structure (keys, duplicates)."""
    problems: List[str] = []
    valid_keys = {theme.value for theme in FeatureTheme}
    for key in FEATURE_THEMES:
        if key not in valid_keys:
            problems.append(
                f"FEATURE_THEMES key {key!r} is not a FeatureTheme value"
            )
    if FeatureTheme.UNMAPPED.value not in FEATURE_THEMES:
        problems.append(
            "FEATURE_THEMES must include an explicit "
            f"{FeatureTheme.UNMAPPED.value!r} bucket"
        )

    counts = Counter(name for names in FEATURE_THEMES.values() for name in names)
    for name, count in sorted(counts.items()):
        if count > 1:
            problems.append(
                f"feature {name!r} appears in {count} FEATURE_THEMES buckets"
            )
    return problems


def validate_registry(specs: Optional[Iterable[Feature]] = None) -> None:
    """Validate the full feature registry and theme table.

    Args:
        specs: Features to check; default all of ``FEATURE_SPECS``.

    Raises:
        ValueError: One or more setup problems; message lists them all.
    """
    problems = validate_theme_table()
    rows: Sequence[Feature] = (
        list(specs) if specs is not None else list(FEATURE_SPECS.values())
    )
    registered = {spec.name for spec in rows}
    themed = {name for names in FEATURE_THEMES.values() for name in names}

    for name in sorted(themed - registered):
        problems.append(
            f"{name}: listed in FEATURE_THEMES but not registered in FEATURE_SPECS"
        )

    for spec in rows:
        problems.extend(validate_feature(spec))

    if problems:
        preview = "; ".join(problems[:20])
        more = f" (+{len(problems) - 20} more)" if len(problems) > 20 else ""
        raise ValueError(
            f"Feature registry validation failed ({len(problems)} issue(s)): "
            f"{preview}{more}"
        )
    logger.info(
        "feature registry ok n=%s themes=%s",
        len(registered),
        len(FEATURE_THEMES),
    )
