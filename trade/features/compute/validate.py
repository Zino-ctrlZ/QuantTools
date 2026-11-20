"""Sanity-check that every named feature is set up correctly before anyone uses it.

When the feature package loads, this runs once. It catches mistakes like a
feature missing from a research theme, asking for a price series we do not
know how to load, or labeling a market feature as if it needed a single stock.

Comment density: orchestration.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, List, Optional, Sequence, Set

from trade.features.compute.specs import FEATURE_SPECS, Feature
from trade.features.compute.symbols import FRED_IV_SERIES, MARKET_SYMBOLS
from trade.features.themes import FEATURE_THEMES, FeatureTheme, map_features_to_themes
from trade.helpers.Logging import setup_logger

logger = setup_logger(__name__)

## These two names mean "use the stock's own price bars," not a market index.
_ASSET_INPUTS = frozenset({"close", "ohlcv"})
## Only three allowed feature kinds: whole market, one stock, or stock+market.
_VALID_SCOPES = frozenset({"market", "asset", "hybrid"})


def _known_input_keys() -> Set[str]:
    """List every price or volatility series a feature is allowed to ask for."""
    ## Combine market tickers, implied-vol series, and the two "this stock" keys.
    return set(MARKET_SYMBOLS) | set(FRED_IV_SERIES) | set(_ASSET_INPUTS)


def validate_feature(spec: Feature) -> List[str]:
    """Check one feature's setup and return a list of plain-English problems.

    An empty list means this feature looks ready to use.

    Args:
        spec: The feature's stored definition (name, formula, inputs, theme).

    Returns:
        Problem strings; empty list means the feature is properly set up.
    """
    problems: List[str] = []
    name = spec.name

    ## --- basic required fields ---
    ## Every feature needs a formula label (used when we change the math later),
    ## a valid kind (market / stock / stock+market), a real calculation function,
    ## and at least one input series to compute from.
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

    ## --- can we actually load what this feature asks for? ---
    ## Reject typos like asking for a ticker that is not on our market list.
    known = _known_input_keys()
    for key in spec.inputs:
        if key not in known:
            problems.append(
                f"{name}: unknown input {key!r} "
                "(expected MARKET_SYMBOLS, FRED_IV_SERIES, or close/ohlcv)"
            )

    ## --- kind of feature vs what it asks for must match ---
    ## A market-wide feature should not demand one stock's bars.
    ## A stock or stock+market feature must demand those bars.
    asset_inputs = _ASSET_INPUTS.intersection(spec.inputs)
    if spec.scope == "market" and asset_inputs:
        problems.append(
            f"{name}: market scope cannot declare asset inputs {sorted(asset_inputs)}"
        )
    if spec.scope in {"asset", "hybrid"} and not asset_inputs:
        problems.append(
            f"{name}: {spec.scope} scope requires close and/or ohlcv in inputs"
        )

    ## --- research theme membership ---
    ## Every feature must sit in exactly one research theme table entry
    ## (momentum, volatility, deprecated, and so on). If it is missing, stop
    ## here — later theme checks would not make sense.
    try:
        table_theme = map_features_to_themes(name)
    except KeyError as exc:
        problems.append(str(exc))
        return problems

    ## The theme stamped on the feature must match the theme table.
    ## Prevents "labeled momentum in code but listed under volatility in the table."
    if spec.theme is not table_theme:
        problems.append(
            f"{name}: registry theme={spec.theme.value!r} disagrees with "
            f"FEATURE_THEMES theme={table_theme.value!r}"
        )

    ## --- soft-retired features ---
    ## If the theme table says "deprecated," the feature must be marked retired.
    ## If it is marked retired, it must live in that deprecated theme bucket.
    ## Keeps "still computing but out of default packs" consistent in both places.
    theme_is_deprecated = table_theme is FeatureTheme.DEPRECATED
    if bool(spec.deprecated) != theme_is_deprecated:
        problems.append(
            f"{name}: deprecated={spec.deprecated} disagrees with "
            f"FEATURE_THEMES theme={table_theme.value!r}"
        )
    return problems


def validate_theme_table() -> List[str]:
    """Check the research-theme table itself for structural mistakes.

    Returns:
        Problem strings about the theme groupings; empty if the table looks fine.
    """
    problems: List[str] = []

    ## --- theme names must be from the official list ---
    ## Stops someone inventing a theme string that the rest of the code does not know.
    valid_keys = {theme.value for theme in FeatureTheme}
    for key in FEATURE_THEMES:
        if key not in valid_keys:
            problems.append(
                f"FEATURE_THEMES key {key!r} is not a FeatureTheme value"
            )

    ## --- required special buckets ---
    ## "unmapped" = features with no research theme on purpose.
    ## "deprecated" = features we are phasing out but have not deleted yet.
    ## Both buckets must exist even when empty, so moves into them always work.
    if FeatureTheme.UNMAPPED.value not in FEATURE_THEMES:
        problems.append(
            "FEATURE_THEMES must include an explicit "
            f"{FeatureTheme.UNMAPPED.value!r} bucket"
        )
    if FeatureTheme.DEPRECATED.value not in FEATURE_THEMES:
        problems.append(
            "FEATURE_THEMES must include an explicit "
            f"{FeatureTheme.DEPRECATED.value!r} bucket"
        )

    ## --- each feature name in only one theme ---
    ## A name cannot be both "momentum" and "volatility" at once.
    counts = Counter(name for names in FEATURE_THEMES.values() for name in names)
    for name, count in sorted(counts.items()):
        if count > 1:
            problems.append(
                f"feature {name!r} appears in {count} FEATURE_THEMES buckets"
            )
    return problems


def validate_registry(specs: Optional[Iterable[Feature]] = None) -> None:
    """Run the full setup check across the theme table and every feature.

    Raises an error listing problems if anything is wrong. Logs success when
    everything looks ready.

    Args:
        specs: Features to check; default is every registered feature.

    Raises:
        ValueError: One or more setup problems; message lists them all.
    """
    ## --- start with the theme table ---
    ## Fix grouping mistakes before looking at individual formulas.
    problems = validate_theme_table()

    ## --- which features are we checking? ---
    ## Usually all of them; tests can pass a smaller subset.
    rows: Sequence[Feature] = (
        list(specs) if specs is not None else list(FEATURE_SPECS.values())
    )
    registered = {spec.name for spec in rows}
    themed = {name for names in FEATURE_THEMES.values() for name in names}

    ## --- names in the theme table must have real formulas ---
    ## Stops "listed under momentum" with no calculation wired up.
    for name in sorted(themed - registered):
        problems.append(
            f"{name}: listed in FEATURE_THEMES but not registered in FEATURE_SPECS"
        )

    ## --- check every feature one by one ---
    for spec in rows:
        problems.extend(validate_feature(spec))

    ## --- report ---
    ## Fail loudly with a short sample of issues (full count in the message).
    ## If nothing is wrong, leave a simple success note in the log.
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
