"""Sleeves, param resolution, rule selectors, and child building for a book.

Validation runs over the whole config first and returns every problem as a
``ConfigIssue``. Only after that passes are children built, each inside a
wrapper that names the sleeve, ticker, and resolved params if construction fails.

Comment density: domain policy

Core Dataclasses:
    Sleeve: One strategy class run over a ticker universe under one name.

Core Functions:
    resolve_params: Layer class defaults, sleeve params, and asset params.
    merge_overrides: Fold ``{(sleeve, ticker): params}`` into new sleeves.
    selector_matches: Whether a rule selector targets a child.
    config_issues: Every problem in a book config, without building anything.
    build_children: Build fresh children and apply selector rules.

Risk/Assumptions:
    Param layers are per sleeve. Two sleeves on the same ticker never share a
    value unless both set it, and identical twins (same class, ticker, and
    resolved params) are rejected because they would double the same trades.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Tuple, Type

from trade.backtester_._helper import REQUIRED
from trade.backtester_._strategy import StrategyBase
from trade.backtester_.data import PTDataset
from trade.backtester_.mixins.market_data import Rule
from trade.backtester_.systematic.errors import ChildBuildError, ConfigIssue, suggest
from trade.helpers.Logging import setup_logger

if TYPE_CHECKING:
    from trade.backtester_._multi_asset_strategy import MultiAssetStrategy

logger = setup_logger(__name__)

## (sleeve name, ticker) -> param name -> value. Input shape for SystematicMultiAsset.with_overrides,
## which folds it into Sleeve.asset_params; it is not a separate param layer.
KeyOverrides = Mapping[Tuple[str, str], Mapping[str, Any]]
## (selector, rule) pairs. Selector grammar is documented on selector_matches.
RuleSpec = Tuple[Tuple[str, Rule], ...]


@dataclass(frozen=True)
class Sleeve:
    """One strategy class run over a ticker universe under one name.

    Attributes:
        name: Unique sleeve name; key prefix in ``"{name}:{ticker}"``. No ``:`` allowed.
        strategy_class: Child ``StrategyBase`` subclass.
        params: Sleeve-wide params (the "global strategy params" layer).
        asset_params: Per-ticker params that beat ``params``, for this sleeve only.
        tickers: Universe for this sleeve; None means every ticker in the book's data.

    Examples:
        >>> Sleeve("mr", MeanRev, params={"window": 20}, asset_params={"TSLA": {"window": 10}})
    """

    name: str
    strategy_class: Type[StrategyBase]
    params: Mapping[str, Any] = field(default_factory=dict)
    asset_params: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    tickers: Optional[Sequence[str]] = None

    def key(self, ticker: str) -> str:
        """Return the simulator strategy key for this sleeve on ``ticker``."""
        return f"{self.name}:{ticker}"

    def universe(self, data: Mapping[str, Any]) -> List[str]:
        """Return this sleeve's tickers, defaulting to every ticker in ``data``."""
        return list(self.tickers) if self.tickers is not None else list(data)

    @classmethod
    def from_multiasset(
        cls,
        multi: "MultiAssetStrategy",
        *,
        name: Optional[str] = None,
    ) -> "Sleeve":
        """Build a sleeve from a ``MultiAssetStrategy`` container.

        Per-ticker ``multi.params`` become this sleeve's ``asset_params``.
        Sleeve-wide ``params`` stay empty so each ticker keeps the values it
        already had on the multi-asset. The universe is the multi-asset's
        param keys (or its data keys if params is empty).

        Args:
            multi: Existing multi-asset container (same strategy class, many tickers).
            name: Sleeve name. Defaults to ``multi.name``.

        Returns:
            A frozen ``Sleeve`` ready to pass into ``SystematicMultiAsset``.

        Raises:
            ValueError: If ``multi`` has no tickers, or ``name`` / ``multi.name`` is empty.

        Examples:
            >>> sleeve = Sleeve.from_multiasset(multi, name="mr")
            >>> book = SystematicMultiAsset([sleeve], multi.data, start_trading_date=multi.start_date)
        """
        sleeve_name = name if name is not None else multi.name
        if not sleeve_name:
            raise ValueError("sleeve name is empty; pass name= or set multi.name")
        tickers = list(multi.params) if multi.params else list(multi.data)
        if not tickers:
            raise ValueError(f"MultiAssetStrategy {multi.name!r} has no tickers to sleeve")
        asset_params = {ticker: dict(params) for ticker, params in multi.params.items()}
        logger.info(
            "Sleeve.from_multiasset name=%s class=%s tickers=%s",
            sleeve_name,
            multi.strategy_class.__name__,
            len(tickers),
        )
        return cls(
            name=sleeve_name,
            strategy_class=multi.strategy_class,
            params={},
            asset_params=asset_params,
            tickers=tickers,
        )


def resolve_params(sleeve: Sleeve, ticker: str) -> Dict[str, Any]:
    """Return one child's params, most specific layer winning.

    Order, weakest first: class ``bt_params`` < ``sleeve.params`` <
    ``sleeve.asset_params[ticker]``. Names are validated by ``config_issues``, not here.

    Args:
        sleeve: Sleeve the child belongs to.
        ticker: Child ticker.

    Returns:
        A fresh dict; mutating it does not touch any layer.
    """
    resolved = deepcopy(dict(sleeve.strategy_class.bt_params))
    for layer in (sleeve.params, sleeve.asset_params.get(ticker, {})):
        resolved.update(deepcopy(dict(layer)))
    return resolved


def merge_overrides(sleeves: Sequence[Sleeve], overrides: KeyOverrides) -> Tuple[Tuple[Sleeve, ...], List[ConfigIssue]]:
    """Return new sleeves with ``overrides`` merged into each sleeve's ``asset_params``.

    Overrides are not a separate layer: ``{("mr", "AAPL"): {"hold": 7}}`` lands
    in ``mr``'s ``asset_params["AAPL"]`` and beats any value already there.
    Ticker and param names are checked later by ``config_issues`` like any
    hand-written ``asset_params``; only unknown sleeve names are reported here.

    Args:
        sleeves: Current sleeves; left unchanged.
        overrides: ``{(sleeve name, ticker): {param: value}}``.

    Returns:
        The merged sleeves, and one issue per override naming no sleeve.
    """
    by_name = {s.name: s for s in sleeves}
    merged: Dict[str, Dict[str, Dict[str, Any]]] = {s.name: {t: dict(p) for t, p in s.asset_params.items()} for s in sleeves}
    issues: List[ConfigIssue] = []
    for (sleeve_name, ticker), layer in overrides.items():
        if sleeve_name not in by_name:
            issues.append(
                ConfigIssue(f"overrides[{sleeve_name!r}, {ticker!r}]", f"no sleeve named {sleeve_name!r}", suggest(sleeve_name, by_name))
            )
            continue
        merged[sleeve_name].setdefault(ticker, {}).update(layer)
    return tuple(replace(s, asset_params=merged[s.name]) for s in sleeves), issues


def selector_matches(selector: str, sleeve: str, ticker: str) -> bool:
    """Return whether ``selector`` targets the child ``sleeve:ticker``.

    Grammar: ``"*"`` every child; ``"AAPL"`` or ``"*:AAPL"`` every sleeve on
    AAPL (plain tickers match ``MultiAssetStrategy.set_rules``); ``"mr:*"``
    every ticker in sleeve mr; ``"mr:AAPL"`` one child.

    Examples:
        >>> selector_matches("mr:*", "mr", "AAPL")
        True
        >>> selector_matches("AAPL", "mom", "MSFT")
        False
    """
    if selector == "*":
        return True
    s_sleeve, _, s_ticker = selector.rpartition(":") if ":" in selector else ("*", "", selector)
    return s_sleeve in ("*", sleeve) and s_ticker in ("*", ticker)


def cls_param_issues(cls: type, given: Mapping[str, Any], *, where: str) -> List[ConfigIssue]:
    """Return one issue per param name in ``given`` that ``cls.bt_params`` does not declare."""
    valid = list(cls.bt_params)
    return [
        ConfigIssue(where, f"unknown param {name!r} for {cls.__name__}", suggest(name, valid) or f"valid: {valid}")
        for name in given
        if name not in cls.bt_params
    ]


def _required_issues(cls: type, resolved: Dict[str, Any], key: str) -> List[ConfigIssue]:
    """Return one issue per child param still set to ``REQUIRED`` after resolution."""
    return [
        ConfigIssue(
            f"child {key!r}",
            f"{cls.__name__} param {k!r} is REQUIRED but no layer set it",
            "set it in the sleeve's params or asset_params",
        )
        for k, v in resolved.items()
        if v is REQUIRED
    ]


def config_issues(
    sleeves: Sequence[Sleeve],
    data: Mapping[str, PTDataset],
    rules: RuleSpec,
) -> List[ConfigIssue]:
    """Return every problem in a book config, without building anything.

    Checks sleeve names, tickers, every param layer, REQUIRED params, rule
    selectors, and twin children.

    Args:
        sleeves: Sleeves in priority order.
        data: Ticker to dataset.
        rules: ``(selector, rule)`` pairs.

    Returns:
        Issues in discovery order; empty when the config is valid.
    """
    issues: List[ConfigIssue] = []

    # --- sleeve names ---
    names = [s.name for s in sleeves]
    if not sleeves:
        issues.append(ConfigIssue("sleeves", "no sleeves given"))
    for name in {n for n in names if names.count(n) > 1}:
        issues.append(ConfigIssue(f"sleeve {name!r}", "name used more than once", "sleeve names must be unique"))
    for name in (n for n in names if ":" in n):
        issues.append(ConfigIssue(f"sleeve {name!r}", "name contains ':'", "':' separates sleeve and ticker in keys"))

    # --- per sleeve: param layers, universe, REQUIRED, twins ---
    children: Dict[Tuple[str, str], Sleeve] = {}
    twins: Dict[Tuple[type, str, str], str] = {}
    for sleeve in sleeves:
        cls = sleeve.strategy_class
        here = f"sleeve {sleeve.name!r}"
        issues += cls_param_issues(cls, sleeve.params, where=f"{here} params")
        universe = sleeve.universe(data)
        for ticker, layer in sleeve.asset_params.items():
            where = f"{here} asset_params[{ticker!r}]"
            if ticker not in universe:
                issues.append(ConfigIssue(where, f"{ticker!r} is not in this sleeve's tickers", suggest(ticker, universe)))
            issues += cls_param_issues(cls, layer, where=where)
        for ticker in universe:
            if ticker not in data:
                issues.append(
                    ConfigIssue(f"{here} tickers", f"{ticker!r} has no data", suggest(ticker, data) or f"data has {sorted(data)}")
                )
                continue
            children[(sleeve.name, ticker)] = sleeve
            resolved = resolve_params(sleeve, ticker)
            issues += _required_issues(cls, resolved, sleeve.key(ticker))
            ## Two sleeves with the same class, ticker, and params would take identical trades
            ## and silently double exposure. repr makes dicts/lists hashable and compares by value.
            fingerprint = (cls, ticker, repr(sorted(resolved.items())))
            if fingerprint in twins:
                issues.append(
                    ConfigIssue(
                        f"child {sleeve.key(ticker)!r}",
                        f"identical class and params to {twins[fingerprint]!r}, so it would take the exact same trades",
                        "change a param in one sleeve, or drop it and raise the other's multiplier",
                    )
                )
            twins.setdefault(fingerprint, sleeve.key(ticker))

    # --- every rule selector must hit at least one child ---
    for selector, _ in rules:
        if not any(selector_matches(selector, s, t) for s, t in children):
            targets = ["*"] + sorted({f"{s}:*" for s, _ in children}) + sorted({t for _, t in children})
            issues.append(
                ConfigIssue(f"rule selector {selector!r}", "matches no child", suggest(selector, targets) or f"try one of {targets}")
            )
    return issues


def build_children(
    sleeves: Sequence[Sleeve],
    data: Mapping[str, PTDataset],
    *,
    start_trading_date: Optional[str],
    tplusn: int,
    rules: RuleSpec = (),
) -> Dict[str, StrategyBase]:
    """Build fresh children and apply selector rules.

    Assumes ``config_issues`` already passed. Every call returns new objects,
    so rules are never stacked twice on the same child.

    Returns:
        Key to child. Insertion order is the same-day priority before ranking.

    Raises:
        ChildBuildError: If a child's constructor or ``setup`` raises.
    """
    children: Dict[str, StrategyBase] = {}
    for sleeve in sleeves:
        for ticker in sleeve.universe(data):
            key = sleeve.key(ticker)
            params = resolve_params(sleeve, ticker)
            try:
                child = sleeve.strategy_class(
                    data=data[ticker], start_trading_date=start_trading_date, ticker=ticker, tplusn=tplusn, **params
                )
            except Exception as exc:
                raise ChildBuildError(key, sleeve.strategy_class, params, exc) from exc
            ## Rules go on after setup, onto each child's own RuleFilter, exactly like
            ## MultiAssetStrategy, because a Bar is bound to one child's data.
            for selector, rule in rules:
                if selector_matches(selector, sleeve.name, ticker):
                    child.rules.add(rule)
            children[key] = child
    return children
