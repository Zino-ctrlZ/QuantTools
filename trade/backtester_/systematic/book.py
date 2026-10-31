"""SystematicMultiAsset: an inheritable book of strategy sleeves sharing one capital pool.

The book keeps a frozen ``BookSpec`` (the recipe) apart from the child
strategies (what gets cooked). Every ``build``/``simulate`` makes fresh children
from the spec, so runs are repeatable, rules never stack, and optimizer workers
can rebuild the exact same book with different per-child params.

Comment density: orchestration

Core Classes:
    PortfolioBook: Abstract base ``Simulator`` checks to route a book.
    SystematicMultiAsset: Default ``PortfolioBook`` users subclass; default hooks change nothing.

Core Dataclasses:
    BookSpec: Frozen description of sleeves, data, and rules.

Processing Flow:
    1. ``__init__`` freezes a ``BookSpec`` and validates all of it at once.
    2. ``set_rules`` swaps in a new, re-validated spec.
    3. ``simulate`` (or ``Simulator(book, ...)``) builds fresh children and runs
       them, asking the book before each fill: rank same-day opens, then ``should_open``
       per candidate against a fresh portfolio snapshot.
    4. ``optimize`` wraps ``ParamOptimizer``: shared book via ``with_trial_params``,
       or ``per_asset=True`` for the solo ticker loop.

Usage:
    >>> book = SystematicMultiAsset([Sleeve("mr", MeanRev)], data, max_open_trades=3)
    >>> book.set_rules({"mr:*": lambda bar: bar.ind("dist") < -0.03})
    >>> result = book.simulate(capital=100_000, percent_of_capital=0.25)
    >>> shared = book.optimize({"max_open_trades": [2, 5]}, simulator_kwargs={"capital": 100_000, "percent_of_capital": 0.25})
    >>> per_asset = book.optimize({"ma_length": [10, 20]}, simulator_kwargs=..., per_asset=True)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields, replace
from datetime import datetime
from typing import Any, ClassVar, Dict, List, Literal, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from trade.backtester_._strategy import StrategyBase, TradeDecision
from trade.backtester_.data import PTDataset
from trade.backtester_.mixins.contract import ParamContractMixin
from trade.backtester_.mixins.market_data import Rule
from trade.backtester_.optimizer.core import OptimizationResult
from trade.backtester_.simulator import BuyingPowerPolicy, Simulator, SimulationResult
from trade.backtester_.simulator.core import OpenCandidate, PortfolioContext, Veto
from trade.backtester_.systematic.core import (
    KeyOverrides,
    RuleSpec,
    Sleeve,
    build_children,
    config_issues,
    merge_overrides,
    resolve_params,
)
from trade.backtester_.systematic.errors import BookConfigError, ConfigIssue
from trade.helpers.Logging import setup_logger

## Same input shapes MultiAssetStrategy.set_rules accepts, plus sleeve selectors as dict keys.
RulesInput = Union[Rule, Sequence[Any], Mapping[str, Any]]

logger = setup_logger(__name__)


class PortfolioBook(ABC):
    """Base for anything ``Simulator`` routes as a book.

    ``Simulator`` routes on ``isinstance(x, PortfolioBook)`` (imported lazily
    inside the engine) and calls these hooks directly. The hook defaults
    approve every child decision in the order given, so a book that only
    implements ``build`` trades exactly like ``Simulator`` on its children.
    Override a hook and call ``super()`` to keep the default as the fallback.
    """

    @abstractmethod
    def build(self) -> Dict[str, StrategyBase]:
        """Return fresh children keyed by strategy key.

        Abstract because only the subclass knows where its children come from.
        Must return new objects on every call, so rules never stack across runs.
        """

    def rank_opens(self, candidates: List[OpenCandidate], ctx: PortfolioContext) -> List[OpenCandidate]:
        """Return same-day candidates in fill priority; omit any to drop them. Default: as given."""
        return candidates

    def should_open(self, candidate: OpenCandidate, ctx: PortfolioContext) -> Union[TradeDecision, Veto]:
        """Return the open to fill, or a ``Veto``. Default: the child's own decision."""
        return candidate.decision

    def should_close(self, key: str, decision: TradeDecision, ctx: PortfolioContext) -> Union[TradeDecision, Veto]:
        """Return the close to fill, or a ``Veto``. Default: the child's own decision."""
        return decision


@dataclass(frozen=True)
class BookSpec:
    """Frozen description of a book: everything needed to rebuild it, nothing built.

    Field names match ``SystematicMultiAsset.__init__`` args, so a spec can be
    splatted back into the constructor.

    Attributes:
        sleeves: Strategy sleeves, in priority order.
        data: Ticker to dataset for children.
        start_trading_date: First date children may trade.
        tplusn: Execution delay for every child.
        rules: ``(selector, rule)`` pairs added to matching children at build.
    """

    sleeves: Tuple[Sleeve, ...]
    data: Mapping[str, PTDataset]
    start_trading_date: Optional[str] = None
    tplusn: int = 0
    rules: RuleSpec = ()

    def as_kwargs(self) -> Dict[str, Any]:
        """Return fields as constructor kwargs (shallow; datasets are shared, not copied)."""
        return {f.name: getattr(self, f.name) for f in fields(self)}


def _flatten(rules: Any) -> List[Rule]:
    """Flatten a callable or nested lists of callables, rejecting anything else.

    Raises:
        TypeError: Naming the offending item and its type.
    """
    if callable(rules):
        return [rules]
    if isinstance(rules, (list, tuple)):
        return [r for item in rules for r in _flatten(item)]
    raise TypeError(f"rule must be a callable or list of callables, got {type(rules).__name__}: {rules!r}")


class SystematicMultiAsset(PortfolioBook, ParamContractMixin):
    """Inheritable book of sleeves sharing one capital pool.

    Subclass and override any hook. Defaults: unlimited open trades, several
    sleeves per asset allowed, insertion-order priority.
    With every default, a run equals ``Simulator`` on the built children.

    Hooks:
        rank_opens: Order same-day candidates; earlier ones get scarce slots.
        should_open: Return the decision to fill, or ``Veto(reason)``.
        should_close: Return the close to fill, or ``Veto(reason)``.
    """

    _required_init_args: ClassVar[Tuple[str, ...]] = tuple(f.name for f in fields(BookSpec))
    ## Book subclasses only spell out the params they add; strategies keep today's strict rule.
    _kwargs_satisfies_contract: ClassVar[bool] = True
    bt_params: ClassVar[Dict[str, Any]] = {"max_open_trades": np.inf, "allow_multiple_per_asset": True}

    def __init__(
        self,
        sleeves: Sequence[Sleeve],
        data: Mapping[str, PTDataset],
        start_trading_date: Optional[str] = None,
        tplusn: int = 0,
        rules: RuleSpec = (),
        max_open_trades: float = np.inf,
        allow_multiple_per_asset: bool = True,
    ) -> None:
        """Freeze the spec and validate all of it.

        Args:
            sleeves: Strategy sleeves, in priority order. Per-ticker params go in
                ``Sleeve.asset_params``; ``with_overrides`` edits them after construction.
            data: Ticker to dataset for children.
            start_trading_date: First date children may trade.
            tplusn: Execution delay for every child.
            rules: ``(selector, rule)`` pairs; usually added later with ``set_rules``.
            max_open_trades: Cap on concurrent positions across the book (pending t+n excluded).
            allow_multiple_per_asset: When False, a ticker held by one sleeve blocks the others.

        Raises:
            BookConfigError: Listing every config problem found, not just the first.
        """
        self.max_open_trades = max_open_trades
        self.allow_multiple_per_asset = allow_multiple_per_asset
        self.spec = BookSpec(
            sleeves=tuple(sleeves),
            data=data,
            start_trading_date=start_trading_date,
            tplusn=tplusn,
            rules=tuple(rules),
        )
        self._validate(self.spec)
        logger.info(
            "%s init sleeves=%s tickers=%s max_open_trades=%s allow_multiple_per_asset=%s",
            type(self).__name__,
            [s.name for s in self.spec.sleeves],
            len(self.spec.data),
            self.max_open_trades,
            self.allow_multiple_per_asset,
        )

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    def _book_issues(self, spec: BookSpec) -> List[ConfigIssue]:
        """Return problems with the book's own params."""
        issues: List[ConfigIssue] = []
        ## `not >=` also catches NaN, which would silently disable the cap.
        if not (self.max_open_trades >= 1):
            issues.append(ConfigIssue("max_open_trades", f"must be >= 1 or np.inf, got {self.max_open_trades!r}"))
        return issues

    def _validate(self, spec: BookSpec) -> None:
        """Raise one ``BookConfigError`` with every problem in ``spec`` and the book params."""
        issues = config_issues(spec.sleeves, spec.data, spec.rules) + self._book_issues(spec)
        if issues:
            logger.error("%s config invalid issues=%s", type(self).__name__, len(issues))
            raise BookConfigError(type(self).__name__, issues)

    # ------------------------------------------------------------------ #
    # Spec editing (each call swaps in a new frozen spec)
    # ------------------------------------------------------------------ #
    def set_rules(self, rules: RulesInput) -> None:
        """Add per-child rules, same shapes as ``MultiAssetStrategy.set_rules``.

        A callable or list of callables applies to every child. A dict maps a
        selector to a rule or list: ``"AAPL"`` (every sleeve on AAPL), ``"mr:*"``
        (every ticker in sleeve mr), ``"mr:AAPL"`` (one child), ``"*"`` (all).
        A list may mix both shapes.

        Raises:
            BookConfigError: If a selector matches no child.
            TypeError: If a rule is not callable.
        """
        items = rules if isinstance(rules, (list, tuple)) else [rules]
        added: List[Tuple[str, Rule]] = []
        for item in items:
            if isinstance(item, Mapping):
                added += [(selector, r) for selector, rs in item.items() for r in _flatten(rs)]
            else:
                added += [("*", r) for r in _flatten(item)]
        spec = replace(self.spec, rules=self.spec.rules + tuple(added))
        self._validate(spec)
        self.spec = spec

    def with_overrides(self, overrides: KeyOverrides) -> "SystematicMultiAsset":
        """Return a new book with per-child params merged into the sleeves' ``asset_params``.

        ``{("mr", "AAPL"): {"hold": 7}}`` becomes ``mr``'s ``asset_params["AAPL"]["hold"]``,
        so there is still one place per child to look. Book params and every rule
        carry over; this book is unchanged.

        Raises:
            BookConfigError: If a key names no sleeve, or the merged sleeves are invalid
                (unknown ticker or param, twins), with the same hints as the constructor.
        """
        sleeves, issues = merge_overrides(self.spec.sleeves, overrides)
        ## Report unknown sleeve names here; tickers and params are checked by the
        ## new book's constructor exactly like hand-written asset_params.
        if issues:
            raise BookConfigError(type(self).__name__, issues)
        return type(self)(**replace(self.spec, sleeves=sleeves).as_kwargs(), **self.get_bt_params())

    def with_params(self, **book_params: Any) -> "SystematicMultiAsset":
        """Return a new book with different book-level params (e.g. ``max_open_trades``)."""
        return type(self)(**self.spec.as_kwargs(), **{**self.get_bt_params(), **book_params})

    # ------------------------------------------------------------------ #
    # Building and running
    # ------------------------------------------------------------------ #
    @property
    def resolved_params(self) -> pd.DataFrame:
        """Return every child's final params, indexed by (sleeve, ticker), without building."""
        rows = {
            (s.name, t): resolve_params(s, t)
            for s in self.spec.sleeves
            for t in s.universe(self.spec.data)
        }
        table = pd.DataFrame.from_dict(rows, orient="index")
        table.index.names = ["sleeve", "ticker"]
        return table

    def build(self) -> Dict[str, StrategyBase]:
        """Return freshly built children keyed ``"{sleeve}:{ticker}"``.

        Raises:
            ChildBuildError: If a child's constructor or ``setup`` raises.
        """
        return build_children(
            self.spec.sleeves,
            self.spec.data,
            start_trading_date=self.spec.start_trading_date,
            tplusn=self.spec.tplusn,
            rules=self.spec.rules,
        )

    def simulate(
        self,
        capital: float,
        *,
        trade_size: Optional[float] = None,
        percent_of_capital: Optional[float] = None,
        max_gross_leverage: float = np.inf,
        min_cash: Optional[float] = 0.0,
        buying_power_policy: BuyingPowerPolicy = BuyingPowerPolicy.SKIP,
        commission_pct: float = 0.0,
        slippage_pct: float = 0.0,
        wipeout_floor: float = 0.0,
        start_date: Optional[Union[datetime, str]] = None,
        risk_free_rate: float = 0.0,
        finalize: bool = True,
        enforce_open_on_signal: bool = False,
        enforce_close_on_signal: bool = False,
        track_exposure: bool = True,
        stats_level: Literal["full", "lean"] = "full",
    ) -> SimulationResult:
        """Run the book in one shared-capital ``Simulator``.

        Same settings as ``Simulator`` (except ``strategy_keys``, which the book
        already owns). Pass exactly one of ``trade_size`` or ``percent_of_capital``.

        Args:
            capital: Starting cash for the shared pool.
            trade_size: Fixed dollar size per open (before any multiplier).
            percent_of_capital: Open size as a share of current equity.
            max_gross_leverage: Cap on total position size over equity; ``np.inf`` = none.
            min_cash: Cash that must remain after an open; ``None`` = no floor.
            buying_power_policy: Skip or shrink opens that break leverage/cash limits.
            commission_pct: Fraction of each fill notional charged as commission.
            slippage_pct: Fraction of each fill notional charged as slippage cost.
            wipeout_floor: Close everything and stop if equity falls to this level.
            start_date: First date to trade and record; ``None`` uses the book/children starts.
            risk_free_rate: Yearly rate for risk-adjusted scores in ``full_stats``.
            finalize: Close leftover positions on the last day when True.
            enforce_open_on_signal: Only matters when a child uses ``tplusn > 0``.
                When False (default), a pending open fills on its due date even if
                the open signal has flipped off. When True, ``should_open`` is
                checked again on the fill bar and the open is dropped unless still
                ``ok``. No effect with ``tplusn=0``.
            enforce_close_on_signal: Same delay/re-check for closes when
                ``tplusn > 0``. When False (default), pending closes fill even if
                the close signal is gone. When True, ``should_close`` must still be
                ``ok`` on the fill bar. Independent of ``enforce_open_on_signal``.
            track_exposure: Keep exposure curves and summaries on the result.
                Forced off when ``stats_level="lean"``.
            stats_level: ``"full"`` builds all summaries; ``"lean"`` keeps equity
                and ranking stats only (see ``Simulator``).

        Returns:
            The shared-capital backtest result.
        """
        logger.info(
            "%s.simulate capital=%s stats_level=%s track_exposure=%s",
            type(self).__name__,
            capital,
            stats_level,
            track_exposure,
        )
        return Simulator(
            self,
            capital,
            trade_size=trade_size,
            percent_of_capital=percent_of_capital,
            max_gross_leverage=max_gross_leverage,
            min_cash=min_cash,
            buying_power_policy=buying_power_policy,
            commission_pct=commission_pct,
            slippage_pct=slippage_pct,
            wipeout_floor=wipeout_floor,
            start_date=start_date,
            risk_free_rate=risk_free_rate,
            finalize=finalize,
            enforce_open_on_signal=enforce_open_on_signal,
            enforce_close_on_signal=enforce_close_on_signal,
            track_exposure=track_exposure,
            stats_level=stats_level,
        ).run()

    # ------------------------------------------------------------------ #
    # Optimize (thin wrappers over ParamOptimizer)
    # ------------------------------------------------------------------ #
    def _tickers(self) -> List[str]:
        """Return data tickers that appear in at least one sleeve universe, in data order."""
        out: List[str] = []
        for ticker in self.spec.data:
            if any(ticker in sleeve.universe(self.spec.data) for sleeve in self.spec.sleeves):
                out.append(ticker)
        return out

    def with_trial_params(self, params: Mapping[str, Any]) -> "SystematicMultiAsset":
        """Return a fresh book with ``params`` applied for one optimizer trial.

        Keys that are book ``bt_params`` (for example ``max_open_trades``) go through
        ``with_params``. Every other key is merged into the sleeve's sleeve-wide
        ``params`` (asset_params still win per child). ``optimize`` only calls this
        on one-sleeve books.

        Args:
            params: One settings mix from the optimizer grid.

        Returns:
            A new book; this instance is unchanged.
        """
        book_keys = set(type(self).bt_params)
        book_params = {k: v for k, v in params.items() if k in book_keys}
        sleeve_params = {k: v for k, v in params.items() if k not in book_keys}

        ## Start from this book's recipe; only replace sleeves when the trial tunes them.
        spec = self.spec
        if sleeve_params:
            sleeve = self.spec.sleeves[0]
            updated = replace(sleeve, params={**dict(sleeve.params), **sleeve_params})
            spec = replace(self.spec, sleeves=(updated,))

        ## Rebuild from the (maybe updated) spec, then overlay book-level trial keys.
        book = type(self)(**spec.as_kwargs(), **self.get_bt_params())
        if not book_params:
            return book
        return book.with_params(**book_params)

    def optimize(
        self,
        space: Mapping[str, Sequence[Any]],
        *,
        simulator_kwargs: Mapping[str, Any],
        per_asset: bool = False,
        **opt_kwargs: Any,
    ) -> Union[OptimizationResult, Dict[str, OptimizationResult]]:
        """Grid-search ``space`` on this book via ``ParamOptimizer``.

        Default (``per_asset=False``) runs one shared-capital search: each trial
        rebuilds this book through ``with_trial_params`` so caps and other hooks
        apply. ``per_asset=True`` runs the same solo loop as
        ``MultiAssetStrategy.optimize``.

        Only one-sleeve books may optimize. For multi-sleeve books, optimize each
        sleeve on its own one-sleeve book.

        Args:
            space: Parameter name to values to try.
            simulator_kwargs: ``Simulator`` settings (must include ``capital`` and
                either ``trade_size`` or ``percent_of_capital``).
            per_asset: When True, optimize each ticker alone; when False, one
                shared-capital grid for the whole book.
            **opt_kwargs: Forwarded to ``ParamOptimizer`` / ``optimize_per_ticker``.

        Returns:
            One ``OptimizationResult`` for the shared book, or ticker to result
            when ``per_asset=True``.

        Raises:
            ValueError: If the book has more than one sleeve, has no tickers, or
                ``per_asset`` is missing ``start_trading_date``.
        """
        ## Import here so optimizer workers never need this module at import time.
        from trade.backtester_.optimizer import ParamOptimizer, optimize_per_ticker

        if len(self.spec.sleeves) != 1:
            raise ValueError(
                f"{type(self).__name__}.optimize requires a one-sleeve book; "
                f"got {[s.name for s in self.spec.sleeves]}"
            )

        if per_asset:
            tickers = self._tickers()
            if not tickers:
                raise ValueError("SystematicMultiAsset has no tickers to optimize")
            if self.spec.start_trading_date is None:
                raise ValueError(
                    "start_trading_date is required for SystematicMultiAsset.optimize(per_asset=True)"
                )
            logger.info(
                "%s.optimize per_asset=True tickers=%s space_keys=%s",
                type(self).__name__,
                tickers,
                list(space),
            )
            sleeve = self.spec.sleeves[0]
            data = {t: self.spec.data[t] for t in tickers}
            base = {t: resolve_params(sleeve, t) for t in tickers}
            return optimize_per_ticker(
                sleeve.strategy_class,
                data,
                space,
                start_trading_date=self.spec.start_trading_date,
                simulator_kwargs=simulator_kwargs,
                base_params=base,
                tplusn=self.spec.tplusn,
                **opt_kwargs,
            )

        logger.info(
            "%s.optimize per_asset=False space_keys=%s",
            type(self).__name__,
            list(space),
        )
        return ParamOptimizer(
            space=space,
            simulator_kwargs=simulator_kwargs,
            book_factory=self.with_trial_params,
            **opt_kwargs,
        ).run()

    # ------------------------------------------------------------------ #
    # Hooks (override in subclasses)
    # ------------------------------------------------------------------ #
    @staticmethod
    def sleeve_of(key: str) -> str:
        """Return the sleeve name from a child key ``"{sleeve}:{ticker}"``.

        Sleeve names cannot contain ``:`` (checked by ``config_issues``), so the
        first ``:`` always separates sleeve from ticker.

        Examples:
            >>> SystematicMultiAsset.sleeve_of("mr:AAPL")
            'mr'
        """
        return key.split(":", 1)[0]

    def should_open(self, candidate: OpenCandidate, ctx: PortfolioContext) -> Union[TradeDecision, Veto]:
        """Apply ``max_open_trades`` and ``allow_multiple_per_asset``, then the base default.

        ``rank_opens`` and ``should_close`` are inherited unchanged from ``PortfolioBook``.
        Veto reasons are ``"category (details)"`` so ``skipped["reason"]`` groups by category.
        """
        if ctx.n_open >= self.max_open_trades:
            return Veto(f"max_open_trades reached ({ctx.n_open}/{self.max_open_trades:g} open)")
        holders = ctx.open_by_ticker.get(candidate.ticker, ())
        if not self.allow_multiple_per_asset and holders:
            return Veto(f"asset already held ({candidate.ticker} by {list(holders)}; allow_multiple_per_asset=False)")
        return super().should_open(candidate, ctx)

    def __repr__(self) -> str:
        """Summarize sleeves, child count, and book params."""
        n = sum(len(s.universe(self.spec.data)) for s in self.spec.sleeves)
        sleeves = [s.name for s in self.spec.sleeves]
        return f"{type(self).__name__}(sleeves={sleeves}, children={n}, {self.get_bt_params()})"
