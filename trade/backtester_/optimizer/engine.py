"""Grid-search a strategy's ``bt_params`` across a book of tickers.

``ParamOptimizer`` tests every mix of the values in its ``space`` dictionary.
Every ticker in ``data`` runs together in one ``Simulator`` on each mix, and the
best mix wins for the whole book. To tune each asset on its own, call it once
per ticker with a one-ticker ``data``, the way ``PTBacktester.position_optimize``
loops over datasets. Trials run in forked workers; scoring, constraints and
picking happen in the parent.

Comment density: orchestration

Core Classes:
    ParamOptimizer: Runs the grid and returns an ``OptimizationResult``.

Processing Flow:
    1. ``__init__`` checks the settings and lists every mix to test (``combos``).
    2. One ``run_trials`` call tests every mix with lean simulator defaults
       (``stats_level="lean"``, ``track_exposure=False``) unless the caller
       already set those keys. Flat runs build one strategy per ticker; with
       ``book_factory``, each mix builds a ``PortfolioBook`` instead.
    3. The parent files each answer under its mix's values (like backtesting.py's
       heatmap), scores it, and marks whether it passes the constraints.
    4. The best passing mix wins, and the best few are re-run here with
       ``stats_level="full"`` and ``track_exposure=True`` so notebooks get
       trade tables and exposure.

Risk/Assumptions:
    Sizing comes entirely from ``simulator_kwargs``. A one-ticker book judged on its
    own signals should size like ``StrategyBase.simulate``
    (``percent_of_capital=1.0, min_cash=None, wipeout_floor=-np.inf``).
    Lean trials omit trade/cash summary keys from ``full_stats``; constraints that
    need those keys must pass ``stats_level="full"`` in ``simulator_kwargs``.
    With ``book_factory``, space keys are not checked against ``bt_params``; the
    factory owns what each mix means (child params, book caps, or both).

Core Functions:
    optimize_per_ticker: One ``ParamOptimizer`` per ticker (MultiAsset / per-asset path).

Usage:
    >>> opt = ParamOptimizer(
    ...     LongUsEqMeanReversionFlat, data=d, start_trading_date="2016-01-01",
    ...     space={"ma_length": [10, 20]}, base_params=MR_ASSET_PARAMS,
    ...     simulator_kwargs={"capital": 100_000, "percent_of_capital": 0.25},
    ...     constraints=[("max_drawdown", "<", 0.25)],
    ... )
    >>> res = opt.run()
    >>> res.best_params
    {'ma_length': 10}
    >>> # Shared-capital book with hooks (caps, rank_opens, …):
    >>> opt = ParamOptimizer(
    ...     space={"max_open_trades": [2, 5]},
    ...     simulator_kwargs={"capital": 100_000, "percent_of_capital": 0.25},
    ...     book_factory=lambda p: template.with_params(**p),
    ... )
    >>> # Per-asset loop (same engine, one ticker at a time):
    >>> by_ticker = optimize_per_ticker(
    ...     LongUsEqMeanReversionFlat, data=d, space={"ma_length": [10, 20]},
    ...     start_trading_date="2016-01-01",
    ...     simulator_kwargs={"capital": 100_000, "percent_of_capital": 1.0, "min_cash": None},
    ... )
"""

from __future__ import annotations

import itertools
import math
import operator
import warnings
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Type, Union

import numpy as np
import pandas as pd

from trade.backtester_._multi_asset_strategy import _setup_strategy
from trade.backtester_._strategy import StrategyBase
from trade.backtester_.data import PTDataset
from trade.backtester_.optimizer._pool import BookBuildFn, Outcome, run_trials, simulate, simulate_book
from trade.backtester_.optimizer.core import OptimizationResult
from trade.backtester_.simulator import SimulationResult
from trade.helpers.Logging import setup_logger

logger = setup_logger(__name__)

## A constraint is either ("stat name", "operator", number), e.g. ("max_drawdown", "<", 0.25),
## or any function that takes the stats and says yes or no.
Constraint = Union[Tuple[str, str, float], Callable[[pd.Series], bool]]
## An objective is a stat name (e.g. "sharpe_ratio") or a function returning one number.
Objective = Union[str, Callable[[pd.Series], float]]
## Builds a PortfolioBook from one settings mix (used instead of a flat strategy list).
BookFactory = BookBuildFn

_OPERATORS: Dict[str, Callable[[Any, Any], bool]] = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}

## Solo sizing matching ``StrategyBase.simulate``: full equity, no cash floor, no wipeout.
SOLO_SIMULATOR_KWARGS: Dict[str, Any] = {
    "capital": 100_000.0,
    "percent_of_capital": 1.0,
    "min_cash": None,
    "wipeout_floor": -np.inf,
}


def optimize_per_ticker(
    strategy_class: Type[StrategyBase],
    data: Mapping[str, PTDataset],
    space: Mapping[str, Sequence[Any]],
    *,
    start_trading_date: str,
    simulator_kwargs: Optional[Mapping[str, Any]] = None,
    base_params: Optional[Mapping[str, Mapping[str, Any]]] = None,
    tplusn: int = 0,
    **opt_kwargs: Any,
) -> Dict[str, OptimizationResult]:
    """Run one ``ParamOptimizer`` per ticker and return results keyed by ticker.

    This is the per-asset path: each ticker is sized and scored alone (default
    solo ``simulator_kwargs`` match ``StrategyBase.simulate``). Shared-capital
    search over many tickers uses ``ParamOptimizer`` directly instead.

    Args:
        strategy_class: ``StrategyBase`` subclass built for each ticker.
        data: Ticker to dataset; iteration order is preserved in the result dict.
        space: Parameter name to values to try (same grid for every ticker).
        start_trading_date: First date each strategy may trade (``YYYY-MM-DD``).
        simulator_kwargs: ``Simulator`` settings per solo run. ``None`` uses
            full-equity solo defaults (``percent_of_capital=1.0``, ``min_cash=None``,
            ``wipeout_floor=-inf``, ``capital=100_000``).
        base_params: Optional ticker to starting settings under the tested mix.
        tplusn: Execution delay in bars.
        **opt_kwargs: Forwarded to each ``ParamOptimizer`` (``objective``,
            ``constraints``, ``n_jobs``, ``keep_top``, …).

    Returns:
        Mapping of ticker to that ticker's ``OptimizationResult``.

    Raises:
        ValueError: If ``data`` is empty (same as a one-ticker ``ParamOptimizer``).
    """
    if not data:
        raise ValueError("data has no tickers")
    sim_kw = dict(simulator_kwargs) if simulator_kwargs is not None else dict(SOLO_SIMULATOR_KWARGS)
    bases = base_params or {}
    logger.info(
        "optimize_per_ticker start tickers=%s combos_space_keys=%s",
        list(data),
        list(space),
    )
    results: Dict[str, OptimizationResult] = {}
    for ticker, dataset in data.items():
        ## One optimizer per ticker so base_params and scoring stay solo.
        results[ticker] = ParamOptimizer(
            strategy_class,
            data={ticker: dataset},
            space=space,
            start_trading_date=start_trading_date,
            simulator_kwargs=sim_kw,
            base_params={ticker: dict(bases[ticker])} if ticker in bases else None,
            tplusn=tplusn,
            **opt_kwargs,
        ).run()
    logger.info("optimize_per_ticker done tickers=%s", list(results))
    return results


class ParamOptimizer:
    """Test every settings mix and pick the best by an objective.

    Two trial shapes:

    - **Flat strategies** (default): ``strategy_class`` + ``data`` build one child
      per ticker and run them in one shared-capital ``Simulator``.
    - **Portfolio book**: pass ``book_factory(params) -> PortfolioBook`` so each
      trial goes through book hooks (``max_open_trades``, ``rank_opens``, …).

    Attributes:
        strategy_class: The ``StrategyBase`` subclass built for each ticker (flat path).
        data: Ticker to dataset (flat path).
        book_factory: Optional ``params -> PortfolioBook`` builder (book path).
        space: Parameter name to the list of values to try.
        start_trading_date: First date each strategy may trade (flat path).
        base_params: Ticker to starting settings that tested values are laid over.
        tplusn: Execution delay in bars, passed to every strategy.
        combos: Every mix to test, as dicts, after ``param_filter``.
        simulator_kwargs: Settings for every ``Simulator`` run.
        objective: Score to rank mixes by.
        constraints: Rules a mix's stats must pass to be allowed to win.
        maximize: True when a higher score is better.
        keep_top: How many of the best passing mixes to re-run in full.
        n_jobs: Worker processes; ``-1`` means one per CPU, ``1`` means none.
        show_progress: Show a progress bar.
    """

    def __init__(
        self,
        strategy_class: Optional[Type[StrategyBase]] = None,
        *,
        data: Optional[Mapping[str, PTDataset]] = None,
        space: Mapping[str, Sequence[Any]],
        simulator_kwargs: Mapping[str, Any],
        start_trading_date: Optional[str] = None,
        book_factory: Optional[BookFactory] = None,
        base_params: Optional[Mapping[str, Mapping[str, Any]]] = None,
        tplusn: int = 0,
        param_filter: Optional[Callable[[Dict[str, Any]], bool]] = None,
        objective: Objective = "sharpe_ratio",
        constraints: Sequence[Constraint] = (),
        maximize: bool = True,
        keep_top: int = 3,
        n_jobs: int = -1,
        show_progress: bool = True,
    ) -> None:
        """Check the settings so mistakes show up before any trial runs.

        Args:
            strategy_class: The ``StrategyBase`` subclass to build for each ticker.
                Required for the flat path; optional when ``book_factory`` is set.
            data: Ticker to dataset. Required for the flat path.
            space: Parameter name to the list of values to try, e.g.
                ``{"ma_length": [10, 20], "exit_days": [10, 15]}``. Every mix is tested.
                An empty dict means "keep the settings fixed, do not tune".
            simulator_kwargs: ``Simulator`` settings such as ``capital``,
                ``percent_of_capital``, ``min_cash`` and ``start_date``. Trials
                default to ``stats_level="lean"`` and ``track_exposure=False``
                when those keys are omitted; ``keep_top`` re-runs force full
                stats and exposure on.
            start_trading_date: First date each strategy may trade (``YYYY-MM-DD``).
                Required for the flat path.
            book_factory: ``book_factory(params) -> PortfolioBook``. When set, each
                trial builds a book and runs ``Simulator(book, ...)`` so hooks apply.
                The factory must return a fresh book every call.
            base_params: Optional ticker to starting settings. Tested values are laid
                on top. Anything still missing comes from the class ``bt_params``.
            tplusn: Execution delay in bars, passed to every strategy.
            param_filter: Optional check on each mix (as a dict). Mixes where it
                returns False are never run, e.g. ``lambda p: p["entry"] < p["exit"]``.
            objective: ``full_stats`` name or a function of the stats Series.
            constraints: Each is ``(stat, operator, value)`` or a function of the stats
                Series. A missing or NaN stat fails the constraint.
            maximize: True when a higher score is better.
            keep_top: How many of the best passing mixes to re-run in full.
            n_jobs: Worker processes; ``-1`` means one per CPU, ``1`` means none.
            show_progress: Show a progress bar.

        Raises:
            ValueError: On a missing flat-path input, no tickers, a setting name not
                in ``bt_params`` (flat path only), a filter that removes every mix,
                or a badly written constraint.
        """
        if book_factory is None:
            if strategy_class is None or data is None or start_trading_date is None:
                raise ValueError(
                    "strategy_class, data, and start_trading_date are required without book_factory"
                )
            if not data:
                raise ValueError("data has no tickers")
            ## A misspelled name would otherwise only fail inside a worker, once per trial,
            ## after the whole grid has already been sent out.
            unknown = [n for n in space if n not in strategy_class.bt_params]
            if unknown:
                raise ValueError(f"{unknown} are not in {strategy_class.__name__}.bt_params")
        for c in constraints:
            if not callable(c) and (len(c) != 3 or c[1] not in _OPERATORS):
                raise ValueError(f"Constraint {c!r} must be (stat, op, value) with op in {list(_OPERATORS)}, or a function")
        self.strategy_class = strategy_class
        self.data = data or {}
        self.book_factory = book_factory
        self.space = space
        self.start_trading_date = start_trading_date
        self.base_params = base_params or {}
        self.tplusn = tplusn
        self.combos: List[Dict[str, Any]] = self._make_combos(param_filter)
        self.simulator_kwargs: Dict[str, Any] = dict(simulator_kwargs)
        self.objective = objective
        self.constraints: Tuple[Constraint, ...] = tuple(constraints)
        self.maximize = maximize
        self.keep_top = int(keep_top)
        self.n_jobs = n_jobs
        self.show_progress = show_progress
        path = "book_factory" if book_factory is not None else f"flat[{strategy_class.__name__}]"
        logger.info(
            "ParamOptimizer ready path=%s combos=%s tickers=%s objective=%s keep_top=%s n_jobs=%s",
            path,
            len(self.combos),
            len(self.data) if book_factory is None else "n/a",
            objective if not callable(objective) else "<callable>",
            self.keep_top,
            n_jobs,
        )

    @property
    def tickers(self) -> List[str]:
        """Return the flat-path tickers in data order (empty on the book path)."""
        return list(self.data)

    def _make_combos(self, param_filter: Optional[Callable[[Dict[str, Any]], bool]]) -> List[Dict[str, Any]]:
        """List every mix of the ``space`` values as dicts, minus filtered ones.

        Raises:
            ValueError: If the filter removed every mix.

        Examples:
            ``space={"a": [1, 2], "b": [1, 2]}`` with ``param_filter=lambda p: p["a"] < p["b"]``
            gives ``[{"a": 1, "b": 2}]``.
        """
        names = list(self.space)
        ## itertools.product walks the grid like an odometer: the last name changes
        ## fastest. An empty space gives one empty mix: run the base settings once.
        combos = [dict(zip(names, values)) for values in itertools.product(*(list(self.space[n]) for n in names))]
        if param_filter is not None:
            combos = [c for c in combos if param_filter(c)]
        if not combos:
            raise ValueError("Nothing to test: param_filter removed every mix")
        return combos

    def _build(self, ticker: str, tested: Dict[str, Any]) -> StrategyBase:
        """Build one ticker's strategy with the tested values laid over its base settings.

        Workers call this too (it is handed to the pool), so a worker builds exactly
        what the parent would. Only used on the flat strategy path.

        Args:
            ticker: Which ticker to build.
            tested: The settings being tried for this ticker.

        Returns:
            A freshly built strategy.

        Raises:
            RuntimeError: If called on the book-factory path.
        """
        if self.strategy_class is None or self.start_trading_date is None:
            raise RuntimeError("_build is only for the flat strategy path")
        ## Start from the ticker's base settings, then put the tested values on top.
        ## _setup_strategy fills anything still missing from the class bt_params.
        params = {**self.base_params.get(ticker, {}), **tested}
        return _setup_strategy(
            self.strategy_class, self.data[ticker], self.start_trading_date, ticker, params, tplusn=self.tplusn
        )

    # ------------------------------------------------------------------ #
    # Main flow                                                             #
    # ------------------------------------------------------------------ #
    def _trial_kwargs(self) -> Dict[str, Any]:
        """Simulator settings for worker trials: lean by default, caller can override.

        Returns:
            A copy of ``simulator_kwargs`` with ``stats_level="lean"`` and
            ``track_exposure=False`` filled in only when the caller left them out.
        """
        kw = dict(self.simulator_kwargs)
        kw.setdefault("stats_level", "lean")
        kw.setdefault("track_exposure", False)
        return kw

    def _full_kwargs(self) -> Dict[str, Any]:
        """Simulator settings for ``keep_top`` re-runs: always rich results.

        Returns:
            A copy of ``simulator_kwargs`` with ``stats_level="full"`` and
            ``track_exposure=True`` forced on so winners keep trade and exposure tables.
        """
        kw = dict(self.simulator_kwargs)
        kw["stats_level"] = "full"
        kw["track_exposure"] = True
        return kw

    def run(self) -> OptimizationResult:
        """Run every mix on the whole book, pick the best, and re-run the top few in full.

        Returns:
            The trials table, winning settings, and full results for the best mixes.
        """
        trial_kw = self._trial_kwargs()
        logger.info(
            "ParamOptimizer.run start combos=%s stats_level=%s track_exposure=%s",
            len(self.combos),
            trial_kw.get("stats_level"),
            trial_kw.get("track_exposure"),
        )
        # --- step 1: every mix runs once (lean stats) ---
        outcomes = self._run(self.combos, trial_kw)

        # --- step 2: score every mix here in the parent ---
        ## Workers only sent numbers back, so scoring is cheap and any objective or
        ## constraint (including notebook lambdas) works without being sent anywhere.
        table = self._table(self.combos, outcomes)
        failed = int(table["error"].notna().sum()) if "error" in table else 0
        ranked = self._ranked(table)
        logger.info(
            "ParamOptimizer scored trials=%s feasible=%s failed=%s",
            len(table),
            int(ranked.shape[0]),
            failed,
        )
        if ranked.empty:
            return self._nothing_passed(table)

        # --- step 3: re-run the best few here to get their full results ---
        ## Runs are repeatable, so re-running here gives exactly what the worker saw,
        ## plus the daily series and trade tables lean trials did not build.
        full_kw = self._full_kwargs()
        top_n = max(self.keep_top, 1)
        logger.debug("ParamOptimizer keep_top re-runs n=%s stats_level=full", top_n)
        top_results = {
            label: self._rerun(self._label_params(label), full_kw)
            for label in ranked.index[:top_n]
        }
        best_label = ranked.index[0]
        best_params = self._label_params(best_label)
        best_score = float(ranked.iloc[0])
        logger.info("ParamOptimizer done best_params=%s best_score=%s", best_params, best_score)
        return OptimizationResult(
            trials=table,
            best_params=best_params,
            best_score=best_score,
            best_result=top_results[best_label],
            top_results=top_results,
        )

    def _run(self, combos: List[Dict[str, Any]], simulator_kwargs: Dict[str, Any]) -> List[Outcome]:
        """Test every mix through the worker pool (flat strategies or book factory)."""
        if self.book_factory is not None:
            return run_trials(
                combos,
                simulator_kwargs,
                book_build=self.book_factory,
                n_jobs=self.n_jobs,
                show_progress=self.show_progress,
            )
        return run_trials(
            combos,
            simulator_kwargs,
            build=self._build,
            tickers=self.tickers,
            n_jobs=self.n_jobs,
            show_progress=self.show_progress,
        )

    def _rerun(self, params: Dict[str, Any], simulator_kwargs: Dict[str, Any]) -> SimulationResult:
        """Run one simulation here in the parent (used for ``keep_top`` full results).

        Args:
            params: The winning settings mix (same dict shape as each combo).
            simulator_kwargs: Settings for the ``Simulator``.

        Returns:
            The ``SimulationResult`` for those settings.
        """
        if self.book_factory is not None:
            return simulate_book(self.book_factory, params, simulator_kwargs)
        ## Flat path: apply the same mix to every ticker in the book.
        return simulate(self._build, {t: params for t in self.tickers}, simulator_kwargs)

    # ------------------------------------------------------------------ #
    # Results table
    # ------------------------------------------------------------------ #
    def _index(self, combos: List[Dict[str, Any]]) -> pd.MultiIndex:
        """Build the table index from the mixes' values, one level per setting name.

        This is how backtesting.py files its results: the row label *is* the mix.
        """
        names = list(self.space)
        if not names:
            ## Nothing is being tuned, so there is one row: the base settings.
            return pd.MultiIndex.from_arrays([["base"]], names=["params"])
        return pd.MultiIndex.from_tuples([tuple(c[n] for n in names) for c in combos], names=names)

    def _label_params(self, label: Tuple[Any, ...]) -> Dict[str, Any]:
        """Turn a row label back into the original settings dict."""
        names = list(self.space)
        if not names:
            return {}
        ## pandas hands label values back as numpy numbers (np.int64(10)), so look up
        ## the user's own dict instead of rebuilding it; numpy and plain numbers
        ## compare equal, so the lookup works.
        values = tuple(label)
        by_values = {tuple(c[n] for n in names): c for c in self.combos}
        return dict(by_values[values])

    def _table(self, combos: List[Dict[str, Any]], outcomes: List[Outcome]) -> pd.DataFrame:
        """File each outcome under its mix, with score, pass flag, error and stats.

        Args:
            combos: The mixes, in the order they were tested.
            outcomes: One outcome per mix, same order.

        Returns:
            DataFrame indexed by the mixes' values.
        """
        rows: List[Dict[str, Any]] = []
        for outcome in outcomes:
            if isinstance(outcome, str):
                ## Failed mixes stay visible in the table, with their error, so a typo
                ## or a bad mix is easy to spot afterwards.
                rows.append({"score": math.nan, "feasible": False, "error": outcome})
                continue
            stats = pd.Series(outcome, dtype=float)
            score = self._score(stats)
            ## A NaN score cannot be ranked, so it can never win.
            rows.append({"score": score, "feasible": self._feasible(stats) and not math.isnan(score), "error": None, **outcome})
        table = pd.DataFrame(rows, index=self._index(combos))
        table["feasible"] = table["feasible"].astype(bool)
        return table

    def _ranked(self, table: pd.DataFrame) -> pd.Series:
        """Return the passing mixes' scores, best first."""
        ## NaN scores were already marked as not passing, so they never show up here.
        ## A stable sort keeps the earlier mix first on a tie, so results repeat exactly.
        return table.loc[table["feasible"], "score"].sort_values(ascending=not self.maximize, kind="stable")

    def _nothing_passed(self, table: pd.DataFrame) -> OptimizationResult:
        """Warn and return a result with no winner."""
        ## Say so plainly rather than quietly crowning a mix that broke the rules.
        failed = int(table["error"].notna().sum())
        msg = f"No trial passed the constraints ({len(table)} trials, {failed} failed with errors)."
        logger.warning(msg)
        warnings.warn(msg, stacklevel=3)
        return OptimizationResult(trials=table, best_params={}, best_score=math.nan, best_result=None)

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #
    def _score(self, stats: pd.Series) -> float:
        """Return the score for one mix's stats; NaN when unusable."""
        raw = self.objective(stats) if callable(self.objective) else stats.get(self.objective, math.nan)
        try:
            return float(raw)
        except (TypeError, ValueError):
            return math.nan

    def _feasible(self, stats: pd.Series) -> bool:
        """Return True when every constraint passes.

        A stat that is missing or NaN fails, because "unknown" should not be allowed
        to win. For example a run with no trades has no win rate.
        """
        for c in self.constraints:
            if callable(c):
                if not bool(c(stats)):
                    return False
                continue
            name, op, limit = c
            value = stats.get(name, math.nan)
            if value is None or pd.isna(value) or not _OPERATORS[op](value, limit):
                return False
        return True
