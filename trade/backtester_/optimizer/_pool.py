"""Run optimizer trials across forked worker processes, the way ``backtesting.py`` does.

Bookkeeping copies ``Backtest.optimize``: the settings to test are plain
``{name: value}`` dicts cut into batches, a worker's task is just a batch
number, and it answers ``(batch_index, values)`` with one value per dict in
that batch, in the same order. The parent pairs the values back up with
``param_batches[batch_index]``, so no extra trial ids are needed.

Workers are started with fork. They begin as copies of the parent, so they
already hold the batches, the datasets and any notebook-made classes or
lambdas; nothing large is sent to them.

Comment density: orchestration

Core Functions:
    run_trials: Run every settings dict and return one outcome per dict, in order.
    simulate: Build flat strategies for some settings and run one ``Simulator``.
    simulate_book: Build a ``PortfolioBook`` for some settings and run one ``Simulator``.
    fork_unsafe_reason: Why forking is not safe right now, or None when it is.

Processing Flow:
    1. ``run_trials`` refuses to start if another run is already going.
    2. Before forking, ``_warm_up`` builds the first settings once so shared market
       data is loaded in the parent and every worker inherits it.
    3. The settings dicts are cut into batches and stored in ``_RUN``.
    4. Without parallel workers each batch runs here, one after another.
    5. Otherwise workers are forked; each runs ``_worker_init`` once, then batches.
    6. ``_mp_task`` simulates each dict in its batch and keeps only the stats.
    7. Answers come back in finish order; each is put back in its batch's place.
    8. ``_RUN`` is emptied again, even after an error or Ctrl-C.

Risk/Assumptions:
    macOS only allows safe forking when ``OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES``
    was set before Python started. Without it the run falls back to one process.
    Workers must not touch the database: trials only simulate, so they never use a
    connection copied from the parent.
    Forking while other threads hold locks (live data feeds, logging threads) can
    leave a worker stuck. Start optimizer runs from a quiet session.
"""

from __future__ import annotations

import math
import multiprocessing as mp
import os
import signal
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import threadpoolctl
from tqdm.auto import tqdm

from trade.backtester_._strategy import StrategyBase
from trade.backtester_.optimizer.core import TickerParams
from trade.backtester_.simulator import SimulationResult, Simulator
from trade.helpers.Logging import setup_logger

## Module logger is created in the parent and inherited by forked workers. Do not
## call setup_logger again inside workers (duplicate handlers / lock fights).
## Workers should log failures at WARNING/ERROR only — not per-trial DEBUG.
logger = setup_logger(__name__)

## One answer per settings dict: its stats as plain floats, or the error text if it crashed.
Outcome = Union[Dict[str, float], str]
## Builds one ticker's strategy from the settings being tried: build(ticker, params).
## The optimizer passes its own method, so workers build exactly what the parent would.
BuildFn = Callable[[str, Dict[str, Any]], StrategyBase]
## Builds a whole PortfolioBook from one settings mix: book_build(params) -> book.
## Used when trials must go through book hooks (caps, rank_opens), not a flat list.
BookBuildFn = Callable[[Dict[str, Any]], Any]

## Batches are sized so each worker gets about four of them. More, smaller batches
## mean the progress bar moves often and no worker sits idle at the end while
## another finishes a big batch.
_BATCHES_PER_WORKER = 4
## One trial takes roughly half a second to a second, so 20 trials is at most about
## 20 seconds of work. Bigger batches would make Ctrl-C and crashes lose more work.
## (backtesting.py caps at 300, but its trials take milliseconds.)
_MAX_BATCH_SIZE = 20


@dataclass(frozen=True)
class _RunState:
    """Everything a worker needs for one run, shared by forking instead of sending.

    Exactly one of ``build`` or ``book_build`` is set. Flat strategy books use
    ``build`` + ``tickers``; portfolio books use ``book_build`` alone.

    Attributes:
        build: Builds one ticker's strategy from the settings being tried, or None.
        book_build: Builds a ``PortfolioBook`` from the settings mix, or None.
        tickers: Tickers that run together in every flat-strategy trial.
        simulator_kwargs: Settings for every ``Simulator`` in this run.
        param_batches: The settings dicts to test, cut into batches.
    """

    build: Optional[BuildFn]
    book_build: Optional[BookBuildFn]
    tickers: Tuple[str, ...]
    simulator_kwargs: Dict[str, Any]
    param_batches: List[List[Dict[str, Any]]]


## The run in progress, or None. The parent fills it in just before starting
## workers. Forked workers are copies of the parent, so they can read it without
## anything being sent to them. Only one run at a time is allowed.
_RUN: Optional[_RunState] = None

## Holds each worker's thread limit so it stays in force for the worker's whole life.
_THREAD_LIMIT: Optional[threadpoolctl.threadpool_limits] = None


def fork_unsafe_reason() -> Optional[str]:
    """Return why forking would be unsafe right now, or None when it is fine.

    Returns:
        A short explanation, or None.
    """
    if "fork" not in mp.get_all_start_methods():
        return "this platform cannot fork processes"
    ## macOS can crash a forked worker if Apple system code was half way through
    ## starting up when the fork happened. This setting turns that check off, but
    ## only when it is present before Python starts; setting it later does nothing.
    if sys.platform == "darwin" and os.environ.get("OBJC_DISABLE_INITIALIZE_FORK_SAFETY") != "YES":
        return (
            "OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES was not set before Python started "
            "(set it in the conda env, the shell, or the notebook kernel's kernel.json)"
        )
    return None


def _make_batches(param_combos: List[Dict[str, Any]], n_workers: int) -> List[List[Dict[str, Any]]]:
    """Cut the settings list into consecutive batches.

    Args:
        param_combos: Every settings dict, in order.
        n_workers: Number of worker processes.

    Returns:
        Batches that together hold every dict exactly once, in the original order.
    """
    ## The cap limits how many dicts sit in one batch, not how many batches there
    ## are. Dicts beyond the cap simply become more batches, so none are dropped.
    ## Example: 1,000 dicts on 8 workers is 50 batches of 20.
    size = min(max(math.ceil(len(param_combos) / (n_workers * _BATCHES_PER_WORKER)), 1), _MAX_BATCH_SIZE)
    return [param_combos[i : i + size] for i in range(0, len(param_combos), size)]


def _worker_init() -> None:
    """Prepare a freshly forked worker before it receives any batch."""
    ## The worker is a copy of the parent, including the parent's Ctrl-C handler from
    ## trade/__init__. Without this line, one Ctrl-C would run that cleanup in every
    ## worker at once. Workers ignore Ctrl-C and let the parent shut them down.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    ## Number-crunching libraries may start one helper thread per CPU core. With
    ## eight workers that would be 64 threads fighting over 8 cores, so each worker
    ## is limited to one.
    global _THREAD_LIMIT
    _THREAD_LIMIT = threadpoolctl.threadpool_limits(1)


def simulate(build: BuildFn, params: TickerParams, simulator_kwargs: Dict[str, Any]) -> SimulationResult:
    """Build one strategy per ticker in ``params`` and run them together in one Simulator.

    Args:
        build: Builds one ticker's strategy from the settings being tried.
        params: Ticker to the settings to try. Only these tickers run.
        simulator_kwargs: Settings for the ``Simulator``.

    Returns:
        The full ``SimulationResult``.
    """
    ## Strategies are built fresh every time. Reusing a built one is unsafe: running
    ## its setup again stacks duplicate filter rules onto it.
    strategies = [build(ticker, tested) for ticker, tested in params.items()]
    return Simulator(strategies, **simulator_kwargs).run()


def simulate_book(
    book_build: BookBuildFn,
    params: Dict[str, Any],
    simulator_kwargs: Dict[str, Any],
) -> SimulationResult:
    """Build one ``PortfolioBook`` from ``params`` and run it in a ``Simulator``.

    Args:
        book_build: ``book_build(params) -> PortfolioBook``. Must return a fresh book
            each call so child rules never stack across trials.
        params: The settings mix being tried for this trial.
        simulator_kwargs: Settings for the ``Simulator``.

    Returns:
        The full ``SimulationResult`` (book hooks apply on every fill).
    """
    return Simulator(book_build(params), **simulator_kwargs).run()


def _to_float(value: Any) -> float:
    """Return ``value`` as a float, or NaN when it is not a number."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _warm_up(
    build: Optional[BuildFn],
    book_build: Optional[BookBuildFn],
    tickers: Sequence[str],
    first: Dict[str, Any],
) -> None:
    """Build the first settings in the parent so workers inherit already-loaded data.

    Args:
        build: Flat-strategy builder, or None when using ``book_build``.
        book_build: Book builder, or None when using ``build``.
        tickers: Tickers in a flat-strategy run.
        first: The first settings dict.
    """
    ## The first strategy built in a fresh session downloads and checks shared market
    ## data (VIX, rates, sector prices) and keeps it in memory, which takes several
    ## seconds. If the parent skips this, every worker downloads it at the same time:
    ## a 24-trial run went from about 10 seconds to over 3 minutes, and workers
    ## ended up using the network and database, which they must not do.
    ## Building once here means every worker starts with that data already loaded.
    try:
        if book_build is not None:
            book_build(first)
        else:
            for ticker in tickers:
                build(ticker, first)  # type: ignore[misc]
    except Exception:
        ## A broken first mix is reported by its worker like any other trial.
        pass


def _mp_task(batch_index: int) -> Tuple[int, List[Outcome]]:
    """Run one batch of settings and return one outcome per settings dict, in order.

    Args:
        batch_index: Which batch of the current run to do.

    Returns:
        The batch number and the outcomes, lined up with ``param_batches[batch_index]``.
    """
    run = _RUN
    outcomes: List[Outcome] = []
    for params in run.param_batches[batch_index]:
        try:
            if run.book_build is not None:
                result = simulate_book(run.book_build, params, run.simulator_kwargs)
            else:
                result = simulate(run.build, {t: params for t in run.tickers}, run.simulator_kwargs)
            ## Keep only the statistics, so a few kilobytes travel back to the parent
            ## instead of daily series and trade tables.
            outcomes.append({k: _to_float(v) for k, v in result.full_stats.items()})
        except Exception as exc:
            ## One bad mix (for example an indicator length longer than the data)
            ## must not stop the whole grid. Its error text comes back instead.
            ## WARNING (not DEBUG): safe volume under fork; parent still sees the row.
            logger.warning("trial failed batch=%s params=%s err=%s", batch_index, params, repr(exc))
            outcomes.append(repr(exc))
    return batch_index, outcomes


def run_trials(
    param_combos: Sequence[Dict[str, Any]],
    simulator_kwargs: Dict[str, Any],
    *,
    n_jobs: int,
    show_progress: bool,
    build: Optional[BuildFn] = None,
    tickers: Sequence[str] = (),
    book_build: Optional[BookBuildFn] = None,
) -> List[Outcome]:
    """Test every settings dict, in parallel when safe.

    Pass exactly one of ``build`` (flat strategy list) or ``book_build``
    (``PortfolioBook``). Flat runs also need ``tickers``.

    Args:
        param_combos: The settings dicts to test.
        simulator_kwargs: Settings for every ``Simulator`` in this run.
        n_jobs: Worker processes. ``-1`` means one per CPU; ``1`` means no workers.
        show_progress: Show a progress bar counting finished batches.
        build: Builds one ticker's strategy from the settings being tried.
        tickers: Tickers that run together in every flat-strategy trial.
        book_build: Builds a ``PortfolioBook`` from each settings mix.

    Returns:
        One outcome per settings dict, in the same order as ``param_combos``.

    Raises:
        ValueError: If both or neither of ``build`` / ``book_build`` are given, or
            ``build`` is given without ``tickers``.
        RuntimeError: If another optimizer run is already in progress.
    """
    global _RUN
    if (build is None) == (book_build is None):
        raise ValueError("Pass exactly one of build or book_build")
    if build is not None and not tickers:
        raise ValueError("tickers is required when using build")
    ## Workers read whatever is in _RUN, so two runs at once would mix up their batches.
    if _RUN is not None:
        raise RuntimeError("Another optimizer run is already in progress; wait for it to finish.")

    n_workers = (os.cpu_count() or 1) if n_jobs == -1 else max(int(n_jobs), 1)
    ## No point starting more workers than there are settings to test.
    n_workers = min(n_workers, len(param_combos)) or 1
    if n_workers > 1:
        reason = fork_unsafe_reason()
        if reason is not None:
            ## Running in one process is slower but cannot crash. That beats a run
            ## that dies half way with a confusing system error.
            msg = f"Optimizer running in one process because {reason}."
            logger.warning(msg)
            warnings.warn(msg, stacklevel=3)
            n_workers = 1
    path = "book" if book_build is not None else f"flat tickers={list(tickers)}"
    logger.info(
        "run_trials start path=%s combos=%s batches_pending workers=%s stats_level=%s",
        path,
        len(param_combos),
        n_workers,
        simulator_kwargs.get("stats_level"),
    )
    if n_workers > 1:
        _warm_up(build, book_build, tickers, param_combos[0])

    param_batches = _make_batches(list(param_combos), n_workers)
    logger.debug("run_trials batches=%s max_batch_size=%s", len(param_batches), max(len(b) for b in param_batches))
    ## Must be filled in before the workers start: they only see what existed at fork time.
    _RUN = _RunState(
        build=build,
        book_build=book_build,
        tickers=tuple(tickers),
        simulator_kwargs=dict(simulator_kwargs),
        param_batches=param_batches,
    )
    ## One slot per batch, so answers can arrive in any order and still land in place.
    answers: List[Optional[List[Outcome]]] = [None] * len(param_batches)
    bar = tqdm(total=len(param_batches), desc="optimize", unit="batch", disable=not show_progress)
    try:
        if n_workers == 1:
            ## The same task function, just called here. This is also the easy path for
            ## debugging, and lets tests check both ways give identical numbers.
            for i in range(len(param_batches)):
                _, answers[i] = _mp_task(i)
                bar.update(1)
        else:
            ## Ask for fork by name: macOS defaults to spawn, which would start every
            ## worker from scratch and re-import everything (about 30 seconds each).
            pool = ProcessPoolExecutor(n_workers, mp_context=mp.get_context("fork"), initializer=_worker_init)
            try:
                futures = [pool.submit(_mp_task, i) for i in range(len(param_batches))]
                for future in as_completed(futures):
                    batch_index, outcomes = future.result()
                    answers[batch_index] = outcomes
                    bar.update(1)
            except BaseException:
                ## On Ctrl-C or a worker crash, drop batches that have not started yet.
                ## Batches already running finish first (workers ignore Ctrl-C).
                pool.shutdown(wait=True, cancel_futures=True)
                raise
            pool.shutdown(wait=True)
    finally:
        bar.close()
        _RUN = None
    ## Batches were cut in order, so laying them end to end restores the original order.
    outcomes = [outcome for batch in answers for outcome in batch]
    n_err = sum(1 for o in outcomes if isinstance(o, str))
    logger.info("run_trials done outcomes=%s errors=%s", len(outcomes), n_err)
    return outcomes
