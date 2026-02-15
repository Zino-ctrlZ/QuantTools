"""
This module provides functionality for parallel processing using multiprocessing and threading.
"""
import signal
from typing import Any, Callable, List, Optional
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import os
import time
import logging
from pathos.multiprocessing import ProcessingPool as Pool
from trade._multiprocessing import ensure_global_start_method, PathosPool
from trade.helpers.Logging import setup_logger, change_logger_stream_level
from trade import get_pool_enabled

logger = setup_logger("trade.helpers.pools", stream_log_level=logging.INFO)

shutdown_event = False
num_workers = int(os.environ.get("NUM_WORKERS", str(cpu_count())).strip())


def _change_global_stream_level(level: int | str) -> None:
    """
    Change the global logger stream level.
    """
    change_logger_stream_level(logger, level)


def _run_processes_pathos(
    func: Callable[..., Any],
    ordered_inputs: List[List[Any]],
    run_type: str,
) -> List[Any]:
    """Run processes using Pathos backend.

    Args:
        func: Function to run in parallel.
        ordered_inputs: Inputs for the function, grouped by argument position.
        run_type: Execution mode: map, amap, uimap, imap.

    Returns:
        Results from the multiprocessing function.

    Raises:
        ValueError: If run_type is not supported.
    """
    global shutdown_event
    ensure_global_start_method()
    try:
        pool = PathosPool(cpu_count())
        pool.restart(force=True)
        if run_type == "map":
            results = pool.map(func, *ordered_inputs)
        elif run_type == "amap":
            results = pool.amap(func, *ordered_inputs)
        elif run_type == "uimap":
            results = pool.uimap(func, *ordered_inputs)
        elif run_type == "imap":
            results = pool.imap(func, *ordered_inputs)
        else:
            raise ValueError(f"Run type {run_type} not recognized")
    except KeyboardInterrupt:
        shutdown_event = True
        shutdown(pool)
        raise
    except Exception as exc:
        logger.error("Error occurred: %s", exc)
        shutdown(pool)
        raise
    finally:
        pool.close()
        if run_type not in ["imap", "uimap"]:
            pool.join()

    return results


def _run_processes_futures(
    func: Callable[..., Any],
    ordered_inputs: List[List[Any]],
    run_type: str,
    max_workers: Optional[int] = None,
) -> List[Any]:
    """Run processes using concurrent.futures backend.

    Args:
        func: Function to run in parallel.
        ordered_inputs: Inputs for the function, grouped by argument position.
        run_type: Execution mode: map, amap, uimap, imap.
        max_workers: Optional override for thread workers.

    Returns:
        Results or futures from the process backend.

    Raises:
        ValueError: If run_type is not supported.
    """
    if run_type not in ["map", "amap", "uimap", "imap"]:
        raise ValueError(f"Run type {run_type} not recognized")

    ensure_global_start_method()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        if run_type == "map":
            return list(executor.map(func, *ordered_inputs))

        futures = [executor.submit(func, *args) for args in zip(*ordered_inputs)]
        if run_type in ["amap", "imap", "uimap"]:
            return futures

    return []


def runProcesses(
    func: Callable[..., Any],
    OrderedInputs: List[List[Any]],
    run_type: str = "map",
    *,
    backend: str = "pathos",
) -> List[Any]:
    """Run parallel execution using a selectable backend.

    Args:
        func: Function to run in parallel.
        OrderedInputs: Inputs to pass to the function.
            Example: [[in1, in1], [in2, in2], [in3, in3]]
        run_type: Execution mode: map, amap, uimap, imap.
        backend: Backend selector: pathos or futures.

    Returns:
        Results or futures depending on the backend and run_type.

    Raises:
        ValueError: If backend is not supported.

    Examples:
        >>> results = runProcesses(func, [[1, 2], [3, 4]], backend='pathos')
        >>> futures = runProcesses(func, [[1, 2], [3, 4]], run_type='imap', backend='futures')
    """
    backend_norm = backend.strip().lower()
    if backend_norm == "pathos":
        return _run_processes_pathos(func, OrderedInputs, run_type)
    if backend_norm in ["futures", "concurrent.futures", "threading"]:
        return _run_processes_futures(func, OrderedInputs, run_type)

    raise ValueError(f"Backend {backend} not recognized")


def shutdown(pool):
    global shutdown_event
    shutdown_event = True
    pool.terminate()


def parallel_apply(data: List[List], func: callable, timeout: int = 60, pool: Pool = None):
    """
    Apply a function to a DataFrame in parallel using multiprocessing.
    """
    global shutdown_event

    ## Set pool
    if pool is None:
        pool = get_pool_enabled()

    # Check if the function is callable
    if not callable(func):
        raise ValueError("Function must be callable")

    # Check if the data is a DataFrame
    if not hasattr(data, "itertuples"):
        raise ValueError("Data must be a DataFrame")

    if pool:
        logger.info("`parrallel_apply` using multiprocessing with %d workers", num_workers)
        logger.info(
            "To change to threading, either set the environment POOL_ENABLED to False, or use `set_pool_enabled(False)` found in trade.__init__"
        )
        logger.info(
            "Logger stream level is set to %s. To change this behavior & reduce stream logs, use `_change_global_stream_level` found in trade.helpers.pools",
            logging.getLevelName(logger.level),
        )
        shutdown_event = False
        try:
            ensure_global_start_method()
            with PathosPool(num_workers) as p:
                p.restart(force=True)
                logger.info("Starting Function with multiprocessing")
                start = time.time()
                # func = reset_signals_wrapper(func)
                results = p.map(func, *data.T.to_numpy())
                logger.info(f"Function completed in {time.time() - start} seconds")
                return results
        except KeyboardInterrupt:
            shutdown_event = True
            shutdown(p)
            raise
        except Exception as e:
            logger.error(f"Error occurred: {e}")
            shutdown_event = True
            shutdown(p)
            raise
        finally:
            shutdown_event = True
            shutdown(p)
            p.close()
            p.join()
        return results

    else:
        logger.info("`parrallel_apply` using threading with %d workers", num_workers)
        logger.info(
            "To change to multiprocessing, either set the environment POOL_ENABLED to True, or use `set_pool_enabled(True)` found in trade.__init__"
        )
        logger.info(
            "Logger stream level is set to %s. To change, use `_change_logger_stream_level` found in trade.helpers.pools",
            logging.getLevelName(logger.level),
        )
        results = [None] * len(data)
        with ThreadPoolExecutor() as executor:
            future_to_idx = {
                executor.submit(func, *row): i for i, row in enumerate(data.itertuples(index=False, name=None))
            }

            for future in as_completed(future_to_idx):
                i = future_to_idx[future]
                try:
                    results[i] = future.result(timeout=timeout)
                except Exception as e:
                    logger.error(f"Failed on row {i} ({data.iloc[i].to_dict()}): {e}")
                    results[i] = 0.0  # or np.nan

        return results


def reset_signals_wrapper(func):
    def wrapped(*args, **kwargs):
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        return func(*args, **kwargs)

    return wrapped


# import multiprocessing as mp
# import time

# ctx = mp.get_context('fork')
# p = ctx.Process(target=save_to_database, args=(request_current, manager.db, manager))
# p.start()
