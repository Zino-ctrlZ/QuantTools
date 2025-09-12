"""
This module provides functionality for parallel processing using multiprocessing and threading.
"""
from typing import List
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time
import logging
from pathos.multiprocessing import ProcessingPool as Pool
from trade._multiprocessing import ensure_global_start_method, PathosPool
from trade.helpers.Logging import setup_logger
from trade import get_pool_enabled

logger = setup_logger('trade.helpers.pools', stream_log_level=logging.INFO)

shutdown_event = False
num_workers = int(os.environ.get('NUM_WORKERS', str(cpu_count())).strip())

def change_logger_stream_level(level):
    """
    Change the logger stream level.
    """
    logger.setLevel(level)
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setLevel(level)

def runProcesses(func, OrderedInputs: List[List], run_type: str = 'map') -> List:
    """
    Run multiprocessing on a given function.

    params:
    --------

    func: Function to run multiprocessing
    OrderedInputs: List of inputs to pass to the function. Must be ordered as below.
        [[input1, input1, input1], [input2, input2, input2], [input3, input3, input3]]

    run_type: Type of multiprocessing to run. Default is 'map'. Other options are 'amap', 'uimap', 'imap'

    returns:
    --------
    List of results from the multiprocessing function.
    if run_type is 'map', results are ordered as the inputs.
    if run_type is 'amap', results are ordered as the inputs. Asynchronous
    if run_type is 'uimap', results are unordered, and a list of futures is returned.
    if run_type is 'imap', results are ordered, and a list of futures is returned. Non blocking.
    """

    global shutdown_event
    ensure_global_start_method()
    try:
        pool = PathosPool(cpu_count())
        pool.restart(force=True)
        if run_type == 'map':
            results = pool.map(func, *OrderedInputs)
        elif run_type == 'amap':
            results = pool.amap(func, *OrderedInputs)
        elif run_type == 'uimap':
            results = pool.uimap(func, *OrderedInputs)
        elif run_type == 'imap':
            results = pool.imap(func, *OrderedInputs)

        else:
            raise ValueError(f'Run type {run_type} not recognized')

    except KeyboardInterrupt as e:

        shutdown_event = True
        shutdown(pool)
        raise

    except Exception as e:
        logger.error('Error occurred: %s', e)
        shutdown(pool)
        raise


    finally:
        pool.close()
        if run_type not in ['imap', 'uimap']:
            pool.join()

    return results



def shutdown(pool):
    global shutdown_event
    shutdown_event
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
    if not hasattr(data, 'itertuples'):
        raise ValueError("Data must be a DataFrame")


    if pool:
        logger.info("`parrallel_apply` using multiprocessing with %d workers", num_workers)
        logger.info("To change to threading, either set the environment POOL_ENABLED to False, or use `set_pool_enabled(False)` found in trade.__init__")
        logger.info("Logger stream level is set to %s. To change this behavior & reduce stream logs, use `change_logger_stream_level` found in trade.helpers.pools", logging.getLevelName(logger.level))
        shutdown_event = False
        try:
            ensure_global_start_method()
            with PathosPool(num_workers) as p:
                p.restart(force = True)
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
        logger.info("To change to multiprocessing, either set the environment POOL_ENABLED to True, or use `set_pool_enabled(True)` found in trade.__init__")
        logger.info("Logger stream level is set to %s. To change, use `change_logger_stream_level` found in trade.helpers.pools", logging.getLevelName(logger.level))
        results = [None] * len(data)
        with ThreadPoolExecutor() as executor:
            future_to_idx = {executor.submit(func, *row): i for i, row in enumerate(data.itertuples(index=False, name=None))}

            for future in as_completed(future_to_idx):
                i = future_to_idx[future]
                try:
                    results[i] = future.result(timeout=timeout)
                except Exception as e:
                    logger.error(f"Failed on row {i} ({data.iloc[i].to_dict()}): {e}")
                    results[i] = 0.0  # or np.nan

        return results

import signal

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