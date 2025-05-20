from typing import List, Dict
from abc import ABC, abstractmethod
from pathos.multiprocessing import ProcessingPool as Pool
from multiprocessing import cpu_count
from pathos.multiprocessing import cpu_count
from pathos.pools import _ProcessPool
from threading import Thread
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed
from trade.helpers.Logging import setup_logger
from trade import POOL_ENABLED
import os, sys
import time
logger = setup_logger('trade.helpers.pools')

shutdown_event = False
num_workers = int(os.environ.get('NUM_WORKERS', str(cpu_count())).strip())

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
    if run_type is 'amap', results are ordered as the inputs.
    if run_type is 'uimap', results are unordered, and a list of futures is returned.
    if run_type is 'imap', results are ordered, and a list of futures is returned
    """

    global shutdown_event
    try:

        pool = Pool(cpu_count())
        pool.restart()
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
        print('Error occured: ', e)
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




def parallel_apply(data, func, timeout=60, pool = POOL_ENABLED):
    """
    Apply a function to a DataFrame in parallel using multiprocessing.
    """
    global shutdown_event

    # Check if the function is callable
    if not callable(func):
        raise ValueError("Function must be callable")
    
    # Check if the data is a DataFrame
    if not hasattr(data, 'itertuples'):
        raise ValueError("Data must be a DataFrame")


    if pool:
        logger.info("Using multiprocessing in parallel_apply")
        shutdown_event = False
        logger.info(f"Using multiprocessing with {cpu_count()} cores")
        try:
            with Pool(num_workers) as p:
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
        logger.info("Using threading in parallel_apply")
        results = [None] * len(data)
        with ThreadPoolExecutor() as executor:
            future_to_idx = {executor.submit(func, *row): i for i, row in enumerate(data.itertuples(index=False, name=None))}

            for future in as_completed(future_to_idx):
                i = future_to_idx[future]
                try:
                    results[i] = future.result(timeout=timeout)
                except Exception as e:
                    print(f"Failed on row {i} ({data.iloc[i].to_dict()}): {e}")
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