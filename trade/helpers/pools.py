from typing import List, Dict
from abc import ABC, abstractmethod
from pathos.multiprocessing import ProcessingPool as Pool
from multiprocessing import cpu_count
from pathos.multiprocessing import cpu_count
from pathos.pools import _ProcessPool
from threading import Thread
from functools import partial
from concurrent.futures import ThreadPoolExecutor

shutdown_event = False

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
        pool.join()

    return results



def shutdown(pool):
    global shutdown_event
    shutdown_event
    shutdown_event = True
    pool.terminate()
