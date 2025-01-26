from typing import List, Dict
from abc import ABC, abstractmethod
from pathos.multiprocessing import ProcessingPool as Pool
from pathos.multiprocessing import cpu_count
from pathos.pools import _ProcessPool
from threading import Thread
from functools import partial
from concurrent.futures import ThreadPoolExecutor

shutdown_event = False

def runProcesses(func, OrderedInputs: List[List], run_type: str = 'map') -> List:
    try:

        pool = Pool(20)
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


    finally:
        pool.close()
        pool.join()

    return results



def shutdown(pool):
    global shutdown_event
    shutdown_event
    shutdown_event = True
    pool.terminate()
