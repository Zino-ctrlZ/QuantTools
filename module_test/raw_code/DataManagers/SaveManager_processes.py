## NOTES:

## 1. All objects are shared between processes using multiprocessing.Manager(). They are gotten from .shared_objs file
## 2. The worker processes are created using multiprocessing.Process and are started in the start_workers method
## 3. Qsize is shared between processes using multiprocessing.Value. But the status method under 'pending_tasks' uses cls._queue.qsize() which is not shared.
## 4. Current Context is set to 'fork' to avoid issues with pickling. This is important for the multiprocessing module to work correctly.
## 5. Disabled using cls.__increase_qsize() and cls.__decrease_qsize() in the enqueue and schedule methods. This is because had issues with accounting for the size of the queue.
## 6. SaveManager Worker uses Threads. Within each thread it sets mp.context as `forkserver` to avoid issues with pickling. 
## 7. Introduced scheduling requests to avoid resource overload in real time.
## 8. Added a timeout for the save function to avoid hanging.
## 9. Added a function to reset signals in the worker process to avoid issues with signal handling.
## 10. Current requests is reset to an empty dictionary when the workers are restarted.

from queue import Full, Empty
from trade.helpers.helper import setup_logger
import os, sys
import json
import pandas as pd
import multiprocessing as mp
from copy import deepcopy
import pickle
from copy import deepcopy
from .Requests import create_request_bulk, construct_request_name
from .shared_obj import (get_shared_queue, 
                         get_shared_dict, 
                         get_shared_list, 
                         get_int_value,
                         get_shared_lock, 
                         get_request_list,
                         )
import multiprocessing as mp
from functools import partial
import traceback
import time
from .SaveManager import( _enqueue, 
                         _schedule,
                         _remove_request_from_list,
                         _status,)
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout


logger = setup_logger("DataManager.SaveManager_process")
timer = setup_logger("DataManager.SaveManager_process.Timer")
# CTX = mp.get_context("fork")
JOB_TIMEOUT = 60 * 10  # 10 minutes
JOB_TIMEOUTS = {
    'bulk': 60 * 60 * 3,  # 3 hours
    'single': 60 * 10,  # 10 minutes
    'chain': 60 * 60 * 1,  # 1 hour
    'default': 60 * 10,  # 10 minutes
}

def flatten_all_dfs(request):
    """
    Flattens all dataframes in the request object.
    """
    for key, value in request.__dict__.items():
        if isinstance(value, pd.DataFrame):
            request.__dict__[key] = value.to_dict(orient="records")
        elif isinstance(value, pd.Series):
            request.__dict__[key] = value.to_dict()
    return request


def save_failed_request(request, filename='requests_json/failed_request.jsonl'):
    """
    Saves the failed request to a JSON file.
    """
    request = flatten_all_dfs(request)
    with open(f'{os.environ["WORK_DIR"]}/module_test/raw_code/DataManagers/{filename}', 'a') as f:
        json.dump(request.__dict__, f, default=str)
        f.write('\n')

def is_pickleable(obj) -> bool:
    try:
        pickle.dumps(obj)
    except Exception:
        return False
    return True

def safe_prepare_request(request):
    if is_pickleable(request):
        return request

    safe_request = deepcopy(request)

    # Dangerous fields
    dangerous_attrs = ['eod', 'intra', 'Stock', 'spot_manager', 'vol_manager', 'chain_manager']

    if hasattr(request, "_non_pickle_fields"):
        dangerous_attrs.extend(request._non_pickle_fields)

    for attr in dangerous_attrs:
        if hasattr(safe_request, attr):
            setattr(safe_request, attr, None)

    if not is_pickleable(safe_request):
        raise ValueError(f"Request {type(request)} still not pickleable after cleaning.")

    return safe_request




#### Save Manager (From ChatGPT)
class ProcessSaveManager:
    MAX_QUEUE_SIZE = 100
    WORKER_COUNT = 4
    
    _queue = None
    _threads = []
    _started = False
    _finished_requests = None
    _lock = None
    _current_requests = None
    _failed_requests = None
    _qsize = None
    _failed_initialization = None
    _initialized = False

    @classmethod
    def auto_setup(cls):
        if not cls._initialized:
            from .shared_obj import setup_shared_objects
            setup_shared_objects()
            cls.initialize()
            cls.start_workers()
            cls._initialized = True
            print("[ProcessSaveManager] Auto-setup completed. Workers started.")



    @classmethod
    def initialize(cls):
        cls._queue = get_shared_queue()
        cls._finished_requests = get_shared_list("finished_requests")
        cls._lock = get_shared_lock()
        cls._current_requests = get_shared_dict()
        cls._failed_requests = get_shared_list("failed_requests")
        cls._qsize = get_int_value()
        cls._failed_initialization = get_shared_list("failed_initialization")


    # @classmethod
    # def _worker(cls):
    #     # cls.initialize()
    #     while True:
    #         from .shared_obj import setup_shared_objects
    #         setup_shared_objects()  # ensure all proxies are re-bound in this process
    #         try:

    #             start = time.time()
    #             thread_name = mp.current_process().name ## Get the name of the current thread
    #             kwargs = cls._queue.get()
    #             cls.__decrease_qsize() ## Decrease after getting the request
    #             if kwargs is None: ## This is the signal to stop the worker 
    #                 break
    #             save_func = kwargs.pop('save_func')                
    #             request = create_request_bulk(**kwargs)
    #             if request is None:
    #                 cls._queue.task_done()
    #                 continue

    #             ## Set current request before processing
    #             with cls._lock: ## Ensures that only one thread can access this block at a time
    #                 cls._current_requests[thread_name] = request
    #                 print(f"[ProcessSaveManager] Current requests: {cls._current_requests}, inside lock")
    #                 print(f"[ProcessSaveManager] Processing save request for {request.symbol} on {request}, thread {thread_name}")
                
    #             print(request)
    #             print(f"Worker {thread_name} got a request")
    #             print(f"Worker {thread_name} got a request: {request}")
    #         except Exception as e:
    #             ## Removing request incase it failed in initialization
    #             _remove_request_from_list(kwargs)
    #             logger.error(f"[ProcessSaveManager] Error getting request from queue: {e}")
    #             logger.error(f"Error processing event: {e}\n{traceback.format_exc()}")
    #             cls._failed_initialization.append(f"{kwargs['tick']}_{kwargs['exp']}: {e}\n{traceback.format_exc()}")
    #             cls.schedule(kwargs)
    #             cls._queue.task_done()
    #             continue

    #         try:
    #             save_func(request)
    #             end = time.time()
    #             timer.info(f"Request Dict: {request.__dict__}, Class Name: {request.__class__.__name__}")
    #             timer.info(f"[ProcessSaveManager] Worker {thread_name} finished processing request in {end - start:.2f} seconds.")
                
    #             with cls._lock:
    #                 cls._finished_requests.append(request)
    #                 del cls._current_requests[thread_name]
                    
    #         except Exception as e:
    #             _remove_request_from_list(kwargs)
    #             logger.error(f"[SaveWorker] Error processing save: {e}")

    #             with cls._lock:
    #                 request.error = f"{e}\n{traceback.format_exc()}"
    #                 request.class_name = request.__class__.__name__
    #                 cls._failed_requests.append(request)
    #                 save_failed_request(request)
    #                 del cls._current_requests[thread_name]
    #         finally:
    #             cls._queue.task_done()

    @classmethod
    def _worker(cls):
        from .shared_obj import setup_shared_objects
        setup_shared_objects()
        while True:
            # 1) Pull a job, but don't block forever
            try:
                kwargs = cls._queue.get(timeout=5)
                job_type = kwargs.get('type_', 'default')
                print(f"[{mp.current_process().name}] Got job: {job_type}")
                timeout = JOB_TIMEOUTS.get(job_type, JOB_TIMEOUT)
                print(f"[{mp.current_process().name}] Timeout for job: {timeout}")
                # cls.__decrease_qsize()
            except Empty as e:
                print(f"[{mp.current_process().name}] Queue is empty, continuing...")
                continue


            # 2) Shutdown signal
            if kwargs is None:
                break

            # 3) Unpack the save function and build request
            save_func = kwargs.pop('save_func')
            try:
                request = create_request_bulk(**kwargs)
                if request is None:
                    cls._queue.task_done()
                    continue

                # track current
                with cls._lock:
                    proc_name = mp.current_process().name
                    cls._current_requests[proc_name] = request

            except Exception as e:
                _remove_request_from_list(kwargs)
                logger.error(f"Init error: {e}\n{traceback.format_exc()}")
                cls._failed_initialization.append(f"{kwargs}: {e}")
                cls.schedule(kwargs)
                cls._queue.task_done()
                continue

            # 4) Execute with per‐job timeout
            start = time.time()
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(save_func, request)
                    future.result(timeout=timeout)

                duration = time.time() - start
                timer.info(f"[{proc_name}] Finished in {duration:.2f}s: {request}")
                print(f"[{proc_name}] Finished in {duration:.2f}s: {request}")

                # mark success
                with cls._lock:
                    cls._finished_requests.append(request)
                    del cls._current_requests[proc_name]

            except FuturesTimeout:
                logger.error(f"[{proc_name}] save_func timeout after {JOB_TIMEOUT}s")
                print(f"[{proc_name}] save_func timeout after {JOB_TIMEOUT}s")
                _remove_request_from_list(kwargs)
                with cls._lock:
                    request.error = f"Timed out after {timeout}s"
                    request.class_name = request.__class__.__name__
                    cls._failed_requests.append(request)
                    save_failed_request(request, filename = 'requests_json/timed_out_requests.jsonl')
                    del cls._current_requests[proc_name]

            except Exception as e:
                logger.error(f"[{proc_name}] save_func error: {e}\n{traceback.format_exc()}")
                _remove_request_from_list(kwargs)
                with cls._lock:
                    request.error = f"{e}"
                    request.class_name = request.__class__.__name__
                    cls._failed_requests.append(request)
                    save_failed_request(request)
                    del cls._current_requests[proc_name]

            finally:
                cls._queue.task_done()

    @classmethod
    def start_workers(cls):
        if cls._started:
            return
        cls.initialize()
        for i in range(cls.WORKER_COUNT):
            t = CTX.Process(target=cls._worker, daemon=False, name =f"SaveWorker-{i}")
            # t = Thread(target=cls._worker, daemon=True, name =f"SaveWorker-{i}")
            t.start()
            t.name = f"SaveWorker-{t.pid}"
            cls._threads.append(t)
        cls._started = True
        logger.info(f"[ProcessSaveManager] Started {cls.WORKER_COUNT} save workers.")


    @classmethod
    def kill_workers(cls): 
        for p in cls._threads:
            p.terminate()
            p.join()
        logger.info("[ProcessSaveManager] All save workers have been killed.")
        print(("[ProcessSaveManager] All save workers have been killed."))

    @classmethod
    def restart_workers(cls):
            cls.kill_workers() 
            cls.reset_variables()
            cls.start_workers()
            ## If restarting workers, we need to clear the current requests
            if cls._current_requests:
                for worker in cls._current_requests:
                    del cls._current_requests[worker]
            logger.info("[ProcessSaveManager] Restarted all save workers.")
            print(("[ProcessSaveManager] Restarted all save workers."))

    @classmethod
    def reset_variables(cls):
        cls.initialize()
        cls._threads = []
        cls._started = False
        logger.info("[ProcessSaveManager] Reset all variables.")
        print(("[ProcessSaveManager] Reset all variables."))

    @classmethod
    def enqueue(cls, kwargs):
        """
        Enqueue a save request.
        :param kwargs: Dictionary of parameters to pass to the save function.
            kwargs should contain both keyword arguments for the save function and the save function itself.

        :return: None
        
        """
        kwargs['save_func'] = partial(kwargs['save_func'], pool = False)
        cls.auto_setup()
        _enqueue(cls, kwargs)
        # cls.__increase_qsize() ## Increase after putting the request

    @classmethod
    def schedule(cls, kwargs):
        """
        Schedule a save request.
        :param kwargs: Dictionary of parameters to pass to the save function.
            kwargs should contain both keyword arguments for the save function and the save function itself.

        :return: None
        """
        _schedule(cls, kwargs)
        
    @classmethod
    def __increase_qsize(cls):
        with cls._lock:
            cls._qsize.value += 1
    
    @classmethod
    def __decrease_qsize(cls):
        with cls._lock:
            cls._qsize.value -= 1

    @classmethod
    def status(cls):
        return _status(cls)
        # return {
        #     "pending_tasks": cls._qsize.value,
        #     "max_queue_size": cls.MAX_QUEUE_SIZE,
        #     "active_processes": sum(t.is_alive() for t in cls._threads),
        #     "total_processes": len(cls._threads),
        #     "current_requests": dict(cls._current_requests.items()),
        #     "num_finished_requests": len(cls._finished_requests),
        #     "num_failed_requests": len(cls._failed_requests),
        #     "failed_initialization": len(cls._failed_initialization),
        # }

## Use to reset signals in the worker process
def reset_signals_in_worker():
    import signal
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_DFL)


if __name__ == "__main__":
    from .shared_obj import get_manager, setup_shared_objects
    setup_shared_objects()
    get_manager()
    ProcessSaveManager.initialize()
    ProcessSaveManager.start_workers()