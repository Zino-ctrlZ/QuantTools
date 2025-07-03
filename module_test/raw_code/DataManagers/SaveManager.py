## Notes:
## Refer to .SaveManager_process.py for all SaveManager related notes

import multiprocessing as mp
from queue import Queue, Full
import threading
from threading import Thread, Lock
from trade.helpers.helper import setup_logger
import os, sys
import json
import pandas as pd
import pickle
from copy import deepcopy
from trade.helpers.helper import check_all_days_available
from .Requests import create_request_bulk,construct_request_name, OptionQueryRequestParameter, BulkOptionQueryRequestParameter
from dbase.database.SQLHelpers import DatabaseAdapter
from functools import partial
from .shared_obj import (get_shared_queue, 
                         get_shared_dict, 
                         get_shared_list, 
                         get_int_value,
                         get_shared_lock, 
                         get_request_list)
import time
print('\n')
print("Scheduled Data Requests will be saved to:", f"{os.environ['WORK_DIR']}/module_test/raw_code/DataManagers/scheduler/requests.jsonl")
_SCHEDULED_NAMES = []
folder_path = f"{os.environ['WORK_DIR']}/module_test/raw_code/DataManagers/scheduler"


## Using names created from current requests, so if older names get into requests.jsonl, we can still process them
## If we used scheduled_names file, we would have skipped them
def construct_current_scheduled_names():
    """
    Load the current scheduled names from the file.
    """
    global _SCHEDULED_NAMES
    try:
        with open(f"{folder_path}/requests.jsonl", "r") as f:
            stripped = [request.strip() for request in f.readlines()]
            try:
                json_list = [json.loads(request) for request in stripped]
                
            except:
                print("Error Loading with list comprehension, will use loop to catch error")
                for request in stripped:
                    try:
                        json_list.append(json.loads(request))
                    except:
                        print(f"Request Threw an error: {request}")

        ## Check if the request has all the keys
        json_list = [req for req in json_list if all(key in req for key in ['start', 'end', 'tick'])]
        _SCHEDULED_NAMES = [construct_request_name(**request) for request in json_list]
        


    except FileNotFoundError:
        raise FileNotFoundError("The file scheduler/requests.jsonl does not exist. Please create it first.")

construct_current_scheduled_names()

logger = setup_logger("DataManagers.SaveManager")
timer  = setup_logger("DataManagers.SaveManager.Timer")
def flatten_all_dfs(request):
    """
    Flattens all dataframes in the request object or dictionaries.
    """
    if not isinstance(request, dict):
        for key, value in request.__dict__.items():
            if isinstance(value, pd.DataFrame):
                request.__dict__[key] = value.to_dict(orient="records")
            elif isinstance(value, pd.Series):
                request.__dict__[key] = value.to_dict()

    elif isinstance(request, dict):
        for k, v in request.items():
            if isinstance(v, pd.DataFrame):
                request[k] = v.to_dict(orient="records")
            elif isinstance(v, pd.Series):
                request[k] = v.to_dict()
    else:
        raise ValueError("Request is not a valid object or dictionary.")
    return request


def save_failed_request(request, filename='requests_json/failed_request.jsonl'):
    """
    Saves the failed request to a JSON file.
    """
    request = flatten_all_dfs(request)
    with open(f'{os.environ["WORK_DIR"]}/module_test/raw_code/DataManagers/{filename}', 'a') as f:
        json.dump(request.__dict__, f, default=str)
        f.write('\n')





#### Save Manager (From ChatGPT)
class SaveManager:
    MAX_QUEUE_SIZE = 100
    WORKER_COUNT = 4
    _queue = Queue(maxsize=MAX_QUEUE_SIZE)
    _threads = None
    _started = False
    _finished_requests = None
    _lock = None
    _current_requests = None
    _failed_requests = None
    _kill_thread = None
    _failed_initialization = None

    @classmethod
    def initialize(cls):
        from .shared_obj import setup_shared_objects
        setup_shared_objects()
        cls._queue = Queue(maxsize=cls.MAX_QUEUE_SIZE)
        cls._threads = []
        cls._started = False
        cls._finished_requests = []
        cls._lock = Lock()
        cls._current_requests = {}
        cls._failed_requests = []
        cls._kill_thread = None
        cls._failed_initialization = []


    @classmethod
    def auto_setup(cls):
        """
        Automatically sets up the shared objects.
        """
        if cls._started:
            return
        cls.initialize()
        cls.start_workers()
        print("[SaveManager] Auto setup complete. Workers started.")

    @classmethod
    def _worker(cls):
        mp.set_start_method('forkserver', force=True) ## Set the start method to forkserver for multiprocessing
        ### Threads have a hard time with fork
        while True:
            ## Assembling the request
            try:
                ## If Error occurs, we want to log it and continue

                start = time.time()
                thread_name = threading.current_thread().name ## Get the name of the current thread
                kwargs = cls._queue.get()
                print(f"{thread_name} is processing request {kwargs}")
                print("Request recieved successfully, current size of queue is", cls._queue.qsize())

                if kwargs is None: ## This is the signal to stop the worker 
                    break
                save_func = kwargs.pop('save_func')
                request = create_request_bulk(**kwargs)
                print("Request created successfully")

                if request is None: ## This means the request has been created before
                    logger.error(f"[SaveWorker] Request is None for {kwargs['tick']}_{kwargs['exp']}")
                    cls._queue.task_done()
                    continue


            except Exception as e:
                logger.error(f"[SaveWorker] Error getting request from queue: {e}")
                cls._failed_initialization.append(f"{kwargs['tick']}_{kwargs['exp']}: {e}")
                cls.schedule(kwargs)
                _remove_request_from_list(kwargs)
                cls._queue.task_done()
                continue
            
            ## Executing the request
            try:
                with cls._lock: ## Ensures that only one thread can access this block at a time
                    cls._current_requests[thread_name] = request
                print("Request is being processed")
                save_func(request)
                end = time.time()
                timer.info(f"Request Dict: {request.__dict__}, Class Name: {request.__class__.__name__}")
                timer.info(f"[SaveWorker] Finished processing save request for {request.symbol} on {request}, thread {thread_name} in {end - start:.2f} seconds")
                print(f"Took {end - start:.2f} seconds to process request {request.symbol} on {request}, thread {thread_name}")
                with cls._lock:
                    cls._finished_requests.append(request)
                    del cls._current_requests[thread_name]
                    
            except Exception as e:
                logger.error(f"[SaveWorker] Error processing save: {e}")
                _remove_request_from_list(kwargs)

                with cls._lock:
                    cls._failed_requests.append(request)
                    request.error = e
                    request.class_name = request.__class__.__name__
                    save_failed_request(request)
                    del cls._current_requests[thread_name]

            finally:
                cls._queue.task_done()

    @classmethod
    def start_workers(cls):
        if cls._started:
            return
        for i in range(cls.WORKER_COUNT):
            t = Thread(target=cls._worker, daemon=True, name =f"SaveWorker-{i}")
            t.start()
            cls._threads.append(t)
        cls._started = True
        logger.info(f"[SaveManager] Started {cls.WORKER_COUNT} save workers.")


    @classmethod
    def kill_workers(cls): 
        def _kill():
            for _ in range(cls.WORKER_COUNT):
                cls._queue.put(None)
            for t in cls._threads:
                t.join()
            cls._started = False
            logger.info("[SaveManager] All save workers have been killed.")
            print(("[SaveManager] All save workers have been killed."))
        if not cls._kill_thread:
            thread = Thread(target=_kill)
            thread.start()
            return thread


    @classmethod
    def restart_workers(cls):
        
        ## We're implementing a restart that's non-blocking.
        if not cls._kill_thread: ## Check if there is a current kill thread running
            kill_thread = cls.kill_workers()
            cls._kill_thread = kill_thread
        else:
            kill_thread = cls._kill_thread

        ## Check if the kill thread is alive
        if kill_thread.is_alive():
            logger.info("[SaveManager] Kill thread is still running. Cannot restart workers yet.")
            print(("[SaveManager] Kill thread is still running. Cannot restart workers yet."))
            return
        else:
            cls._kill_thread = None
            cls._threads = []
            cls.initialize() ## Reset the variables
            cls.start_workers()
            logger.info("[SaveManager] Restarted all save workers.")
            print(("[SaveManager] Restarted all save workers."))

    @classmethod
    def enqueue(cls, kwargs):
        """
        Enqueue a save request.
        :param kwargs: Dictionary of parameters to pass to the save function.
            kwargs should contain both keyword arguments for the save function and the save function itself.

        :return: None
        
        """
        cls.auto_setup()
        _enqueue(cls, kwargs)

    @classmethod
    def schedule(cls, kwargs):
        """
        Schedule a request to be run later
        """
        _schedule(cls, kwargs)        
    @classmethod
    def status(cls):
        return _status(cls)

def _status(cls):
    """
    Get the status of the SaveManager/ProcessSaveManager.
    :return: Dictionary with the status of the SaveManager/ProcessSaveManager.
    """
    return {
        "pending_tasks": cls._queue.qsize(), # if cls.__name__ == "SaveManager" else cls._qsize.value,
        "max_queue_size": cls.MAX_QUEUE_SIZE,
        "active_workers": sum(t.is_alive() for t in cls._threads),
        "total_workers": len(cls._threads) ,
        "current_requests": cls._current_requests if cls.__name__ == "SaveManager" else dict(cls._current_requests.items()),
        "num_finished_requests": len(cls._finished_requests),
        "num_failed_requests": len(cls._failed_requests),
        "num_failed_initialization": len(cls._failed_initialization),
    }


def _enqueue(cls, kwargs):
    """
    Enqueue a save request.
    :param kwargs: Dictionary of parameters to pass to the save function.
        kwargs should contain both keyword arguments for the save function and the save function itself.

    :return: None
    """
    kwargs = deepcopy(kwargs)
    if 'save_func' not in kwargs:
        raise ValueError("save_func not in kwargs")
    try:
        kwargs['_requests'] = get_request_list()
        print_attr = {x: y for x, y in kwargs.items() if x not in ['set_attributes']}
        print(f"[{cls.__name__}] Enqueueing save request for {kwargs['tick']} on {print_attr}")
        cls._queue.put(kwargs, block=False)  # Will raise Full if over limit
    except Full:
        logger.warning(f"[{cls.__name__}] Save queue full (max {cls.MAX_QUEUE_SIZE}). Task ignored.")


def is_already_scheduled(kwargs):
    """
    Check if a request is already scheduled.
    :param kwargs: Dictionary of parameters to pass to the save function.
        kwargs should contain both keyword arguments for the save function and the save function itself.
    :return: True if the request is already scheduled, False otherwise.
    """
    
    # construct_current_scheduled_names() ## Update the scheduled names
    global _SCHEDULED_NAMES
    request_name = construct_request_name(**kwargs)
    if request_name in _SCHEDULED_NAMES:
        logger.info(f"[SaveManager] Request {request_name} is already scheduled.")
    else:
        logger.info(f"[SaveManager] Request {request_name} is not scheduled.")
        _SCHEDULED_NAMES.append(request_name)
    return request_name in _SCHEDULED_NAMES

def write_to_requests_jsonl(kwargs):
    """
    Write the request to a jsonl file.
    """
    with open(f"{folder_path}/requests.jsonl", "a") as f:
        json.dump(kwargs, f, default=str)
        f.write('\n')

def format_schedule_kwargs(kwargs):
    """
    Format the kwargs for the schedule function.
    """

    if kwargs['type_'] in ['bulk', 'single']:
        kwargs['save_func'] = 'save_to_database'
    elif kwargs['type_'] in ['chain']:
        kwargs['save_func'] = 'save_chain_data'

    if kwargs['type_'] == 'chain':
        kwargs['set_attributes']['post_processed_data'] = kwargs['set_attributes']['post_processed_data'].to_dict(orient="records")
    
    return kwargs


def _schedule(cls, kwargs):
    """
    _schedule saves request in a jsonl file to be ran at a latter time
    :param kwargs: Dictionary of parameters to pass to the save function.
        kwargs should contain both keyword arguments for the save function and the save function itself.
    """
    global _SCHEDULED_NAMES
    print_attr = {x: y for x, y in kwargs.items() if x not in ['set_attributes', '_requests', 'save_func']}
    logger.info(f"[{cls.__name__}] is scheduling a request for {kwargs['tick']} on {print_attr}")
    if kwargs['type_'] in ['bulk', 'single']:
        kwargs['save_func'] = 'save_to_database'
    elif kwargs['type_'] in ['chain']:
        kwargs['save_func'] = 'save_chain_data'

    if kwargs['type_'] == 'chain':
        if isinstance(kwargs['set_attributes']['post_processed_data'], pd.DataFrame):
            kwargs['set_attributes']['post_processed_data'] = kwargs['set_attributes']['post_processed_data'].to_dict(orient="records")
        elif not isinstance(kwargs['set_attributes']['post_processed_data'], list):
            logger.error(f"[{cls.__name__}] post_processed_data is not a DataFrame or list. It's a {type(kwargs['set_attributes']['post_processed_data'])}.")
            return False
    
    if is_already_scheduled(kwargs):
        logger.info("Already scheduled, not scheduling")
        logger.warning(f"[{cls.__name__}] Request {kwargs['tick']} on {print_attr} is already scheduled. Ignoring.")
        return False
    write_to_requests_jsonl(kwargs)

    ## Considering not doing a data availability check
        ## Reason 1: It takes ~5-10 seconds PER check
        ## Reason 2: Want to avoid too many threads
    ## If it is already in db, the processing script will catch it
    # def save_func(kwargs):
    #     if is_single_already_in_db(kwargs):
    #         logger.info(f"[{cls.__name__}] Request {kwargs['tick']} on {print_attr} is already in DB, not scheduling")
    #         return
    #     write_to_requests_jsonl(kwargs)
    #     with open(f"{folder_path}/scheduled_names.txt", "a") as f:
    #         request_name = construct_request_name(**kwargs)
    #         f.write(f"{request_name}\n")
    # schedule_thread = Thread(target=save_func, args=(kwargs,))
    # schedule_thread.start()


def _remove_request_from_list(kwargs):
    """
    Remove a request from the request list.
    :param kwargs: Dictionary of parameters to pass to the save function.
        kwargs should contain both keyword arguments for the save function and the save function itself.
    """
    req_name = construct_request_name(**kwargs)
    if req_name in get_request_list():
        get_request_list().remove(req_name)
        logger.info(f"[SaveManager] Removed request {req_name} from request list.")
    else:
        logger.warning(f"[SaveManager] Request {req_name} not found in request list.")



def enqueue_save_request(
        tick: str,
        exp: str,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        save_func: callable,
        **kwargs
):
    """
    Enqueue a save request.
    :param tick: Ticker symbol
    :param exp: Expiration date
    :param start: Start date
    :param end: End date
    :param save_func: Function to save the data
    :param kwargs: Additional arguments for the save function
    """
    kwargs['tick'] = tick
    kwargs['exp'] = exp
    kwargs['start'] = start
    kwargs['end'] = end
    kwargs['save_func'] = save_func


def is_single_already_in_db(req):
    """
    Check if the data has already been saved to the database.
    """
    from .DataManagers import init_query
    # Check if the data has already been saved
    db = DatabaseAdapter()
    if isinstance(req,OptionQueryRequestParameter):
        pass
    elif isinstance(req, dict):
        req["_requests"] = []
        req = create_request_bulk(**req)
    else:
        ## Anything else, we return false for convenience
        return False
    
    database_data = init_query(data_request=req, db=db, query_category='single')
    database_data['Datetime'] = pd.to_datetime(database_data['datetime'])
    bool_check = check_all_days_available(database_data, req.start_date, req.end_date)
    return bool_check
    