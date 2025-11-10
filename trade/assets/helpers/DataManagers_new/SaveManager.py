from queue import Queue, Full
import threading
from threading import Thread, Lock
from trade.helpers.helper import setup_logger
import os, sys
import json
import pandas as pd


logger = setup_logger("SaveManager.py")



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


def save_failed_request(request, filename='failed_request.jsonl'):
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
    _threads = []
    _started = False
    _finished_requests = []
    _lock = Lock()
    _current_requests = {}
    _failed_requests = []

    @classmethod
    def _worker(cls):
        while True:
            thread_name = threading.current_thread().name ## Get the name of the current thread
            pack = cls._queue.get()
            request, save_func = pack
            if request is None:
                break
            try:
                with cls._lock: ## Ensures that only one thread can access this block at a time
                    cls._current_requests[thread_name] = request
                save_func(request)
                
                with cls._lock:
                    cls._finished_requests.append(request)
                    del cls._current_requests[thread_name]
                    
            except Exception as e:
                logger.error(f"[SaveWorker] Error processing save: {e}")

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
        for _ in range(cls.WORKER_COUNT):
            t = Thread(target=cls._worker, daemon=True)
            t.start()
            cls._threads.append(t)
        cls._started = True
        logger.info(f"[SaveManager] Started {cls.WORKER_COUNT} save workers.")

    @classmethod
    def enqueue(cls, data_request, save_func):
        try:
            print(f"[SaveManager] Enqueueing save request for {data_request.symbol} on {data_request}")
            cls._queue.put((data_request, save_func), block=False)  # Will raise Full if over limit
        except Full:
            logger.warning(f"[SaveManager] Save queue full (max {cls.MAX_QUEUE_SIZE}). Task ignored.")
        
    @classmethod
    def status(cls):
        return {
            "pending_tasks": cls._queue.qsize(),
            "max_queue_size": cls._queue.maxsize,
            "active_threads": sum(t.is_alive() for t in cls._threads),
            "total_threads": len(cls._threads),
            "current_requests": cls._current_requests,
            "num_finished_requests": len(cls._finished_requests),
            "num_failed_requests": len(cls._failed_requests),
        }

SaveManager.start_workers()