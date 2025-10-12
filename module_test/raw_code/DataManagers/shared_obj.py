# shared_objects.py

## Using spawn context, we set manager only after the first import is done 
import multiprocessing as mp
from trade.helpers.Logging import setup_logger
from trade._multiprocessing import MP_CONTEXT, ensure_global_start_method
logger = setup_logger('DataManagers.shared_obj.py')

_ctx = MP_CONTEXT
_manager = None

_shared_queue = None
_shared_dict = None
_shared_lists = {}
_shared_lock = None
_shared_value = None
_request_list = None

def setup_shared_objects():
    global _manager, _shared_queue, _shared_dict, _shared_lists, _shared_lock, _shared_value, _request_list
    ensure_global_start_method()
    _manager = _ctx.Manager()
    _shared_queue = _manager.Queue()
    _shared_dict = _manager.dict()
    _shared_lists = {}
    _shared_lock = _manager.Lock()
    _shared_value = _manager.Value('i', 0)
    _request_list = _manager.list()

def get_manager():
    global _manager
    if _manager is None:
        raise RuntimeError("Manager must be created first with setup_shared_objects() before starting workers.")
    return _manager

def get_shared_queue():
    if _shared_queue is None:
        raise RuntimeError("Shared queue not initialized. Call setup_shared_objects() first.")
    return _shared_queue

def get_shared_dict():
    return _shared_dict

def get_shared_list(name: str):
    global _shared_lists
    if name not in _shared_lists:
        _shared_lists[name] = get_manager().list()
    return _shared_lists[name]

def get_shared_lock():
    return _shared_lock

def get_int_value():
    return _shared_value

def get_request_list():
    return _request_list

def reset_request_list():
    global _request_list
    _request_list = get_manager().list()