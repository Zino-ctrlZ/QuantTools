import time
from functools import wraps

def log_time(logger):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            logger.info(f'{func.__name__} took {end - start} seconds')
            return result
        return wrapper
    return decorator
