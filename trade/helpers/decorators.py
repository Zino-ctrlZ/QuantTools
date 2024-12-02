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

def log_error(logger):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error('')
                logger.error(f'{func.__name__} raise an error: {e}', exc_info = True)
                logger.error(f'args {args}, kwargs: {kwargs}')
                raise e
        return wrapper
    return decorator