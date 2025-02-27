import time
from functools import wraps
import inspect

def log_time(logger):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            logger.info(f'{func.__name__} took {end - start} seconds')
            logger.info(f'args {args}, kwargs: {kwargs}')
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



def log_error_with_stack(logger, raise_exception=True):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                stack = inspect.stack()
                filtered_stack = [
                    frame.function for frame in stack
                    if not any(excluded in frame.function for excluded in (
                        "<", "run_", "_run", "do_execute", "execute_request", "dispatch_shell",
                        "process_one", "dispatch_queue", "start", "launch_instance"
                    ))
                ]
                call_chain = " -> ".join(filtered_stack+[func.__name__])
                logger.error(f'Error: {e}')
                logger.error(f'Call Chain: {call_chain}')
                logger.error(f'Variables: {args}, {kwargs}')
                if raise_exception:
                    raise e
        return wrapper
    return decorator