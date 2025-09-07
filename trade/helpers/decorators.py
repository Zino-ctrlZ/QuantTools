import time
from functools import wraps
import inspect
import cProfile
import pstats
import io
import traceback
import pandas as pd
import numbers

def log_time(logger):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            args = [arg for arg in args if not isinstance(arg, (type(None), pd.DataFrame, bytes))]
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
                logger.error("Traceback:\n" + traceback.format_exc())
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
                logger.error('')
                logger.error(f'{func.__name__} raise an error: {e}\n{traceback.format_exc()}', exc_info = True)
                logger.error(f'args {args}, kwargs: {kwargs}')
                
                stack = inspect.stack()
                filtered_stack = [
                    frame.function for frame in stack
                    if not any(excluded in frame.function for excluded in (
                        "<", "run_", "_run", "do_execute", "execute_request", "dispatch_shell",
                        "process_one", "dispatch_queue", "start", "launch_instance"
                    ))
                ]
                call_chain = " -> ".join(filtered_stack+[func.__name__])
                logger.error(f'Call Chain: {call_chain}')
                if raise_exception:
                    raise e
        return wrapper
    return decorator


def cProfiler(func):
    """Decorator to profile a function to measure its execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        results = func(*args, **kwargs)
        profiler.disable()
        stream  = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream).sort_stats('cumulative')
        stats.print_stats()
        return results, stream.getvalue()
    return wrapper

def cprofiler_func(func, *args, **kwargs):
    """Function to profile a function to measure its execution time."""
    profiler = cProfile.Profile()
    profiler.enable()
    results = func(*args, **kwargs)
    profiler.disable()
    stream  = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats('cumulative')
    stats.print_stats()
    return results, stream.getvalue()



def copy_doc(from_func):
    def decorator(to_func):
        to_func.__doc__ = from_func.__doc__
        return to_func
    return decorator


def class_math_operation(cls):
    """
    Compatible with dataclasses only
    """
    def make_op(op):
        def f(self, other):
            ## support number operation
            if isinstance(other, numbers.Number):
                other = cls(**{field: other for field in self.__dataclass_fields__})
                
            elif not isinstance(other, cls):
                return NotImplemented
            data = {}
            for field in self.__dataclass_fields__:
                v1, v2 = getattr(self, field), getattr(other, field)
                if not isinstance(v1, (int, float)) or not isinstance(v2, (int, float)):
                    data[field] = None

                else:
                    data[field]= op(v1,v2)
            return cls(**data)
        return f
    
    import operator
    cls.__add__ = make_op(operator.add)
    cls.__sub__ = make_op(operator.sub)
    cls.__mul__ = make_op(operator.mul)
    cls.__truediv__ = make_op(operator.truediv)
    cls.__floordiv__ = make_op(operator.floordiv)
    return cls