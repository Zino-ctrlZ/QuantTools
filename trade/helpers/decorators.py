# pylint: disable=broad-exception-caught
import time
import numbers
import asyncio
from logging import Logger
from functools import wraps
import inspect
import cProfile
import pstats
import io
import traceback
import pandas as pd
from trade.helpers.Logging import setup_logger
_logger = setup_logger('trade.helpers.decorators')
time_logger = setup_logger('trade.helpers.decorators.time_logger')
failed_logger = setup_logger('trade.helpers.decorators.failed_logger')

class PrintingLogger:
    """
    A simple logger that prints messages to the console.
    """
    def __init__(self, fmt: str = None):
        self.name = "PrintingLogger"
        self.format = fmt or "%(asctime)s %(levelname)s: %(message)s"

    
    def info(self, msg: str, exc_info: bool = False, *args, **kwargs):
        print(f"[INFO] {msg}")
        if exc_info:
            traceback.print_exc()

    def warning(self, msg: str, exc_info: bool = False, *args, **kwargs):
        print(f"[WARNING] {msg}")
        if exc_info:
            traceback.print_exc()

    def error(self, msg: str, exc_info: bool = False, *args, **kwargs):
        print(f"[ERROR] {msg}")
        if exc_info:
            traceback.print_exc()

    def critical(self, msg: str, exc_info: bool = False, *args, **kwargs):
        print(f"[CRITICAL] {msg}")
        if exc_info:
            traceback.print_exc()

    def debug(self, msg: str, exc_info: bool = False, *args, **kwargs):
        print(f"[DEBUG] {msg}")
        if exc_info:
            traceback.print_exc()



def log_time(logger: Logger=None):
    """
    Log the execution time of the decorated function.
    Args:
        logger: The logger instance to use for logging execution time.
    Returns:
        A decorator that logs execution time for the decorated function.
    """
    if logger is None:
        logger = time_logger

    def decorator(func):
        iscoroutine = asyncio.iscoroutinefunction(func)
        if iscoroutine:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                start = time.time()
                result = await func(*args, **kwargs)
                end = time.time()
                args = [arg for arg in args if not isinstance(arg, (type(None), pd.DataFrame, bytes))]
                logger.info(f'{func.__name__} took {end - start} seconds')
                logger.info(f'args {args}, kwargs: {kwargs}')
                return result
            return wrapper
        else:
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


def log_error(logger: Logger=None, raise_exception=True):
    """
    Log errors that occur in the decorated function.
    Args:
        logger: The logger instance to use for logging errors.
    Returns:
        A decorator that logs errors for the decorated function.
    
    Notes
    """
    if logger is None:
        logger = failed_logger

    def decorator(func):
        iscoroutine = asyncio.iscoroutinefunction(func)
        if iscoroutine:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    logger.error(f'\n{func.__name__} raised an error: {e}', exc_info = True)
                    logger.error(f'args {args}, kwargs: {kwargs}')
                    logger.error("Traceback:\n" + traceback.format_exc())
                    if raise_exception:
                        raise e
            return wrapper
        else:
            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.error(f'\n{func.__name__} raised an error: {e}', exc_info = True)
                    logger.error(f'args {args}, kwargs: {kwargs}')
                    logger.error("Traceback:\n" + traceback.format_exc())
                    raise e
            return wrapper
    return decorator



def log_error_with_stack(logger: Logger = None, raise_exception=True):
    """
    Log errors that occur in the decorated function along with the call stack.
    Args:
        logger: The logger instance to use for logging errors.
        raise_exception (bool): Whether to re-raise the exception after logging.
    Returns:
        A decorator that logs errors and call stack for the decorated function.
    """

    def decorator(func):
        def _log_exception(e, args, kwargs):
            logger.error(f'\n{func.__name__} raised an error:', exc_info=True)
            logger.error("Traceback:\n" + traceback.format_exc())
            logger.error(f'args {args}, kwargs: {kwargs}')

            stack = inspect.stack()
            filtered_stack = [
                frame.function for frame in stack
                if not any(excluded in frame.function for excluded in (
                    "<", "run_", "_run", "do_execute", "execute_request", "dispatch_shell",
                    "process_one", "dispatch_queue", "start", "launch_instance"
                ))
            ]
            call_chain = " -> ".join(filtered_stack + [func.__name__])
            logger.error(f'Call Chain: {call_chain}')
            if raise_exception:
                raise e

        iscoroutine = asyncio.iscoroutinefunction(func)
        if iscoroutine:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    _log_exception(e, args, kwargs)
            return wrapper
        else:
            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    _log_exception(e, args, kwargs)
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


class classproperty(property):
    def __get__(self, obj, cls):
        return self.fget(cls)
    def __set__(self, obj, value):
        return self.fset(type(obj), value)