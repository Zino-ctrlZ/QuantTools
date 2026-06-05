from concurrent.futures import ThreadPoolExecutor
from os import cpu_count
from typing import Callable, Any, Optional

from trade.helpers.Logging import setup_logger

logger = setup_logger("trade.helpers.threads")


def runThreads(
    func: Callable[..., Any],
    OrderedInputs: list[list],
    run_type: str = "map",
    block: bool = True,
    thread_name_prefix: str = "",
    show_progress: bool = False,
    progress_desc: Optional[str] = None,
) -> list:
    """
    Run multithreading on a given function.

    params:
    --------
    func: Function to run in multiple threads.
    OrderedInputs: List of inputs to pass to the function. Example:
                   [[input1, input1, input1], [input2, input2, input2], [input3, input3, input3]]

    run_type: Type of multithreading execution. Default is 'map'.
    block: Boolean flag to indicate if results should be returned. Default is True.
                If False, the function will return a list of futures.
                Note: Returning a list of futures is non blocking.
    thread_name_prefix: Prefix for the thread names. Default is empty string.
    show_progress: If True, show a tqdm progress bar while collecting results.
                   Only supported when block=True.
    progress_desc: Optional progress bar description.

    returns:
    --------
    List of results from the threading function.
    """

    global shutdown_event
    try:
        if show_progress and not block:
            raise ValueError("show_progress is only supported when block=True")

        num_threads = min(max(len(OrderedInputs[0]), 1), cpu_count())  # Limit threads to CPU cores or available inputs
        results = []
        if block:
            with ThreadPoolExecutor(
                max_workers=num_threads, thread_name_prefix=thread_name_prefix + "_thread"
            ) as executor:
                if run_type == "map":
                    mapped_results = executor.map(func, *OrderedInputs)
                    if show_progress:
                        try:
                            from tqdm.auto import tqdm
                        except ImportError as import_error:
                            raise ImportError("tqdm is required when show_progress=True") from import_error

                        total_tasks = len(OrderedInputs[0]) if OrderedInputs else 0
                        results = list(tqdm(mapped_results, total=total_tasks, desc=progress_desc))
                    else:
                        results = list(mapped_results)
                else:
                    raise ValueError(f"Run type {run_type} not recognized")
        else:
            executor = ThreadPoolExecutor(max_workers=num_threads, thread_name_prefix=thread_name_prefix)
            results = executor.map(func, *OrderedInputs)
            executor.shutdown(wait=False)

    except KeyboardInterrupt:
        shutdown_event = True
        raise
    except Exception as e:
        logger.error("Error occurred: %s", e)
        raise
    return results if block else results
