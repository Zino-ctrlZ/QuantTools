from concurrent.futures import ThreadPoolExecutor
from os import cpu_count

def runThreads(func, OrderedInputs: list[list], 
               run_type: str = 'map', block = True, 
               thread_name_prefix = '') -> list:
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

    returns:
    --------
    List of results from the threading function.
    """

    global shutdown_event
    try:
        num_threads = min(max(len(OrderedInputs[0]), 1), cpu_count())  # Limit threads to CPU cores or available inputs
        results = []
        if block:
            with ThreadPoolExecutor(max_workers=num_threads, thread_name_prefix=thread_name_prefix+'_thread') as executor:
                if run_type == 'map':
                    results = executor.map(func, *OrderedInputs)
                else:
                    raise ValueError(f'Run type {run_type} not recognized')
        else:
            executor = ThreadPoolExecutor(max_workers=num_threads, thread_name_prefix= thread_name_prefix)
            results = executor.map(func, *OrderedInputs)
            executor.shutdown(wait=False)

    except KeyboardInterrupt:
        shutdown_event = True
        raise
    except Exception as e:
        logger.error('Error occurred: ', e)
        raise
    return list(results) if block else results
