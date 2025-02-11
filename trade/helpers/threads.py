from concurrent.futures import ThreadPoolExecutor
from os import cpu_count

def runThreads(func, OrderedInputs: list[list], run_type: str = 'map') -> list:
    """
    Run multithreading on a given function.

    params:
    --------
    func: Function to run in multiple threads.
    OrderedInputs: List of inputs to pass to the function.
    
    run_type: Type of multithreading execution. Default is 'map'.

    returns:
    --------
    List of results from the threading function.
    """

    global shutdown_event
    try:
        num_threads = min(len(OrderedInputs[0]), cpu_count())  # Limit threads to CPU cores or available inputs
        results = []

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            if run_type == 'map':
                results = list(executor.map(func, *OrderedInputs))
            else:
                raise ValueError(f'Run type {run_type} not recognized')

    except KeyboardInterrupt:
        shutdown_event = True
        raise
    except Exception as e:
        print('Error occurred: ', e)
        raise

    return results
