import os
import numpy as np
from itertools import chain
from trade.helpers.helper import get_parrallel_apply
from typing import List, Union

parrallel_apply = get_parrallel_apply() ## Using system to pick btwmeen multiprocessing and threading

def vector_batch_processor(callable, *args, **kwargs):
    """
    Process a list of inputs in parallel using multiprocessing. This processor assumes the callable works with vectorization
    Underlying ass
    
    Parameters:
    - callable: Function to call with each set of inputs
    - *args: ordered inputs to the callable, where each input is a list or array of values.
        - Each input should be a list or numpy array that can be split into chunks for parallel processing.
        - Anything else will be treated as a single input for each process.
    - **kwargs: Additional keyword arguments (not used in this implementation).
        - Will raise ValueError if kwargs are provided. Only allowed keyword is 'num_process' to specify the number of processes.


    How this works:
    - The function splits each input argument into chunks based on the number of processes specified. The underlying assumption is that the callable can handle vectorized inputs, meaning it can process arrays or lists of values at once.
    - The function then runs the callable in parallel across the specified number of processes.
    - If a single input is provided (not a list or array), it will be replicated across all processes.
    - Why this function?:
        - Vectorization is great to speed up work, but spreading this vectorization across multiple processes can further enhance performance, especially for computationally intensive tasks.

    - Ensure the callable returns only ONE result per call, as this function will flatten the results from all processes into a single list.

    
    Returns:
    List of results from the callable.
    """
    num_process = kwargs.pop('num_process', None)
    if num_process is None:
        num_process = os.cpu_count() or 8

    if kwargs:
        raise ValueError("kwargs are not supported in vector_batch_processor. Use only *args for inputs.")


    ordered_inputs = []
    for arg in args:
        if isinstance(arg, (list, np.ndarray)):
            split_arg = np.array_split(np.asarray(arg, dtype=object), num_process, )
            # print(split_arg )
            ordered_inputs.append(split_arg)
        else:
            split_arg = [arg] * num_process
            ordered_inputs.append(split_arg)
    
    results = parrallel_apply(
        func=callable,
        OrderedInputs=ordered_inputs,
        run_type='map')
    res = list(chain.from_iterable(results))
    
    return res
    