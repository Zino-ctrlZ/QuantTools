#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test the timeit decorator."""
import time
from trade.helpers.decorators import timeit


@timeit
def sample_function(x, y, operation="add"):
    """Sample function to test timeit decorator."""
    time.sleep(0.1)  # Simulate some work
    if operation == "add":
        return x + y
    elif operation == "multiply":
        return x * y
    return None


@timeit
def another_function(name):
    """Another sample function."""
    time.sleep(0.05)
    return f"Hello, {name}!"


if __name__ == "__main__":
    print("Testing timeit decorator...")
    
    # Test with positional args
    result1 = sample_function(5, 3)
    print(f"Result 1: {result1}")
    
    # Test with kwargs
    result2 = sample_function(5, 3, operation="multiply")
    print(f"Result 2: {result2}")
    
    # Test another function
    result3 = another_function("World")
    print(f"Result 3: {result3}")
    
    # Test with keyword arguments only
    result4 = sample_function(x=10, y=20, operation="add")
    print(f"Result 4: {result4}")
    
    print("\nTest completed! Check the CSV file at GEN_CACHE_PATH after script exits.")
    print("The timeit metadata should be saved automatically via atexit.")
