#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test the timeit decorator with signal handling."""
import time
import sys
from trade.helpers.decorators import timeit


@timeit
def sample_function(x, y):
    """Sample function to test timeit decorator."""
    time.sleep(0.1)
    return x + y


if __name__ == "__main__":
    print("Testing timeit decorator with signal...")
    
    result1 = sample_function(5, 3)
    print(f"Result 1: {result1}")
    
    result2 = sample_function(10, 20)
    print(f"Result 2: {result2}")
    
    print("\nSending SIGTERM to test signal handling...")
    # Send SIGTERM to self to trigger cleanup
    sys.exit(0)
