import time
from functools import wraps

def time_it(func):
    """
    Decorator to measure the execution time of a function.
    Returns a tuple of (result, execution_time_in_seconds).
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        return result, execution_time
    return wrapper
