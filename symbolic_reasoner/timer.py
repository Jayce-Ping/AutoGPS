import time
import functools

def timer(func):
    """
    A decorator that prints the time a function takes.
    Works for both regular functions and generator functions.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        
        # If the result is a generator
        if hasattr(result, '__iter__') and hasattr(result, '__next__'):
            # Gather all the items in the generator
            items = []
            try:
                while True:
                    items.append(next(result))
            except StopIteration:
                pass
            
            end_time = time.time()
            execution_time = (end_time - start_time)
            # print(f"Generator Function: <{func.__name__}>     Execution time: <{execution_time:.2f} s>")
            
            # Return the items in the generator
            return (item for item in items)
        
        # If the result is not a generator
        else:
            end_time = time.time()
            execution_time = (end_time - start_time)
            # print(f"Function: <{func.__name__}>     Execution time: <{execution_time:.2f} s>")
            return result
            
    return wrapper