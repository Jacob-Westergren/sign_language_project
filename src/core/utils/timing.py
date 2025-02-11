import time
from contextlib import contextmanager

@contextmanager
def timing(description: str = "Operation"):
    start = time.time() # runs when entering the context manager
    yield # runs when exiting the context manager
    elapsed = time.time() - start 
    print(f"{description}: {elapsed:.2f} seconds") 
