# def timethis(func):
#     pass
#
# @timethis
# def countdown(n):
#     while n > 0:
#         n -= 1

import time
from functools import wraps

def timethis(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(func.__name__+': {:.5f} sec'.format(end-start))
        return result
    return wrapper

@timethis
def countdown(n):
    while n > 0:
        n -= 1

countdown(100000)