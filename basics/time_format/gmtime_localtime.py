#!/usr/bin/python
import time

t = (2009, 2, 17, 17, 3, 38, 1, 48, 0)
t = time.mktime(t)
print time.strftime("%b %d %Y %H:%M:%S", time.gmtime(t))

t = time.time()
print time.strftime("%b %d %Y %H:%M:%S", time.gmtime(t))
print time.strftime("%b %d %Y %H:%M:%S", time.localtime(t))

# outcome:
#     Feb 17 2009 09:03:38
#     May 16 2017 07:34:02
#     May 16 2017 15:34:02
