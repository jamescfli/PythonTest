from __future__ import print_function

# list comprehensions: for + if
num =[1, 4, -5, 10, -7, 2, 3, -1, 0]
filtered_and_squared_1 = map(lambda x: x**2, filter(lambda x: x>0, num))
print(filtered_and_squared_1)

filtered_and_squared_2 = [x**2 for x in num if x>0]
print(filtered_and_squared_2)
# .. drawback: load the list/array once in memory as a whole

# so we use generator instead by () rather than []
filtered_and_squared_3 = (x**2 for x in num if x>0)
print(filtered_and_squared_3)   # <generator object <genexpr> at 0x...>
for item in filtered_and_squared_3:
    print(item)
# .. diff is subtle unless the list is huge, but generator is preferred in general

# use zip()
alist =['a1', 'a2', 'a3']
blist =['1', '2', '3']
for a, b in zip(alist, blist):
    print(a, b)

import os
def tree(top):
    for path, names, fnames in os.walk(top):    # walk gives a generator
        for fname in fnames:
            yield os.path.join(path, fname)

for name in tree('.'):
    print(name)
