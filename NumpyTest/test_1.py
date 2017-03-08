import numpy as np


print(np.__version__)
np.show_config()        # include openblas atlas etc.

foo = np.zeros((10,10))     # float64, 8 bytes
print("%d bytes" % (foo.size * foo.itemsize))   # 800