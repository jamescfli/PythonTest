import numpy as np


print(np.__version__)
np.show_config()        # include openblas atlas etc.

foo = np.zeros((10,10))     # float64, 8 bytes
print("%d bytes" % (foo.size * foo.itemsize))   # 800

np.info(np.add)

np.arange(10, 50)

nz = np.nonzero([1,2,0,0,4,0])
print nz

np.eye(3,4)
# array([[ 1.,  0.,  0.,  0.],
#        [ 0.,  1.,  0.,  0.],
#        [ 0.,  0.,  1.,  0.]])

foo = np.random.random((3, 3))
foo.min()
foo.max()

foo = np.ones((5,5))
foo = np.pad(foo, pad_width=1, mode='constant', constant_values=0)
print(foo)

print(0.3 == 3 * 0.1)   # False
# 3*0.1 = 0.30000000000000004

np.diag(1+np.arange(4),k=-1)
# where The default is 0. Use
#   k>0 for diagonals above the main diagonal,
#   and k<0 for diagonals below the main diagonal.


foo = np.zeros((8,8),dtype=int)
foo[1::2,::2] = 1
foo[::2,1::2] = 1
print(foo)

print(np.unravel_index(100,(6,7,8)))