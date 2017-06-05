import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

X = np.random.random((100, 2))
print X[:, 0].max()
print X[:, 0].min()
print X[:, 1].max()
print X[:, 1].min()

Y = np.random.randint(0, 2, size=(100,), dtype=np.uint8)
print np.unique(Y)
print Y.dtype

# # OK
# plt.scatter(X[:, 0], X[:, 1], s=5, c=Y, cmap=colors.ListedColormap(['red', 'blue']))
# plt.show()

Y = Y[:, np.newaxis]
print Y.shape
Y = np.hstack([Y, 1-Y])
print np.sum(Y, axis=1)

print X.__class__
print Y.__class__

# OK
plt.scatter(X[:, 0], X[:, 1], s=50, c=Y[:, 1], cmap=colors.ListedColormap(['orange', 'blue']))
plt.show()

# X = np.matrix(X)
# plt.scatter(X[:, 0], X[:, 1], s=50, c=Y[:, 1], cmap=colors.ListedColormap(['orange', 'blue']))
# plt.show()
# # Issue: ValueError: Masked arrays must be 1-D
# # Solution: X = np.array(X)
