import numpy as np


Z = np.tile( np.array([[0, 0, 1],[1, 0, 0]]), (4,4))
print(Z)

Z = np.random.random((5,5))
Zmax, Zmin = Z.max(), Z.min()
Z = (Z - Zmin)/(Zmax - Zmin)
print(Z)