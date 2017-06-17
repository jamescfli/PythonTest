# filled contour
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


xlist = np.linspace(-3.0, 3.0, 3)
ylist = np.linspace(-3.0, 3.0, 4)
X, Y = np.meshgrid(xlist, ylist)    # x goes with columns, y goes with rows
print(xlist)
print(ylist)
print(X)
print(Y)

Z = np.sqrt(X**2 + Y**2)
print(Z)

plt.figure()

# cp = plt.contour(X, Y, Z)
# plt.clabel(cp, inline=True, fontsize=10)
# plt.title('contour plot')
# plt.xlabel('x (cm)')
# plt.ylabel('y (cm)')
# plt.show()

# # change color, line style
# cp = plt.contour(X, Y, Z, colors='black', linestyles='dashed')
# plt.clabel(cp, inline=True, fontsize=10)
# plt.title('contour plot')
# plt.xlabel('x (cm)')
# plt.ylabel('y (cm)')
# plt.show()

# # filled contour
# cp = plt.contourf(X, Y, Z)  # color could be set with colors=c
# plt.colorbar(cp)
# plt.title('filled contours plot')
# plt.xlabel('x (cm)')
# plt.ylabel('y (cm)')
# plt.show()

# levels
levels = [0.0, 0.2, 0.5, 0.9, 1.5, 2.5, 3.5]
contour = plt.contour(X, Y, Z, levels, colors='k')
plt.clabel(contour, colors = 'k', fmt = '%2.1f', fontsize=12)
contour_filled = plt.contourf(X, Y, Z, levels)
plt.colorbar(contour_filled)
plt.title('Plot from level list')
plt.xlabel('x (cm)')
plt.ylabel('y (cm)')
plt.show()