import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

# prepare data
delta = 0.025
x = np.arange(-3.0, +3.0, delta)
y = np.arange(-2.0, +2.0, delta)
X, Y = np.meshgrid(x, y)
Z1 = mlab.bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
Z2 = mlab.bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
# diff of Gaussians
Z = 10.0 * (Z2 - Z1)

# # labels are draw over the line segments of the contour, removing the lines beneath
# # the label
# plt.figure()
# CS = plt.contour(X, Y, Z)
# plt.clabel(CS, inline=1, fontsize=10)
# plt.title('Simplest default with labels')
# plt.show()

# # force all the contours to be the same color
# plt.figure()
# CS = plt.contour(X, Y, Z, 6, colors='k')     # negative contours will be dashed by default
# plt.clabel(CS, fontsize=9, inline=1)
# plt.title('Single color - negative with dashed line')
# plt.show()

# use a colormap to specify the colors; the default
# colormap will be used for the contour lines
plt.figure()
im = plt.imshow(Z, interpolation='bilinear', origin='lower', cmap=cm.gray, extent=(-3, 3, -2, 2))
levels = np.arange(-1.2, 1.6, 0.2)
CS = plt.contour(Z, levels,
                 origin='lower',    # upper / lower, position for (0,0)
                 linewidths=2,
                 extent=(-3, 3, -2, 2))     # (left, right, bottom, top) range of the displayed image
# thicken the zero contour
zc = CS.collections[6]
plt.setp(zc, linewidth=4)
plt.clabel(CS, levels[1::2], # label every second level
           inline=1, fmt='%1.1f', fontsize=14)
# make a colorbar for contour lines
CB = plt.colorbar(CS, shrink=0.8, extend='both')
plt.title('Lines with colorbar')
plt.flag()

CBI = plt.colorbar(im, orientation='horizontal', shrink=0.8)

l, b, w, h = plt.gca().get_position().bounds    # get current axis
ll, bb, ww, hh = CB.ax.get_position().bounds
CB.ax.set_position([ll, b + 0.1*h, ww, h*0.8])

plt.show()