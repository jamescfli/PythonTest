import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors


def plot(X, Y, pred_func):
    # determine canvas borders, in np.matrix
    mins = np.min(X, axis=0)         # np.min (which is just an alias for np.amin)
    mins -= 0.1 * np.abs(mins)
    maxs = np.amax(X, axis=0)
    maxs += 0.1 * maxs

    ## generate dense grid
    xs, ys = np.meshgrid(np.linspace(mins[0], maxs[0], 300),
                         np.linspace(mins[1], maxs[1], 300))


    # evaluate model on the dense grid
    Z = pred_func(np.c_[xs.flatten(), ys.flatten()])    # column stack
    Z = Z.reshape(xs.shape)

    # Plot the contour and training examples
    plt.contourf(xs, ys, Z, cmap=plt.cm.Spectral)   # draw contour lines and filled contours
    plt.scatter(X[:, 0], X[:, 1], c=Y[:,1], s=50, cmap=colors.ListedColormap(['orange', 'blue']))
    plt.show()
