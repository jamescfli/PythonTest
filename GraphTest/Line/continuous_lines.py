# Note that IPython are perfectly compatible with matplotlib
# Automatically shutdown python when exit
# However, when exiting PyCharm, the Python icon from matplotlib still exists
# The issue has been solved by installing IPython

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

fig = plt.figure()

# sub figure 1
subfig_1 = fig.add_subplot(211)

x_1 = [10,24,23,23,3]
y_1 = [12,2,3,4,2]

line_1 = Line2D(x_1,y_1)

subfig_1.add_line(line_1)
subfig_1.set_xlim(min(x_1), max(x_1))
subfig_1.set_ylim(min(y_1), max(y_1))

# sub figure 2
subfig_2 = fig.add_subplot(212)

points_x = np.arange(10.0).reshape(10,1)
points_y = np.sin(points_x).reshape(10,1)

points = np.hstack([points_x, points_y])    # two columns 10*2 with x and y values

(x_2, y_2) = zip(*points)   # x_2 and y_2 are tuples

line_2 = Line2D(x_2,y_2)

subfig_2.add_line(line_2)
subfig_2.set_xlim(min(x_2), max(x_2))
subfig_2.set_ylim(min(y_2), max(y_2))

plt.show()
