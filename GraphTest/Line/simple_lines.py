# The code does not work properly
import matplotlib.pyplot as plt
import numpy as np
from IPython.html.widgets import interact   # ImportError: cannot import name interact
# .. ShimWarning: The `IPython.html` package has been deprecated. You should import from `notebook` instead.
# `IPython.html.widgets` has moved to `ipywidgets`.

def plot_sine(frequency=1.0, amplitude=1.0):
    plt.ylim(-1.0, 1.0);
    x = np.linspace(0, 10, 1000)
    plt.plot(x, amplitude*np.sin(x*frequency));

interact(plot_sine, frequency=(0.5, 10.0), amplitude=(0.0, 1.0))