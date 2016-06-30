# run the following code in IPython
import matplotlib.pyplot as plt

plt.plot([1,2,3,4],[1,4,9,16], 'ro')
plt.interactive(True)   # o.w. the command line will freeze
plt.show()