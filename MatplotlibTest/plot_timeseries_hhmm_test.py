import numpy as np
import datetime
import matplotlib.pyplot as plt
from matplotlib import dates

data = np.random.random(720,)       # from 10am to 10pm
# I can represent minutes as integers:
mins = np.arange(720, dtype=np.int) + 10*60     # from 10 am
# convert to datetime
times=np.array([datetime.datetime(2017, 3, 4, int(p/60), p%60) for p in mins])
# and plot for every 20 samples:
plt.plot(times[1::20], data[1::20])     # one value per 20 min

# generate a formatter, using the fields required
fmtr = dates.DateFormatter("%H:%M")
# need a handle to the current axes to manipulate it
ax = plt.gca()
# set this formatter to the axis
ax.xaxis.set_major_formatter(fmtr)

plt.show()