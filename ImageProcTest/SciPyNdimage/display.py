# Use matplotlib and imshow to display an image inside a matplotlib figure

from scipy import misc
f = misc.face(gray=True)

import matplotlib.pyplot as plt
plt.imshow(f, cmap=plt.cm.gray)

# Increase contrast by setting min and max values
plt.imshow(f, cmap=plt.cm.gray, vmin=30, vmax=200)
# Remove axes and ticks
plt.axis('off')

# Draw contour lines
plt.contour(f, [50, 200])

# Fine inspection of intensity variations with 'nearest'
plt.imshow(f[320:340, 510:530], cmap=plt.cm.gray)
plt.imshow(f[320:340, 510:530], cmap=plt.cm.gray, interpolation='nearest')