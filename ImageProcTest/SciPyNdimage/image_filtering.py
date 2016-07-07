from scipy import misc
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt

# Blurring
face = misc.face(gray=True)
# Gaussian filter
blurred_face = ndimage.gaussian_filter(face, sigma=3)
very_blurred = ndimage.gaussian_filter(face, sigma=5)
# Uniform filter
local_mean = ndimage.uniform_filter(face, size=11)

# Sharpening
face = misc.face(gray=True).astype(float)
blurred_f = ndimage.gaussian_filter(face, 3)
# Increase weight of edges by adding an approximation of the Laplacian
filter_blurred_f = ndimage.gaussian_filter(blurred_f, 1)
alpha = 30
sharpened = blurred_f + alpha * (blurred_f - filter_blurred_f)

# Denoising
f = face[230:290, 220:320]
noisy = f + 0.4 * f.std() * np.random.random(f.shape)
plt.imshow(noisy, cmap='gray')