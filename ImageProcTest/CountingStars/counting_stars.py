from matplotlib import pyplot as plt
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from math import sqrt
from skimage.color import rgb2gray
import glob
from skimage.io import imread

# Unix style pathname pattern expansion, e.g. glob.glob(r"C:\Users\Tavish\Deskto\pwint_sky.gif")[0]
# String is to be treated as a raw string, exactly the string literals marked by a 'r'
example_file = glob.glob("Assets/wint_sky.gif")[0]  # Out: 'Assets/wint_sky.gif'
im = imread(example_file, as_grey=True)
plt.imshow(im, cmap=plt.get_cmap('gray'))
plt.show()

# Counting: search continuous objects in the picture
# Blobs are found using the Laplacian of Gaussian (LoG) method. For each blob found,
# the method returns its coordinates and the standard deviation of the Gaussian kernel
# that detected the blob.
# threshold: The absolute lower bound for scale space maxima. Local maxima smaller than thresh are ignored.
# Reduce this to detect blobs with less intensities.
# Output: A 2d array with each row representing 3 values, (y,x,sigma) where (y,x) are coordinates of the blob
# and sigma is the standard deviation
blobs_log = blob_log(im, max_sigma=30, num_sigma=10, threshold=.1) # len(blobs_log) = 308
# .. output - first two are coordinates, and the third is the area of the object
# Compute radii in the 3rd column
blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)
numrows = len(blobs_log)
print("Number of stars : ", numrows)

# Check whether there is a missing
fig, ax = plt.subplots(1, 1)
plt.imshow(im, cmap=plt.get_cmap('gray'))
for blob in blobs_log:
    y, x, r = blob
    c = plt.Circle((x,y), r+5, color='lime', linewidth=2, fill=False)
    ax.add_patch(c)