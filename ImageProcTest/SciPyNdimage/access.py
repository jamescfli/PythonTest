from scipy import misc

# Writing an array to a file
f = misc.face() # f.shape = (768, 1024, 3)
misc.imsave('../Assets/face.png', f)

import matplotlib.pyplot as plt
plt.imshow(f)
plt.show()

# Creating a numpy array from an image file
face = misc.imread('../Assets/face.png')
type(face)
face.shape, face.dtype  # dtype is uint8 for 8-bit images

import numpy as np
# Opening raw files (camera, 3-D images)
face.tofile('../Assets/face.raw')
face_from_raw = np.fromfile('../Assets/face.raw', dtype=np.uint8)   # shape (2359296,)
face_from_raw.shape
face_from_raw.shape = (768, 1024, 3)
# use np.memmap for memory mapping
face_memmap = np.memmap('../Assets/face.raw', dtype=np.uint8, shape=(768, 1024, 3))

# Working on a list of image files
for i in range(10):
    im = np.random.randint(0, 255+1, 10000).reshape((100, 100))
    im.max()
    misc.imsave('../Assets/random_%02d.png' % i, im)
from glob import glob
filelist = glob('../Assets/random*.png')
filelist.sort()