from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage


face = misc.face(gray=True)
face[0, 40] # 127
face[10:13, 20:23]

# Mask
lx, ly = face.shape
X, Y = np.ogrid[0:lx, 0:ly] # X.shape (768, 1), Y.shape (1, 1024)
mask = (X - lx / 2) ** 2 + (Y - ly / 2) ** 2 > lx * ly / 4
face[mask] = 0
# Fancy indexing
face[range(400), range(400)] = 255

plt.imshow(face, cmap='gray')

# Geometrical transformations
# Cropping
crop_face = face[lx / 4 : -lx / 4, ly / 4 : - ly / 4]
plt.imshow(crop_face, cmap='gray')
# Upside down
flip_ud_face = np.flipud(face)
# Rotation
rotate_face = ndimage.rotate(face, 45)
rotate_face_noreshape = ndimage.rotate(face, 45, reshape=False)