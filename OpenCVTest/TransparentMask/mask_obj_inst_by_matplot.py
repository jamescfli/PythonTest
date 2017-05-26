import cv2
import matplotlib.pyplot as plt
import numpy as np

image_src = cv2.imread('JackMa.jpg')
image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2RGB)
height, width, _ = image_src.shape
image_mask = np.zeros(image_src.shape, dtype=image_src.dtype)
image_mask[(height/4):(height/4*3), (width/4):(width/4*3), :] = (255, 20, 147)
# Hot Pink	    255-105-180	ff69b4
# Deep Pink	    255-20-147	ff1493
# Pink	        255-192-203	ffc0cb
# Light Pink	255-182-193	ffb6c1

plt.figure()
plt.subplot(1,2,1)
plt.imshow(image_src)
plt.subplot(1,2,2)
plt.imshow(image_src)
plt.imshow(image_mask, cmap='jet', alpha=0.4)
plt.show()
