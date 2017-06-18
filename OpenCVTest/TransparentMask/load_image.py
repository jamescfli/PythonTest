import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help = "Path to the image")
args = vars(ap.parse_args())    # dict with one set of key-value

image = cv2.imread(args['image'])

cv2.imshow("image", image)
cv2.waitKey(0)      # activate image and press any key to close
