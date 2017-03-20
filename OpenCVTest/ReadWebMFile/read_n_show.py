import cv2
import os
import numpy


# video_filepath = './video_clips/b.mp4'      # no problem for mp4
video_filepath = './video_clips/b.webm'
# .. Issue: VIDEOIO(cvCreateFileCapture_AVFoundation (filename)): raised unknown C++ exception!

print("loading {}".format(video_filepath))
# cap = cv2.VideoCapture(video_filepath)
cap = cv2.VideoCapture(video_filepath, cv2.CAP_FFMPEG)  # after brew install ffmpeg
print("capture finished")

# output_shape = (480, 960)
# # const char* filename, int fourcc, double fps, CvSize frame_size, int is_color=1 (gray or color)
# out = cv2.VideoWriter('../videos/output.mp4', -1, 60.0, output_shape[::-1])
# print('finish init video writer')

frame_counter = 0
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        frame_counter += 1
    else:
        break
print frame_counter     # 2473 frames for b.mp4

cap.release()
# out.release()
cv2.destroyAllWindows()