import cv2


# video_filepath = './video_clips/b.mp4'      # no problem for .mp4 in general, but this one does not work
# video_filepath = './video_clips/b.webm'
# video_filepath = './video_clips/test.webm'
video_filepath = './video_out/b_640x1280_15fps.mp4'
# video_filepath = './video_out/b_640x1280_60fps.mp4'

# .. Issue: VIDEOIO(cvCreateFileCapture_AVFoundation (filename)): raised unknown C++ exception!

print("loading {}".format(video_filepath))
cap = cv2.VideoCapture(video_filepath)
# cap = cv2.VideoCapture(video_filepath, cv2.CAP_FFMPEG)  # after brew install ffmpeg
print("capture finished")

output_shape = (480, 960)
# const char* filename, int fourcc, double fps, CvSize frame_size, int is_color=1 (gray or color)
# forcecc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
# forcecc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
forcecc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('./video_out/output.avi', -1, 30.0, output_shape[::-1], isColor=True)
print('finish init video writer')

frame_counter = 0
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        frame_counter += 1
        out.write(frame)
    else:
        break
print frame_counter     # 2473 frames for b.mp4

cap.release()
out.release()
cv2.destroyAllWindows()