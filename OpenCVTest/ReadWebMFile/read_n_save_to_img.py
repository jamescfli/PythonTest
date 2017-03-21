import cv2
import os, shutil

video_filepath = './video_clips/b.webm'
# video_filepath = './video_out/b_640x1280_15fps.mp4'     # 2909 frames

print("loading {}".format(video_filepath))
cap = cv2.VideoCapture(video_filepath)
print("capture finished")

output_shape = (480, 960)

img_out_tmp_path = './img_out_tmp/'
def clear_img_tmp_folder(path_to_folder):
    for file in os.listdir(path_to_folder):
        file_path = os.path.join(path_to_folder, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            # elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print e
clear_img_tmp_folder(img_out_tmp_path)

frame_counter = 0
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        frame_counter += 1
        cv2.imwrite('{}frame_{:05d}.png'.format(img_out_tmp_path, frame_counter), frame)
    else:
        break
print('Total # of frames: {}'.format(frame_counter))

cap.release()
