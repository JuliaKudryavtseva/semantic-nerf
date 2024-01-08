import cv2
import os

os.makedirs('assets/teatime', exist_ok=True)

vidcap = cv2.VideoCapture('teatime_render.mp4')
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite("assets/teatime/frame_%d.jpg" % count, image)     # save frame as JPEG file      
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1