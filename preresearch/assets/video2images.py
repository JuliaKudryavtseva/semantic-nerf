import cv2
import os
import argparse


# parsing args
def parse_args():

    parser = argparse.ArgumentParser(description ='args for algorithm which makes frame consistant')

    parser.add_argument('--in-name', default='teatime_render', type=str, help='Name of experiment.')
    parser.add_argument('--out-name', default='teatime', type=str,  help='Name of experiment.')
    return parser.parse_args()

if __name__ == '__main__':
  args = parse_args()
      
  os.makedirs(args.out_name, exist_ok=True)

  vidcap = cv2.VideoCapture(f'{args.in_name}.mp4')
  success,image = vidcap.read()
  count = 0
  while success:
    cv2.imwrite(f"{args.out_name}/frame_%d.jpg" % count, image)     # save frame as JPEG file      
    success,image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1