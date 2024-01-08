import colorsys
import os
import json
import numpy as np
from tqdm import tqdm
from PIL import Image
import argparse
import random
import cv2
from PIL import Image
import matplotlib.pyplot as plt


def visualise_frame_masks(masks):
    final_mask = Image.fromarray((255*masks).astype('uint8'), mode="L")
    return final_mask
    
# parsing args
def parse_args():

    parser = argparse.ArgumentParser(description ='args for algorithm which makes frame consistant')

    parser.add_argument('--exp-name', type=str, help='Name of experiment.')
    parser.add_argument('--out-name', type=str,  help='Name of experiment.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    INPUT_PATH = os.path.join('segmentation_results', args.exp_name)
    SAVE_PATH = 'vis_prompt'


    os.makedirs(SAVE_PATH, exist_ok=True)

   
    frames = sorted(os.listdir(INPUT_PATH), key=lambda x: int(x.split('_')[1].split('.')[0])) # sort in right order fo frames
   
    image_list = []
    for frame in tqdm(frames):
        masks = np.load(os.path.join(INPUT_PATH, frame)) 
        
        vis_masks = visualise_frame_masks(masks)
        image_list.append(vis_masks)

    
    image_list[0].save(
        os.path.join(SAVE_PATH, f"{args.out_name}.gif"), 
        save_all=True, append_images=image_list[1:],  duration=400)