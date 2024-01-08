import json
import os
from tqdm import tqdm
from PIL import Image
import argparse
import time 

import numpy as np
from lang_sam import LangSAM
import torch
import matplotlib.pyplot as plt



# parsing args
def parse_args():

    parser = argparse.ArgumentParser(description ='args for algorithm which makes frame consistant')

    parser.add_argument('--data-path', type=str, default='dataset/assets/teatime',  help='Path to the data.')
    parser.add_argument('--text', type=str, default='bear', help='Here you can specify text prompt for the experiment.')
    return parser.parse_args()


if __name__ == '__main__':

    # ------ init SAM model ------ 
    model = LangSAM(sam_type="vit_h")
    # ---------------------------- 

    args = parse_args()


    video_path = args.data_path
    
    frames = sorted(os.listdir(video_path), key=lambda x: int(x.split('_')[1].split('.')[0])) # sort in right order fo frames

    OUTPUT_NAME = args.text 

    save_path = os.path.join('segmentation_results', OUTPUT_NAME)   # output

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join('vis_prompt', 'sem_seg'), exist_ok=True)
    
    print('Experiment name: ', args.text, '\nInput path: ', video_path, 'Output path: ', save_path)

    results = {}

    for image_pil in tqdm(frames):

        # read image
        image_path = os.path.join(video_path, image_pil)
            
        image = Image.open(image_path)
        image = image.convert('RGB')
        H, W, C = np.array(image).shape

        # SAM segmentation
        masks, boxes, phrases, logits = model.predict(image, args.text)
        
        if len(masks) == 0:
            mask = np.zeros((H, W))
        else:
            ind_rel = torch.argmax(logits)
            mask = masks[ind_rel].squeeze().numpy()

      
        # save numpy masks in save_path
        save_path_mask = os.path.join(save_path, image_pil.split('.')[0])
        np.save(save_path_mask, mask)
