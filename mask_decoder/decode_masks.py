import json
import os
from tqdm.contrib import tzip
from PIL import Image
import argparse
import time 
import torch
import matplotlib.pyplot as plt

import numpy as np
from lang_sam import LangSAM

import warnings
warnings.filterwarnings("ignore")


def visualise_frame_masks(mask):
    # mask = mask[0].cpu().detach().numpy()
    final_mask = Image.fromarray((255*mask).astype('uint8'), mode="L")
    return final_mask

# parsing args
def parse_args():

    parser = argparse.ArgumentParser(description ='args for algorithm which makes frame consistant')

    parser.add_argument('--data-path', type=str, default='teatime',  help='Path to the data.')
    parser.add_argument('--exp-name', type=str, default='teatime', help='Here you can specify the name of the experiment.')
    parser.add_argument('--text-prompt', type=str, default='brown teddy bear', help='Here you can specify text prompt.')
    return parser.parse_args()


if __name__ == '__main__':

    # ------ init SAM model ------ 
    CHECKPOINT_PATH = "/weights/sam_vit_h_4b8939.pth"
    model = LangSAM(sam_type="vit_h", ckpt_path=CHECKPOINT_PATH)
    # ---------------------------- 


    # ------ args parsing ---------
    args = parse_args()
    OUTPUT_NAME = args.exp_name
    TEXT_PROMPT = args.text_prompt
    # ---------------------------- 


    # ----------- pathes ---------
    # input
    video_path = os.path.join('dataset', OUTPUT_NAME, 'images')  
    frames = sorted(os.listdir(video_path), key=lambda x: int(x.split('.')[0].split('_')[1]))

    feature_path = os.path.join('dataset', OUTPUT_NAME, 'segmentation_results/seg_features') 
    feature_frames = sorted(os.listdir(feature_path), key=lambda x: int(x.split('_')[1])) 
    

    # output
    save_path = os.path.join('assets', OUTPUT_NAME, 'vis_prompt', TEXT_PROMPT)   
    os.makedirs(save_path, exist_ok=True)

    # ---------------------------- 
    
    print('Experiment name: ', args.exp_name, '\nInput path: ', video_path, 'Output path: ', save_path)

    
    


    image_list = []
    for frame_pil, frame_feature in tzip(frames, feature_frames):
        # read image
        image_pill=Image.open(os.path.join(video_path, frame_pil)).convert("RGB")


        frame_feature_path = os.path.join(feature_path, frame_feature)
        image_features = torch.load(frame_feature_path, map_location='cpu')
        
        # Text SAM segmentation
        masks, boxes, phrases, logits = model.predict(image_featues=image_features, 
                                                      image_pil=image_pill,
                                                      text_prompt=TEXT_PROMPT)

        
        if len(masks) == 0:
            H, W = np.array(image_pill).shape[:2]
            mask = np.zeros((H, W))
        else:
            # ind_rel = torch.argmax(logits)
            # mask = masks[ind_rel].squeeze().numpy()
            masks = [mask.squeeze().numpy() for mask in masks]
            mask = (np.array(masks).mean(axis=0) > 0).astype(int) # concat all masks in one frame

        vis_masks = visualise_frame_masks(mask)
        plt.imshow(vis_masks)
        plt.savefig(os.path.join(save_path, frame_pil))


        image_list.append(vis_masks)



    image_list[0].save(
        os.path.join(save_path, f"{args.text_prompt}.gif"), 
        save_all=True, append_images=image_list[1:],  duration=400)