import json
import os
from tqdm.contrib import tzip
from PIL import Image
import argparse
import time 
import torch
import matplotlib.pyplot as plt
import cv2

import numpy as np
import gzip

from lang_sam import LangSAM

import warnings
warnings.filterwarnings("ignore")


def resize(large_features, fmap):
    large_dict = {fi: [] for fi in range(len(fmap.flatten()))}
    hw = large_features.reshape(-1, 256).shape[0]
    for i, fi in zip(range(hw), fmap.flatten()):
        large_dict[fi].append(large_features.reshape(-1, 256)[i][None, ...])

    # one of side is always 64
    new_dim = int((fmap.max()+1)/64)
    list_large_f = []
    for ind in range(new_dim*64):
        large_f = torch.tensor(
            np.concatenate(large_dict[ind], axis=0).mean(0)[None, ...]
        )
        list_large_f.append(large_f)
    #choose dims
    if np.argmax(fmap.shape):
        new_h, new_w = new_dim, 64
        fill_h, fill_w = 64-new_dim, 64
    else:
        new_h, new_w = 64, new_dim
        fill_h, fill_w =64, 64-new_dim

    features = torch.cat(list_large_f, dim=0).reshape(new_h, new_w, 256).permute(2, 0, 1).unsqueeze(0)

    # coeffs = features.mean(dim=(2, 3)).repeat_interleave(fill_h*fill_w, 1).reshape(1, 256, fill_h, fill_w)
    # coeffs*torch.ones((1, 256, fill_h, fill_w))

    features = torch.cat([features, rest_features], dim=2)
    return features

# feature_list=[]
# for ind in range(256):
#     features = image_features[ind, :, :].detach().numpy()
#     sam_feature = cv2.resize( features, (64, 64), interpolation = cv2.INTER_AREA )

def visualise_frame_masks(mask, image_pil):
    red_ch = mask[:, :, np.newaxis]
    green_ch = np.zeros_like(mask)[:, :, np.newaxis]
    blue_ch =  np.zeros_like(mask)[:, :, np.newaxis]

    colored_mask = np.concatenate([255*red_ch, green_ch, blue_ch], axis=2)

    result = (0.5*np.array(image_pil)).astype(int) +(0.5*colored_mask).astype(int)
    # final_mask = Image.fromarray((255*mask).astype('uint8'), mode="L")
    return result


def load_npy_gz(filename):
    with gzip.open(filename, 'rb') as f:
        array = np.load(f)
    return array

# parsing args
def parse_args():

    parser = argparse.ArgumentParser(description ='args for algorithm which makes frame consistant')

    parser.add_argument('--data-path', type=str, default='teatime',  help='Path to the data.')
    parser.add_argument('--exp-name', type=str, default='teatime', help='Here you can specify the name of the experiment.')
    parser.add_argument('--text-prompt', type=str, default='brown teddy bear', help='Here you can specify text prompt.')
    parser.add_argument('--separate', type=str, default='False', help='Save every masks.')

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
    video_path = os.path.join('renders/test/rgb')  
    frames = sorted(os.listdir(video_path), key=lambda x: int(x.split('.')[0].split('_')[1]))

    feature_path = os.path.join('renders/test/raw-sam_features_background') 
    feature_frames = sorted(os.listdir(feature_path), key=lambda x: int(x.split('.')[0].split('_')[1])) 

    # output
    save_path = os.path.join('assets', OUTPUT_NAME, 'vis_prompt', TEXT_PROMPT)   
    os.makedirs(save_path, exist_ok=True)

    fmap = np.load('data/teatime/segmentation_results/features_map.npy')
    rest_features = torch.tensor(np.load('data/teatime/segmentation_results/rest_features.npy'))

    # ---------------------------- 
    
    print('Experiment name: ', args.exp_name, '\nInput path: ', video_path, 'Output path: ', save_path)

    image_list = []
    for frame_pil, frame_feature in tzip(frames, feature_frames):
        # read image
        image_pill=Image.open(os.path.join(video_path, frame_pil)).convert("RGB")


        frame_feature_path = os.path.join(feature_path, frame_feature)
        image_features = load_npy_gz(frame_feature_path)
        
        # RESIZE HERE
        sam_feature=resize(image_features, fmap)
        
        # Text SAM segmentation
        masks, boxes, phrases, logits = model.predict(image_featues=sam_feature, 
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

        # smoothing
        opening = cv2.morphologyEx(mask.astype('uint8'), cv2.MORPH_OPEN, np.ones((10,10), np.uint8))
        mask = cv2.filter2D(opening,-1, np.ones((20,20),np.float32)/400)


        vis_masks = visualise_frame_masks(mask, image_pill)

        if args.separate:

            fig, ax = plt.subplots()
            ax.imshow(vis_masks)
            ax.axis('off')

            for box, logit in zip(boxes, logits):
                x_min, y_min, x_max, y_max = box
                confidence_score = round(logit.item(), 2)  # Convert logit to a scalar before rounding
                box_width = x_max - x_min
                box_height = y_max - y_min

                # Draw bounding box
                rect = plt.Rectangle((x_min, y_min), box_width, box_height, fill=False, edgecolor='red', linewidth=2)
                ax.add_patch(rect)
            plt.savefig(os.path.join(save_path, frame_pil))


        image_list.append(vis_masks)