import json
import os
from tqdm.contrib import tzip
from tqdm import tqdm
from PIL import Image
import argparse
import time 
import torch
import matplotlib.pyplot as plt
import cv2

import numpy as np
import pandas as pd
import gzip

from lang_sam import LangSAM
from eval_metrics import calulate_ious, get_gt_masks

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


def resize_pool(large_features, fmap, frame_name):
    # one of side is always 64
    new_dim = int((fmap.max()+1)/64)
    
    #choose dims
    if np.argmax(fmap.shape):
        new_h, new_w = new_dim, 64
        fill_h, fill_w = 64-new_dim, 64
    else:
        new_h, new_w = 64, new_dim
        fill_h, fill_w =64, 64-new_dim

    # resizing
    large_features = torch.tensor(large_features).permute(2, 0, 1)
    large_features = large_features.unsqueeze(0)
    resize_pool_fn = torch.nn.AdaptiveAvgPool2d((new_h, new_w))
    features = resize_pool_fn(large_features)

    # frame_name = frame_name.replace('.jpg', '_rest_features.npy')
    # rest_features = torch.tensor(np.load(f'data/{DATA_PATH}/segmentation_results/rest_features/{frame_name}'))

    # features = torch.cat([features, rest_features], dim=2)
    return features



def visualise_frame_masks(mask, image_pil):
    red_ch = mask[:, :, np.newaxis]
    green_ch = np.zeros_like(mask)[:, :, np.newaxis]
    blue_ch =  np.zeros_like(mask)[:, :, np.newaxis]

    colored_mask = np.concatenate([255*red_ch, green_ch, blue_ch], axis=2)

    result = (0.5*np.array(image_pil)).astype(int) +(0.5*colored_mask).astype(int)
    save_vis(result)


def save_vis(vis_masks):
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
    parser.add_argument('--resize', type=bool, default=True)
    parser.add_argument('--reg', type=bool)

    return parser.parse_args()


if __name__ == '__main__':

    DATA_PATH = os.environ['DATA_PATH']

    # ------ init SAM model ------ 
    CHECKPOINT_PATH = "/weights/sam_vit_h_4b8939.pth"
    model = LangSAM(sam_type="vit_h", ckpt_path=CHECKPOINT_PATH)
    # ---------------------------- 


    # ------ args parsing ---------
    args = parse_args()
    OUTPUT_NAME = args.exp_name
    TEXT_PROMPT = args.text_prompt
    # ---------------------------- 

    if args.reg is None:
        reg = False
    else:
        reg = True

    if reg:
        REG = 'reg'
    else:
        REG = 'no_reg'

    # ----------- pathes ---------
    # input
    video_path = os.path.join(f'renders/{DATA_PATH}/{REG}/test/rgb')  
    frames = sorted(os.listdir(video_path), key=lambda x: int(x.split('.')[0].split('_')[1]))

    # compressed SAM feature 
    compress_sam_feature_path = os.path.join(f'data/{DATA_PATH}/compress_features') 
    sam_feature_path = f'renders/{DATA_PATH}/{REG}/test/raw-sam_features'

    # output
    save_path = os.path.join('assets', OUTPUT_NAME, 'vis_prompt', TEXT_PROMPT)   
    os.makedirs(save_path, exist_ok=True)

    # results
    save_results_path = os.path.join('assets', OUTPUT_NAME)   
    os.makedirs(save_results_path, exist_ok=True)

    # rest features
    fmap = np.load(f'data/{DATA_PATH}/segmentation_results/features_map.npy')
    rest_features = torch.tensor(np.load(f'data/{DATA_PATH}/segmentation_results/rest_features.npy'))

    # ---------------------------- 
    

    print('\n\n')
    print(' === Experiment name: ', args.exp_name, '\nInput path: ', video_path, ' | Output path: ', save_path, ' === \n')
    print('With regularization: ', reg)
    print(' === Decode masks === ')

    all_masks = []

    for frame_pil in tqdm(frames):

        # read image
        image_pill=Image.open(os.path.join(video_path, frame_pil)).convert("RGB")

        # load sam_features
        if args.resize:
            frame_name = frame_pil.replace('.jpg', '.npy.gz')
            frame_name_path = os.path.join(sam_feature_path, frame_name)
            sam_feature = load_npy_gz(frame_name_path)
            sam_feature = resize_pool(sam_feature, fmap, frame_pil)

        else:
            frame_name = frame_pil.replace('jpg', 'pt')
            sam_feature = torch.load(os.path.join(compress_sam_feature_path, frame_name), map_location='cpu')

        sam_feature = torch.cat([sam_feature, rest_features], dim=2)

        
        # Text SAM segmentation
        masks, boxes, phrases, logits = model.predict(image_featues=sam_feature, 
                                                      image_pil=image_pill,
                                                      text_prompt=TEXT_PROMPT)

        
        if len(masks) == 0:
            H, W = np.array(image_pill).shape[:2]
            mask = np.zeros((H, W))
        else:
            mask = [mask.squeeze().numpy() for mask in masks]
            mask = (np.array(mask).mean(axis=0) > 0).astype(int) # concat all masks in one frame

        # smoothing
        opening = cv2.morphologyEx(mask.astype('uint8'), cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
        mask = cv2.filter2D(opening,-1, np.ones((10,10),np.float32)/100)

        # visualization
        vis_masks = visualise_frame_masks(mask, image_pill)
        all_masks.append(mask)

    print(' === Evaluate metrics: IOUs === ')

    
    # get ground truth masks
    gt_masks_path = f'data/{DATA_PATH}/ground_truth'
    gt_masks = get_gt_masks(gt_masks_path, TEXT_PROMPT, frames, all_masks[0].shape)

    # calculate metrics
    IOUs, IOUS_per_frame  = calulate_ious(pred_masks=all_masks, gt_masks=gt_masks)

    # create df
    columns = ['text prompt', 'reg', 'IOU mean'] + frames
    try:
        df_ious = pd.read_csv(f'assets/{DATA_PATH}/results.csv')
    except FileNotFoundError:
        df_ious = pd.DataFrame(columns=columns)


    df_ious_curr = pd.DataFrame([[TEXT_PROMPT, reg, IOUs] + IOUS_per_frame], columns=columns)
    
    # drop if there is eval
    
    if (TEXT_PROMPT in df_ious['text prompt'].values):
        text_prompt_ind = int(df_ious[df_ious['text prompt']==TEXT_PROMPT].index[0])
        if (reg == df_ious.loc[text_prompt_ind, 'reg']):
            df_ious.drop(text_prompt_ind, axis=0, inplace = True)


    df_ious = pd.concat([df_ious, df_ious_curr], ignore_index = True)

    # save df
    metrics_path = os.path.join(save_results_path, 'results.csv')
    df_ious = df_ious.sort_values(by=['text prompt', 'reg'])
    df_ious.to_csv(metrics_path, index=False)


    print(f'{IOUs=}\n')