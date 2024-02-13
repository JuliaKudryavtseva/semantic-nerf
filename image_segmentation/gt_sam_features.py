import json
import os
from tqdm import tqdm
from PIL import Image
import argparse
import time 
import torch
import cv2

import pickle
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide

resizer = ResizeLongestSide(1024)


# parsing args
def parse_args():

    parser = argparse.ArgumentParser(description ='args for algorithm which makes frame consistant')

    parser.add_argument('--data-path', type=str, default='teatime',  help='Path to the data.')
    parser.add_argument('--exp-name', type=str, default='teatime', help='Here you can specify the name of the experiment.')
    return parser.parse_args()


if __name__ == '__main__':

    # ------ init SAM model ------ 
    MODEL_TYPE = "vit_h"
    CHECKPOINT_PATH = "weights/sam_vit_h_4b8939.pth"
    DEVICE = "cuda"
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
    sam_features_generator = SamPredictor(sam)
    # ---------------------------- 

    args = parse_args()


    video_path = os.path.join('dataset', args.data_path, 'images') # input
    frames = sorted(os.listdir(video_path), key=lambda x: int(x.split('_')[1].split('.')[0]))

    OUTPUT_NAME = args.exp_name 
    save_path = os.path.join('dataset', OUTPUT_NAME, 'segmentation_results')   # output


    # save sam features in save_path
    path2frame_save = os.path.join(save_path, 'seg_features')
    os.makedirs(path2frame_save, exist_ok=True)
    
    print('Experiment name: ', args.exp_name, '\nInput path: ', video_path, 'Output path: ', save_path)

    results = {}

    for image_pil in tqdm(frames):
        # read image
        image_path = os.path.join(video_path, image_pil)
            
        image =Image.open(image_path)
        image = image.convert('RGB')
        image_array = np.array(image)

        H, W, _ = image_array.shape
        
        # SAM segmentation
        sam_features_generator.set_image(image_array)
        sam_features = sam_features_generator.get_image_embedding()

        # resizer
        h, w = resizer.get_preprocess_shape(image_array.shape[0], image_array.shape[1], 1024)
        h_new, w_new = int(np.ceil(h/16)), int(np.ceil(w/16))
        sam_features= sam_features[:, :, :h_new, :w_new]


        sam_features = sam_features[0].flatten(1).cpu().detach().numpy()
        sam_feature_dict={}
        for pix_ind in range(h_new*w_new):
            sam_feature_dict[pix_ind] = sam_features[:, pix_ind]

        # save sam_features
        features_frame_name = image_pil.split('.')[0] + '_enc_features.pkl'
        features_frame_path = os.path.join(path2frame_save, features_frame_name)

        with open((features_frame_path), 'wb') as f:
            pickle.dump(sam_feature_dict, f)

        results[image_pil] = os.path.join('seg_features', features_frame_name)


    # RESIZE map
    x = np.arange(h_new*w_new).reshape(h_new, w_new).astype('float32')
    y = cv2.resize(x, (W, H), interpolation = cv2.INTER_NEAREST)

    frame_map_path = os.path.join(save_path, 'features_map.npy') 
    np.save(frame_map_path, y.astype(int))

    with open(os.path.join(save_path, f'{OUTPUT_NAME}.json') , 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    