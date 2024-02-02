import matplotlib.pyplot as plt
import os
import json
import numpy as np
from tqdm import tqdm
from PIL import Image
import argparse
import torch



# parsing args
def parse_args():

    parser = argparse.ArgumentParser(description ='args for algorithm which makes frame consistant')

    parser.add_argument('--exp-name', default='teatime', type=str, help='Name of experiment.')
    parser.add_argument('--out-name',  default='teatime', type=str,  help='Name of experiment.')

    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()

    data_path = f'dataset/{args.exp_name}/segmentation_results'
    json_file = f'{args.exp_name}.json'

    output_path = f'assets/{args.exp_name}/vis_features'
    os.makedirs(output_path, exist_ok=True)

    with open(os.path.join(data_path, json_file)) as f:
            ANNOTATIONS = json.load(f)

    for frame_name, frame_path in tqdm(ANNOTATIONS.items()):
        path = os.path.join(data_path, frame_path)

        features = torch.load(path)

        plt.imshow(features[4, :, :].cpu().detach().numpy())
        plt.savefig(os.path.join(output_path, frame_name))
        plt.clf()
