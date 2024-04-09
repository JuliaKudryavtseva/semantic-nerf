import os
import pickle
import gzip
import matplotlib.pyplot as plt

import numpy as np
import cv2 

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms 


class CompressDataset(Dataset):
    def __init__(self, 
    paths
    ):
        self.model_input_names = sorted(
            os.listdir(paths['model_input_names_path']), key=lambda x: int(x.split('_')[1].split('.')[0])
            )

        self.model_input_names_path = paths['model_input_names_path']
        self.sam_features_path = paths['sam_features_path']

        self.fmap = np.load(os.path.join(paths['sam_features_path'].replace('seg_features', ''), 'features_map.npy'))


    def __len__(self):
        return len(self.model_input_names)

    def __getitem__(self, idx):
        input_name = self.model_input_names[idx]
        inp_path = os.path.join(self.model_input_names_path, input_name)
        inp = load_npy_gz(inp_path)

        # inp = torch.tensor(inp).permute(2, 0, 1) # batch_size, 256, H, W
        # inp_resized = resize_pool(inp, self.fmap)

        target_ft_name = input_name.split('.')[0] + '_enc_features.pkl'
        target_path = os.path.join(self.sam_features_path, target_ft_name)
        target = load_pkl(target_path)

        # target = torch.tensor(np.array(list(target.values())))
        # target = target.reshape(-1, 64, 256).permute(2, 0, 1)

        return inp, target


def collate_fn(batch):
    inps = []
    targets = []

    for (inp, target) in batch:
        inp = torch.tensor(inp).permute(2, 0, 1).unsqueeze(0) # batch_size, 256, H, W
        inps.append(inp)

        _, _, height, width = inp.shape
        target = cv2.resize(
            np.array(list(target.values())).reshape(-1, 64, 256), 
            (width, height), 
            interpolation = cv2.INTER_NEAREST
            )
        target = torch.tensor(target).permute(2, 0, 1).unsqueeze(0)


        # target = torch.tensor(np.array(list(target.values())))
        # target = target.reshape(-1, 64, 256).permute(2, 0, 1).unsqueeze(0)

        targets.append(target)

    inps, targets = torch.cat(inps), torch.cat(targets)
    return inps, targets



def get_dataloader(batch_size, paths, num_workers=0):
    training_dataset = CompressDataset(paths)
    return DataLoader(
        training_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        drop_last=True, 
        collate_fn=collate_fn, 
        num_workers=num_workers)


def load_npy_gz(filename):
    with gzip.open(filename, 'rb') as f:
        array = np.load(f)
    return array

def load_pkl(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data





if __name__ == '__main__':
    import time

    DATA_PATH = os.environ['DATA_PATH']
    REGULAR = os.environ['REGULAR']
    fmap = np.load(f'data/{DATA_PATH}/segmentation_results/features_map.npy')
    

    paths = {
        'model_input_names_path': f'renders/{DATA_PATH}/{REGULAR}/val/raw-sam_features', 
        'sam_features_path': f'data/{DATA_PATH}/segmentation_results/seg_features'}

    dataloader = get_dataloader(1, paths, num_workers=0)
    
    print(f'{len(dataloader)=}')

    start_time = time.time()
    for ind, (inp, target) in enumerate(dataloader, 1):
        print(inp.shape, target.shape)
        break

    end_time = time.time()
    print("Total execution time: {} seconds".format(end_time - start_time))
