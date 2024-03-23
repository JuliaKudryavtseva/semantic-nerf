import os
import pickle
import gzip
import matplotlib.pyplot as plt

import numpy as np
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
            )[:6]

        self.model_input_names_path = paths['model_input_names_path']
        self.sam_features_path = paths['sam_features_path']

        self.fmap = np.load('../data/teatime/segmentation_results/features_map.npy')


    def __len__(self):
        return len(self.model_input_names)

    def __getitem__(self, idx):
        input_name = self.model_input_names[idx]
        inp_path = os.path.join(self.model_input_names_path, input_name)
        inp = load_npy_gz(inp_path)

        # target_ft_name = self.sam_features_names[idx]
        target_ft_name = input_name.split('.')[0] + '_enc_features.pkl'
        target_path = os.path.join(self.sam_features_path, target_ft_name)
        target = load_pkl(target_path)

        return inp, target


def collate_fn(batch):
    fmap = np.load('../data/teatime/segmentation_results/features_map.npy')
    inps = []
    targets = []
    h_new, w_new = 48, 64
    for (inp, target) in batch:

        inp = torch.tensor(inp).permute(2, 0, 1).unsqueeze(0) # batch_size, 256, H, W
        inps.append(inp)

        inp_resized = resize_pool(inp, fmap)

        target = torch.tensor(np.array(list(target.values())))
        target = target.reshape(-1, 64, 256).permute(2, 0, 1).unsqueeze(0)
        targets.append(target-inp_resized)

    inps, targets = torch.cat(inps), torch.cat(targets)
    return inps, targets



def get_dataloader(batch_size, paths, num_workers=0):
    training_dataset = CompressDataset(paths)
    return DataLoader(
        training_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
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

def resize_pool(large_features, fmap):
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
    resize_pool_fn = torch.nn.AdaptiveAvgPool2d((new_h, new_w))
    features = resize_pool_fn(large_features)
    return features




if __name__ == '__main__':
    import time

    fmap = np.load('../data/teatime/segmentation_results/features_map.npy')
    
    start_time = time.time()

    paths = {
        'model_input_names_path': '../renders/train/raw-sam_features', 
        'sam_features_path':'../data/teatime/segmentation_results/seg_features'}
    dataloader = get_dataloader(1, paths, num_workers=0)
    
    print(f'{len(dataloader)=}')

    for ind, (inp, target) in enumerate(dataloader, 1):
        print(inp.shape, target.shape)


        # avr_ft = resize_pool(inp[0].permute(2, 1, 0), fmap)


        # === current mse:  tensor(0.0098)  === 
        # print('\n === current mse: ', torch.mean((inp-target)**2), ' === \n')
        print('\n === current mse: ', torch.mean(target**2), ' === \n')


        # plt.imshow((avr_ft[0, 4, ...] - target[0, 4, ...])**2)
        # plt.savefig(f'vis/{ind}.jpg')
        # plt.clf()

        plt.imshow(target[0, 4, ...])
        plt.savefig('target.jpg')
        plt.clf()

        plt.imshow(inp[0, 4, ...])
        plt.savefig('inp.jpg')
        plt.clf()

        break

    end_time = time.time()
    print("Total execution time: {} seconds".format(end_time - start_time))
