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
    model_input_names_path='../renders/train/raw-sam_features', 
    sam_features_path='../data/teatime/segmentation_results/seg_features'
    ):
        self.model_input_names = sorted(
            os.listdir(model_input_names_path), key=lambda x: int(x.split('_')[1].split('.')[0])
            )
        self.sam_features_names = sorted(
            os.listdir(sam_features_path),  key=lambda x: int(x.split('_')[1])
            )

        self.model_input_names_path = model_input_names_path
        self.sam_features_path = sam_features_path


    def __len__(self):
        return len(self.model_input_names)

    def __getitem__(self, idx):
        input_name = self.model_input_names[idx]
        inp_path = os.path.join(self.model_input_names_path, input_name)
        inp = load_npy_gz(inp_path)

        target_ft_name = self.sam_features_names[idx]
        target_path = os.path.join(self.sam_features_path, target_ft_name)
        target = load_pkl(target_path)

        return inp, target


def collate_fn(batch):
    inps = []
    targets = []
    h_new, w_new = 48, 64
    for (inp, target) in batch:

        inp = torch.tensor(inp).permute(2, 0, 1).unsqueeze(0)
        inps.append(inp)

        target = torch.tensor(list(target.values()))
        target = target.reshape(-1, 64, 256).permute(2, 0, 1).unsqueeze(0)
        targets.append(target)

    return torch.cat(inps), torch.cat(targets)



def get_dataloader(batch_size):
    training_dataset = CompressDataset()
    return DataLoader(training_dataset, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=collate_fn)


def load_npy_gz(filename):
    with gzip.open(filename, 'rb') as f:
        array = np.load(f)
    return array

def load_pkl(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

if __name__ == '__main__':
    dataloader = get_dataloader(1)
    for (inp, target) in dataloader:
        print(inp.shape, target.shape)

        # plt.imshow(inp[0, 4, ...])
        # plt.savefig('inp.jpg')
        # plt.clf()

        # plt.imshow(target[0, 4, ...])
        # plt.savefig('target.jpg')
        # plt.clf()

        break