
import torch.nn as nn
import torch

import cv2
from compress_block import Compression
from data import get_dataloader

import matplotlib.pyplot as plt
import os
import numpy as np   




def eval(model, test_loader, frame_names, metrics):
    model.eval()
    batch_metrics = []

    ada_layer = get_ada_layer()

    for frame_name, (model_input, target) in zip(frame_names, test_loader):
        model_input = model_input.cuda()

        target = target.cuda()
        model_input = model_input.cuda()



        output, ada_pool= model(model_input)

        # project to 64 * 64
        # output = ada_layer(output)
        # target = ada_layer(target)

        # save 
        torch.save(ada_pool, os.path.join(compress_ft_path, f'{frame_name}.pt'))
        print(frame_name)

        # mse_before = metrics(ada_pool, target).item()
        # mse_after = metrics(output, target).item()
        # delta = mse_after-mse_before
        # batch_metrics.append(mse_after)
        # print(f'{mse_before=}, {mse_after=}, {delta=}')

        del model_input
        del target
        del output

    return np.mean(batch_metrics)


def load_checkpoint(model, path='./checkpoints/compress_block.pth'):
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    model_number_of_params = sum(p.numel() for p in model.parameters())
    print(1e-6*model_number_of_params, ' mil.')



def get_ada_layer():
    fmap = np.load(f'data/{DATA_PATH}/segmentation_results/features_map.npy')

    # one of side is always 64
    new_dim = int((fmap.max()+1)/64)
    
    #choose dims
    if np.argmax(fmap.shape):
        new_h, new_w = new_dim, 64
        fill_h, fill_w = 64-new_dim, 64
    else:
        new_h, new_w = 64, new_dim
        fill_h, fill_w =64, 64-new_dim

    ada_layer = torch.nn.AdaptiveAvgPool2d((new_h, new_w))
    return ada_layer


if __name__ == '__main__':
    DATA_PATH = os.environ['DATA_PATH']
    REGULAR = os.environ['REGULAR']

    paths = {
        'model_input_names_path': 'renders/{DATA_PATH}/{REGULAR}/test/raw-sam_features', 
        'sam_features_path':f'data/{DATA_PATH}/segmentation_results/seg_features'
        }

    test_dataloader = get_dataloader(1, paths)

    config = {
        'h_old': 738, 'h_new': 48, 
        'w_old': 994, 'w_new': 64, 
        'kernel_size_conv': 3, 
        'kernel_size_pool': 2
        }

    model = Compression(config)
    load_checkpoint(model)
    model = model.cuda()

    compress_ft_path = os.path.join(f"data/{DATA_PATH}/{REGULAR}/", 'compress_features')
    os.makedirs(compress_ft_path, exist_ok=True)

    frame_names = [frame.split('.')[0] for frame in os.listdir(paths['model_input_names_path'])]
    frame_names = sorted(frame_names, key=lambda x: int(x.split('_')[1].split('.')[0]))
    mean_metrics = eval(model, test_dataloader, frame_names, metrics=nn.MSELoss())
    print(f'{mean_metrics=}')




