
import torch.nn as nn
import torch

from compress_block import Compression
from data import get_dataloader

import matplotlib.pyplot as plt
import os
import numpy as np


def eval(model, test_loader, metrics):
    model.eval()
    batch_metrics = []
    for (model_input, target) in test_loader:
        model_input = model_input.cuda()
        target = target.cuda()

        output = model(model_input)

        metrics_val = metrics(output, target)
        batch_metrics.append(metrics_val.item())

        fig, ax = plt.subplots(nrows=1, ncols=2)
        ax[0].imshow(output[0, 4, ...].cpu().detach().numpy())
        ax[1].imshow(target[0, 4, ...].cpu().detach().numpy())
        plt.savefig('eval_target.png')
        plt.clf()

        del model_input
        del target
        del output

    return np.mean(batch_metrics), batch_metrics

def load_checkpoint(model, path='./checkpoints/compress_block.pth'):
    checkpoint = torch.load(path, map_location='cpu')
    return model.load_state_dict(checkpoint['model_state_dict'])


if __name__ == '__main__':


    paths = {
        'model_input_names_path': '../renders/train/raw-sam_features', 
        'sam_features_path':'../data/teatime/segmentation_results/seg_features'
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

    mean_metrics, all_metrics = eval(model, test_dataloader, metrics=nn.MSELoss())
    print(f'{mean_metrics=}, {all_metrics=}')

