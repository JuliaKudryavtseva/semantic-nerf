import torch
import torch.nn as nn
from torch import optim

import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from compress_block import Compression
from data import get_dataloader


def train_epoch(model, train_loader, optimizer, loss_fn):
    model.train()
    running_loss = []
    for (model_input, target) in train_loader:
        model_input = model_input.cuda()
        target = target.cuda()

        output = model(model_input)

        loss = loss_fn(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss.append(loss.item())

        del model_input
        del target
        
    return model, np.mean(running_loss)


def train_model(model, train_loader, epochs, lr):

    optimizer = optim.SGD(model.parameters(), lr=lr)
    # scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-4, max_lr=lr, step_size_up=5, mode="triangular2")
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=lr, 
        steps_per_epoch=int(0.5*epochs), 
        epochs=epochs, 
        anneal_strategy='linear'
        )

    loss_fn = nn.MSELoss()

    train_losses = []
    model = model.cuda()
    best_loss = float('inf')


    pbar = tqdm(range(epochs))
    for epoch in tqdm(range(epochs)):
        model.train()
        model, epoch_loss = train_epoch(model, train_loader, optimizer, loss_fn)
        train_losses.append(epoch_loss)
        scheduler.step()

        if epoch_loss < best_loss:
            save_checkpoint(model, optimizer, 'compress_block', epoch)


        lr = optimizer.param_groups[0]["lr"]
        pbar.set_description(f'{epoch_loss=}, {lr=}')

        plt.plot(list(range(1, len(train_losses)+1)), train_losses)
        plt.savefig('loss.png')
        plt.clf()

    return train_losses


def save_checkpoint(model, optimizer, filename, EPOCH):
    save_path = os.path.join(os.getcwd(), 'checkpoints')
    os.makedirs(save_path, exist_ok=True)

    name_checkpoint = f'{filename}.pth'
    save_path = os.path.join(save_path, name_checkpoint)
    torch.save({
        'epoch': EPOCH,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, save_path) 


if __name__ == '__main__':
    import time


    paths = {
        'model_input_names_path': '../renders/train/raw-sam_features', 
        'sam_features_path':'../data/teatime/segmentation_results/seg_features'
        }

    train_dataloader = get_dataloader(3, paths)

    config = {
        'h_old': 738, 'h_new': 48, 
        'w_old': 994, 'w_new': 64, 
        'kernel_size_conv': 3, 
        'kernel_size_pool': 2
        }

    model = Compression(config)

    start_time = time.time()
    train_loss = train_model(model, train_dataloader, epochs=30, lr=1e-1)
    end_time = time.time()

    time_seconds = end_time - start_time
    np.save('losses.npy', np.array([time_seconds/60] + train_loss))

