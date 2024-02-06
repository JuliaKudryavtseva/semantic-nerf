import torch.nn as nn
import torch
import os

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_size = 1024

    def forward(self, x):
        return x