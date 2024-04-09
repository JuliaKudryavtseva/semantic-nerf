import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt


class Compression(nn.Module):
    def __init__(self, config):
        super(Compression, self).__init__()

        self.config = config
        self.kernel_size = get_kernel(config['h_old'], config['h_new'] )+1
        # conv 
        
        hidden_dim = 256
        self.convs = nn.Conv2d(256, hidden_dim, kernel_size=self.kernel_size, stride=self.kernel_size, bias=True)
        self.upsample = nn.Upsample(size=(config['h_old'], config['w_old']), mode='nearest')

        self.convs_transpose = nn.ConvTranspose2d(
            hidden_dim, 256, 
            kernel_size=self.kernel_size+2, 
            stride=self.kernel_size, 
            padding=0, 
            bias=True
            )
        

        # ada avr pool
        self.adapool = nn.AdaptiveAvgPool2d((config['h_new'], config['w_new']))
        self.h_new, self.w_new  = config['h_new'], config['w_new']

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        _, _, h, w = inp.shape

        x_avr = self.adapool(inp)
        x_avr_up = self.upsample(x_avr)

        x = self.convs(inp)
        x = self.convs_transpose(x)

        x = x[:, :, :h, :w]
        x += x_avr_up
       
        return x, x_avr


def get_kernel(H, h):
    k = H/h
    k_floor = int(k)
    k_ceil = math.ceil(k)
    h_cail = int((H - k_ceil) / k_ceil + 1)
    if h_cail > h:
        return k_ceil
    else:
        return k_floor



if __name__ == '__main__':

    config = {
        'h_old': 738, 'h_new': 48, 
        'w_old': 994, 'w_new': 64, 
        'kernel_size_conv': 3, 
        'kernel_size_pool': 2
        }

    large_ft = torch.rand(1, 256, 738, 994)

    model = Compression(config)
    output, _ = model(large_ft)
    print(output.shape)
    assert output.shape == (1, 256, 48, 64)
    print('\n === Compression is ready! :) === \n')