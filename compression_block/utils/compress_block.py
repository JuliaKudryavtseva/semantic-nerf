import torch
import torch.nn as nn
import torch.nn.functional as F


class CompressionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size_conv, kernel_size_pool):
        super(CompressionBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size_conv, padding=0, stride=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.AvgPool2d(kernel_size=kernel_size_pool),

            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size_conv, padding=0, stride=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.block(x)
        return x


class Compression(nn.Module):
    def __init__(self, config):
        super(Compression, self).__init__()

        self.config = config
        self.n_blocks = min(get_n_blocks(config['h_old'], config['h_new']), get_n_blocks(config['w_old'], config['w_new']))

        # conv layers
        self.layers = nn.ModuleList([
            CompressionBlock(
                256, 256, 
                kernel_size_conv=config['kernel_size_conv'], 
                kernel_size_pool=config['kernel_size_pool']) for _ in range(self.n_blocks)
            ])

        # adaptive avr pool
        self.adapool = nn.AdaptiveAvgPool2d(
            (config['h_new'], config['w_new'])
            )
        self.maxpool = nn.AdaptiveMaxPool2d(
            (config['h_new'], config['w_new'])
            )

        self.head = nn.Conv2d(256, 256, kernel_size=1)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:

        x = inp
        # conv layers
        for layers in self.layers:
            x = layers(x)

        # # to the size of h_ew, w_new
        x = self.adapool(x) # (batch_size, 256, h_new, w_new)

        x = self.head(x)
        return x





def block(h, kernel_conv, kernel_pool):
    h = h - (kernel_conv - 1)
    h =  int((h-kernel_pool)/kernel_pool + 1)
    h = h - (kernel_conv - 1)
    return h


def get_n_blocks(h_old, h_new, kernel_conv=3, kernel_pool=2):
    n_blocks = 0
    while h_old > h_new:
        h_old = block(h_old, kernel_conv=kernel_conv, kernel_pool=kernel_pool)
        n_blocks+=1
        
    n_blocks = n_blocks - 1
    return n_blocks


if __name__ == '__main__':

    config = {
        'h_old': 738, 'h_new': 48, 
        'w_old': 994, 'w_new': 64, 
        'kernel_size_conv': 3, 
        'kernel_size_pool': 2
        }

    large_ft = torch.rand(1, 256, 738, 994)
    rest_ft = torch.rand(1, 256, 64-48, 64)

    model = Compression(config)
    output = model(large_ft, rest_ft)

    assert output.shape == (1, 256, 48, 64)
    print('\n === Compression is ready! :) === \n')