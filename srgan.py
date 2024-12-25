import torch
import torch.nn as nn
from typing import List 

disc_config = [
    (3, 64, 1),
    (3, 64, 2),
    (3, 128, 1),
    (3, 128, 2),
    (3, 256, 1),
    (3, 256, 2),
    (3, 512, 1),
    (3, 512, 2),
]


class ConvBlock(nn.Module):
    def __init__(self, 
                 in_channels: int=3,
                 out_channels: int=64,
                 kernel_size: int=3,
                 stride: int=1,
                 bn_fn: nn.Module=nn.Identity(),
                 act_fn: nn.Module=nn.Identity(),
                 **kwargs
                 ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, bias= not isinstance(bn_fn, nn.BatchNorm2d), **kwargs)
        self.bn = bn_fn
        self.act = act_fn

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    def __init__(self, block1: ConvBlock, block2: ConvBlock):
        super().__init__()
        self.block1 = block1
        self.block2 = block2
    def forward(self, x):
        out = self.block2(self.block1(x))
        return x + out

class Generator(nn.Module):
    def __init__(self, in_channels: int=3, num_channels: int=64, num_blocks: int=16):
        super().__init__()
        self.pre_block = ConvBlock(in_channels=in_channels, out_channels=num_channels, 
                                   kernel_size=9, stride=1, padding=4,  
                                   act_fn=nn.PReLU(num_parameters=num_channels))
        self.res_blocks = self.make_residual_layers(num_blocks, num_channels)
        self.conv = ConvBlock(in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=1, padding=1, bn_fn=nn.BatchNorm2d(num_channels))
        self.upsample = self.make_upsample_blocks(channels=num_channels, scale_factor=2) 
        self.final = nn.Conv2d(in_channels=num_channels, out_channels=in_channels, kernel_size=9, stride=1, padding=4)

    def make_residual_layers(self, num_blocks: int=16, channels: int=64):
        blocks = []
        for _ in range(num_blocks):
            block1 = ConvBlock(in_channels=channels, 
                               out_channels=channels, 
                               kernel_size=3, 
                               stride=1, 
                               padding=1,
                               bn_fn=nn.BatchNorm2d(channels),
                               act_fn=nn.PReLU(num_parameters=channels)
                               )
            block2 = ConvBlock(in_channels=channels, 
                               out_channels=channels, 
                               kernel_size=3, 
                               stride=1, 
                               padding=1,
                               bn_fn=nn.BatchNorm2d(channels),
                               act_fn=nn.PReLU(num_parameters=channels)
                               )
            blocks.append(ResBlock(block1, block2))
        return nn.Sequential(*blocks)
            
    def make_upsample_blocks(self, channels: int=64, scale_factor: int=2):
        upsample_blocks = nn.Sequential(
            nn.Conv2d(channels, channels * scale_factor**2, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels * scale_factor**2, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.PReLU(channels)
        )
        return upsample_blocks

    def forward(self, x):
        pre = self.pre_block(x)
        out = self.res_blocks(pre)
        out = self.conv(out) + pre
        out = self.upsample(out)
        out = self.final(out)
        return torch.tanh(out) 


class Discriminator(nn.Module):
    def __init__(self, in_channels: int=3, disc_config: List=None):
        super().__init__()
        assert disc_config is not None, "must provide disc_config for discrimiator"
        
        self.blocks = self.make_layers(in_channels, disc_config)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(512*6*6, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1)
        )
    
    def make_layers(self, in_channels: int=3, disc_config: List=None):
        blocks = []
        for idx, (kernel_size, out_channels, stride) in enumerate(disc_config):
            blocks.append(
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=1,
                    bn_fn=nn.BatchNorm2d(out_channels) if idx!=0 else nn.Identity(),
                    act_fn=nn.LeakyReLU(0.2, inplace=True)
                )
            )
            in_channels = out_channels
        return nn.Sequential(*blocks)

    def forward(self, x):
        return self.head(self.blocks(x))
        

def test():
    lr = 24 # HR = lr * scale_factor**2
    x = torch.randn((5, 3, lr, lr))
    gen = Generator()
    gen_out = gen(x)
    disc = Discriminator(disc_config=disc_config)
    print(disc)
    disc_out = disc(gen_out)

    print(gen_out.shape)
    print(disc_out.shape)


if __name__ == "__main__":
    test()