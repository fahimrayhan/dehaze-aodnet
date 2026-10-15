"""
ESPCN: Efficient Sub-Pixel Convolutional Neural Network
Paper: Shi, W. et al. "Real-Time Single Image and Video Super-Resolution
Using an Efficient Sub-Pixel Convolutional Neural Network." CVPR 2016.

Key idea: do all feature extraction at the LOW input resolution, and only
expand to the high-res grid in the very last layer via a "sub-pixel
convolution" (pixel shuffle): the last conv produces out_ch * r^2 channels
at low-res, which are then rearranged (no learned weights needed for this
step -- it's a pure reshape) into out_ch channels at r times the spatial
resolution. This is why ESPCN is so much faster than architectures that
upsample early and then convolve at high-res: all the expensive conv
compute happens on the small feature maps, not the big output grid.
"""

import torch
import torch.nn as nn


class ESPCN(nn.Module):
    def __init__(self, scale_factor=4, channels=3, base_ch=64):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv1 = nn.Conv2d(channels, base_ch, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(base_ch, base_ch // 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(base_ch // 2, channels * (scale_factor ** 2),
                                kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.tanh = nn.Tanh()
        self._init_weights()

    def _init_weights(self):
        # standard ESPCN init: orthogonal for the sub-pixel conv layer helps
        # avoid checkerboard artifacts from pixel shuffle
        for m in [self.conv1, self.conv2]:
            nn.init.kaiming_normal_(m.weight, nonlinearity="tanh")
            nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.conv3.weight)
        nn.init.zeros_(self.conv3.bias)

    def forward(self, x):
        x = self.tanh(self.conv1(x))
        x = self.tanh(self.conv2(x))
        x = self.conv3(x)
        x = self.pixel_shuffle(x)
        return torch.clamp(x, 0, 1)
