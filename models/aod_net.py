"""
AOD-Net: All-in-One Dehazing Network
Paper: Li, B. et al. "AOD-Net: All-in-One Dehazing Network." ICCV 2017.

I(x) = J(x) t(x) + A (1 - t(x))  reformulated into one learned variable K(x):
    J(x) = K(x) I(x) - K(x) + b
so the whole dehazing step is one tiny CNN + an algebraic recombination.
<10KB of weights, real-time even on CPU.
"""

import torch
import torch.nn as nn


class AODNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(3, 3, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(6, 3, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(6, 3, kernel_size=7, padding=3)
        self.conv5 = nn.Conv2d(12, 3, kernel_size=3, padding=1)
        self.b = 1.0
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        cat1 = torch.cat([x1, x2], dim=1)
        x3 = self.relu(self.conv3(cat1))
        cat2 = torch.cat([x2, x3], dim=1)
        x4 = self.relu(self.conv4(cat2))
        cat3 = torch.cat([x1, x2, x3, x4], dim=1)
        k = self.relu(self.conv5(cat3))
        out = k * x - k + self.b
        out = self.relu(out)
        return torch.clamp(out, 0, 1)
