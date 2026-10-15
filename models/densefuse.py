"""
DenseFuse: A Fusion Approach to Infrared and Visible Images
Paper: Li, H. & Wu, X.J. IEEE Trans. Image Processing, 2019.

Shared encoder/decoder trained as a plain autoencoder (unsupervised,
input==target). At inference: encode RGB-Y and thermal separately with the
SAME encoder, combine features with an L1-norm activity rule (favors
whichever modality has stronger local signal), decode once.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseBlock(nn.Module):
    def __init__(self, in_ch, growth_ch=16, n_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()
        ch = in_ch
        for _ in range(n_layers):
            self.layers.append(nn.Sequential(
                nn.Conv2d(ch, growth_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ))
            ch += growth_ch
        self.out_channels = ch

    def forward(self, x):
        feats = [x]
        for layer in self.layers:
            y = layer(torch.cat(feats, dim=1))
            feats.append(y)
        return torch.cat(feats, dim=1)


class Encoder(nn.Module):
    def __init__(self, in_ch=1, base_ch=16):
        super().__init__()
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.dense = DenseBlock(base_ch, growth_ch=16, n_layers=3)
        self.out_channels = self.dense.out_channels

    def forward(self, x):
        return self.dense(self.conv_in(x))


class Decoder(nn.Module):
    def __init__(self, in_ch, out_ch=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, out_ch, kernel_size=3, padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class DenseFuseAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(in_ch=1)
        self.decoder = Decoder(in_ch=self.encoder.out_channels, out_ch=1)

    def forward(self, x):
        return self.decoder(self.encoder(x))


def l1_norm_fusion(feat_rgb, feat_thermal, window=1):
    act_rgb = F.avg_pool2d(feat_rgb.abs(), kernel_size=2 * window + 1, stride=1, padding=window)
    act_thermal = F.avg_pool2d(feat_thermal.abs(), kernel_size=2 * window + 1, stride=1, padding=window)
    total = act_rgb + act_thermal + 1e-8
    return (act_rgb / total) * feat_rgb + (act_thermal / total) * feat_thermal


class DenseFuseInference(nn.Module):
    def __init__(self, autoencoder: DenseFuseAutoencoder):
        super().__init__()
        self.encoder = autoencoder.encoder
        self.decoder = autoencoder.decoder

    def forward(self, y_rgb, y_thermal):
        feat_rgb = self.encoder(y_rgb)
        feat_thermal = self.encoder(y_thermal)
        fused_feat = l1_norm_fusion(feat_rgb, feat_thermal)
        return self.decoder(fused_feat)
