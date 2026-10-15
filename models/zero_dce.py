"""
Zero-DCE++: Learning to Enhance Low-Light Image via Zero-Reference
Deep Curve Estimation
Paper: Li, C., Guo, C., Loy, C.C. "Learning to Enhance Low-Light Image via
Zero-Reference Deep Curve Estimation." IEEE TPAMI 2021 (lightweight "++"
follow-up to the CVPR 2020 Zero-DCE paper).

Key idea: instead of learning a direct mapping dark->bright (which needs
paired ground truth that's expensive/impossible to get perfectly), the
network predicts a per-pixel HIGHER-ORDER CURVE that is applied to the
image iteratively:

    LE_n(x) = LE_{n-1}(x) + A(x) * (LE_{n-1}(x)^2 - LE_{n-1}(x))

applied n=8 times, where A(x) is a learned per-pixel adjustment map in
[-1, 1]. Because this is just a pixelwise polynomial, it's trivially fast
and -- crucially -- it is trained with NO paired data at all: four
"zero-reference" losses constrain the output using only the input image
itself:
  1. Spatial consistency  -- preserves local contrast/edges between input
     and output (output shouldn't lose structure relative to the input)
  2. Exposure control      -- pushes local patch brightness toward a
     well-exposed target level (not too dark, not blown out)
  3. Color constancy       -- gray-world assumption: the three channel
     averages should converge (corrects color casts introduced by
     brightening)
  4. Illumination smoothness -- total-variation penalty on the predicted
     curve maps, keeping the enhancement spatially smooth (no blotchy
     artifacts)

Zero-DCE++ (vs. the original Zero-DCE) uses depthwise separable
convolutions and shares ONE curve-parameter map across all 8 iterations
(instead of a fresh map per iteration), cutting parameters and compute
substantially while keeping quality close to the original -- this is why
it hits such high FPS even on modest GPUs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, padding=padding, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class ZeroDCEPP(nn.Module):
    def __init__(self, n_iter=8, base_ch=32):
        super().__init__()
        self.n_iter = n_iter
        self.relu = nn.ReLU(inplace=True)

        self.e_conv1 = DepthwiseSeparableConv(3, base_ch)
        self.e_conv2 = DepthwiseSeparableConv(base_ch, base_ch)
        self.e_conv3 = DepthwiseSeparableConv(base_ch, base_ch)
        self.e_conv4 = DepthwiseSeparableConv(base_ch, base_ch)
        self.e_conv5 = DepthwiseSeparableConv(base_ch * 2, base_ch)
        self.e_conv6 = DepthwiseSeparableConv(base_ch * 2, base_ch)
        # single shared 3-channel curve map, reused for all n_iter steps
        # (this sharing is the "++" efficiency trick vs. original Zero-DCE)
        self.e_conv7 = DepthwiseSeparableConv(base_ch * 2, 3)

    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], dim=1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], dim=1)))
        a = torch.tanh(self.e_conv7(torch.cat([x1, x6], dim=1)))  # curve params in [-1,1]

        enhanced = x
        for _ in range(self.n_iter):
            enhanced = enhanced + a * (torch.pow(enhanced, 2) - enhanced)

        return torch.clamp(enhanced, 0, 1), a


# ---------------------------------------------------------------------------
# Zero-reference losses (no ground-truth needed -- train on raw dark images)
# ---------------------------------------------------------------------------
class SpatialConsistencyLoss(nn.Module):
    """Keeps local gradients between neighboring regions similar before/after
    enhancement, so output doesn't lose or invent structure."""
    def __init__(self):
        super().__init__()
        kernel_left = torch.tensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]], dtype=torch.float32)
        kernel_right = torch.tensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]], dtype=torch.float32)
        kernel_up = torch.tensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32)
        kernel_down = torch.tensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=torch.float32)
        for name, k in [("left", kernel_left), ("right", kernel_right),
                         ("up", kernel_up), ("down", kernel_down)]:
            self.register_buffer(f"kernel_{name}", k.view(1, 1, 3, 3))
        self.pool = nn.AvgPool2d(4)

    def forward(self, enhanced, original):
        org_mean = original.mean(dim=1, keepdim=True)
        enh_mean = enhanced.mean(dim=1, keepdim=True)
        org_pool = self.pool(org_mean)
        enh_pool = self.pool(enh_mean)

        loss = 0.0
        for name in ["left", "right", "up", "down"]:
            k = getattr(self, f"kernel_{name}")
            d_org = F.conv2d(org_pool, k, padding=1)
            d_enh = F.conv2d(enh_pool, k, padding=1)
            loss = loss + torch.mean((d_org - d_enh) ** 2)
        return loss


class ExposureLoss(nn.Module):
    """Pushes local patch mean brightness toward a well-exposed target
    (paper default ~0.6 on a 0-1 scale)."""
    def __init__(self, patch_size=16, target_exposure=0.6):
        super().__init__()
        self.pool = nn.AvgPool2d(patch_size)
        self.target = target_exposure

    def forward(self, enhanced):
        mean_gray = enhanced.mean(dim=1, keepdim=True)
        patch_means = self.pool(mean_gray)
        return torch.mean((patch_means - self.target) ** 2)


class ColorConstancyLoss(nn.Module):
    """Gray-world assumption: average R, G, B channel intensities should
    converge toward each other, correcting color casts from brightening."""
    def forward(self, enhanced):
        mean_rgb = enhanced.mean(dim=[2, 3])
        r, g, b = mean_rgb[:, 0], mean_rgb[:, 1], mean_rgb[:, 2]
        return torch.mean((r - g) ** 2 + (g - b) ** 2 + (b - r) ** 2)


class IlluminationSmoothnessLoss(nn.Module):
    """Total-variation penalty on the predicted curve map, keeping the
    enhancement spatially smooth (prevents blotchy/noisy artifacts)."""
    def forward(self, curve_params):
        h, w = curve_params.shape[2], curve_params.shape[3]
        tv_h = torch.pow(curve_params[:, :, 1:, :] - curve_params[:, :, :-1, :], 2).sum()
        tv_w = torch.pow(curve_params[:, :, :, 1:] - curve_params[:, :, :, :-1], 2).sum()
        batch = curve_params.size(0)
        return (tv_h + tv_w) / (batch * h * w)


class ZeroDCELoss(nn.Module):
    """Combined zero-reference loss, weights per the paper's ablation."""
    def __init__(self, w_spa=1.0, w_exp=10.0, w_col=5.0, w_tv=200.0):
        super().__init__()
        self.spa = SpatialConsistencyLoss()
        self.exp = ExposureLoss()
        self.col = ColorConstancyLoss()
        self.tv = IlluminationSmoothnessLoss()
        self.w_spa, self.w_exp, self.w_col, self.w_tv = w_spa, w_exp, w_col, w_tv

    def forward(self, enhanced, original, curve_params):
        l_spa = self.spa(enhanced, original)
        l_exp = self.exp(enhanced)
        l_col = self.col(enhanced)
        l_tv = self.tv(curve_params)
        total = self.w_spa * l_spa + self.w_exp * l_exp + self.w_col * l_col + self.w_tv * l_tv
        return total, {"spatial": l_spa.item(), "exposure": l_exp.item(),
                        "color": l_col.item(), "smoothness": l_tv.item()}
