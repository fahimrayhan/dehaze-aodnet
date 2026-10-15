"""
Shared utilities: SSIM loss, synthetic haze generation, condition scoring
(haze/darkness) implemented as GPU tensor ops so the live pipeline never
has to round-trip through numpy/CPU just to decide which branch to run.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2


# ---------------------------------------------------------------------------
# SSIM loss (used for DenseFuse training)
# ---------------------------------------------------------------------------
def _gaussian_window(window_size=11, sigma=1.5):
    coords = torch.arange(window_size).float() - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    window_2d = g.unsqueeze(0) * g.unsqueeze(1)
    return window_2d.unsqueeze(0).unsqueeze(0)


def ssim(img1, img2, window_size=11):
    channels = img1.size(1)
    window = _gaussian_window(window_size).to(img1.device)
    window = window.expand(channels, 1, window_size, window_size)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channels)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channels)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channels) - mu1_mu2

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def ssim_loss(img1, img2, window_size=11):
    return 1 - ssim(img1, img2, window_size)


# ---------------------------------------------------------------------------
# Synthetic haze generation (for AOD-Net training pairs)
# ---------------------------------------------------------------------------
def synthesize_haze(clear_img, beta=None, A=None):
    """clear_img: HxWx3 uint8 RGB. Returns (hazy uint8, transmission map)."""
    h, w = clear_img.shape[:2]
    if beta is None:
        beta = np.random.uniform(0.8, 1.6)
    if A is None:
        A = np.random.uniform(0.7, 1.0)

    depth = np.tile(np.linspace(1.0, 0.2, h).reshape(h, 1), (1, w))
    depth += np.random.normal(0, 0.03, size=(h, w))
    depth = np.clip(depth, 0.05, 1.0)

    t = np.exp(-beta * depth)
    t3 = np.stack([t] * 3, axis=-1)

    clear_f = clear_img.astype(np.float32) / 255.0
    hazy_f = clear_f * t3 + A * (1 - t3)
    hazy_img = np.clip(hazy_f * 255, 0, 255).astype(np.uint8)
    return hazy_img, t.astype(np.float32)


def make_synthetic_haze_dataset(clear_image_paths, out_dir, size=256):
    import os
    os.makedirs(os.path.join(out_dir, "hazy"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "clear"), exist_ok=True)
    for i, path in enumerate(clear_image_paths):
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size, size))
        hazy, _ = synthesize_haze(img)
        cv2.imwrite(os.path.join(out_dir, "clear", f"{i:05d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(out_dir, "hazy", f"{i:05d}.png"), cv2.cvtColor(hazy, cv2.COLOR_RGB2BGR))
    print(f"Generated {len(clear_image_paths)} synthetic haze pairs in {out_dir}")


def make_synthetic_lowlight_dataset(clear_image_paths, out_dir, size=256):
    """Darken clear images (gamma + gain reduction + noise) to make
    zero-reference training data pairs look realistic -- note Zero-DCE
    training itself only needs the DARK image (no ground truth target),
    but having the clear original alongside lets you visually sanity-check
    enhancement quality by eye during training."""
    import os
    os.makedirs(os.path.join(out_dir, "dark"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "clear"), exist_ok=True)
    for i, path in enumerate(clear_image_paths):
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size, size)).astype(np.float32) / 255.0

        gamma = np.random.uniform(2.0, 3.5)
        gain = np.random.uniform(0.15, 0.4)
        dark = np.power(img, gamma) * gain
        noise = np.random.normal(0, 0.01, dark.shape)
        dark = np.clip(dark + noise, 0, 1)

        cv2.imwrite(os.path.join(out_dir, "clear", f"{i:05d}.png"),
                    cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(out_dir, "dark", f"{i:05d}.png"),
                    cv2.cvtColor((dark * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    print(f"Generated {len(clear_image_paths)} synthetic low-light images in {out_dir}")


# ---------------------------------------------------------------------------
# GPU-resident condition scoring (no CPU round-trip)
# ---------------------------------------------------------------------------
def gpu_haze_score(rgb_tensor, patch=15):
    """rgb_tensor: 1x3xHxW float in [0,1] on GPU. Returns a scalar tensor.
    Cheap approximation of the dark-channel heuristic entirely in torch,
    evaluated on a small downsampled copy so it stays nearly free."""
    small = F.interpolate(rgb_tensor, size=(64, 64), mode="area")
    min_ch = small.min(dim=1, keepdim=True).values
    dark = -F.max_pool2d(-min_ch, kernel_size=patch // 2 * 2 + 1, stride=1,
                          padding=patch // 2)  # erosion via negated max-pool
    dc_mean = dark.mean()

    maxc = small.max(dim=1, keepdim=True).values
    minc = small.min(dim=1, keepdim=True).values
    sat = (maxc - minc) / (maxc + 1e-6)
    sat_mean = sat.mean()

    gray = small.mean(dim=1, keepdim=True)
    contrast = gray.std()

    return dc_mean * (1 - sat_mean) * (1 - torch.clamp(contrast, 0, 1))


def gpu_brightness_score(rgb_tensor):
    """rgb_tensor: 1x3xHxW float in [0,1] on GPU. Returns mean brightness in [0,1]."""
    return rgb_tensor.mean()
