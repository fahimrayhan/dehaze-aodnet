"""
Dataset classes for all four training stages: AOD-Net (dehazing),
Zero-DCE++ (low-light), DenseFuse (fusion), ESPCN (super-resolution).
"""

import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def _load_rgb(path, size=256):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    return img


def _load_gray(path, size=256):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        img_c = cv2.imread(path)
        img = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (size, size))
    return img


class AODDataset(Dataset):
    """root_dir/hazy/*.png + root_dir/clear/*.png, matching filenames."""
    def __init__(self, root_dir, size=256):
        self.hazy_files = sorted(glob.glob(os.path.join(root_dir, "hazy", "*")))
        self.clear_files = sorted(glob.glob(os.path.join(root_dir, "clear", "*")))
        assert len(self.hazy_files) == len(self.clear_files), "Mismatched hazy/clear counts"
        self.size = size

    def __len__(self):
        return len(self.hazy_files)

    def __getitem__(self, idx):
        hazy = _load_rgb(self.hazy_files[idx], self.size).astype(np.float32) / 255.0
        clear = _load_rgb(self.clear_files[idx], self.size).astype(np.float32) / 255.0
        return torch.from_numpy(hazy).permute(2, 0, 1), torch.from_numpy(clear).permute(2, 0, 1)


class ZeroDCEDataset(Dataset):
    """Zero-reference training needs ONLY dark images -- no paired ground
    truth required by the loss itself. Pass a folder of dark/night images
    (real or synthetic via utils.make_synthetic_lowlight_dataset)."""
    def __init__(self, image_paths, size=256):
        self.paths = image_paths
        self.size = size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = _load_rgb(self.paths[idx], self.size).astype(np.float32) / 255.0
        return torch.from_numpy(img).permute(2, 0, 1)


class DenseFuseDataset(Dataset):
    """Single-image dataset for autoencoder pretraining (any grayscale-
    convertible images -- mix your visible + thermal frames)."""
    def __init__(self, image_paths, size=256):
        self.paths = image_paths
        self.size = size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        gray = _load_gray(self.paths[idx], self.size).astype(np.float32) / 255.0
        return torch.from_numpy(gray).unsqueeze(0)


class RGBTPairDataset(Dataset):
    """root/visible/xxx.jpg + root/thermal/xxx.jpg, matching basenames."""
    def __init__(self, root_dir, size=256):
        vis_dir = os.path.join(root_dir, "visible")
        thm_dir = os.path.join(root_dir, "thermal")
        self.vis_files = sorted(glob.glob(os.path.join(vis_dir, "*")))
        self.thm_files = sorted(glob.glob(os.path.join(thm_dir, "*")))
        assert len(self.vis_files) == len(self.thm_files), "Mismatched visible/thermal counts"
        self.size = size

    def __len__(self):
        return len(self.vis_files)

    def __getitem__(self, idx):
        rgb = _load_rgb(self.vis_files[idx], self.size)
        thm = _load_gray(self.thm_files[idx], self.size)

        ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
        y = ycrcb[:, :, 0].astype(np.float32) / 255.0
        thm = thm.astype(np.float32) / 255.0

        y_t = torch.from_numpy(y).unsqueeze(0)
        thm_t = torch.from_numpy(thm).unsqueeze(0)
        rgb_t = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
        return rgb_t, y_t, thm_t


class SRDataset(Dataset):
    """Generates (low-res, high-res) pairs on the fly by downsampling any
    folder of clear, high-resolution images -- standard SR training recipe
    (no separate LR dataset needed, downsampling the HR image IS the LR input)."""
    def __init__(self, image_paths, hr_size=256, scale_factor=4):
        self.paths = image_paths
        self.hr_size = hr_size
        self.lr_size = hr_size // scale_factor
        self.scale_factor = scale_factor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        hr = _load_rgb(self.paths[idx], self.hr_size)
        lr = cv2.resize(hr, (self.lr_size, self.lr_size), interpolation=cv2.INTER_CUBIC)
        hr = hr.astype(np.float32) / 255.0
        lr = lr.astype(np.float32) / 255.0
        return torch.from_numpy(lr).permute(2, 0, 1), torch.from_numpy(hr).permute(2, 0, 1)
