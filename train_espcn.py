"""
Train ESPCN for super-resolution (upscales the small working-resolution
output of the enhancement/fusion stages back to full camera resolution,
replacing plain bicubic resize).

Trains on any folder of clear, reasonably high-res images -- the dataset
class downsamples each one on the fly to create the (low-res, high-res)
training pair, so you don't need a separate low-res dataset.

Usage:
    python train_espcn.py --image_dir /content/train/visible \
        --scale_factor 4 --epochs 30 --batch_size 16
"""

import argparse
import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.espcn import ESPCN
from dataset import SRDataset
from utils import ssim_loss


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    paths = []
    for d in args.image_dir:
        paths += glob.glob(os.path.join(d, "*"))
    print(f"Found {len(paths)} images")

    dataset = SRDataset(paths, hr_size=args.hr_size, scale_factor=args.scale_factor)
    n_val = max(1, int(0.1 * len(dataset)))
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                               num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)

    model = ESPCN(scale_factor=args.scale_factor).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    ssim_lambda = 2.0  # lighter weight than DenseFuse -- MSE/PSNR is the primary SR metric

    os.makedirs(args.out_dir, exist_ok=True)
    best_val = float("inf")

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for lr_img, hr_img in train_loader:
            lr_img, hr_img = lr_img.to(device, non_blocking=True), hr_img.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred = model(lr_img)
                loss = mse(pred, hr_img) + ssim_lambda * ssim_loss(pred, hr_img)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * lr_img.size(0)
        train_loss /= len(train_set)

        model.eval()
        val_loss, val_psnr = 0.0, 0.0
        with torch.no_grad():
            for lr_img, hr_img in val_loader:
                lr_img, hr_img = lr_img.to(device), hr_img.to(device)
                pred = model(lr_img)
                loss = mse(pred, hr_img) + ssim_lambda * ssim_loss(pred, hr_img)
                val_loss += loss.item() * lr_img.size(0)
                mse_val = torch.mean((pred - hr_img) ** 2).item()
                val_psnr += (10 * torch.log10(torch.tensor(1.0 / max(mse_val, 1e-10)))).item() * lr_img.size(0)
        val_loss /= len(val_set)
        val_psnr /= len(val_set)

        print(f"Epoch {epoch+1}/{args.epochs} | train_loss={train_loss:.5f} "
              f"| val_loss={val_loss:.5f} | val_PSNR={val_psnr:.2f}dB")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(args.out_dir, f"espcn_x{args.scale_factor}_best.pth"))
            print(f"  -> saved new best checkpoint (val_loss={val_loss:.5f})")

    torch.save(model.state_dict(), os.path.join(args.out_dir, f"espcn_x{args.scale_factor}_last.pth"))
    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", nargs="+", required=True)
    parser.add_argument("--out_dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hr_size", type=int, default=256)
    parser.add_argument("--scale_factor", type=int, default=4, choices=[2, 3, 4])
    args = parser.parse_args()
    train(args)
