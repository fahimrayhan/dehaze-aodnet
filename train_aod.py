"""
Train AOD-Net for dehazing.

Usage:
    python -c "
    import glob
    from utils import make_synthetic_haze_dataset
    clear_paths = glob.glob('/content/train/visible/*')[:2000]
    make_synthetic_haze_dataset(clear_paths, '/content/synthetic_haze')
    "
    python train_aod.py --data_root /content/synthetic_haze --epochs 20 --batch_size 16
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.aod_net import AODNet
from dataset import AODDataset


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    dataset = AODDataset(args.data_root, size=args.img_size)
    n_val = max(1, int(0.1 * len(dataset)))
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                               num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)

    model = AODNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_loss = float("inf")
    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for hazy, clear in train_loader:
            hazy, clear = hazy.to(device, non_blocking=True), clear.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred = model(hazy)
                loss = criterion(pred, clear)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * hazy.size(0)
        train_loss /= len(train_set)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for hazy, clear in val_loader:
                hazy, clear = hazy.to(device), clear.to(device)
                pred = model(hazy)
                val_loss += criterion(pred, clear).item() * hazy.size(0)
        val_loss /= len(val_set)

        print(f"Epoch {epoch+1}/{args.epochs} | train_loss={train_loss:.5f} | val_loss={val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.out_dir, "aod_net_best.pth"))
            print(f"  -> saved new best checkpoint (val_loss={val_loss:.5f})")

    torch.save(model.state_dict(), os.path.join(args.out_dir, "aod_net_last.pth"))
    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img_size", type=int, default=256)
    args = parser.parse_args()
    train(args)
