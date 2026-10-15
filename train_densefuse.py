"""
Train DenseFuse in two stages.

STAGE 1 (required): autoencoder pretraining on a mix of your visible +
    thermal frames (unsupervised, input==target).
STAGE 2 (recommended): fine-tune on your RGBT pairs with a self-supervised
    proxy loss (fused output should stay structurally close to BOTH sources).

Usage:
    python train_densefuse.py --stage autoencoder \
        --image_dir /content/train/visible /content/train/thermal \
        --epochs 15 --batch_size 16

    python train_densefuse.py --stage finetune \
        --rgbt_root /content/train \
        --pretrained ./checkpoints/densefuse_ae_best.pth \
        --epochs 10 --batch_size 8
"""

import argparse
import os
import glob
import torch
from torch.utils.data import DataLoader

from models.densefuse import DenseFuseAutoencoder, l1_norm_fusion
from dataset import DenseFuseDataset, RGBTPairDataset
from utils import ssim_loss


def train_autoencoder(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Stage 1: Autoencoder] Training on {device}")

    all_paths = []
    for d in args.image_dir:
        all_paths += glob.glob(os.path.join(d, "*"))
    print(f"Found {len(all_paths)} images across {len(args.image_dir)} folder(s)")

    dataset = DenseFuseDataset(all_paths, size=args.img_size)
    n_val = max(1, int(0.1 * len(dataset)))
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                               num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)

    model = DenseFuseAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    mse = torch.nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    ssim_lambda = 5.0

    os.makedirs(args.out_dir, exist_ok=True)
    best_val = float("inf")

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for img in train_loader:
            img = img.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                recon = model(img)
                loss = mse(recon, img) + ssim_lambda * ssim_loss(recon, img)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * img.size(0)
        train_loss /= len(train_set)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for img in val_loader:
                img = img.to(device)
                recon = model(img)
                loss = mse(recon, img) + ssim_lambda * ssim_loss(recon, img)
                val_loss += loss.item() * img.size(0)
        val_loss /= len(val_set)

        print(f"Epoch {epoch+1}/{args.epochs} | train_loss={train_loss:.5f} | val_loss={val_loss:.5f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(args.out_dir, "densefuse_ae_best.pth"))
            print(f"  -> saved new best checkpoint (val_loss={val_loss:.5f})")

    torch.save(model.state_dict(), os.path.join(args.out_dir, "densefuse_ae_last.pth"))
    print("Stage 1 complete.")


def finetune_fusion(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Stage 2: Fusion fine-tune] Training on {device}")

    ae = DenseFuseAutoencoder().to(device)
    ae.load_state_dict(torch.load(args.pretrained, map_location=device))

    dataset = RGBTPairDataset(args.rgbt_root, size=args.img_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                         num_workers=2, pin_memory=True)

    optimizer = torch.optim.Adam(ae.parameters(), lr=args.lr * 0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(args.epochs):
        ae.train()
        epoch_loss = 0.0
        for rgb, y_rgb, thm in loader:
            y_rgb, thm = y_rgb.to(device, non_blocking=True), thm.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                feat_rgb = ae.encoder(y_rgb)
                feat_thm = ae.encoder(thm)
                fused_feat = l1_norm_fusion(feat_rgb, feat_thm)
                fused = ae.decoder(fused_feat)
                loss = 0.5 * ssim_loss(fused, y_rgb) + 0.5 * ssim_loss(fused, thm)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item() * y_rgb.size(0)

        epoch_loss /= len(dataset)
        print(f"Epoch {epoch+1}/{args.epochs} | fusion_loss={epoch_loss:.5f}")

    torch.save(ae.state_dict(), os.path.join(args.out_dir, "densefuse_finetuned.pth"))
    print("Stage 2 complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["autoencoder", "finetune"], required=True)
    parser.add_argument("--image_dir", nargs="+")
    parser.add_argument("--rgbt_root", type=str)
    parser.add_argument("--pretrained", type=str)
    parser.add_argument("--out_dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=256)
    args = parser.parse_args()

    if args.stage == "autoencoder":
        assert args.image_dir, "--image_dir required for autoencoder stage"
        train_autoencoder(args)
    else:
        assert args.rgbt_root and args.pretrained, "--rgbt_root and --pretrained required"
        finetune_fusion(args)
