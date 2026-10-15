"""
Train Zero-DCE++ for low-light enhancement.

Needs ONLY dark/night images -- no paired ground truth (the four
zero-reference losses constrain training using just the input itself).
Use real night footage if you have it (extract frames from your dataset's
night sequences), or synthesize dark images from any clear-image folder:

    python -c "
    import glob
    from utils import make_synthetic_lowlight_dataset
    clear_paths = glob.glob('/content/train/visible/*')[:2000]
    make_synthetic_lowlight_dataset(clear_paths, '/content/synthetic_lowlight')
    "
    python train_zerodce.py --image_dir /content/synthetic_lowlight/dark --epochs 30 --batch_size 16
"""

import argparse
import os
import glob
import torch
from torch.utils.data import DataLoader

from models.zero_dce import ZeroDCEPP, ZeroDCELoss
from dataset import ZeroDCEDataset


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    paths = []
    for d in args.image_dir:
        paths += glob.glob(os.path.join(d, "*"))
    print(f"Found {len(paths)} dark/night images")

    dataset = ZeroDCEDataset(paths, size=args.img_size)
    n_val = max(1, int(0.05 * len(dataset)))
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                               num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)

    model = ZeroDCEPP(n_iter=args.n_iter).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = ZeroDCELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    os.makedirs(args.out_dir, exist_ok=True)
    best_val = float("inf")

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for img in train_loader:
            img = img.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                enhanced, curve = model(img)
                loss, parts = criterion(enhanced, img, curve)
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
                enhanced, curve = model(img)
                loss, _ = criterion(enhanced, img, curve)
                val_loss += loss.item() * img.size(0)
        val_loss /= len(val_set)

        print(f"Epoch {epoch+1}/{args.epochs} | train_loss={train_loss:.5f} | val_loss={val_loss:.5f} "
              f"| last_batch_breakdown={parts}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(args.out_dir, "zerodce_best.pth"))
            print(f"  -> saved new best checkpoint (val_loss={val_loss:.5f})")

    torch.save(model.state_dict(), os.path.join(args.out_dir, "zerodce_last.pth"))
    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", nargs="+", required=True,
                         help="One or more folders of dark/night images")
    parser.add_argument("--out_dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--n_iter", type=int, default=8, help="Curve iterations (paper default 8)")
    args = parser.parse_args()
    train(args)
