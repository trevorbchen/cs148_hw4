"""
scripts/train_vp.py  —  Part 5: Train a VP score model on FashionMNIST
=======================================================================

Usage::
    python scripts/train_vp.py --config configs/vp_fashionmnist.yaml
    python scripts/train_vp.py --beta_min 0.01 --beta_max 5.0 --epochs 50
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from diffusion.unet import UNet
from diffusion.vp import VPSDE


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",    type=str,   default=None)
    p.add_argument("--beta_min",  type=float, default=0.01)
    p.add_argument("--beta_max",  type=float, default=5.0)
    p.add_argument("--T",         type=int,   default=1000)
    p.add_argument("--epochs",    type=int,   default=50)
    p.add_argument("--lr",        type=float, default=1e-4)
    p.add_argument("--batch_size",type=int,   default=128)
    p.add_argument("--save_dir",  type=str,   default="runs/vp")
    p.add_argument("--device",    type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    # Early stopping
    p.add_argument("--patience",  type=int,   default=10,
                   help="Stop if val loss does not improve for this many epochs. 0 = disabled.")
    return p.parse_args()


def build_dataloader(batch_size: int):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),   # map to [-1, 1]
    ])
    train_ds = datasets.FashionMNIST("data", train=True,  download=True, transform=tf)
    val_ds   = datasets.FashionMNIST("data", train=False, download=True, transform=tf)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_dl, val_dl


def score_loss(sde: VPSDE, model: torch.nn.Module, x0: torch.Tensor, device) -> torch.Tensor:
    """DSM training loss for the VP score model.

    Samples a random continuous time t ~ Uniform(0, 1), noises x0 to x_t
    via the VP marginal, and regresses the score (or equivalently the noise).

    Args:
        sde:    VPSDE instance.
        model:  Score network s_θ.
        x0:     Clean images, shape (B, 1, 28, 28) in [-1, 1].
        device: Compute device.

    Returns:
        Scalar loss.
    """
    B = x0.shape[0]
    t = torch.rand(B, device=device)          # t ~ Uniform(0, 1)
    x_t, eps = sde.marginal(x0, t)            # forward noising
    eps_pred = model(x_t, t)                  # model predicts eps (epsilon parameterization)
    return F.mse_loss(eps_pred, eps)


def main():
    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)

    sde   = VPSDE(beta_min=args.beta_min, beta_max=args.beta_max, T=args.T)
    model = UNet(in_channels=1, base_channels=64).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_dl, val_dl = build_dataloader(args.batch_size)

    train_losses, val_losses = [], []
    best_val = math.inf
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        # --- Training ---
        model.train()
        running = 0.0
        for x0, _ in train_dl:
            x0 = x0.to(device)
            loss = score_loss(sde, model, x0, device)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * x0.size(0)
        train_loss = running / len(train_dl.dataset)

        # --- Validation ---
        model.eval()
        running = 0.0
        with torch.no_grad():
            for x0, _ in val_dl:
                x0 = x0.to(device)
                loss = score_loss(sde, model, x0, device)
                running += loss.item() * x0.size(0)
        val_loss = running / len(val_dl.dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch:3d}/{args.epochs} | train {train_loss:.4f} | val {val_loss:.4f}")

        # Checkpoint
        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Path(args.save_dir) / "best.pt")
        else:
            patience_counter += 1

        if args.patience > 0 and patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs).")
            break

    # Save loss curves
    import numpy as np
    np.save(Path(args.save_dir) / "train_losses.npy", np.array(train_losses))
    np.save(Path(args.save_dir) / "val_losses.npy",   np.array(val_losses))
    print(f"Training complete. Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
