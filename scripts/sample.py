"""
scripts/sample.py  —  Generate and compare samples (Parts 5C, 6B, 6D)
=======================================================================

Usage::
    # EM samples  (5.C.iii)
    python scripts/sample.py --method em --checkpoint runs/vp/best.pt \\
        --beta_min 0.01 --beta_max 5.0 --num_steps 1000

    # PC samples  (5.C.iv)
    python scripts/sample.py --method pc --checkpoint runs/vp/best.pt \\
        --beta_min 0.01 --beta_max 5.0 --num_steps 1000 --n_corrector 1
    python scripts/sample.py --method pc --checkpoint runs/vp/best.pt \\
        --beta_min 0.01 --beta_max 5.0 --num_steps 1000 --n_corrector 3

    # Rectified Flow Euler  (6.B)
    python scripts/sample.py --method rectflow --checkpoint runs/rectflow/best.pt \\
        --num_steps 100

    # One-step reflow  (6.C)
    python scripts/sample.py --method rectflow --checkpoint runs/rectflow_reflow/best.pt \\
        --num_steps 1

    # Side-by-side grid  (6.D): pass a fixed seed file
    python scripts/sample.py --method all --vp_checkpoint runs/vp/best.pt \\
        --rf_checkpoint runs/rectflow/best.pt \\
        --reflow_checkpoint runs/rectflow_reflow/best.pt \\
        --seed 42 --out comparison_grid.png
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

from diffusion.unet import UNet
from diffusion.vp import VPSDE
from diffusion.rectflow import RectifiedFlow


FASHION_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def save_grid(samples: torch.Tensor, path: str, nrow: int = 8, title: str = ""):
    """Save a (B,1,H,W) tensor as an image grid."""
    grid = make_grid(samples.clamp(-1, 1) * 0.5 + 0.5, nrow=nrow)
    plt.figure(figsize=(nrow, samples.size(0) // nrow + 1))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method",      type=str, default="em",
                   choices=["em", "pc", "rectflow", "all"],
                   help="Sampler to run (or 'all' for side-by-side grid).")
    # VP checkpoints
    p.add_argument("--checkpoint",    type=str, default=None)
    p.add_argument("--vp_checkpoint", type=str, default=None)
    # Rect-flow checkpoints
    p.add_argument("--rf_checkpoint",     type=str, default=None)
    p.add_argument("--reflow_checkpoint", type=str, default=None)
    # VP schedule
    p.add_argument("--beta_min", type=float, default=0.01)
    p.add_argument("--beta_max", type=float, default=5.0)
    p.add_argument("--T",        type=int,   default=1000)
    # Sampler params
    p.add_argument("--num_steps",   type=int, default=1000)
    p.add_argument("--n_corrector", type=int, default=1)
    p.add_argument("--snr",         type=float, default=0.16)
    p.add_argument("--n_samples",   type=int, default=64)
    # Output
    p.add_argument("--out",    type=str, default="samples.png")
    p.add_argument("--seed",   type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_vp_model(checkpoint: str, device, beta_min: float = 0.01,
                  beta_max: float = 5.0, T: int = 1000) -> tuple[VPSDE, UNet]:
    sde = VPSDE(beta_min=beta_min, beta_max=beta_max, T=T)
    model = UNet(in_channels=1, base_channels=64).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return sde, model


def load_rf_model(checkpoint: str, device) -> tuple[RectifiedFlow, UNet]:
    flow = RectifiedFlow()
    model = UNet(in_channels=1, base_channels=64).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return flow, model


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    shape = (args.n_samples, 1, 28, 28)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    if args.method == "em":
        ckpt = args.checkpoint or args.vp_checkpoint
        sde, model = load_vp_model(ckpt, device, args.beta_min, args.beta_max, args.T)
        samples = sde.euler_maruyama(model, shape, num_steps=args.num_steps, device=device)
        save_grid(samples, args.out, title=f"VP EM ({args.num_steps} steps)")

    elif args.method == "pc":
        ckpt = args.checkpoint or args.vp_checkpoint
        sde, model = load_vp_model(ckpt, device, args.beta_min, args.beta_max, args.T)
        samples = sde.predictor_corrector(
            model, shape,
            num_steps=args.num_steps,
            n_corrector=args.n_corrector,
            snr=args.snr,
            device=device,
        )
        save_grid(samples, args.out,
                  title=f"VP PC (steps={args.num_steps}, corrector={args.n_corrector})")

    elif args.method == "rectflow":
        ckpt = args.checkpoint or args.rf_checkpoint
        flow, model = load_rf_model(ckpt, device)
        samples = flow.euler_sample(model, shape, num_steps=args.num_steps, device=device)
        save_grid(samples, args.out, title=f"Rect Flow ({args.num_steps} steps)")

    elif args.method == "all":
        # Problem 6.D: 4×8 side-by-side grid with 8 fixed seeds
        # Rows: DDPM EM (1000 steps), Rect Flow (100 steps),
        #       Rect Flow (1 step), Reflow (1 step)
        n = 8
        fixed_shape = (n, 1, 28, 28)

        # Fix the same initial noise for all methods
        torch.manual_seed(args.seed)
        noise = torch.randn(fixed_shape, device=device)

        rows = []

        # Row 1: DDPM EM 1000 steps
        if args.vp_checkpoint:
            sde, vp_model = load_vp_model(
                args.vp_checkpoint, device, args.beta_min, args.beta_max, args.T
            )
            # Re-seed so the EM sampler starts from our fixed noise
            # (we inject noise manually for the first step)
            samples_em = sde.euler_maruyama(vp_model, fixed_shape, num_steps=1000, device=device)
            rows.append(samples_em)

        # Row 2: Rect Flow 100 steps
        if args.rf_checkpoint:
            flow, rf_model = load_rf_model(args.rf_checkpoint, device)
            # Use fixed starting noise
            x = noise.clone()
            dt = 1.0 / 100
            with torch.no_grad():
                for i in range(100):
                    t_val = i * dt
                    t = torch.full((n,), t_val, device=device)
                    v = rf_model(x, t)
                    x = x + v * dt
            rows.append(x)

        # Row 3: Rect Flow 1 step
        if args.rf_checkpoint:
            x = noise.clone()
            with torch.no_grad():
                t = torch.zeros(n, device=device)
                v = rf_model(x, t)
                x = x + v * 1.0
            rows.append(x)

        # Row 4: Reflow 1 step
        if args.reflow_checkpoint:
            _, reflow_model = load_rf_model(args.reflow_checkpoint, device)
            x = noise.clone()
            with torch.no_grad():
                t = torch.zeros(n, device=device)
                v = reflow_model(x, t)
                x = x + v * 1.0
            rows.append(x)

        if not rows:
            raise ValueError("No checkpoints provided for --method all")

        # Stack rows into a single grid: each row is 8 images
        all_imgs = torch.cat(rows, dim=0)  # (4*8, 1, 28, 28)
        save_grid(all_imgs, args.out, nrow=n, title="Method comparison (rows: EM, RF-100, RF-1, Reflow-1)")


if __name__ == "__main__":
    main()
