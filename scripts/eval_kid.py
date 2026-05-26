"""
scripts/eval_kid.py  —  Part 6B: KID evaluation
=================================================
Compute KID (Kernel Inception Distance) for each method and step count
to fill in the table in Problem 6.B.

Requires: pip install torch-fidelity

Usage::
    python scripts/eval_kid.py \\
        --vp_checkpoint  runs/vp/best.pt \\
        --rf_checkpoint  runs/rectflow/best.pt \\
        --beta_min 0.01 --beta_max 5.0 \\
        --n_samples 1000 --device cuda

The script prints a markdown table with KID mean ± std for each
(method, num_steps) combination.
"""

from __future__ import annotations

import argparse
import os
import tempfile

import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image

try:
    import torch_fidelity
except ImportError:
    raise ImportError(
        "torch-fidelity is required. Install with: pip install torch-fidelity"
    )

from diffusion.unet import UNet
from diffusion.vp import VPSDE
from diffusion.rectflow import RectifiedFlow


STEP_COUNTS = [1, 5, 10, 50, 100, 200, 1000]
METHODS = ["rectflow", "ddim", "em"]


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vp_checkpoint", type=str, required=True)
    p.add_argument("--rf_checkpoint", type=str, required=True)
    p.add_argument("--beta_min",  type=float, default=0.01)
    p.add_argument("--beta_max",  type=float, default=5.0)
    p.add_argument("--T",         type=int,   default=1000)
    p.add_argument("--n_samples", type=int,   default=1000)
    p.add_argument("--device",    type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def save_samples_to_dir(samples: torch.Tensor, directory: str):
    """Save (B,1,H,W) samples to individual PNG files for torch-fidelity."""
    os.makedirs(directory, exist_ok=True)
    samples = (samples.clamp(-1, 1) * 0.5 + 0.5)  # [0,1]
    for i, img in enumerate(samples):
        save_image(img, os.path.join(directory, f"{i:05d}.png"))


def compute_kid(generated_dir: str, real_dir: str) -> dict:
    metrics = torch_fidelity.calculate_metrics(
        input1=generated_dir,
        input2=real_dir,
        kid=True,
        kid_subset_size=min(1000, len(os.listdir(generated_dir))),
        verbose=False,
    )
    return metrics


def main():
    args = get_args()
    device = torch.device(args.device)

    device = torch.device(args.device)

    # --- Load models ---
    sde = VPSDE(beta_min=args.beta_min, beta_max=args.beta_max, T=args.T)
    vp_model = UNet(in_channels=1, base_channels=64).to(device)
    vp_model.load_state_dict(torch.load(args.vp_checkpoint, map_location=device))
    vp_model.eval()

    flow = RectifiedFlow()
    rf_model = UNet(in_channels=1, base_channels=64).to(device)
    rf_model.load_state_dict(torch.load(args.rf_checkpoint, map_location=device))
    rf_model.eval()

    # --- Build real-data reference directory ---
    real_dir = tempfile.mkdtemp(prefix="kid_real_")
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    val_ds = datasets.FashionMNIST("data", train=False, download=True, transform=tf)
    real_samples = torch.stack([val_ds[i][0] for i in range(args.n_samples)])
    save_samples_to_dir(real_samples, real_dir)

    shape = (args.n_samples, 1, 28, 28)
    results = {}  # (method, steps) -> KID dict

    for steps in STEP_COUNTS:
        print(f"\n=== steps={steps} ===")

        # Rectified Flow (Euler ODE)
        print("  rectflow ...", end="", flush=True)
        samples = flow.euler_sample(rf_model, shape, num_steps=steps, device=device)
        gen_dir = tempfile.mkdtemp(prefix=f"kid_rf_{steps}_")
        save_samples_to_dir(samples, gen_dir)
        kid = compute_kid(gen_dir, real_dir)
        results[("rectflow", steps)] = kid
        print(f" KID={kid['kernel_inception_distance_mean']:.4f}")

        # DDIM (deterministic reverse SDE — zero noise in EM predictor)
        # Implemented as EM with noise zeroed out (deterministic ODE approximation)
        if steps <= 200:  # skip 1000-step DDIM for speed
            print("  ddim ...", end="", flush=True)
            with torch.no_grad():
                import math as _math
                dt = 1.0 / steps
                t1 = torch.ones(shape[0], device=device)
                sigma1 = sde.sigma(t1).view(shape[0], 1, 1, 1)
                x = sigma1 * torch.randn(shape, device=device)
                for i in range(steps):
                    t_val = 1.0 - i * dt
                    t = torch.full((shape[0],), t_val, device=device)
                    beta_t = sde.beta(t).view(shape[0], 1, 1, 1)
                    sigma_t = sde.sigma(t).view(shape[0], 1, 1, 1)
                    eps_pred = vp_model(x, t)
                    score = -eps_pred / (sigma_t + 1e-6)
                    # Deterministic (no noise term)
                    drift = 0.5 * beta_t * x + beta_t * score
                    x = x + drift * dt
                samples_ddim = x.clamp(-1.0, 1.0)
            gen_dir = tempfile.mkdtemp(prefix=f"kid_ddim_{steps}_")
            save_samples_to_dir(samples_ddim, gen_dir)
            kid = compute_kid(gen_dir, real_dir)
            results[("ddim", steps)] = kid
            print(f" KID={kid['kernel_inception_distance_mean']:.4f}")
        else:
            results[("ddim", steps)] = None

        # DDPM EM (stochastic)
        if steps == 1000:
            print("  em (baseline) ...", end="", flush=True)
            samples_em = sde.euler_maruyama(vp_model, shape, num_steps=steps, device=device)
            gen_dir = tempfile.mkdtemp(prefix=f"kid_em_{steps}_")
            save_samples_to_dir(samples_em, gen_dir)
            kid = compute_kid(gen_dir, real_dir)
            results[("em", steps)] = kid
            print(f" KID={kid['kernel_inception_distance_mean']:.4f}")

    # --- Print markdown table ---
    print("\n| Steps | Flow Matching | DDIM | DDPM EM |")
    print("|-------|--------------|------|---------|")
    for steps in STEP_COUNTS:
        rf_kid = results.get(("rectflow", steps))
        ddim_kid = results.get(("ddim", steps))
        em_kid = results.get(("em", steps))

        def fmt(d):
            if d is None:
                return "—"
            mean = d["kernel_inception_distance_mean"]
            std  = d["kernel_inception_distance_std"]
            return f"{mean:.4f} ± {std:.4f}"

        em_str = fmt(em_kid) if em_kid else ("*(baseline)*" if steps == 1000 else "—")
        print(f"| {steps} | {fmt(rf_kid)} | {fmt(ddim_kid)} | {em_str} |")


if __name__ == "__main__":
    main()
