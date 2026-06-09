"""
Train or solve one (config, seed) pair and save results to results/.

PINN usage (via SLURM array):
    python train.py --job_id 0              # config 0, seed SEEDS[0]
    python train.py --job_id 42 --n_seeds 8

FD usage (run once per config; deterministic, no seed needed):
    python train.py --fd_config_idx 0
    python train.py --fd_config_idx 1
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from methods import (
    FeatureMap2D, FourierFeatureMap2D,
    Sin, VanillaPINN, MoEModel,
    train as run_training,
    fd_solve,
)
from configs import CONFIGS, FD_CONFIGS, SEEDS, DOMAIN

# ---------------------------------------------------------------------------
# Network architecture — matches the working notebook
# N_FREQ=5 → IN_DIM=22 for both feature maps (det: 2+4*5, rff: 2*11)
# ---------------------------------------------------------------------------

N_FREQ     = 5    # for FeatureMap2D (deterministic)
N_FREQ_RFF = 11   # for FourierFeatureMap2D → out_dim = 2*11 = 22
IN_DIM     = 22   # shared network input size


def _expert(use_sin=False):
    Act = Sin if use_sin else nn.Tanh
    return nn.Sequential(
        nn.Linear(IN_DIM, 10), Act(),
        nn.Linear(10, 10),     Act(),
        nn.Linear(10, 12),     Act(),
        nn.Linear(12, 10),     Act(),
        nn.Linear(10, 1),
    )


def build_model(cfg, device):
    Y, T    = DOMAIN["Y"], DOMAIN["T"]
    fm_type = cfg.get("feature_map", "deterministic")

    if fm_type == "fourier_rff":
        fm = FourierFeatureMap2D(Y, T, n_features=N_FREQ_RFF, sigma=1.0).to(device)
    else:
        fm = FeatureMap2D(Y, T, n_freq_y=N_FREQ, n_freq_t=N_FREQ).to(device)

    if cfg["use_moe"]:
        # Expert 0: Tanh, Expert 1: Sin, Expert 2: Tanh  (mirrors the notebook)
        experts = [_expert(False), _expert(True), _expert(False)]
        gating_net = nn.Sequential(
            nn.Linear(IN_DIM, 10), nn.Tanh(),
            nn.Linear(10, 10),     nn.Tanh(),
            nn.Linear(10, 12),     nn.Tanh(),
            nn.Linear(12, 10),     nn.Tanh(),
            nn.Linear(10, len(experts)),
        )
        model = MoEModel(fm, gating_net, experts,
                         gating=cfg["moe_gating"],
                         k=cfg.get("moe_k", 2))
    else:
        model = VanillaPINN(fm, _expert(False))

    return model.to(device)


# ---------------------------------------------------------------------------
# Evaluation against exact solution
# ---------------------------------------------------------------------------

def evaluate(model, device, n_grid=100):
    """Returns {"l2_rel": float, "max_err": float} on a uniform grid."""
    Y, T = DOMAIN["Y"], DOMAIN["T"]
    c, v = DOMAIN["c"], DOMAIN["v"]

    y_np = np.linspace(0.0, Y, n_grid)
    t_np = np.linspace(0.0, T, n_grid)
    yg, tg = np.meshgrid(y_np, t_np)

    y_t = torch.tensor(yg.reshape(-1, 1), dtype=torch.float32, device=device)
    t_t = torch.tensor(tg.reshape(-1, 1), dtype=torch.float32, device=device)

    with torch.no_grad():
        _, _, u_pred = model(y_t, t_t)

    u_pred  = u_pred.cpu().numpy().reshape(n_grid, n_grid)
    u_exact = np.exp(-v * tg) * np.sin(yg - c * tg)

    err = u_pred - u_exact
    return {
        "l2_rel":  float(np.linalg.norm(err) / np.linalg.norm(u_exact)),
        "max_err": float(np.max(np.abs(err))),
    }


# ---------------------------------------------------------------------------
# PINN runner
# ---------------------------------------------------------------------------

def run_pinn(config_idx, seed, out_dir):
    cfg = CONFIGS[config_idx]
    print(f"[PINN] Config [{config_idx:02d}]: {cfg['name']}  |  Seed: {seed}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model      = build_model(cfg, device)
    all_params = list(model.parameters())
    n_params   = sum(p.numel() for p in all_params)
    print(f"Parameters: {n_params:,}")

    # Mid-training evaluation callback (uses a coarse grid to keep overhead low)
    eval_fn = lambda m, e: evaluate(m, device, n_grid=100)

    t0   = time.time()
    hist = run_training(model, all_params, DOMAIN, cfg, eval_fn=eval_fn)
    total_time = time.time() - t0

    # Final high-resolution evaluation
    final = evaluate(model, device, n_grid=200)
    print(f"L2 rel: {final['l2_rel']:.4e}  |  Max err: {final['max_err']:.4e}")
    print(f"Total wall time: {total_time:.1f}s")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"cfg{config_idx:02d}_seed{seed}.npz")

    np.savez(
        out_path,
        # --- Identity ---
        solver          = "pinn",
        config_name     = cfg["name"],
        config_idx      = config_idx,
        seed            = seed,

        # --- Method flags ---
        feature_map          = cfg.get("feature_map", "deterministic"),
        use_moe              = cfg["use_moe"],
        moe_gating           = cfg.get("moe_gating") or "",
        use_softadapt        = cfg["use_softadapt"],
        use_adaptive_refine  = cfg["use_adaptive_refine"],
        use_lbfgs            = cfg["use_lbfgs"],

        # --- Domain ---
        Y   = DOMAIN["Y"],
        T   = DOMAIN["T"],
        c   = DOMAIN["c"],
        v   = DOMAIN["v"],
        N_f = DOMAIN["N_f"],

        # --- Model size ---
        n_params = n_params,

        # --- Per-epoch loss history ---
        hist_total = hist["total"],
        hist_pde   = hist["pde"],
        hist_ini   = hist["ini"],
        hist_load  = hist["load"],

        # --- Per-epoch wall time (seconds from start of training) ---
        hist_wall_time = hist["wall_time"],

        # --- Mid-training error snapshots (every eval_every epochs) ---
        eval_epochs   = hist["eval_epochs"],
        eval_l2_rel   = hist["eval_l2_rel"],
        eval_max_err  = hist["eval_max_err"],

        # --- Final errors (high-res 200×200 grid) ---
        l2_rel_final   = final["l2_rel"],
        max_err_final  = final["max_err"],

        # --- Timing ---
        total_time_sec = total_time,
    )
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Finite-difference runner
# ---------------------------------------------------------------------------

def run_fd(fd_config_idx, out_dir):
    cfg = FD_CONFIGS[fd_config_idx]
    print(f"[FD] Config: {cfg['name']}  N_y={cfg['N_y']}  N_t={cfg['N_t']}")

    result = fd_solve(DOMAIN, N_y=cfg["N_y"], N_t=cfg["N_t"])

    print(f"L2 rel: {result['l2_rel_final']:.4e}  |  Max err: {result['max_err_final']:.4e}")
    print(f"Solve time: {result['solve_time_sec']:.3f}s")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{cfg['name']}.npz")

    np.savez(
        out_path,
        # --- Identity ---
        solver      = "fd",
        config_name = cfg["name"],

        # --- Domain ---
        Y = DOMAIN["Y"],
        T = DOMAIN["T"],
        c = DOMAIN["c"],
        v = DOMAIN["v"],

        # --- Grid parameters ---
        N_y = result["N_y"],
        N_t = result["N_t"],
        dt  = result["dt"],

        # --- Solution at final time ---
        u_final       = result["u_final"],
        u_exact_final = result["u_exact_final"],
        y_grid        = result["y_grid"],

        # --- Error at every time step ---
        t_grid          = result["t_grid"],
        l2_rel_history  = result["l2_rel_history"],
        max_err_history = result["max_err_history"],

        # --- Final summary errors ---
        l2_rel_final  = result["l2_rel_final"],
        max_err_final = result["max_err_final"],

        # --- Timing ---
        solve_time_sec = result["solve_time_sec"],
    )
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    # PINN mode
    parser.add_argument("--job_id",  type=int, default=None,
                        help="SLURM array task ID → (config_idx, seed_idx)")
    parser.add_argument("--n_seeds", type=int, default=len(SEEDS))
    parser.add_argument("--out_dir", type=str, default="results")

    # FD mode
    parser.add_argument("--fd_config_idx", type=int, default=None,
                        help="Run a finite-difference config instead of PINN")

    args = parser.parse_args()

    if args.fd_config_idx is not None:
        if args.fd_config_idx >= len(FD_CONFIGS):
            sys.exit(f"fd_config_idx {args.fd_config_idx} out of range (max {len(FD_CONFIGS)-1})")
        run_fd(args.fd_config_idx, args.out_dir)

    elif args.job_id is not None:
        n_seeds    = args.n_seeds
        config_idx = args.job_id // n_seeds
        seed_idx   = args.job_id % n_seeds

        if config_idx >= len(CONFIGS):
            sys.exit(
                f"job_id {args.job_id} out of range — "
                f"max valid job_id is {len(CONFIGS) * n_seeds - 1}"
            )
        run_pinn(config_idx, SEEDS[seed_idx], args.out_dir)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
