"""
Train or solve one (config, seed) pair and save results to results/.

PINN usage (via LSF loop in submit_all.sh):
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
    FeatureMap, ScaledInputs,
    Sin, VanillaPINN, MoEModel,
    train as run_training,
    numerical_solve,
)
from configs import CONFIGS, FD_CONFIGS, SEEDS, DOMAIN

# ---------------------------------------------------------------------------
# Network architecture
# ---------------------------------------------------------------------------

N_FEATURES = 11   # RFF frequencies → out_dim = 2 * N_FEATURES = 22

# Widths chosen so vanilla and MoE have the same total parameter count.
# Vanilla always uses w=64.  MoE expert width depends on use_rff because
# removing RFF shrinks each of the 3 expert first layers (22→w vs 3→w),
# creating a larger gap than vanilla's single first layer.
#   RFF  (in_dim=22): vanilla=14 017,  MoE 3×34+gate=13 580  (−3%)
#   noRFF(in_dim= 3): vanilla=12 801,  MoE 3×36+gate=12 646  (−1%)
_VANILLA_WIDTH          = 64
_MOE_EXPERT_WIDTH_RFF   = 34
_MOE_EXPERT_WIDTH_NORFF = 36


def _expert(in_dim, width, use_sin=False):
    Act = Sin if use_sin else nn.Tanh
    return nn.Sequential(
        nn.Linear(in_dim, width), Act(),
        nn.Linear(width,  width), Act(),
        nn.Linear(width,  width), Act(),
        nn.Linear(width,  width), Act(),
        nn.Linear(width,  1),
    )


def build_model(cfg, device):
    T    = DOMAIN["T"]
    X_lo = DOMAIN.get("X_lo", 0.0);  X_hi = DOMAIN["X_hi"]
    Y_lo = DOMAIN.get("Y_lo", 0.0);  Y_hi = DOMAIN["Y_hi"]
    bounds = [(X_lo, X_hi), (Y_lo, Y_hi), (0.0, T)]

    if cfg.get("use_rff", True):
        fm = FeatureMap(bounds, n_features=N_FEATURES, sigma=1.0).to(device)
    else:
        fm = ScaledInputs(bounds).to(device)
    in_dim = fm.out_dim

    if cfg["use_moe"]:
        moe_w = _MOE_EXPERT_WIDTH_RFF if cfg.get("use_rff", True) else _MOE_EXPERT_WIDTH_NORFF
        experts = [_expert(in_dim, moe_w, False),
                   _expert(in_dim, moe_w, True),
                   _expert(in_dim, moe_w, False)]
        gating_net = nn.Sequential(
            nn.Linear(in_dim, 16), nn.Tanh(),
            nn.Linear(16, len(experts)),
        )
        model = MoEModel(fm, gating_net, experts)
    else:
        model = VanillaPINN(fm, _expert(in_dim, _VANILLA_WIDTH))

    return model.to(device)


# ---------------------------------------------------------------------------
# Evaluation against exact solution
# ---------------------------------------------------------------------------

def evaluate(model, device, n_grid=100):
    """
    Evaluate L2 relative error and total (absolute) L2 norm over the full
    (x, y) spatial domain at t = 0, T/2, T.  Returns NaN when no exact
    solution is available.
    """
    exact_fn = DOMAIN.get("exact_fn")
    if exact_fn is None:
        return {"l2_rel": float("nan"), "max_err": float("nan")}

    X_lo = DOMAIN.get("X_lo", 0.0);  X_hi = DOMAIN["X_hi"]
    Y_lo = DOMAIN.get("Y_lo", 0.0);  Y_hi = DOMAIN["Y_hi"]
    T    = DOMAIN["T"]

    x_np = np.linspace(X_lo, X_hi, n_grid)
    y_np = np.linspace(Y_lo, Y_hi, n_grid)
    xg, yg = np.meshgrid(x_np, y_np)          # (n_grid, n_grid) spatial grid

    err_all    = []
    u_exact_all = []

    for t_val in [0.0, T / 2.0, T]:
        tg = np.full_like(xg, t_val)

        x_t = torch.tensor(xg.reshape(-1, 1), dtype=torch.float32, device=device)
        y_t = torch.tensor(yg.reshape(-1, 1), dtype=torch.float32, device=device)
        t_t = torch.tensor(tg.reshape(-1, 1), dtype=torch.float32, device=device)

        with torch.no_grad():
            _, _, u_pred = model(x_t, y_t, t_t)

        u_pred_np  = u_pred.cpu().numpy().ravel()
        u_exact_np = exact_fn(xg, yg, tg, DOMAIN).ravel()

        err_all.append(u_pred_np - u_exact_np)
        u_exact_all.append(u_exact_np)

    err_all     = np.concatenate(err_all)
    u_exact_all = np.concatenate(u_exact_all)

    l2_rel  = float(np.linalg.norm(err_all) / np.linalg.norm(u_exact_all))
    max_err = float(np.linalg.norm(err_all))
    return {"l2_rel": l2_rel, "max_err": max_err}


def solution_grid(model, device, n_grid=200):
    """
    Evaluate model and exact solution on an (x, y) grid at t = 0, T/2, T.

    Returns dict with:
        grid_x        : (n_grid,)       x coordinates
        grid_y        : (n_grid,)       y coordinates
        grid_t_vals   : (3,)            [0, T/2, T]
        grid_u_pred   : (3, n_grid, n_grid)
        grid_u_exact  : (3, n_grid, n_grid)  — NaN where exact_fn returns None
    """
    exact_fn = DOMAIN.get("exact_fn")
    X_lo = DOMAIN.get("X_lo", 0.0);  X_hi = DOMAIN["X_hi"]
    Y_lo = DOMAIN.get("Y_lo", 0.0);  Y_hi = DOMAIN["Y_hi"]
    T    = DOMAIN["T"]

    x_np = np.linspace(X_lo, X_hi, n_grid)
    y_np = np.linspace(Y_lo, Y_hi, n_grid)
    xg, yg = np.meshgrid(x_np, y_np)   # (n_grid, n_grid), axes: [y-idx, x-idx]

    t_vals = [0.0, T / 2.0, T]
    u_pred_snaps  = []
    u_exact_snaps = []

    for t_val in t_vals:
        tg = np.full_like(xg, t_val)

        x_t = torch.tensor(xg.reshape(-1, 1), dtype=torch.float32, device=device)
        y_t = torch.tensor(yg.reshape(-1, 1), dtype=torch.float32, device=device)
        t_t = torch.tensor(tg.reshape(-1, 1), dtype=torch.float32, device=device)

        with torch.no_grad():
            _, _, u = model(x_t, y_t, t_t)
        u_pred_snaps.append(u.cpu().numpy().reshape(n_grid, n_grid))

        if exact_fn is not None:
            ue = exact_fn(xg, yg, tg, DOMAIN)
            u_exact_snaps.append(ue if ue is not None else np.full_like(xg, np.nan))
        else:
            u_exact_snaps.append(np.full_like(xg, np.nan))

    return {
        "grid_x":       x_np,
        "grid_y":       y_np,
        "grid_t_vals":  np.array(t_vals),
        "grid_u_pred":  np.array(u_pred_snaps),   # (3, n_grid, n_grid)
        "grid_u_exact": np.array(u_exact_snaps),  # (3, n_grid, n_grid)
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
    print(f"L2 rel: {final['l2_rel']:.4e}  |  L2 abs: {final['max_err']:.4e}")
    print(f"Total wall time: {total_time:.1f}s")

    grid = solution_grid(model, device, n_grid=200)

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
        use_moe              = cfg["use_moe"],
        use_rff              = cfg.get("use_rff", False),
        use_fd_deriv         = cfg.get("use_fd_deriv", True),
        use_softadapt        = cfg["use_softadapt"],
        use_adaptive_refine  = cfg["use_adaptive_refine"],
        use_lbfgs            = cfg["use_lbfgs"],

        # --- Domain ---
        X_lo = DOMAIN.get("X_lo", 0.0),
        X_hi = DOMAIN["X_hi"],
        Y_lo = DOMAIN.get("Y_lo", 0.0),
        Y_hi = DOMAIN["Y_hi"],
        T    = DOMAIN["T"],
        N_f  = DOMAIN["N_f"],

        # --- Model size ---
        n_params = n_params,

        # --- Per-epoch loss history ---
        hist_total = hist["total"],
        hist_pde   = hist["pde"],
        hist_ini   = hist["ini"],
        hist_load  = hist["load"],
        hist_bc    = hist.get("bc", []),

        # --- Per-epoch wall time (seconds from start of training) ---
        hist_wall_time = hist["wall_time"],

        # --- Mid-training error snapshots (every eval_every epochs) ---
        eval_epochs   = hist["eval_epochs"],
        eval_l2_rel   = hist["eval_l2_rel"],
        eval_max_err  = hist["eval_max_err"],

        # --- Final errors (high-res 200×200 grid) ---
        l2_rel_final   = final["l2_rel"],
        max_err_final  = final["max_err"],

        # --- Solution grids at t = 0, T/2, T ---
        grid_x       = grid["grid_x"],
        grid_y       = grid["grid_y"],
        grid_t_vals  = grid["grid_t_vals"],
        grid_u_pred  = grid["grid_u_pred"],
        grid_u_exact = grid["grid_u_exact"],

        # --- Timing ---
        total_time_sec = total_time,
    )
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Finite-difference runner
# ---------------------------------------------------------------------------

def run_fd(fd_config_idx, out_dir):
    cfg = FD_CONFIGS[fd_config_idx]
    print(f"[FD] Config: {cfg['name']}  N_y={cfg['N_y']}")

    result = numerical_solve(DOMAIN, N_y=cfg["N_y"], N_t_plot=cfg.get("N_t_plot", 1000))

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
        N_y     = result["N_y"],
        N_t_plot= result["N_t_plot"],

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
    parser.add_argument("--job_id",     type=int, default=None,
                        help="LSF job index → (config_idx, seed_idx)")
    parser.add_argument("--n_seeds",    type=int, default=len(SEEDS))
    parser.add_argument("--config_idx", type=int, default=None,
                        help="Config index (use with --seed for direct invocation)")
    parser.add_argument("--seed",       type=int, default=None,
                        help="Explicit seed value (use with --config_idx)")
    parser.add_argument("--out_dir",    type=str, default="results")

    # FD mode
    parser.add_argument("--fd_config_idx", type=int, default=None,
                        help="Run a finite-difference config instead of PINN")

    args = parser.parse_args()

    if args.fd_config_idx is not None:
        if args.fd_config_idx >= len(FD_CONFIGS):
            sys.exit(f"fd_config_idx {args.fd_config_idx} out of range (max {len(FD_CONFIGS)-1})")
        run_fd(args.fd_config_idx, args.out_dir)

    elif args.config_idx is not None and args.seed is not None:
        if args.config_idx >= len(CONFIGS):
            sys.exit(f"config_idx {args.config_idx} out of range (max {len(CONFIGS)-1})")
        run_pinn(args.config_idx, args.seed, args.out_dir)

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
