"""
Lightweight local smoke-test.
Runs 4 quick PINN configs + 1 FD config to verify the full pipeline works.

Usage:
    python test_local.py

Expected runtime on CPU laptop: ~2-5 minutes total.
"""

import math
import time
import os

import numpy as np
import torch
import torch.nn as nn

from methods import (
    FeatureMap2D, FourierFeatureMap2D,
    Sin, VanillaPINN, MoEModel,
    train as run_training,
    fd_solve,
)
from configs import DOMAIN

# ---------------------------------------------------------------------------
# Small test configs (override the heavy production settings)
# ---------------------------------------------------------------------------

_TEST_BASE = {
    "solver":        "pinn",
    "adam_epochs":   200,
    "adam_lr":       1e-3,
    "adam_step_size": 100,
    "adam_gamma":    0.5,
    "base_weights":  {"pde": 1.0, "ini": 10.0, "load": 1e-2},
    "softadapt_beta": 5.0,
    "refine_every":  50,
    "n_candidates":  500,
    "replace_fraction": 0.2,
    "refine_gamma":  50.0,
    "lbfgs_max_iter": 50,
    "eval_every":    50,
    "log_every":     100,
}

# Small domain so collocation points cover it well with fewer samples
_TEST_DOMAIN = dict(DOMAIN)
_TEST_DOMAIN["N_f"] = 300

TEST_CONFIGS = [
    {**_TEST_BASE, "name": "vanilla_det",
     "feature_map": "deterministic", "use_moe": False, "moe_gating": None,
     "use_softadapt": False, "use_adaptive_refine": False, "use_lbfgs": False},

    {**_TEST_BASE, "name": "vanilla_rff",
     "feature_map": "fourier_rff", "use_moe": False, "moe_gating": None,
     "use_softadapt": True,  "use_adaptive_refine": False, "use_lbfgs": False},

    {**_TEST_BASE, "name": "moe_cont_det",
     "feature_map": "deterministic", "use_moe": True, "moe_gating": "continuous",
     "use_softadapt": False, "use_adaptive_refine": True,  "use_lbfgs": False},

    {**_TEST_BASE, "name": "moe_bin_rff_lbfgs",
     "feature_map": "fourier_rff", "use_moe": True, "moe_gating": "binary",
     "use_softadapt": False, "use_adaptive_refine": False, "use_lbfgs": True},
]

# ---------------------------------------------------------------------------
# Tiny network (faster than production; same shape but narrower)
# ---------------------------------------------------------------------------

N_FREQ     = 5
N_FREQ_RFF = 11
IN_DIM     = 22


def _expert(use_sin=False):
    Act = Sin if use_sin else nn.Tanh
    return nn.Sequential(
        nn.Linear(IN_DIM, 8), Act(),
        nn.Linear(8, 8),      Act(),
        nn.Linear(8, 1),
    )


def build_model(cfg, device):
    Y, T    = _TEST_DOMAIN["Y"], _TEST_DOMAIN["T"]
    fm_type = cfg.get("feature_map", "deterministic")

    if fm_type == "fourier_rff":
        fm = FourierFeatureMap2D(Y, T, n_features=N_FREQ_RFF, sigma=1.0).to(device)
    else:
        fm = FeatureMap2D(Y, T, n_freq_y=N_FREQ, n_freq_t=N_FREQ).to(device)

    if cfg["use_moe"]:
        experts = [_expert(False), _expert(True), _expert(False)]
        gating  = nn.Sequential(
            nn.Linear(IN_DIM, 8), nn.Tanh(),
            nn.Linear(8, 8),      nn.Tanh(),
            nn.Linear(8, len(experts)),
        )
        model = MoEModel(fm, gating, experts, gating=cfg["moe_gating"])
    else:
        model = VanillaPINN(fm, _expert(False))

    return model.to(device)


def evaluate(model, device, n_grid=100):
    Y, T = _TEST_DOMAIN["Y"], _TEST_DOMAIN["T"]
    c, v = _TEST_DOMAIN["c"], _TEST_DOMAIN["v"]
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
# Run
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n{'='*60}")

    os.makedirs("results_test", exist_ok=True)
    results = []

    # ---- PINN configs -------------------------------------------------------
    seed = 1234
    torch.manual_seed(seed)
    np.random.seed(seed)

    for i, cfg in enumerate(TEST_CONFIGS):
        print(f"\n[{i+1}/{len(TEST_CONFIGS)}] {cfg['name']}")
        print("-" * 40)

        torch.manual_seed(seed)
        np.random.seed(seed)

        model      = build_model(cfg, device)
        all_params = list(model.parameters())
        n_params   = sum(p.numel() for p in all_params)
        print(f"  Parameters : {n_params}")

        eval_fn = lambda m, e: evaluate(m, device, n_grid=50)

        t0   = time.time()
        hist = run_training(model, all_params, _TEST_DOMAIN, cfg, eval_fn=eval_fn)
        elapsed = time.time() - t0

        final = evaluate(model, device, n_grid=100)
        print(f"  L2 rel     : {final['l2_rel']:.4e}")
        print(f"  Max err    : {final['max_err']:.4e}")
        print(f"  Wall time  : {elapsed:.2f}s")
        print(f"  Eval snapshots recorded: {len(hist['eval_epochs'])}")

        # Quick sanity checks
        assert len(hist["total"])     == cfg["adam_epochs"], "hist_total length mismatch"
        assert len(hist["wall_time"]) == cfg["adam_epochs"], "wall_time length mismatch"
        assert len(hist["eval_epochs"]) > 0,                 "no eval snapshots recorded"

        out_path = f"results_test/{cfg['name']}_seed{seed}.npz"
        np.savez(out_path,
            config_name    = cfg["name"],
            seed           = seed,
            hist_total     = hist["total"],
            hist_pde       = hist["pde"],
            hist_ini       = hist["ini"],
            hist_load      = hist["load"],
            hist_wall_time = hist["wall_time"],
            eval_epochs    = hist["eval_epochs"],
            eval_l2_rel    = hist["eval_l2_rel"],
            eval_max_err   = hist["eval_max_err"],
            l2_rel_final   = final["l2_rel"],
            max_err_final  = final["max_err"],
            total_time_sec = elapsed,
        )
        print(f"  Saved: {out_path}")
        results.append((cfg["name"], final["l2_rel"], final["max_err"], elapsed))

    # ---- FD reference -------------------------------------------------------
    print(f"\n[FD] fd_N128")
    print("-" * 40)
    t0     = time.time()
    fd_res = fd_solve(_TEST_DOMAIN, N_y=128, N_t=500)
    elapsed = time.time() - t0
    print(f"  L2 rel    : {fd_res['l2_rel_final']:.4e}")
    print(f"  Max err   : {fd_res['max_err_final']:.4e}")
    print(f"  Wall time : {elapsed:.3f}s")

    out_path = "results_test/fd_N128.npz"
    np.savez(out_path,
        solver          = "fd",
        config_name     = "fd_N128",
        Y = _TEST_DOMAIN["Y"], T = _TEST_DOMAIN["T"],
        c = _TEST_DOMAIN["c"], v = _TEST_DOMAIN["v"],
        N_y             = fd_res["N_y"],
        N_t             = fd_res["N_t"],
        dt              = fd_res["dt"],
        u_final         = fd_res["u_final"],
        u_exact_final   = fd_res["u_exact_final"],
        y_grid          = fd_res["y_grid"],
        t_grid          = fd_res["t_grid"],
        l2_rel_history  = fd_res["l2_rel_history"],
        max_err_history = fd_res["max_err_history"],
        l2_rel_final    = fd_res["l2_rel_final"],
        max_err_final   = fd_res["max_err_final"],
        solve_time_sec  = fd_res["solve_time_sec"],
    )
    print(f"  Saved: {out_path}")
    results.append(("fd_N128", fd_res["l2_rel_final"], fd_res["max_err_final"], elapsed))

    # ---- Summary ------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"{'Config':<25}  {'L2 rel':>10}  {'Max err':>10}  {'Time (s)':>10}")
    print(f"{'-'*25}  {'-'*10}  {'-'*10}  {'-'*10}")
    for name, l2, mx, t in results:
        print(f"{name:<25}  {l2:>10.4e}  {mx:>10.4e}  {t:>10.2f}")
    print(f"\nAll .npz files written to results_test/")
    print("All assertions passed — pipeline looks good.")


if __name__ == "__main__":
    main()
