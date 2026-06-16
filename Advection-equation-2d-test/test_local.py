"""
Local test of the accuracy fixes from the boundary-condition investigation,
applied on top of the best / median / worst configs from the main sweep
ranking (results/, 34 configs x 5 seeds, ranked by rank_configs() in
plot.py).

These changes are scoped to this script only — configs.py / methods.py
default behaviour for the production sweep (train.py) is unchanged:

  - Dirichlet boundary loss using the closed-form exact_fn at the four
    edges of the truncated box (DOMAIN["bc_fn"] is None for every CONFIGS
    entry — this problem was treated as a pure Cauchy/IC-only problem with
    no boundary constraint at all).
  - Hard-constrained IC: u = ic_fn(x,y) + t * network(x,y,t), so the
    initial condition is satisfied exactly instead of via a soft loss term
    (zero added parameters — same network, reparametrized output).
  - Causal/temporal weighting of the PDE residual (methods.py's
    causal_eps/n_causal_bins, default 0.0/off everywhere else).
  - Constant learning rate (adam_gamma=1.0) instead of the StepLR decay.
  - Fresh collocation points every epoch (methods.py's resample_every,
    default 0/off everywhere else) instead of reusing one fixed batch of
    N_f points for the whole run, to avoid overfitting to that fixed set.

Usage:
    python test_local.py

Expected runtime on CPU laptop: ~3-6 minutes total (3 configs).
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn

from configs import CONFIGS, DOMAIN
from train import build_model, evaluate, solution_grid

# ---------------------------------------------------------------------------
# Configs selected from the main sweep ranking — see investigation notes.
# ---------------------------------------------------------------------------

BEST_NAME   = "vanilla_rff1_fd0_sa0_ar0_lb1"   # rank  1/34, median L2 = 4.67e-02
MEDIAN_NAME = "vanilla_rff0_fd0_sa1_ar1_lb0"   # rank 18/34, median L2 = 1.64e-01
WORST_NAME  = "vanilla_rff0_fd0_sa0_ar0_lb0"   # rank 34/34, median L2 = 2.37e-01

SELECTED = [("best", BEST_NAME), ("median", MEDIAN_NAME), ("worst", WORST_NAME)]

SEED = 1234

# If True, hard-constrains u(x,y,0) = ic_fn(x,y) exactly (HardICWrapper) so the
# 'ini' loss is always ~0 and its weight has no effect. If False, the IC is
# enforced the normal way, as a weighted loss term balanced against pde/bc.
USE_HARD_IC = False

# ---------------------------------------------------------------------------
# Lightweight overrides + the fixes under test
# ---------------------------------------------------------------------------

_TEST_OVERRIDES = {
    "adam_epochs":    5000,
    "adam_step_size": 1250,
    "adam_gamma":     1.0,    # constant LR throughout (StepLR decay disabled)
    "resample_every": 1,      # redraw all collocation points fresh every epoch
    "base_weights":   {"pde": 1.0, "ini": 10.0, "load": 1e-2},
    "refine_every":   50,
    "n_candidates":   2000,
    "lbfgs_max_iter": 50,
    "eval_every":     50,
    "log_every":      100,
    "causal_eps":     1.0,    # enable causal/temporal weighting of the PDE loss
    "n_causal_bins":  10,
}


def _dirichlet_bc_fn(model, device, domain):
    """
    Dirichlet boundary loss using the closed-form exact_fn (valid on all of
    R^2 for this manufactured problem) at the four edges of the truncated
    box. The production bc_fn in configs.py implements a Robin+Neumann
    condition that doesn't match this IC/exact_fn and is disabled anyway
    (DOMAIN["bc_fn"] = None) — this replaces it for the test only.
    """
    X_lo, X_hi = domain["X_lo"], domain["X_hi"]
    Y_lo, Y_hi = domain["Y_lo"], domain["Y_hi"]
    T          = domain["T"]
    exact_fn   = domain["exact_fn"]
    N = 200

    def _edge_loss(x, y, t):
        _, _, u_pred = model(x, y, t)
        u_exact_np = exact_fn(x.detach().cpu().numpy(), y.detach().cpu().numpy(),
                               t.detach().cpu().numpy(), domain)
        u_exact = torch.as_tensor(u_exact_np, dtype=u_pred.dtype, device=u_pred.device)
        return torch.mean((u_pred - u_exact) ** 2)

    t = T * torch.rand(N, 1, device=device)
    y = Y_lo + (Y_hi - Y_lo) * torch.rand(N, 1, device=device)
    loss_left  = _edge_loss(torch.full((N, 1), X_lo, device=device), y, t)
    loss_right = _edge_loss(torch.full((N, 1), X_hi, device=device), y, t)

    t = T * torch.rand(N, 1, device=device)
    x = X_lo + (X_hi - X_lo) * torch.rand(N, 1, device=device)
    loss_bot = _edge_loss(x, torch.full((N, 1), Y_lo, device=device), t)
    loss_top = _edge_loss(x, torch.full((N, 1), Y_hi, device=device), t)

    return (loss_left + loss_right + loss_bot + loss_top) / 4.0


class HardICWrapper(nn.Module):
    """
    Reparametrizes the network output as u = ic_fn(x,y) + t * network(x,y,t)
    so the initial condition is satisfied exactly (zero added parameters)
    instead of via a soft loss term.
    """

    def __init__(self, base_model, ic_fn):
        super().__init__()
        self.base_model = base_model
        self.ic_fn = ic_fn

    def forward(self, x, y, t):
        gates, expert_vals, u_raw = self.base_model(x, y, t)
        u = self.ic_fn(x, y) + t * u_raw
        return gates, expert_vals, u


_TEST_DOMAIN = {
    **DOMAIN,
    "N_f":       15000,
    "bc_fn":     _dirichlet_bc_fn,
    "bc_weight": 10.0,
}


def make_test_cfg(cfg):
    return {**cfg, **_TEST_OVERRIDES}


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def main():
    name_to_cfg = {c["name"]: c for c in CONFIGS}
    missing = [name for _, name in SELECTED if name not in name_to_cfg]
    if missing:
        raise ValueError(f"Config name(s) not found in CONFIGS: {missing}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Running {len(SELECTED)} configs (best / median / worst from main sweep ranking)")
    print("=" * 60)

    os.makedirs("results_test", exist_ok=True)

    from methods import train as run_training

    results = []
    for i, (label, name) in enumerate(SELECTED):
        test_cfg = make_test_cfg(name_to_cfg[name])
        print(f"\n[{i+1}/{len(SELECTED)}] ({label}) {test_cfg['name']}")
        print("-" * 40)

        torch.manual_seed(SEED)
        np.random.seed(SEED)

        base_model = build_model(test_cfg, device)
        model      = (HardICWrapper(base_model, DOMAIN["ic_fn"]).to(device)
                      if USE_HARD_IC else base_model)
        all_params = list(model.parameters())
        n_params   = sum(p.numel() for p in all_params)
        print(f"  Parameters : {n_params:,}")

        eval_fn = lambda m, e: evaluate(m, device, n_grid=100)

        t0   = time.time()
        hist = run_training(model, all_params, _TEST_DOMAIN, test_cfg, eval_fn=eval_fn)
        elapsed = time.time() - t0

        final = evaluate(model, device, n_grid=100)
        grid  = solution_grid(model, device, n_grid=100)
        print(f"  L2 rel     : {final['l2_rel']:.4e}")
        print(f"  L2 abs     : {final['max_err']:.4e}")
        print(f"  Wall time  : {elapsed:.2f}s")

        out_path = f"results_test/{label}_{test_cfg['name']}_seed{SEED}.npz"
        np.savez(
            out_path,
            config_name          = test_cfg["name"],
            rank_label            = label,
            seed                 = SEED,
            use_moe              = test_cfg["use_moe"],
            use_rff              = test_cfg.get("use_rff", False),
            use_fd_deriv         = test_cfg.get("use_fd_deriv", True),
            use_softadapt        = test_cfg["use_softadapt"],
            use_adaptive_refine  = test_cfg["use_adaptive_refine"],
            use_lbfgs            = test_cfg["use_lbfgs"],
            n_params             = n_params,
            hist_total           = hist["total"],
            hist_pde             = hist["pde"],
            hist_ini             = hist["ini"],
            hist_load            = hist["load"],
            hist_wall_time       = hist["wall_time"],
            eval_epochs          = hist["eval_epochs"],
            eval_l2_rel          = hist["eval_l2_rel"],
            eval_max_err         = hist["eval_max_err"],
            l2_rel_final         = final["l2_rel"],
            max_err_final        = final["max_err"],
            grid_x               = grid["grid_x"],
            grid_y               = grid["grid_y"],
            grid_t_vals          = grid["grid_t_vals"],
            grid_u_pred          = grid["grid_u_pred"],
            grid_u_exact         = grid["grid_u_exact"],
            total_time_sec       = elapsed,
        )
        print(f"  Saved      : {out_path}")
        results.append((label, test_cfg["name"], final["l2_rel"], elapsed))

    # ---- Summary ------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"{'Rank':<8} {'Config':<32} {'L2 rel':>10}  {'Time (s)':>10}")
    print(f"{'-'*8} {'-'*32} {'-'*10}  {'-'*10}")
    for label, name, l2, t in results:
        print(f"{label:<8} {name:<32} {l2:>10.4e}  {t:>10.2f}")
    print(f"\nResults written to results_test/")


if __name__ == "__main__":
    main()
