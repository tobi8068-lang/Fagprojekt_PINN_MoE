"""
Sweep configurations.

DOMAIN      — PDE / domain parameters + all equation definitions (edit at top).
SEEDS       — 5 seeds used for the main sweep.
CONFIGS     — 22 PINN configs: 2 model types × 11 method combos.
FD_CONFIGS  — 2 upwind finite-difference reference runs (no seed needed).

Feature map : Random Fourier Features (Tancik et al. 2020) — fixed for all configs.
Model types : vanilla | moe_cont  (continuous softmax gating)
Method combos: all (FD, SA, AR, LB) tuples with at most 2 active  → 11 combos
  FD = use finite-difference derivatives instead of autograd

Main sweep  : 22 configs × 5 seeds = 110 jobs
Follow-up   : fill in TOP_N_NAMES after ranking, then set SWEEP = "followup"
              → adds all-4-enhancement variants for those configs

Job ID n maps to:  config_idx = n // len(SEEDS),  seed = SEEDS[n % len(SEEDS)]
Reference jobs:    python train.py --fd_config_idx 0
"""

import math
import itertools
import numpy as np
import torch

# ===========================================================================
# *** EQUATION SETUP — edit here to change PDE / ICs / BCs ***
# ===========================================================================

# --- Domain extents & PDE coefficients ------------------------------------
_X_LO, _X_HI = -1.0,  8.0  # x training region
_Y_LO, _Y_HI = -7.0,  7.0  # y training region — Cauchy problem, symmetric about y=0
_T  = 1.0    # time horizon [0, T]
_Dt = 1.0    # diffusion coefficient in y
_Dl = 1.0    # diffusion coefficient in x
_v  = 3.0    # advection speed in x

# --- PDE residual ----------------------------------------------------------
# Return the residual r such that r = 0 is the strong form of your PDE.
# Arguments: u_t, u_y, u_yy, u_x, u_xx — PyTorch tensors of shape (N, 1)
#            domain                      — the full DOMAIN dict
def pde_fn(u_t, u_y, u_yy, u_x, u_xx, domain):
    """u_t + v·u_x - Dl·u_xx - Dt·u_yy = 0   (2-D advection-diffusion)"""
    return u_t + domain["v"] * u_x - domain["Dl"] * u_xx - domain["Dt"] * u_yy

# --- Initial condition  u(x, y, 0) = ic_fn(x, y) -------------------------
def ic_fn(x, y):
    """PyTorch tensor inputs — used in the PINN loss."""
    p = 1.0
    return torch.exp(-(((x - p) ** 2) / 4) - (y ** 2 / 4))

def ic_fn_np(x, y):
    """NumPy array inputs — used in the finite-difference reference solver."""
    p = 1.0
    return np.exp(-(((x - p) ** 2) / 4) - (y ** 2 / 4))

# --- Exact / reference solution  u(x, y, t) -------------------------------
# Valid on all of R² (unbounded domain). Gaussian IC advected by v in x and
# diffused independently in each direction:
#   u = 1/sqrt((1+Dl*t)(1+Dt*t)) * exp(-(x-p-v*t)^2/(4(1+Dl*t)) - y^2/(4(1+Dt*t)))
def exact_fn(x, y, t, domain):
    """NumPy array inputs."""
    Dl, Dt, v = domain["Dl"], domain["Dt"], domain["v"]
    p = 1.0
    dx = 1.0 + Dl * t
    dy = 1.0 + Dt * t
    return (1.0 / np.sqrt(dx * dy)) * np.exp(
        -((x - p - v * t) ** 2) / (4.0 * dx) - (y ** 2) / (4.0 * dy)
    )

# --- Boundary conditions --------------------------------------------------
# bc_fn(model, device, domain) -> scalar loss tensor, or None for no BC loss.
#
# Dirichlet data from the known closed-form exact_fn at the four edges of the
# truncated box. Valid because exact_fn is defined on all of R^2, so it gives
# the true boundary value at any (x_boundary, y, t) or (x, y_boundary, t) —
# this problem was previously treated as a pure Cauchy/IC-only problem with
# no boundary constraint at all (bc_fn = None below), which left the model
# free to drift near the domain edges as t grows since nothing penalized it
# for doing so (verified: error concentrated at the edges and grew over time,
# while the PDE residual + IC loss were already converged to ~1e-6).
# A previous version of this function implemented a Robin(Danckwerts)+Neumann
# condition instead, which doesn't actually hold for this Gaussian Cauchy
# solution (e.g. (u - Dl*u_x) at x=X_lo evaluates to u*(-t/(2(1+t))), not 0)
# and was never enabled anyway.

def bc_fn(model, device, domain):
    X_lo, X_hi = domain["X_lo"], domain["X_hi"]
    Y_lo, Y_hi = domain["Y_lo"], domain["Y_hi"]
    T        = domain["T"]
    exact_fn = domain["exact_fn"]
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


# --- BC loss weight (only used when bc_fn is not None) --------------------
_bc_weight = 10.0

# ===========================================================================

# ---------------------------------------------------------------------------
# Domain dict — passed to all training and solver functions
# ---------------------------------------------------------------------------

DOMAIN = {
    "X_lo":     _X_LO,
    "X_hi":     _X_HI,
    "Y_lo":     _Y_LO,
    "Y_hi":     _Y_HI,
    "T":        _T,
    "v":        _v,    # advection speed in x
    "Dl":       _Dl,   # diffusion coefficient in x
    "Dt":       _Dt,   # diffusion coefficient in y
    "K":        3.0,   # load-balancing scale for MoE gate entropy loss
    "N_f":      15000, # number of collocation points
    # equation callables
    "pde_fn":   pde_fn,
    "ic_fn":    ic_fn,
    "ic_fn_np": ic_fn_np,
    "exact_fn": exact_fn,
    "bc_fn":    bc_fn,  # Dirichlet via exact_fn at the 4 edges — see bc_fn above
    "bc_weight": _bc_weight,
}

# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------

_ALL_SEEDS = [1234, 2345, 3456, 4567, 5678]

# ---------------------------------------------------------------------------
# Fixed training hyperparameters
# ---------------------------------------------------------------------------

_TRAIN_BASE = {
    "adam_epochs":       10000,
    "adam_lr":           1e-3,
    "adam_step_size":    2500,
    "adam_gamma":        1.0,    # constant LR (was 0.5 step decay) — validated to help
    "base_weights":      {"pde": 1.0, "ini": 10.0, "load": 1e-2},  # pde was 10.0
    "softadapt_beta":    5.0,
    "refine_every":      100,
    "n_candidates":      8000,
    "replace_fraction":  0.2,
    "refine_gamma":      50.0,
    "lbfgs_max_iter":    600,
    "eval_every":        200,
    "log_every":         200,
    "causal_eps":        1.0,    # causal/temporal weighting of the PDE loss
    "n_causal_bins":     10,
    "resample_every":    1,      # redraw collocation points every epoch;
                                  # ignored for use_adaptive_refine=True configs
                                  # (AR manages the collocation set instead — see methods.train)
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _method_combos(max_enhancements):
    """All (FD, SA, AR, LB, RFF) tuples with at most max_enhancements True values."""
    return [
        (fd, sa, ar, lb, rff)
        for fd, sa, ar, lb, rff in itertools.product([False, True], repeat=5)
        if (fd + sa + ar + lb + rff) <= max_enhancements
    ]


def _make_cfg(use_moe, m_tag, use_fd, use_sa, use_ar, use_lb, use_rff):
    model_tag = f"moe_{m_tag}" if use_moe else "vanilla"
    cfg = dict(_TRAIN_BASE)
    cfg["base_weights"] = dict(_TRAIN_BASE["base_weights"])
    cfg.update({
        "solver":              "pinn",
        "use_moe":             use_moe,
        "use_rff":             use_rff,
        "use_fd_deriv":        use_fd,
        "use_softadapt":       use_sa,
        "use_adaptive_refine": use_ar,
        "use_lbfgs":           use_lb,
        "name": (f"{model_tag}"
                 f"_rff{int(use_rff)}"
                 f"_fd{int(use_fd)}"
                 f"_sa{int(use_sa)}"
                 f"_ar{int(use_ar)}"
                 f"_lb{int(use_lb)}"),
    })
    return cfg


# Model variants: (use_moe, short_tag)
_MODELS = [
    (False, ""),
    (True,  "cont"),
]

# ---------------------------------------------------------------------------
# Main sweep — 0, 1, or 2 enhancements active
# ---------------------------------------------------------------------------

def _build_main():
    configs = []
    for use_moe, m_tag in _MODELS:
        for fd, sa, ar, lb, rff in _method_combos(max_enhancements=2):
            configs.append(_make_cfg(use_moe, m_tag, fd, sa, ar, lb, rff))
        # All-flags-on ceiling config (not reachable from max_enhancements=2)
        configs.append(_make_cfg(use_moe, m_tag, True, True, True, True, True))
    return configs   # 2 models × (16 combos + 1 all-on) = 34


# ---------------------------------------------------------------------------
# Follow-up sweep — re-run top-N configs with all 4 enhancements enabled
#
# After running the main sweep:
#   1. Run: python plot.py --results results
#   2. Copy the top config names from the printed table into TOP_N_NAMES
#   3. Set SWEEP = "followup" and re-submit
# ---------------------------------------------------------------------------

TOP_N_NAMES = [
    # e.g. "moe_cont_fd1_sa1_ar0_lb0",
]


def _build_followup():
    """For each name in TOP_N_NAMES, generate the all-4-enhancements variant."""
    main_lookup = {cfg["name"]: cfg for cfg in _build_main()}
    configs = []
    for name in TOP_N_NAMES:
        if name not in main_lookup:
            raise ValueError(f"'{name}' not found in main configs — check spelling")
        base = main_lookup[name]
        cfg = dict(base)
        cfg["base_weights"] = dict(base["base_weights"])
        cfg.update({
            "use_rff":             True,
            "use_fd_deriv":        True,
            "use_softadapt":       True,
            "use_adaptive_refine": True,
            "use_lbfgs":           True,
            "name":                name.split("_rff")[0] + "_rff1_fd1_sa1_ar1_lb1",
        })
        if not (base["use_rff"] and base["use_fd_deriv"] and base["use_softadapt"]
                and base["use_adaptive_refine"] and base["use_lbfgs"]):
            configs.append(cfg)
    return configs


# ---------------------------------------------------------------------------
# *** SET YOUR SWEEP HERE ***
#   "main"     → 34 configs, 5 seeds  (170 jobs)
#   "followup" → len(TOP_N_NAMES) configs, 5 seeds  (fill in TOP_N_NAMES first)
# ---------------------------------------------------------------------------

SWEEP = "main"

# ---------------------------------------------------------------------------

if SWEEP == "main":
    CONFIGS = _build_main()
    SEEDS   = _ALL_SEEDS
elif SWEEP == "followup":
    CONFIGS = _build_followup()
    SEEDS   = _ALL_SEEDS
else:
    raise ValueError(f"Unknown SWEEP value: '{SWEEP}'")

# ---------------------------------------------------------------------------
# Reference solver configs
# ---------------------------------------------------------------------------

FD_CONFIGS = [
    {"solver": "fd", "name": "ref_N512",  "N_y": 512,  "N_t_plot": 1000},
    {"solver": "fd", "name": "ref_N1024", "N_y": 1024, "N_t_plot": 1000},
]

# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Sweep         : {SWEEP}")
    print(f"PINN configs  : {len(CONFIGS)}")
    print(f"Seeds         : {SEEDS}")
    print(f"Total jobs    : {len(CONFIGS) * len(SEEDS)}")
    print(f"FD configs    : {len(FD_CONFIGS)}")
    print()
    for i, cfg in enumerate(CONFIGS):
        print(f"[{i:02d}] {cfg['name']}")
