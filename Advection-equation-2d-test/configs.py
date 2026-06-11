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
# Pattern:  for each boundary edge, sample N_bc random (position, time) pairs,
#           evaluate model there, compare to a target, return mean MSE.
#           Replace the u_exact block with torch.zeros_like(u_pred) for u=0 BCs.
#
# Here we use the known exact solution as Dirichlet data on all four edges.
# This is valid because exact_fn is defined on all of R² so it gives the true
# boundary value at any (x_boundary, y, t) or (x, y_boundary, t).

def bc_fn(model, device, domain):
    """
    BCs from eq. (2.27):
      Robin (Danckwerts) inlet at x = X_lo:  (C - b*C_x)(X_lo, y, t) = 0
      Neumann at y = Y_lo:                    C_y(x, Y_lo, t) = 0
      Neumann at y = Y_hi:                    C_y(x, Y_hi, t) = 0
    where b = Dl (= D_L/V in dimensionless form with V=1).
    Derivatives are computed via autograd regardless of the FD flag in training.
    """
    X_lo, X_hi = domain["X_lo"], domain["X_hi"]
    Y_lo, Y_hi = domain["Y_lo"], domain["Y_hi"]
    T = domain["T"]
    b = domain["Dl"]
    N = 200

    # --- Robin at x = X_lo: (C - b * dC/dx) = 0 ----------------------------
    t = T * torch.rand(N, 1, device=device)
    y = Y_lo + (Y_hi - Y_lo) * torch.rand(N, 1, device=device)
    x_left = torch.full((N, 1), X_lo, device=device, requires_grad=True)
    _, _, u = model(x_left, y, t)
    u_x = torch.autograd.grad(u.sum(), x_left, create_graph=True)[0]
    loss_robin = torch.mean((u - b * u_x) ** 2)

    # --- Neumann at y = Y_lo: dC/dy = 0 ------------------------------------
    t = T * torch.rand(N, 1, device=device)
    x = X_lo + (X_hi - X_lo) * torch.rand(N, 1, device=device)
    y_bot = torch.full((N, 1), Y_lo, device=device, requires_grad=True)
    _, _, u = model(x, y_bot, t)
    u_y = torch.autograd.grad(u.sum(), y_bot, create_graph=True)[0]
    loss_neumann_bot = torch.mean(u_y ** 2)

    # --- Neumann at y = Y_hi: dC/dy = 0 ------------------------------------
    t = T * torch.rand(N, 1, device=device)
    x = X_lo + (X_hi - X_lo) * torch.rand(N, 1, device=device)
    y_top = torch.full((N, 1), Y_hi, device=device, requires_grad=True)
    _, _, u = model(x, y_top, t)
    u_y = torch.autograd.grad(u.sum(), y_top, create_graph=True)[0]
    loss_neumann_top = torch.mean(u_y ** 2)

    return (loss_robin + loss_neumann_bot + loss_neumann_top) / 3.0


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
    "bc_fn":    None,   # Cauchy problem — IC only, no spatial BCs
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
    "adam_gamma":        0.5,
    "base_weights":      {"pde": 10.0, "ini": 10.0, "load": 1e-2},
    "softadapt_beta":    5.0,
    "refine_every":      100,
    "n_candidates":      8000,
    "replace_fraction":  0.2,
    "refine_gamma":      50.0,
    "lbfgs_max_iter":    600,
    "eval_every":        200,
    "log_every":         200,
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
