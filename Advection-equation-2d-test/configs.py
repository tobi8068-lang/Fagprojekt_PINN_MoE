"""
Sweep configurations.

DOMAIN   — PDE parameters and equation callables (edit at top to change PDE).
SEEDS    — 5 seeds for the main sweep.
CONFIGS  — 34 PINN configs: 2 model types × (16 combos + 1 all-on).

Main sweep: 34 configs × 5 seeds = 170 jobs.
Job ID n  : config_idx = n // len(SEEDS),  seed = SEEDS[n % len(SEEDS)].
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
# Dirichlet BC: penalizes deviation from exact_fn on all 4 edges.
# Without this the model drifts near domain boundaries as t grows.

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
    "resample_every":    1,      # redraw all N_f collocation points every epoch (AR configs ignore this)
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


CONFIGS = _build_main()
SEEDS   = _ALL_SEEDS

# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"PINN configs  : {len(CONFIGS)}")
    print(f"Seeds         : {SEEDS}")
    print(f"Total jobs    : {len(CONFIGS) * len(SEEDS)}")
    print()
    for i, cfg in enumerate(CONFIGS):
        print(f"[{i:02d}] {cfg['name']}")
