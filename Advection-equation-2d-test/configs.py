"""
Sweep configurations.

DOMAIN      — PDE / domain parameters, shared by every run.
SEEDS       — first 5 seeds used for the main sweep.
CONFIGS     — 42 PINN configs: 3 model types × 2 feature maps × 7 method combos.
FD_CONFIGS  — 2 finite-difference reference runs (deterministic, no seed needed).

Model types : vanilla | moe_continuous | moe_kgating  (top-2-of-3 sparse gating)
Feature maps: deterministic | fourier_rff
Method combos: all (SA, AR, LB) tuples with at most 2 enhancements active  → 7 combos

Main sweep  : 42 configs × 5 seeds = 210 jobs
Follow-up   : fill in TOP_N_NAMES after ranking, then set SWEEP = "followup"
              → adds all-3-enhancement variants for those configs (≤ N × 5 more jobs)

Job ID n maps to:  config_idx = n // len(SEEDS),  seed = SEEDS[n % len(SEEDS)]
FD jobs:           python train.py --fd_config_idx 0
"""

import math
import itertools

# ---------------------------------------------------------------------------
# Domain
# ---------------------------------------------------------------------------

DOMAIN = {
    "Y":   2.0 * math.pi,
    "T":   3.0,
    "c":   1.0,
    "v":   1.0,
    "K":   3.0,
    "N_f": 2000,
}

# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------

_ALL_SEEDS = [1234, 42, 100, 999, 2025, 7, 314, 777]

# ---------------------------------------------------------------------------
# Fixed training hyperparameters
# ---------------------------------------------------------------------------

_TRAIN_BASE = {
    "adam_epochs":       10000,
    "adam_lr":           1e-3,
    "adam_step_size":    2500,
    "adam_gamma":        0.5,
    "base_weights":      {"pde": 1.0, "ini": 10.0, "load": 1e-2},
    "softadapt_beta":    5.0,
    "refine_every":      400,
    "n_candidates":      8000,
    "replace_fraction":  0.2,
    "refine_gamma":      50.0,
    "lbfgs_max_iter":    600,
    "eval_every":        500,
    "log_every":         500,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _method_combos(max_enhancements):
    """All (SA, AR, LB) tuples with at most max_enhancements True values."""
    return [
        (sa, ar, lb)
        for sa, ar, lb in itertools.product([False, True], repeat=3)
        if (sa + ar + lb) <= max_enhancements
    ]


def _make_cfg(feature_map, fm_tag, use_moe, gating, g_tag, use_sa, use_ar, use_lb):
    model_tag = f"moe_{g_tag}" if use_moe else "vanilla"
    cfg = dict(_TRAIN_BASE)
    cfg["base_weights"] = dict(_TRAIN_BASE["base_weights"])
    cfg.update({
        "solver":              "pinn",
        "feature_map":         feature_map,
        "use_moe":             use_moe,
        "moe_gating":          gating,
        "moe_k":               2,          # k for top-k gating (ignored otherwise)
        "use_softadapt":       use_sa,
        "use_adaptive_refine": use_ar,
        "use_lbfgs":           use_lb,
        "name": f"{model_tag}_{fm_tag}_sa{int(use_sa)}_ar{int(use_ar)}_lb{int(use_lb)}",
    })
    return cfg


# Model variants: (use_moe, gating_str, short_tag)
_MODELS = [
    (False, None,         ""),
    (True,  "continuous", "cont"),
    (True,  "topk",       "kgat"),
]

_FEATURE_MAPS = [
    ("deterministic", "det"),
    ("fourier_rff",   "rff"),
]

# ---------------------------------------------------------------------------
# Main sweep — 0 or 1 or 2 enhancements active
# ---------------------------------------------------------------------------

def _build_main():
    configs = []
    for fm_type, fm_tag in _FEATURE_MAPS:
        for use_moe, gating, g_tag in _MODELS:
            for sa, ar, lb in _method_combos(max_enhancements=2):
                configs.append(_make_cfg(fm_type, fm_tag,
                                         use_moe, gating, g_tag, sa, ar, lb))
    return configs   # 2 FMs × 3 models × 7 combos = 42


# ---------------------------------------------------------------------------
# Follow-up sweep — re-run top-N configs with all 3 enhancements enabled
#
# After running the main sweep:
#   1. Run: python plot.py --results results
#   2. Copy the top ~10 config names from the printed table into TOP_N_NAMES
#   3. Set SWEEP = "followup" and re-submit
# ---------------------------------------------------------------------------

TOP_N_NAMES = [
    # Paste config names here after the main sweep, e.g.:
    # "moe_cont_det_sa1_ar1_lb0",
    # "vanilla_rff_sa0_ar1_lb0",
]


def _build_followup():
    """For each name in TOP_N_NAMES, generate the all-3-enhancements variant."""
    main_lookup = {cfg["name"]: cfg for cfg in _build_main()}
    configs = []
    for name in TOP_N_NAMES:
        if name not in main_lookup:
            raise ValueError(f"'{name}' not found in main configs — check spelling")
        base = main_lookup[name]
        # Clone and force all enhancements on
        cfg = dict(base)
        cfg["base_weights"] = dict(base["base_weights"])
        cfg.update({
            "use_softadapt":       True,
            "use_adaptive_refine": True,
            "use_lbfgs":           True,
            "name": name.rsplit("_sa", 1)[0] + "_sa1_ar1_lb1",
        })
        # Avoid duplicates (skip if main sweep already included all-3-on)
        if not (base["use_softadapt"] and base["use_adaptive_refine"] and base["use_lbfgs"]):
            configs.append(cfg)
    return configs


# ---------------------------------------------------------------------------
# *** SET YOUR SWEEP HERE ***
#   "main"     → 42 configs, 5 seeds  (210 jobs)
#   "followup" → len(TOP_N_NAMES) configs, 5 seeds  (fill in TOP_N_NAMES first)
# ---------------------------------------------------------------------------

SWEEP = "main"

# ---------------------------------------------------------------------------

if SWEEP == "main":
    CONFIGS = _build_main()
    SEEDS   = _ALL_SEEDS[:5]
elif SWEEP == "followup":
    CONFIGS = _build_followup()
    SEEDS   = _ALL_SEEDS[:5]
else:
    raise ValueError(f"Unknown SWEEP value: '{SWEEP}'")

# ---------------------------------------------------------------------------
# Finite-difference reference
# ---------------------------------------------------------------------------

FD_CONFIGS = [
    {"solver": "fd", "name": "fd_N512",  "N_y": 512,  "N_t": 3000},
    {"solver": "fd", "name": "fd_N1024", "N_y": 1024, "N_t": 6000},
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
