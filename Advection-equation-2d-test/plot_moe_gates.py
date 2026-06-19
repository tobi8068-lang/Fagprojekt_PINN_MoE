"""
Plot MoE expert gate weights from a single result file that contains grid_gates.

Usage:
    python plot_moe_gates.py                                        # auto-finds results_best_moe/
    python plot_moe_gates.py --npz results_best_moe/cfg23_seed1234.npz
    python plot_moe_gates.py --out_dir figures_moe

Layout  (1 row):
    Expert mixing map at t = 0, T/2, T.
    Each pixel is a colour-blend of the three expert colours weighted
    by their gate values (8× contrast enhancement around uniform = 1/3).
    Exact-solution contours overlaid for spatial reference.
"""

import argparse
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from plot import pretty_name

plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

# One colour per expert (orange, blue, green) — same as the rest of the report
EXPERT_RGB = np.array([
    [0.91, 0.49, 0.16],   # Expert 1 — orange
    [0.30, 0.47, 0.83],   # Expert 2 — blue
    [0.24, 0.70, 0.34],   # Expert 3 — green
], dtype=float)

EXPERT_HEX = ["#e87c2a", "#4c78d4", "#3db356"]

# Amplification factor: deviations from 1/3 are small (~0.1), so we stretch
# them to fill the colour space and reveal spatial structure.
CONTRAST = 8.0


def blend_colors(gate_snap):
    """
    gate_snap : (ny, nx, n_exp)
    Returns    : (ny, nx, 3)  RGB image in [0, 1]

    Amplifies deviations from uniform mixing (1/n_exp) by CONTRAST, clips to
    [0,1], renormalises so weights still sum to 1, then blends expert colours.
    """
    n_exp   = gate_snap.shape[-1]
    uniform = 1.0 / n_exp
    g = uniform + (gate_snap - uniform) * CONTRAST
    g = np.clip(g, 0.0, 1.0)
    g /= g.sum(axis=-1, keepdims=True)
    return np.einsum("...e,ec->...c", g, EXPERT_RGB[:n_exp])


def plot_gates(d, save_dir):
    gates    = d["grid_gates"]          # (3, ny, nx, n_experts)
    x        = d["grid_x"]
    y        = d["grid_y"]
    t_vals   = d["grid_t_vals"]
    u_exact  = d["grid_u_exact"]        # (3, ny, nx)
    cfg_name = pretty_name(str(d["config_name"]))
    l2_final = float(d["l2_rel_final"])

    n_t, ny, nx, n_exp = gates.shape
    uniform = 1.0 / n_exp

    fig, axes = plt.subplots(1, n_t, figsize=(5 * n_t, 4.5),
                             constrained_layout=True)

    for col, (t_val, gate_snap, ue) in enumerate(zip(t_vals, gates, u_exact)):
        ax  = axes[col]
        rgb = blend_colors(gate_snap)

        extent = [x.min(), x.max(), y.min(), y.max()]
        ax.imshow(rgb, origin="lower", extent=extent,
                  aspect="auto", interpolation="bilinear")

        if not np.all(np.isnan(ue)):
            lvls = np.linspace(ue.max() * 0.1, ue.max() * 0.9, 5)
            ax.contour(x, y, ue, levels=lvls,
                       colors="white", linewidths=0.8, alpha=0.7)

        ax.set_title(f"t = {t_val:.2f}", fontsize=11)
        ax.set_xlabel("x")
        ax.set_ylabel("y" if col == 0 else "")

    patches = [mpatches.Patch(color=EXPERT_HEX[i], label=f"Expert {i+1}")
               for i in range(n_exp)]
    axes[0].legend(handles=patches, loc="upper left", fontsize=8, framealpha=0.7)

    fig.suptitle(
        f"MoE expert mixing map — {cfg_name}  ({CONTRAST:.0f}× contrast, "
        f"white contours = exact solution)\n"
        f"L2 rel = {l2_final:.2e}   |   uniform gate = {uniform:.3f}",
        fontsize=11,
    )

    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, "moe_gates.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def load_npz(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    if "grid_gates" not in d:
        sys.exit(f"No grid_gates found in {npz_path}. "
                 "Re-run with the updated train.py.")
    return d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz",     type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="figures")
    args = parser.parse_args()

    if args.npz:
        path = args.npz
    else:
        candidates = sorted(glob.glob("results_best_moe/*.npz"))
        if not candidates:
            sys.exit("No .npz files found in results_best_moe/. "
                     "Run run_best_moe.py first, or pass --npz <path>.")
        path = candidates[0]
        print(f"Using: {path}")

    plot_gates(load_npz(path), args.out_dir)


if __name__ == "__main__":
    main()
