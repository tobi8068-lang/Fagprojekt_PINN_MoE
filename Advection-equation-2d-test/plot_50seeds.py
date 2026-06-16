"""
Visualize the 50-seed re-runs of the two high-variance MoE configs
(moe_cont_rff0_fd0_sa0_ar1_lb1 and moe_cont_rff0_fd0_sa1_ar0_lb1).

Usage:
    python plot_50seeds.py                            # reads results_50seeds/, writes figures_50seeds/
    python plot_50seeds.py --results results_50seeds  # explicit
"""
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot import load_results, pretty_name, _save

plt.rcParams.update({"font.size": 10, "figure.dpi": 150})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results_50seeds")
    parser.add_argument("--figures", default="figures_50seeds")
    args = parser.parse_args()

    pinn_df, _ = load_results(args.results)
    if pinn_df.empty:
        raise SystemExit(f"No results found in '{args.results}'.")

    names  = pinn_df["name"].unique().tolist()
    groups = [pinn_df[pinn_df["name"] == n]["l2_rel"].values for n in names]

    header = f"\n{'Config':<55} {'N':>4} {'Median L2':>10} {'Mean L2':>10} {'Std L2':>10} {'Min L2':>10} {'Max L2':>10}"
    print(header)
    print("-" * len(header))
    for n, vals in zip(names, groups):
        print(f"{pretty_name(n):<55} {len(vals):>4} "
              f"{np.median(vals):>10.3e} {np.mean(vals):>10.3e} {np.std(vals):>10.3e} "
              f"{vals.min():>10.3e} {vals.max():>10.3e}")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(names)))
    bp = ax.boxplot(
        groups, labels=[pretty_name(n) for n in names],
        patch_artist=True, notch=False,
        medianprops={"color": "black", "lw": 1.8},
        flierprops={"marker": ".", "markersize": 4, "alpha": 0.5},
    )
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c);  patch.set_alpha(0.8)

    ax.set_yscale("log")
    ax.set_ylabel("L2 relative error")
    ax.set_title(f"50-seed distribution  ({sum(len(g) for g in groups)} runs total)")
    ax.grid(True, axis="y", ls=":", alpha=0.4)
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    _save(fig, args.figures, "50seeds_boxplot.png")


if __name__ == "__main__":
    main()
