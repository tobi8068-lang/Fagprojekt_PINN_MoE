"""
Visualize the 50-seed re-runs of the two high-variance MoE configs
(moe_cont_rff0_fd0_sa0_ar1_lb1 and moe_cont_rff0_fd0_sa1_ar0_lb1) and compare
each against its original 5-seed run, to see whether the wide error bars were
a consistent property of the config or just an unlucky seed draw.

Usage:
    python plot_50seeds.py
    python plot_50seeds.py --seeds50 results_50seeds --main results_run1
"""
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot import load_results, pretty_name, _save

plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

# light / dark shade per config, reused across the two panels
COLORS = [("#F7A07E", "#C44B1A"), ("#8ad6c2", "#1a8a6b")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds50", default="results_50seeds",
                         help="Directory with the 50-seed re-runs")
    parser.add_argument("--main", default="results_run1",
                         help="Directory with the original 5-seed sweep "
                              "(use 'results' if you haven't archived it yet)")
    parser.add_argument("--figures", default="figures_50seeds")
    args = parser.parse_args()

    df50, _   = load_results(args.seeds50)
    df5, _    = load_results(args.main)
    if df50.empty:
        raise SystemExit(f"No results found in '{args.seeds50}'.")

    names = df50["name"].unique().tolist()

    header = f"\n{'Config':<45} {'N5':>3} {'5-seed median':>14} {'N50':>4} {'50-seed median':>15} {'50-seed mean':>13} {'50-seed std':>12}"
    print(header)
    print("-" * len(header))
    for name in names:
        v50 = df50[df50["name"] == name]["l2_rel"].values
        v5  = df5[df5["name"] == name]["l2_rel"].values
        med5 = f"{np.median(v5):.3e}" if len(v5) else "n/a"
        print(f"{pretty_name(name):<45} {len(v5):>3} {med5:>14} {len(v50):>4} "
              f"{np.median(v50):>15.3e} {np.mean(v50):>13.3e} {np.std(v50):>12.3e}")

    fig, axes = plt.subplots(1, len(names), figsize=(6.5 * len(names), 5.5), sharey=True)
    if len(names) == 1:
        axes = [axes]
    rng = np.random.default_rng(0)

    for ax, name, (c0, c1) in zip(axes, names, COLORS):
        v50 = df50[df50["name"] == name]["l2_rel"].values
        v5  = df5[df5["name"] == name]["l2_rel"].values

        groups = [v5, v50] if len(v5) else [v50]
        xpos   = [1, 2] if len(v5) else [1]
        labels = ["5-seed\n(original)", "50-seed\n(new)"] if len(v5) else ["50-seed\n(new)"]
        colors = [c0, c1] if len(v5) else [c1]

        bp = ax.boxplot(
            groups, positions=xpos, widths=0.5, labels=labels,
            patch_artist=True, notch=False,
            medianprops={"color": "black", "lw": 1.8},
            flierprops={"marker": ".", "markersize": 4, "alpha": 0.4},
        )
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c);  patch.set_alpha(0.8)

        # Overlay raw points so the small-N 5-seed group isn't hidden by the box
        for x, vals in zip(xpos, groups):
            jitter = x + rng.uniform(-0.08, 0.08, size=len(vals))
            ax.scatter(jitter, vals, color="black", s=14, alpha=0.5, zorder=5)

        ax.set_yscale("log")
        ax.set_xlim(0.5, 2.5)
        ax.set_title(pretty_name(name), fontsize=10)
        ax.grid(True, axis="y", ls=":", alpha=0.4)

    axes[0].set_ylabel("L2 relative error")
    fig.suptitle("Original 5 seeds vs new 50 seeds — same config", y=1.02)
    fig.tight_layout()
    _save(fig, args.figures, "50seeds_vs_5seeds.png")


if __name__ == "__main__":
    main()
