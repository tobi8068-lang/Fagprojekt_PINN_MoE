"""
Load all sweep results and generate ranked summary + comparison figures.

Usage:
    python plot.py                        # reads results/, writes figures/
    python plot.py --results results_test # point at the local test output
    python plot.py --top 10               # how many configs to show (default 15)

Figures produced:
    figures/1_ranking.png       — bar chart of top-N configs by mean L2 (± std)
    figures/2_convergence.png   — error vs epoch and vs wall-clock time for top 5
    figures/3_toggles.png       — boxplot: effect of each method toggle on L2
    figures/4_time_vs_error.png — scatter: training cost vs final accuracy
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # works on HPC (no display needed)
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 10, "figure.dpi": 150})


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _s(d, key, default=""):
    """Safely extract a scalar from an npz dict entry."""
    if key not in d:
        return default
    v = d[key]
    return v.item() if hasattr(v, "item") else v


def load_results(results_dir="results"):
    """
    Read all .npz files in results_dir.
    Returns (pinn_df, fd_df) — one row per run.
    pinn_df has a '_path' column so convergence plots can reload the arrays.
    """
    pinn_rows, fd_rows = [], []

    for path in sorted(glob.glob(os.path.join(results_dir, "*.npz"))):
        d = np.load(path, allow_pickle=True)
        solver = str(_s(d, "solver", "pinn"))

        if solver == "fd":
            fd_rows.append({
                "name":       str(_s(d, "config_name")),
                "l2_rel":     float(_s(d, "l2_rel_final")),
                "max_err":    float(_s(d, "max_err_final")),
                "time_sec":   float(_s(d, "solve_time_sec")),
                "N_y":        int(_s(d, "N_y", 0)),
                "N_t":        int(_s(d, "N_t", 0)),
            })
        else:
            # Recompute errors from saved grids (consistent metric regardless of
            # which evaluate() version was used during training).
            if "grid_u_pred" in d and "grid_u_exact" in d:
                u_pred  = d["grid_u_pred"].astype(float)   # (3, ny, nx)
                u_exact = d["grid_u_exact"].astype(float)  # (3, ny, nx)
                err     = (u_pred - u_exact).ravel()
                l2_rel  = float(np.linalg.norm(err) / np.linalg.norm(u_exact.ravel()))
                max_err = float(np.max(np.abs(err)))
            else:
                l2_rel  = float(_s(d, "l2_rel_final"))
                max_err = float(_s(d, "max_err_final"))

            pinn_rows.append({
                "name":                str(_s(d, "config_name")),
                "seed":                int(_s(d, "seed", 0)),
                "feature_map":         str(_s(d, "feature_map", "deterministic")),
                "use_moe":             bool(_s(d, "use_moe", False)),
                "moe_gating":          str(_s(d, "moe_gating", "")),
                "use_softadapt":       bool(_s(d, "use_softadapt", False)),
                "use_adaptive_refine": bool(_s(d, "use_adaptive_refine", False)),
                "use_lbfgs":           bool(_s(d, "use_lbfgs", False)),
                "l2_rel":              l2_rel,
                "max_err":             max_err,
                "time_sec":            float(_s(d, "total_time_sec", 0)),
                "n_params":            int(_s(d, "n_params", 0)),
                "_path":               path,
            })

    pinn_df = pd.DataFrame(pinn_rows) if pinn_rows else pd.DataFrame()
    fd_df   = pd.DataFrame(fd_rows)   if fd_rows   else pd.DataFrame()
    return pinn_df, fd_df


def rank_configs(pinn_df):
    """Aggregate over seeds; return DataFrame sorted by median L2 (ascending)."""
    agg = (
        pinn_df
        .groupby("name", sort=False)
        .agg(
            mean_l2            = ("l2_rel",              "mean"),
            median_l2          = ("l2_rel",              "median"),
            std_l2             = ("l2_rel",              "std"),
            min_l2             = ("l2_rel",              "min"),
            mean_time          = ("time_sec",            "mean"),
            n_seeds            = ("seed",                "count"),
            feature_map        = ("feature_map",         "first"),
            use_moe            = ("use_moe",             "first"),
            moe_gating         = ("moe_gating",          "first"),
            use_softadapt      = ("use_softadapt",       "first"),
            use_adaptive_refine= ("use_adaptive_refine", "first"),
            use_lbfgs          = ("use_lbfgs",           "first"),
        )
        .reset_index()
        .sort_values("median_l2")
        .reset_index(drop=True)
    )
    agg.index += 1   # 1-based rank
    return agg


def print_ranking(ranked, fd_df, top=20):
    header = f"\n{'Rank':<5} {'Config':<38} {'Median L2':>10} {'Mean L2':>10} {'Std L2':>10} {'Min L2':>10} {'Time (s)':>9} {'Seeds':>6}"
    print(header)
    print("-" * len(header))
    for rank, row in ranked.head(top).iterrows():
        std = f"{row['std_l2']:.2e}" if not np.isnan(row["std_l2"]) else "  n/a  "
        print(
            f"{rank:<5} {row['name']:<38} "
            f"{row['median_l2']:>10.3e} {row['mean_l2']:>10.3e} {std:>10} {row['min_l2']:>10.3e} "
            f"{row['mean_time']:>8.0f}s {row['n_seeds']:>6}"
        )
    if not fd_df.empty:
        print("\nFinite-difference reference:")
        for _, row in fd_df.iterrows():
            print(f"       {row['name']:<38} L2={row['l2_rel']:.3e}  time={row['time_sec']:.3f}s  N_y={row['N_y']}")
    print()


# ---------------------------------------------------------------------------
# Figure 1 — Ranking bar chart
# ---------------------------------------------------------------------------

def fig_ranking(ranked, fd_df, top=15, save_dir="figures"):
    top_df = ranked.head(top).iloc[::-1]   # reverse so best is at top

    fig, ax = plt.subplots(figsize=(9, max(4, top * 0.42)))

    colors = plt.cm.RdYlGn(np.linspace(0.85, 0.15, len(top_df)))
    bars = ax.barh(
        top_df["name"], top_df["median_l2"],
        xerr=top_df["std_l2"].fillna(0),
        color=colors, edgecolor="white", height=0.65, capsize=3,
    )
    for bar, val in zip(bars, top_df["median_l2"]):
        ax.text(bar.get_width() * 1.04, bar.get_y() + bar.get_height() / 2,
                f"{val:.2e}", va="center", fontsize=8)

    if not fd_df.empty:
        for _, row in fd_df.iterrows():
            ax.axvline(row["l2_rel"], color="steelblue", lw=1.5, ls="--",
                       label=f"FD {row['name']} ({row['l2_rel']:.2e})")
        ax.legend(fontsize=8)

    ax.set_xlabel("Median L2 relative error  (error bars = std across seeds)")
    ax.set_title(f"Config ranking — top {top}")
    ax.margins(y=0.02)
    fig.tight_layout()
    _save(fig, save_dir, "1_ranking.png")


# ---------------------------------------------------------------------------
# Figure 2 — Convergence curves (top 5 configs)
# ---------------------------------------------------------------------------

def fig_convergence(pinn_df, ranked, top=5, save_dir="figures"):
    top_names = ranked.head(top)["name"].tolist()
    colors    = plt.cm.tab10(np.linspace(0, 0.9, top))

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    (ax_l2_ep, ax_l2_t), (ax_li_ep, ax_li_t) = axes

    min_max_wt = np.inf   # shortest run's final wall-clock time → x-axis cutoff

    for color, name in zip(colors, top_names):
        subset = pinn_df[pinn_df["name"] == name]
        all_l2, all_li, all_ep, all_wt = [], [], [], []

        for _, row in subset.iterrows():
            d  = np.load(row["_path"], allow_pickle=True)
            ep = d["eval_epochs"].astype(float)
            l2 = d["eval_l2_rel"].astype(float)
            wt = d["hist_wall_time"].astype(float)
            if len(ep) == 0:
                continue
            li = d["eval_max_err"].astype(float) if "eval_max_err" in d else np.full_like(l2, np.nan)
            # epoch 0 is before training → wall time = 0; epochs ≥1 index into wt
            wt_at_eval = np.array([0.0 if e == 0 else wt[int(e) - 1] for e in ep])
            all_ep.append(ep); all_l2.append(l2); all_li.append(li); all_wt.append(wt_at_eval)

        if not all_l2:
            continue

        n      = min(len(x) for x in all_l2)
        ep_ref = all_ep[0][:n]
        wt_ref = np.mean([x[:n] for x in all_wt], axis=0)
        lbl    = name[:32]

        min_max_wt = min(min_max_wt, wt_ref[-1])

        for (ax_ep, ax_t), all_vals, ylabel in [
            ((ax_l2_ep, ax_l2_t), all_l2, "L2"),
            ((ax_li_ep, ax_li_t), all_li, "Linf"),
        ]:
            mat  = np.array([x[:n] for x in all_vals])
            mean = mat.mean(axis=0)
            std  = mat.std(axis=0)
            for ax, xs in [(ax_ep, ep_ref), (ax_t, wt_ref)]:
                ax.semilogy(xs, mean, color=color,
                            label=lbl if (ax is ax_l2_ep) else None, lw=1.8)
                ax.fill_between(xs,
                                np.maximum(mean - std, 1e-12),
                                mean + std,
                                color=color, alpha=0.12)

    if np.isfinite(min_max_wt):
        ax_l2_t.set_xlim(left=0, right=min_max_wt)
        ax_li_t.set_xlim(left=0, right=min_max_wt)

    for ax, xlabel, ylabel, title in [
        (ax_l2_ep, "Epoch",          "L2 relative error",   f"Top {top} — L2 vs epoch"),
        (ax_l2_t,  "Wall-clock (s)", "L2 relative error",   f"Top {top} — L2 vs time"),
        (ax_li_ep, "Epoch",          "L∞ absolute error",   f"Top {top} — L∞ vs epoch"),
        (ax_li_t,  "Wall-clock (s)", "L∞ absolute error",   f"Top {top} — L∞ vs time"),
    ]:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which="both", ls=":", alpha=0.4)

    ax_l2_ep.legend(fontsize=7, loc="upper right")

    fig.tight_layout()
    _save(fig, save_dir, "2_convergence.png")


# ---------------------------------------------------------------------------
# Figure 3 — Method toggle effect (boxplots)
# ---------------------------------------------------------------------------

def fig_toggles(pinn_df, save_dir="figures"):
    toggles = [
        ("feature_map",         "Feature map",      {"deterministic": "det", "fourier_rff": "rff"}),
        ("use_moe",             "MoE",              {False: "Vanilla", True: "MoE"}),
        ("use_softadapt",       "SoftAdapt",        {False: "Off", True: "On"}),
        ("use_adaptive_refine", "Adaptive refine",  {False: "Off", True: "On"}),
        ("use_lbfgs",           "L-BFGS",           {False: "Off", True: "On"}),
    ]

    fig, axes = plt.subplots(1, len(toggles), figsize=(14, 4.5), sharey=True)
    colors = ["#5599ee", "#ee7755"]

    for ax, (col, title, labels) in zip(axes, toggles):
        groups = {labels[k]: pinn_df[pinn_df[col] == k]["l2_rel"].values
                  for k in labels}
        keys = list(groups.keys())
        bp = ax.boxplot(
            [groups[k] for k in keys], labels=keys,
            patch_artist=True, notch=False,
            medianprops={"color": "black", "lw": 1.8},
            flierprops={"marker": ".", "markersize": 3, "alpha": 0.4},
        )
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c);  patch.set_alpha(0.75)
        ax.set_yscale("log")
        ax.set_title(title)
        ax.grid(True, axis="y", ls=":", alpha=0.4)
        if ax == axes[0]:
            ax.set_ylabel("L2 relative error")

    fig.suptitle("Effect of each method toggle — all configs, all seeds", y=1.01)
    fig.tight_layout()
    _save(fig, save_dir, "3_toggles.png", bbox_inches="tight")


# ---------------------------------------------------------------------------
# Figure 4 — Training time vs accuracy scatter
# ---------------------------------------------------------------------------

def fig_time_vs_error(pinn_df, fd_df, save_dir="figures"):
    MODEL_STYLES = {
        "vanilla_det":  ("#4c78d4", "o"),
        "vanilla_rff":  ("#a8c0f0", "o"),
        "moe_cont_det": ("#e87c2a", "s"),
        "moe_cont_rff": ("#f5c08a", "s"),
        "moe_bin_det":  ("#3db356", "^"),
        "moe_bin_rff":  ("#8ee8a0", "^"),
    }

    def model_key(row):
        if not row["use_moe"]:
            return f"vanilla_{row['feature_map'][:3]}"
        g = "cont" if row["moe_gating"] == "continuous" else "bin"
        return f"moe_{g}_{row['feature_map'][:3]}"

    df = pinn_df.copy()
    df["model_key"] = df.apply(model_key, axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))

    for key, grp in df.groupby("model_key"):
        c, m = MODEL_STYLES.get(key, ("gray", "o"))
        ax.scatter(grp["time_sec"], grp["l2_rel"],
                   label=key, color=c, marker=m,
                   s=30, alpha=0.65, edgecolors="white", lw=0.4)

    if not fd_df.empty:
        ymin = df["l2_rel"].min()
        for _, row in fd_df.iterrows():
            ax.axhline(row["l2_rel"], color="steelblue", lw=1.2, ls="--")
            ax.text(df["time_sec"].min() * 0.95, row["l2_rel"] * 0.88,
                    f"FD {row['name']}", fontsize=7, color="steelblue", va="top")

    ax.set_xscale("log");  ax.set_yscale("log")
    ax.set_xlabel("Training time (s)  [log scale]")
    ax.set_ylabel("Final L2 relative error  [log scale]")
    ax.set_title("Accuracy vs training cost  (each dot = one seed)")
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax.grid(True, which="both", ls=":", alpha=0.35)

    fig.tight_layout()
    _save(fig, save_dir, "4_time_vs_error.png")


# ---------------------------------------------------------------------------
# Figure 5 — Solution heatmaps (best config, t = 0 / T/2 / T)
# ---------------------------------------------------------------------------

def fig_heatmaps(pinn_df, ranked, save_dir="figures"):
    """
    3 rows (t = 0, T/2, T) × 3 columns (exact | predicted | |error|)
    for the best-ranked config (first seed available).
    If exact solution is NaN the exact column is skipped.
    """
    best_name = ranked.iloc[0]["name"]
    row       = pinn_df[pinn_df["name"] == best_name].iloc[0]
    d         = np.load(row["_path"], allow_pickle=True)

    if "grid_u_pred" not in d:
        print("  Skipping heatmap figure — no grid data in .npz (re-run training).")
        return

    x       = d["grid_x"]
    y       = d["grid_y"]
    t_vals  = d["grid_t_vals"]
    u_pred  = d["grid_u_pred"]   # (3, ny, nx)
    u_exact = d["grid_u_exact"]  # (3, ny, nx)

    has_exact = not np.all(np.isnan(u_exact))
    n_cols    = 3 if has_exact else 2
    col_titles = (["Exact", "Predicted", "|Error|"] if has_exact
                  else ["Predicted", "Mean |Error| (no exact)"])

    fig, axes = plt.subplots(
        3, n_cols,
        figsize=(4.5 * n_cols, 10),
        constrained_layout=True,
    )
    if axes.ndim == 1:
        axes = axes[:, np.newaxis]   # keep 2-D indexing

    vmin = min(u_pred.min(), u_exact[~np.isnan(u_exact)].min() if has_exact else u_pred.min())
    vmax = max(u_pred.max(), u_exact[~np.isnan(u_exact)].max() if has_exact else u_pred.max())

    for row_idx, t_val in enumerate(t_vals):
        up = u_pred[row_idx]
        ue = u_exact[row_idx]
        err = np.abs(up - ue) if has_exact else np.zeros_like(up)

        col_data = ([ue, up, err] if has_exact else [up, err])

        for col_idx, data in enumerate(col_data):
            ax  = axes[row_idx, col_idx]
            cmap = "RdBu_r" if col_idx < (n_cols - 1) else "Reds"
            vlo  = vmin      if col_idx < (n_cols - 1) else 0.0
            vhi  = vmax      if col_idx < (n_cols - 1) else err.max() or 1.0

            im = ax.pcolormesh(x, y, data, cmap=cmap, vmin=vlo, vmax=vhi, shading="auto")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=11)
            ax.set_ylabel(f"t = {t_val:.2f}" if col_idx == 0 else "y")
            ax.set_xlabel("x")

    fig.suptitle(f"Solution heatmaps — {best_name}", fontsize=12, y=1.01)
    _save(fig, save_dir, "5_heatmaps.png", bbox_inches="tight")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(fig, save_dir, filename, **kwargs):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    fig.savefig(path, **kwargs)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results",
                        help="Directory containing .npz result files")
    parser.add_argument("--figures", default="figures",
                        help="Directory to write figures into")
    parser.add_argument("--top",     type=int, default=15,
                        help="How many configs to show in ranking plot")
    args = parser.parse_args()

    print(f"Loading results from: {args.results}")
    pinn_df, fd_df = load_results(args.results)

    if pinn_df.empty:
        sys.exit(f"No PINN results found in '{args.results}'.")

    ranked = rank_configs(pinn_df)
    print_ranking(ranked, fd_df, top=args.top)

    print("Generating figures...")
    fig_ranking(ranked,    fd_df,  top=args.top,  save_dir=args.figures)
    fig_convergence(pinn_df, ranked, top=5,         save_dir=args.figures)
    fig_toggles(pinn_df,                            save_dir=args.figures)
    fig_time_vs_error(pinn_df, fd_df,               save_dir=args.figures)
    fig_heatmaps(pinn_df, ranked,                   save_dir=args.figures)

    print(f"\nDone. All figures in {args.figures}/")


if __name__ == "__main__":
    main()
