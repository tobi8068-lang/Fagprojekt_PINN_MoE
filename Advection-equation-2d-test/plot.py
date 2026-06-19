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
import re
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


def _smooth(arr, w):
    """Centered moving average with edge-value padding to avoid boundary dip."""
    arr = np.asarray(arr, dtype=float)
    if len(arr) <= w:
        return arr.copy()
    pad    = w // 2
    padded = np.concatenate([np.full(pad, arr[0]), arr, np.full(pad, arr[-1])])
    return np.convolve(padded, np.ones(w) / w, mode="valid")[:len(arr)]


_MOD_LABELS = [
    ("rff", "Fourier Features"),
    ("fd",  "Finite Diff."),
    ("sa",  "SoftAdapt"),
    ("ar",  "Adaptive Refine"),
    ("lb",  "L-BFGS"),
]
_NAME_RE = re.compile(r"^(vanilla|moe_\w+?)_rff(\d)_fd(\d)_sa(\d)_ar(\d)_lb(\d)$")


def pretty_name(name):
    """e.g. 'moe_cont_rff1_fd0_sa1_ar0_lb0' -> 'MoE with Fourier Features, SoftAdapt'."""
    m = _NAME_RE.match(name)
    if not m:
        return name
    model_tag, rff, fd, sa, ar, lb = m.groups()
    model = "MoE" if model_tag.startswith("moe") else "Vanilla"
    flags = dict(zip(["rff", "fd", "sa", "ar", "lb"], [rff, fd, sa, ar, lb]))
    mods  = [label for key, label in _MOD_LABELS if flags[key] == "1"]
    return f"{model} (baseline)" if not mods else f"{model} with " + ", ".join(mods)


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
                "feature_map":         "fourier_rff" if bool(_s(d, "use_rff", False)) else "deterministic",
                "use_moe":             bool(_s(d, "use_moe", False)),
                "moe_gating":          str(_s(d, "moe_gating", "")),
                "use_fd_deriv":        bool(_s(d, "use_fd_deriv", False)),
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

    fig, ax = plt.subplots(figsize=(11, max(4, top * 0.42)))

    colors = plt.cm.RdYlGn(np.linspace(0.85, 0.15, len(top_df)))
    bars = ax.barh(
        top_df["name"].map(pretty_name), top_df["median_l2"],
        xerr=top_df["std_l2"].fillna(0),
        color=colors, edgecolor="white", height=0.65, capsize=3,
    )
    ax.tick_params(axis="y", labelsize=8)
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


# Epoch-axis placeholder for the post-L-BFGS result; must exceed adam_epochs (10000).
_LBFGS_TICK = 11500


def _inject_lbfgs(d, ep, l2, wt_at_eval):
    """Append the post-L-BFGS eval point at _LBFGS_TICK; handles missing or duplicate entry."""
    if not bool(_s(d, "use_lbfgs", False)) or "l2_rel_final" not in d:
        return ep, l2, wt_at_eval, False
    l2_final = float(_s(d, "l2_rel_final"))
    t_final  = float(_s(d, "total_time_sec", wt_at_eval[-1]))
    # New runs record the point at adam_epochs (same x as last Adam eval); remove it.
    if len(ep) >= 2 and ep[-1] == ep[-2]:
        ep = ep[:-1]; l2 = l2[:-1]; wt_at_eval = wt_at_eval[:-1]
    ep         = np.append(ep, _LBFGS_TICK)
    l2         = np.append(l2, l2_final)
    wt_at_eval = np.append(wt_at_eval, t_final)
    return ep, l2, wt_at_eval, True


def _fix_lbfgs_ticks(ax):
    """Replace the sentinel x-tick with an 'L-BFGS' label and add a separator."""
    ax.axvline(10050, color="gray", lw=0.9, ls="--", alpha=0.55)
    ax.set_xticks([0, 2000, 4000, 6000, 8000, 10000, _LBFGS_TICK])
    ax.set_xticklabels(["0", "2k", "4k", "6k", "8k", "10k", "L-BFGS"])
    ax.tick_params(axis="x", which="major", pad=4)
    plt.setp(ax.get_xticklabels()[-2:], rotation=30, ha="right")


# ---------------------------------------------------------------------------
# Figure 2 — Convergence curves (top 5 configs)
# ---------------------------------------------------------------------------

def fig_convergence(pinn_df, ranked, fd_df=None, top=5, save_dir="figures"):
    top_names = ranked.head(top)["name"].tolist()
    colors    = plt.cm.tab10(np.linspace(0, 0.9, top))

    fig, (ax_l2_ep, ax_l2_t) = plt.subplots(1, 2, figsize=(13, 5))

    min_max_wt  = np.inf
    has_lbfgs   = False
    term_labels = []   # (x, y, text, color) added after loop to avoid overlap

    for color, name in zip(colors, top_names):
        subset = pinn_df[pinn_df["name"] == name]
        all_l2, all_ep, all_wt = [], [], []
        final_l2_vals = []

        for _, row in subset.iterrows():
            d  = np.load(row["_path"], allow_pickle=True)
            ep = d["eval_epochs"].astype(float)
            l2 = d["eval_l2_rel"].astype(float)
            wt = d["hist_wall_time"].astype(float)
            if len(ep) == 0:
                continue
            wt_at_eval = np.array([0.0 if e == 0 else wt[int(e) - 1] for e in ep])
            ep, l2, wt_at_eval, injected = _inject_lbfgs(d, ep, l2, wt_at_eval)
            has_lbfgs = has_lbfgs or injected
            # Snap terminal point to the 200×200 recomputed L2 used in ranking
            l2[-1] = row["l2_rel"]
            wt_at_eval[-1] = row["time_sec"]
            final_l2_vals.append(row["l2_rel"])
            all_ep.append(ep); all_l2.append(l2); all_wt.append(wt_at_eval)

        if not all_l2:
            continue

        n      = min(len(x) for x in all_l2)
        ep_ref = all_ep[0][:n]
        wt_ref = np.mean([x[:n] for x in all_wt], axis=0)
        lbl    = pretty_name(name)
        min_max_wt = min(min_max_wt, wt_ref[-1])

        mat    = np.array([x[:n] for x in all_l2])
        sm_mat = np.array([_smooth(r, w=5) for r in mat])
        sm     = sm_mat.mean(axis=0)
        # Snap final point to true value — smoothing window would otherwise blend the L-BFGS drop.
        final_mean = float(np.mean(final_l2_vals))
        sm_mat[:, -1] = np.array(final_l2_vals)
        sm[-1] = final_mean
        log_std = np.std(np.log(np.maximum(sm_mat, 1e-30)), axis=0)
        for ax, xs in [(ax_l2_ep, ep_ref), (ax_l2_t, wt_ref)]:
            ax.semilogy(xs, sm, color=color,
                        label=lbl if ax is ax_l2_ep else None, lw=1.8)
            ax.fill_between(xs,
                            sm * np.exp(-log_std),
                            sm * np.exp(+log_std),
                            color=color, alpha=0.12)

        term_labels.append((ep_ref[-1], final_mean, f"{final_mean:.2e}", color))

    if np.isfinite(min_max_wt):
        ax_l2_t.set_xlim(left=0, right=min_max_wt)

    for x, y, txt, color in term_labels:
        ax_l2_ep.annotate(txt, xy=(x, y), xytext=(5, 0),
                          textcoords="offset points",
                          color=color, fontsize=6, va="center")

    if fd_df is not None and not fd_df.empty:
        for _, row in fd_df.iterrows():
            for ax in (ax_l2_ep, ax_l2_t):
                ax.axhline(row["l2_rel"], color="steelblue", lw=1.2, ls="--",
                           label=f"FD {row['name']} ({row['l2_rel']:.2e})")

    for ax, xlabel, title in [
        (ax_l2_ep, "Epoch",          f"Top {top} — L2 relative vs epoch (smoothed, w=5)"),
        (ax_l2_t,  "Wall-clock (s)", f"Top {top} — L2 relative vs time (smoothed, w=5)"),
    ]:
        ax.set_xlabel(xlabel)
        ax.set_ylabel("L2 relative error")
        ax.set_title(title)
        ax.grid(True, which="both", ls=":", alpha=0.4)

    if has_lbfgs:
        _fix_lbfgs_ticks(ax_l2_ep)
    ax_l2_ep.legend(fontsize=6, loc="upper right")

    fig.tight_layout()
    _save(fig, save_dir, "2_convergence.png")


# ---------------------------------------------------------------------------
# Figure 6 — Curated convergence: top-3, best vanilla, best MoE, median
# ---------------------------------------------------------------------------

def fig_convergence_curated(pinn_df, ranked, fd_df=None, save_dir="figures"):
    van_ranked = ranked[ranked["use_moe"] == False]
    moe_ranked = ranked[ranked["use_moe"] == True]
    best_van   = van_ranked.iloc[0]["name"];  van_rank = int(van_ranked.index[0])
    best_moe   = moe_ranked.iloc[0]["name"];  moe_rank = int(moe_ranked.index[0])
    med_row    = ranked.iloc[len(ranked) // 2]
    med_name   = med_row["name"];             med_rank = int(med_row.name)

    selected = {
        best_moe: (f"#{moe_rank} {pretty_name(best_moe)} (best MoE)", "--"),
        best_van: (f"#{van_rank} {pretty_name(best_van)} (best Vanilla)", "-"),
        med_name: (f"#{med_rank} {pretty_name(med_name)} (median)", ":"),
    }
    names_ordered = ranked["name"].tolist()
    ordered = sorted(selected.keys(), key=lambda n: names_ordered.index(n))
    colors  = ["#e87c2a", "#4c78d4", "#555555"]

    fig, (ax_ep, ax_t) = plt.subplots(1, 2, figsize=(13, 5))
    has_lbfgs    = False
    term_ep      = []   # (x_ep, y, txt, color)
    term_t       = []   # (x_t,  y, txt, color)

    for color, name in zip(colors, ordered):
        lbl, ls = selected[name]
        subset  = pinn_df[pinn_df["name"] == name]
        all_l2, all_ep, all_wt = [], [], []
        final_l2_vals = []

        for _, row in subset.iterrows():
            d  = np.load(row["_path"], allow_pickle=True)
            ep = d["eval_epochs"].astype(float)
            l2 = d["eval_l2_rel"].astype(float)
            wt = d["hist_wall_time"].astype(float)
            if len(ep) == 0:
                continue
            wt_at_eval = np.array([0.0 if e == 0 else wt[int(e) - 1] for e in ep])
            ep, l2, wt_at_eval, injected = _inject_lbfgs(d, ep, l2, wt_at_eval)
            has_lbfgs = has_lbfgs or injected
            # Snap terminal point to the 200×200 recomputed L2 used in ranking
            l2[-1] = row["l2_rel"]
            wt_at_eval[-1] = row["time_sec"]
            final_l2_vals.append(row["l2_rel"])
            all_ep.append(ep); all_l2.append(l2); all_wt.append(wt_at_eval)

        if not all_l2:
            continue

        n      = min(len(x) for x in all_l2)
        ep_ref = all_ep[0][:n]
        wt_ref = np.mean([x[:n] for x in all_wt], axis=0)

        mat    = np.array([x[:n] for x in all_l2])
        sm_mat = np.array([_smooth(r, w=3) for r in mat])
        sm     = sm_mat.mean(axis=0)
        # Snap final point to true value — smoothing would otherwise raise sm[-1].
        final_mean = float(np.mean(final_l2_vals))
        sm_mat[:, -1] = np.array(final_l2_vals)
        sm[-1] = final_mean
        log_std = np.std(np.log(np.maximum(sm_mat, 1e-30)), axis=0)
        for ax, xs in [(ax_ep, ep_ref), (ax_t, wt_ref)]:
            ax.semilogy(xs, sm, color=color, ls=ls,
                        label=lbl if ax is ax_ep else None, lw=1.8)
            ax.fill_between(xs,
                            sm * np.exp(-log_std),
                            sm * np.exp(+log_std),
                            color=color, alpha=0.15)

        # Vertical end-of-run marker on time axis
        ax_t.axvline(wt_ref[-1], color=color, lw=0.8, ls=":", alpha=0.6)

        term_ep.append((ep_ref[-1], final_mean, f"{final_mean:.2e}", color))
        term_t.append( (wt_ref[-1], final_mean, f"{final_mean:.2e}", color))

    for x, y, txt, color in term_ep:
        ax_ep.annotate(txt, xy=(x, y), xytext=(5, 0),
                       textcoords="offset points",
                       color=color, fontsize=7, va="center")
    for x, y, txt, color in term_t:
        ax_t.annotate(txt, xy=(x, y), xytext=(5, 0),
                      textcoords="offset points",
                      color=color, fontsize=7, va="center")

    if fd_df is not None and not fd_df.empty:
        for _, row in fd_df.iterrows():
            for ax in (ax_ep, ax_t):
                ax.axhline(row["l2_rel"], color="steelblue", lw=1.2, ls="--",
                           label=f"FD {row['name']} ({row['l2_rel']:.2e})")

    for ax, xlabel, title in [
        (ax_ep, "Epoch",          "L2 relative vs epoch (smoothed, w=3)"),
        (ax_t,  "Wall-clock (s)", "L2 relative vs time (smoothed, w=3)"),
    ]:
        ax.set_xlabel(xlabel)
        ax.set_ylabel("L2 relative error")
        ax.set_title(title)
        ax.grid(True, which="both", ls=":", alpha=0.4)

    if has_lbfgs:
        _fix_lbfgs_ticks(ax_ep)
    ax_ep.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    _save(fig, save_dir, "6_convergence_curated.png")


# ---------------------------------------------------------------------------
# Figure 3 — Method toggle effect (boxplots)
# ---------------------------------------------------------------------------

def fig_toggles(pinn_df, save_dir="figures"):
    # Each entry: (column, title, {key: short_label})
    # For boolean toggles keys are False/True; for feature_map keys are strings.
    toggles = [
        ("feature_map",         "Fourier features",  {"deterministic": "Off", "fourier_rff": "On"}),
        ("use_fd_deriv",        "Finite diff.",      {False: "Off", True: "On"}),
        ("use_softadapt",       "SoftAdapt",         {False: "Off", True: "On"}),
        ("use_adaptive_refine", "Adaptive refine",   {False: "Off", True: "On"}),
        ("use_lbfgs",           "L-BFGS",            {False: "Off", True: "On"}),
    ]

    # 4 colours: vanilla-off, vanilla-on, moe-off, moe-on
    CV0, CV1 = "#7EB8F7", "#1A6BB5"   # vanilla: light / dark blue
    CM0, CM1 = "#F7A07E", "#C44B1A"   # moe:     light / dark orange

    van = pinn_df[pinn_df["use_moe"] == False]
    moe = pinn_df[pinn_df["use_moe"] == True]

    n_panels = 1 + len(toggles)   # MoE overview + one per method toggle
    fig, axes = plt.subplots(1, n_panels, figsize=(24,5), sharey=True)

    # --- Panel 0: Vanilla vs MoE (overview, 2 boxes) -------------------------
    ax = axes[0]
    bp = ax.boxplot(
        [van["l2_rel"].values, moe["l2_rel"].values],
        labels=["Vanilla", "MoE"],
        patch_artist=True, notch=False,
        medianprops={"color": "black", "lw": 1.8},
        flierprops={"marker": ".", "markersize": 3, "alpha": 0.4},
    )
    for patch, c in zip(bp["boxes"], [CV0, CM0]):
        patch.set_facecolor(c);  patch.set_alpha(0.85)
    ax.set_yscale("log");  ax.set_title("MoE")
    ax.set_ylabel("L2 relative error")
    ax.grid(True, axis="y", ls=":", alpha=0.4)

    # --- Panels 1+: 4 boxes each (V-off, V-on, M-off, M-on) -----------------
    for ax, (col, title, labels) in zip(axes[1:], toggles):
        keys = list(labels.keys())
        k0, k1 = keys
        data = [
            van[van[col] == k0]["l2_rel"].values,
            van[van[col] == k1]["l2_rel"].values,
            moe[moe[col] == k0]["l2_rel"].values,
            moe[moe[col] == k1]["l2_rel"].values,
        ]
        xpos   = [1, 2, 3.3, 4.3]
        xlbls  = [f"V·{labels[k0]}", f"V·{labels[k1]}",
                  f"M·{labels[k0]}", f"M·{labels[k1]}"]
        colors4 = [CV0, CV1, CM0, CM1]

        bp = ax.boxplot(
            data, positions=xpos, widths=0.65, labels=xlbls,
            patch_artist=True, notch=False,
            medianprops={"color": "black", "lw": 1.8},
            flierprops={"marker": ".", "markersize": 3, "alpha": 0.4},
        )
        for patch, c in zip(bp["boxes"], colors4):
            patch.set_facecolor(c);  patch.set_alpha(0.85)
        ax.set_xlim(0.3, 5.0)
        ax.set_yscale("log");  ax.set_title(title)
        ax.grid(True, axis="y", ls=":", alpha=0.4)

    # Legend
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=CV0, alpha=0.85, label="Vanilla · Off"),
        Patch(facecolor=CV1, alpha=0.85, label="Vanilla · On"),
        Patch(facecolor=CM0, alpha=0.85, label="MoE · Off"),
        Patch(facecolor=CM1, alpha=0.85, label="MoE · On"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=4,
               fontsize=13, bbox_to_anchor=(0.5, 1.08))

    fig.suptitle("Effect of each method toggle — all configs, all seeds", y=1.14,fontsize=21)
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

    fig.suptitle(f"Solution heatmaps — {pretty_name(best_name)}", fontsize=12, y=1.01)
    _save(fig, save_dir, "5_heatmaps.png", bbox_inches="tight")


# ---------------------------------------------------------------------------
# Figure 7 — loss component breakdown for curated configs
# ---------------------------------------------------------------------------

def fig_loss_breakdown(pinn_df, ranked, save_dir="figures"):
    van_ranked = ranked[ranked["use_moe"] == False]
    moe_ranked = ranked[ranked["use_moe"] == True]
    best_van   = van_ranked.iloc[0]["name"];  van_rank = int(van_ranked.index[0])
    best_moe   = moe_ranked.iloc[0]["name"];  moe_rank = int(moe_ranked.index[0])
    med_row    = ranked.iloc[len(ranked) // 2]
    med_name   = med_row["name"];             med_rank = int(med_row.name)

    selected = {
        best_moe: (f"#{moe_rank} {pretty_name(best_moe)} (best MoE)", "--"),
        best_van: (f"#{van_rank} {pretty_name(best_van)} (best Vanilla)", "-"),
        med_name: (f"#{med_rank} {pretty_name(med_name)} (median)", ":"),
    }
    names_ordered = ranked["name"].tolist()
    ordered = sorted(selected.keys(), key=lambda n: names_ordered.index(n))
    colors  = ["#e87c2a", "#4c78d4", "#555555"]

    components = [("pde", "PDE loss"), ("ini", "IC loss"), ("bc", "BC loss")]
    fig, axes  = plt.subplots(1, 3, figsize=(15, 4), sharey=False)

    W = 300  # smoothing window for per-epoch loss history

    for color, name in zip(colors, ordered):
        lbl, ls = selected[name]
        subset  = pinn_df[pinn_df["name"] == name]

        all_losses = {key: [] for key, _ in components}
        epoch_ref  = None

        for _, row in subset.iterrows():
            d = np.load(row["_path"], allow_pickle=True)
            for key, _ in components:
                arr_key = f"hist_{key}"
                if arr_key in d and len(d[arr_key]) > 0:
                    all_losses[key].append(np.array(d[arr_key], dtype=float))
            if epoch_ref is None and "hist_pde" in d and len(d["hist_pde"]) > 0:
                epoch_ref = np.arange(len(d["hist_pde"]))

        if epoch_ref is None:
            continue

        for ax, (key, title) in zip(axes, components):
            arrs = all_losses[key]
            if not arrs:
                continue
            n    = min(len(a) for a in arrs)
            mat  = np.array([a[:n] for a in arrs])
            mean = mat.mean(axis=0)
            sm   = _smooth(mean, w=W)
            xs   = epoch_ref[:n]
            ax.semilogy(xs, mean, color=color, ls=ls, lw=0.5, alpha=0.15)
            ax.semilogy(xs, sm,   color=color, ls=ls, lw=1.8,
                        label=lbl if ax is axes[0] else None)

    for ax, (_, title) in zip(axes, components):
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, which="both", ls=":", alpha=0.4)

    axes[0].legend(fontsize=7, loc="upper right")
    fig.suptitle(f"Loss component breakdown — curated configs (smoothed, w={W})", fontsize=11)
    fig.tight_layout()
    _save(fig, save_dir, "7_loss_breakdown.png")


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
    fig_convergence(pinn_df, ranked, fd_df=fd_df, top=5, save_dir=args.figures)
    fig_convergence_curated(pinn_df, ranked, fd_df=fd_df, save_dir=args.figures)
    fig_loss_breakdown(pinn_df, ranked,              save_dir=args.figures)
    fig_toggles(pinn_df,                            save_dir=args.figures)
    fig_time_vs_error(pinn_df, fd_df,               save_dir=args.figures)
    fig_heatmaps(pinn_df, ranked,                   save_dir=args.figures)

    print(f"\nDone. All figures in {args.figures}/")


if __name__ == "__main__":
    main()
