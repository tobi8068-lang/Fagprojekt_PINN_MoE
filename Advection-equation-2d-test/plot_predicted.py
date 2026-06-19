"""
Plot the exact solution at t = 0, T/2, T on a slightly expanded domain
so the Gaussian tails are never clipped at the boundary.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

# =============================================================================
# Settings — tweak here
# =============================================================================

SAVE_PATH = "figures/predicted.png"
DPI       = 250
PANEL_W   = 8.0        # inches per panel
ROW_GAP   = 0.02       # gap between panels as fraction of panel width

# How much to pad beyond the training domain on each side
X_PAD = 2.0
Y_PAD = 2.0

# Grid resolution
NX, NY = 300, 300

# =============================================================================

# Domain / PDE parameters (mirror configs.py — no import needed)
X_LO, X_HI = -1.0,  8.0
Y_LO, Y_HI = -7.0,  7.0
T  = 1.0
v  = 3.0
Dl = 1.0
Dt = 1.0
p  = 1.0   # initial x-centre of the Gaussian IC

T_VALS = [0.0, T / 2, T]

CMAP = mcolors.LinearSegmentedColormap.from_list("dtu_contrast", [
    (0.00, 0.85, 0.85, 0.0),   # cyan, fully transparent
    (0.00, 0.85, 0.85, 1.0),   # cyan, fully opaque
    (1.00, 1.00, 1.00, 1.0),   # white at peak
])


def exact(x, y, t):
    dx = 1.0 + Dl * t
    dy = 1.0 + Dt * t
    return (1.0 / np.sqrt(dx * dy)) * np.exp(
        -((x - p - v * t) ** 2) / (4.0 * dx) - (y ** 2) / (4.0 * dy)
    )


def main():
    x = np.linspace(X_LO - X_PAD, X_HI + X_PAD, NX)
    y = np.linspace(Y_LO - Y_PAD, Y_HI + Y_PAD, NY)
    X, Y = np.meshgrid(x, y)

    frames = [exact(X, Y, t) for t in T_VALS]
    vmin = min(f.min() for f in frames)
    vmax = max(f.max() for f in frames)

    x_range = (X_HI + X_PAD) - (X_LO - X_PAD)
    y_range = (Y_HI + Y_PAD) - (Y_LO - Y_PAD)
    panel_h = PANEL_W * (y_range / x_range)

    fig, axes = plt.subplots(
        1, len(frames),
        figsize=(PANEL_W * len(frames), PANEL_W),
        constrained_layout=False,
    )
    fig.patch.set_facecolor("none")
    for ax in axes:
        ax.set_facecolor("none")

    for ax, u in zip(axes, frames):
        ax.pcolormesh(x, y, u, cmap=CMAP, vmin=vmin, vmax=vmax, shading="auto")
        ax.axis("off")

    fig.subplots_adjust(wspace=ROW_GAP, left=0, right=1, top=1, bottom=0)

    os.makedirs(os.path.dirname(SAVE_PATH) or ".", exist_ok=True)
    fig.savefig(SAVE_PATH, dpi=DPI, bbox_inches="tight", pad_inches=0.02,
                transparent=True)
    plt.close(fig)
    print(f"Saved: {SAVE_PATH}")


if __name__ == "__main__":
    main()
