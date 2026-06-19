"""
Classical finite-difference baseline for the 1D advection-diffusion equation.

PDE  : C_t = -v*C_x + D*C_xx
IC   : C(x,0) = sin(x)
BC   : Periodic — C(0,t) = C(2π,t),  C_x(0,t) = C_x(2π,t)
Exact: C(x,t)  = exp(-D*t) * sin(x - v*t)
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator
import time

# ---------------------------------------------------------------------------
# PDE parameters — must match the 1D ablation study
# ---------------------------------------------------------------------------
V = 1.0   # advection speed
D = 1.0   # diffusion coefficient
T = 1.0   # time horizon

# ---------------------------------------------------------------------------
# Solver settings
# ---------------------------------------------------------------------------
NX     = 400   # periodic grid points on [0, 2π);  dx = 2π/NX
RTOL   = 1e-6
ATOL   = 1e-8
N_EVAL = 200   # evaluation grid size — matches the PINN's 200×200 grid
SEEDS  = [1234, 2345, 3456, 4567, 5678]


def exact_fn(x, t):
    return np.exp(-D * t) * np.sin(x - V * t)


def solve_fd(seed=None):
    """
    Run one FD solve.  seed only seeds numpy (the deterministic FD solution
    is unaffected).  Returns (l2_rel, runtime_seconds).
    """
    if seed is not None:
        np.random.seed(seed)

    # Periodic spatial grid: NX equally-spaced points on [0, 2π)
    x  = np.linspace(0.0, 2.0 * np.pi, NX, endpoint=False)
    dx = x[1] - x[0]

    u0 = np.sin(x)  # IC

    def rhs(t_val, u):
        u_plus  = np.roll(u, -1)   # u_{i+1}  (periodic wrap)
        u_minus = np.roll(u,  1)   # u_{i-1}  (periodic wrap)
        adv  = -V * (u_plus - u_minus) / (2.0 * dx)
        diff =  D * (u_plus - 2.0 * u + u_minus) / dx**2
        return adv + diff

    # t_eval at 200 uniformly-spaced time levels in [0, T]
    t_eval = np.linspace(0.0, T, N_EVAL)

    t0  = time.perf_counter()
    sol = solve_ivp(
        rhs,
        t_span=(0.0, T),
        y0=u0,
        method="RK45",
        t_eval=t_eval,
        rtol=RTOL,
        atol=ATOL,
        dense_output=False,
    )
    runtime = time.perf_counter() - t0

    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")

    # sol.y has shape (NX, N_EVAL)

    # Evaluation grid: 200 x-points × 200 t-points, matching the PINN's grid
    x_ev = np.linspace(0.0, 2.0 * np.pi, N_EVAL)

    interp = RegularGridInterpolator(
        (x, t_eval), sol.y, method="linear",
        bounds_error=False, fill_value=None,
    )

    # meshgrid layout: (N_EVAL, N_EVAL) — rows vary in t, cols vary in x
    xg_e, tg_e = np.meshgrid(x_ev, t_eval)
    query      = np.column_stack([xg_e.ravel(), tg_e.ravel()])
    u_fd_ev    = interp(query).reshape(N_EVAL, N_EVAL)

    u_exact = exact_fn(xg_e, tg_e)

    err    = (u_fd_ev - u_exact).ravel()
    ref    = u_exact.ravel()
    l2_rel = float(np.linalg.norm(err) / np.linalg.norm(ref))

    return l2_rel, runtime


def main():
    print("Classical 1D FD baseline — Method of Lines + RK45")
    print(f"  PDE     : C_t = -{V}*C_x + {D}*C_xx")
    print(f"  IC/BC   : C(x,0) = sin(x),  periodic on [0, 2*pi]")
    print(f"  Domain  : [0, 2*pi] x [0, {T}]")
    print(f"  Grid    : NX = {NX},  dx = {2*np.pi/NX:.5f}")
    print(f"  Tols    : rtol={RTOL:.0e}  atol={ATOL:.0e}")
    print(f"  Seeds   : {SEEDS}")
    print()

    errors   = []
    runtimes = []

    for seed in SEEDS:
        l2_rel, rt = solve_fd(seed)
        errors.append(l2_rel)
        runtimes.append(rt)
        print(f"  seed {seed}: L2_rel = {l2_rel:.4e}   time = {rt:.3f}s")

    errors   = np.array(errors)
    runtimes = np.array(runtimes)

    print()
    print(f"Relative L2 error  : {errors.mean():.4e} ± {errors.std():.4e}")
    print(f"Median runtime      : {np.median(runtimes):.3f}s")


if __name__ == "__main__":
    main()
