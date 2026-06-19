

import sys
import os
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configs import DOMAIN, SEEDS

# ---------------------------------------------------------------------------
# Solver settings
# ---------------------------------------------------------------------------
NX   = 100   # grid points in x  (domain width  9,  dx ≈ 0.091)
NY   = 150   # grid points in y  (domain height 14, dy ≈ 0.094)
RTOL = 1e-6
ATOL = 1e-8
N_EVAL = 200  # evaluation grid size — matches train.py evaluate(n_grid=200)


def solve_fd(seed=None):
    """
    Run one FD solve.  seed is passed to numpy.random.seed for reproducibility
    only; it does not affect the deterministic FD solution.

    Returns (l2_rel, runtime_seconds).
    """
    if seed is not None:
        np.random.seed(seed)

    X_lo, X_hi = DOMAIN["X_lo"], DOMAIN["X_hi"]
    Y_lo, Y_hi = DOMAIN["Y_lo"], DOMAIN["Y_hi"]
    T        = DOMAIN["T"]
    Dl       = DOMAIN["Dl"]
    Dt       = DOMAIN["Dt"]
    v        = DOMAIN["v"]
    ic_fn_np = DOMAIN["ic_fn_np"]
    exact_fn = DOMAIN["exact_fn"]

    # Spatial grid: U[i, j] is the value at (y[i], x[j])
    x  = np.linspace(X_lo, X_hi, NX)
    y  = np.linspace(Y_lo, Y_hi, NY)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    xg, yg = np.meshgrid(x, y)   # (NY, NX), default 'xy' indexing

    # Evolve interior nodes only; boundaries are held at the exact solution
    u0_int = ic_fn_np(xg, yg)[1:-1, 1:-1].ravel()   # shape: (NY-2)*(NX-2)

    def rhs(t_val, u_flat):
        U = np.empty((NY, NX))
        U[1:-1, 1:-1] = u_flat.reshape(NY - 2, NX - 2)

        # Time-dependent Dirichlet BCs from exact solution (matches PINN bc_fn)
        U[:,  0] = exact_fn(x[0],  y,     t_val, DOMAIN)   # left
        U[:, -1] = exact_fn(x[-1], y,     t_val, DOMAIN)   # right
        U[0,  :] = exact_fn(x,     y[0],  t_val, DOMAIN)   # bottom
        U[-1, :] = exact_fn(x,     y[-1], t_val, DOMAIN)   # top

        # Second-order central differences on interior nodes → (NY-2) × (NX-2)
        adv_x  = v  * (U[1:-1, 2:] - U[1:-1, :-2]) / (2.0 * dx)
        diff_x = Dl * (U[1:-1, 2:] - 2.0 * U[1:-1, 1:-1] + U[1:-1, :-2]) / dx**2
        diff_y = Dt * (U[2:, 1:-1] - 2.0 * U[1:-1, 1:-1] + U[:-2, 1:-1]) / dy**2

        # u_t = -v·u_x + Dl·u_xx + Dt·u_yy
        return (-adv_x + diff_x + diff_y).ravel()

    t_snap = np.array([0.0, T / 2.0, T])

    t0  = time.perf_counter()
    sol = solve_ivp(
        rhs,
        t_span=(0.0, T),
        y0=u0_int,
        method="RK45",
        t_eval=t_snap,
        rtol=RTOL,
        atol=ATOL,
        dense_output=False,
    )
    runtime = time.perf_counter() - t0

    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")

    # Evaluation grid — same as PINN (200×200 over full domain)
    x_ev = np.linspace(X_lo, X_hi, N_EVAL)
    y_ev = np.linspace(Y_lo, Y_hi, N_EVAL)
    xg_e, yg_e = np.meshgrid(x_ev, y_ev)   # (N_EVAL, N_EVAL)
    query = np.column_stack([yg_e.ravel(), xg_e.ravel()])  # (y, x) pairs for interp

    err_list = []
    ref_list = []

    for k, t_val in enumerate(t_snap):
        # Reconstruct full grid (interior + BCs) at this time snapshot
        U = np.empty((NY, NX))
        U[1:-1, 1:-1] = sol.y[:, k].reshape(NY - 2, NX - 2)
        U[:,  0] = exact_fn(x[0],  y,     t_val, DOMAIN)
        U[:, -1] = exact_fn(x[-1], y,     t_val, DOMAIN)
        U[0,  :] = exact_fn(x,     y[0],  t_val, DOMAIN)
        U[-1, :] = exact_fn(x,     y[-1], t_val, DOMAIN)

        # Bilinear interpolation onto the evaluation grid
        # (RegularGridInterpolator axes are ordered (y, x) to match U's layout)
        interp  = RegularGridInterpolator((y, x), U, method="linear",
                                          bounds_error=False, fill_value=None)
        u_fd_ev = interp(query).reshape(N_EVAL, N_EVAL)

        tg_e    = np.full_like(xg_e, t_val)
        u_exact = exact_fn(xg_e, yg_e, tg_e, DOMAIN)

        err_list.append((u_fd_ev - u_exact).ravel())
        ref_list.append(u_exact.ravel())

    err_all = np.concatenate(err_list)
    ref_all = np.concatenate(ref_list)
    l2_rel  = float(np.linalg.norm(err_all) / np.linalg.norm(ref_all))

    return l2_rel, runtime


def main():
    print("Classical FD baseline — Method of Lines + RK45")
    print(f"  Grid  : {NX} × {NY}   "
          f"dx = {(DOMAIN['X_hi'] - DOMAIN['X_lo']) / (NX - 1):.4f}   "
          f"dy = {(DOMAIN['Y_hi'] - DOMAIN['Y_lo']) / (NY - 1):.4f}")
    print(f"  Tols  : rtol={RTOL:.0e}  atol={ATOL:.0e}")
    print(f"  Seeds : {SEEDS}")
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
