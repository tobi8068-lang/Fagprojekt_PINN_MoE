import math
import time
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class FeatureMap(nn.Module):
    """
    Random Fourier Features (Tancik et al. 2020).

    Draws a random frequency matrix B ~ N(0, σ²) at init (fixed thereafter).
    Maps d coordinates to R^{2m}, normalising each to [0, 1] via its (lo, hi) bounds.
    bounds : list of (lo, hi) per dimension, e.g. [(-10,10), (-10,10), (0,3)].

    The inner product γ(x)^T γ(x') ≈ exp(-‖x-x'‖²/2σ²) approximates a
    Gaussian kernel, giving the network a spectral bias towards smooth
    functions and helping it learn high-frequency components of the PDE solution.
    """

    def __init__(self, bounds, n_features=11, sigma=1.0, rff_seed=0):
        super().__init__()
        n_dims = len(bounds)
        # Fixed generator so B is independent of the training seed —
        # feature map quality should not be a source of run variance.
        rng = torch.Generator().manual_seed(rff_seed)
        B = torch.randn(n_dims, n_features, generator=rng) * sigma   # (d, m)
        self.register_buffer("B", B)
        lo = torch.tensor([b[0] for b in bounds], dtype=torch.float32)
        hi = torch.tensor([b[1] for b in bounds], dtype=torch.float32)
        self.register_buffer("lo", lo)
        self.register_buffer("hi", hi)

    @property
    def out_dim(self):
        return 2 * self.B.shape[1]

    def forward(self, *coords):
        # coords: d tensors each of shape (N, 1); normalise to [0, 1]
        x    = torch.cat([(c - l) / (h - l) for c, l, h in zip(coords, self.lo, self.hi)], dim=1)
        proj = 2.0 * math.pi * (x @ self.B)
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=1)   # (N, 2m)


class ScaledInputs(nn.Module):
    """Scales (x, y, t) linearly to [-1, 1]. Drop-in replacement for FeatureMap
    when RFF features are not wanted."""

    def __init__(self, bounds):
        super().__init__()
        lo = torch.tensor([b[0] for b in bounds], dtype=torch.float32)
        hi = torch.tensor([b[1] for b in bounds], dtype=torch.float32)
        self.register_buffer("lo", lo)
        self.register_buffer("hi", hi)

    @property
    def out_dim(self):
        return len(self.lo)

    def forward(self, *coords):
        return torch.cat([2*(c-l)/(h-l) - 1 for c,l,h in zip(coords, self.lo, self.hi)], dim=1)


FeatureMap2D = FeatureMap  # backward-compat alias


# ---------------------------------------------------------------------------
# Method 0a — Vanilla PINN (baseline, no gating)
# ---------------------------------------------------------------------------

class VanillaPINN(nn.Module):
    """
    Single-network PINN. Wraps a plain nn.Sequential so it shares the same
    (gates, expert_vals, u) interface as MoEModel.

    Returns trivial gates (ones) so the rest of the training code is unchanged;
    the load-balancing loss weight should be set to 0.0 in the config when
    using this model.
    """

    def __init__(self, feature_map: FeatureMap2D, network: nn.Module):
        super().__init__()
        self.feature_map = feature_map
        self.network = network

    def forward(self, *coords):
        feats = self.feature_map(*coords)
        u     = self.network(feats)
        N     = u.shape[0]
        gates = torch.ones(N, 1, device=u.device, dtype=u.dtype)
        return gates, u.unsqueeze(1), u


# ---------------------------------------------------------------------------
# Method 0b — MoE PINN (configurable gating)
# ---------------------------------------------------------------------------

class MoEModel(nn.Module):
    """
    Mixture-of-Experts PINN with continuous softmax gating.
    Each expert contributes a weighted fraction to every collocation point.
    Controlled by config key: use_moe=True
    """

    def __init__(self, feature_map: FeatureMap2D, gating_net: nn.Module, experts: list):
        super().__init__()
        self.feature_map = feature_map
        self.gating_net  = gating_net
        self.experts     = nn.ModuleList(experts)

    def forward(self, *coords):
        feats       = self.feature_map(*coords)
        logits      = self.gating_net(feats)
        gates       = torch.softmax(logits, dim=1)
        expert_outs = torch.stack(
            [e(feats).squeeze(-1) for e in self.experts], dim=1
        )
        u = torch.sum(gates * expert_outs, dim=1, keepdim=True)
        return gates, expert_outs, u


# ---------------------------------------------------------------------------
# PDE residual and loss computation
# ---------------------------------------------------------------------------

def pde_residual(model, x, y, t, domain_params, use_fd=True, h=1e-2, pde_fn=None):
    """
    PDE residual — supports u_t, u_y, u_yy, u_x, u_xx for any 2D+time PDE.

    pde_fn(u_t, u_y, u_yy, u_x, u_xx, domain_params) -> residual tensor.
    Defaults to 1-D advection-diffusion if not provided (expects domain["c"], domain["v"]).
    use_fd=True  : 8 forward passes, no create_graph.
    use_fd=False : exact derivatives via autograd.
    """
    if pde_fn is None:
        pde_fn = lambda ut, uy, uyy, ux, uxx, d: ut + d["c"] * uy - d["v"] * uyy

    zeros_t = torch.zeros_like(t)

    if use_fd:
        gates, expert_vals, u   = model(x,     y,     t      )
        _, _,            u_xp   = model(x + h, y,     t      )
        _, _,            u_xm   = model(x - h, y,     t      )
        _, _,            u_yp   = model(x,     y + h, t      )
        _, _,            u_ym   = model(x,     y - h, t      )
        _, _,            u_tp   = model(x,     y,     t + h  )
        _, _,            u_tm   = model(x,     y,     t - h  )
        _, _,        ini_pred   = model(x,     y,     zeros_t)

        u_x  = (u_xp - u_xm) / (2.0 * h)
        u_xx = (u_xp - 2.0 * u + u_xm) / (h * h)
        u_y  = (u_yp - u_ym) / (2.0 * h)
        u_yy = (u_yp - 2.0 * u + u_ym) / (h * h)
        u_t  = (u_tp - u_tm) / (2.0 * h)
    else:
        x = x.clone().detach().requires_grad_(True)
        y = y.clone().detach().requires_grad_(True)
        t = t.clone().detach().requires_grad_(True)

        gates, expert_vals, u = model(x, y, t)

        u_t  = torch.autograd.grad(u,   t, grad_outputs=torch.ones_like(u),   create_graph=True)[0]
        u_x  = torch.autograd.grad(u,   x, grad_outputs=torch.ones_like(u),   create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_y  = torch.autograd.grad(u,   y, grad_outputs=torch.ones_like(u),   create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]

        _, _, ini_pred = model(x, y, zeros_t)

    r = pde_fn(u_t, u_y, u_yy, u_x, u_xx, domain_params)

    return u, u_y, u_t, u_yy, u_x, u_xx, r, gates, expert_vals, ini_pred


def compute_losses(model, x_batch, y_batch, t_batch, domain_params, K,
                   ic_fn=None, use_fd=True, pde_fn=None, bc_fn=None):
    """
    Returns unweighted individual losses and gates.

    ic_fn  : u(x,y,0) callable (torch tensors) — from DOMAIN["ic_fn"].
    pde_fn : residual callable — from DOMAIN["pde_fn"].
    bc_fn  : optional boundary loss callable bc_fn(model, device, domain_params).
    K      : load-balancing scale (0 for vanilla).
    """
    if ic_fn is None:
        ic_fn = lambda x, y: torch.sin(y)

    _, _, _, _, _, _, r, gates, _, ini_pred = pde_residual(
        model, x_batch, y_batch, t_batch, domain_params, use_fd=use_fd, pde_fn=pde_fn,
    )

    loss_pde  = torch.mean(r ** 2)
    loss_ini  = torch.mean((ic_fn(x_batch, y_batch) - ini_pred) ** 2)
    gbar      = torch.mean(gates, dim=0)
    loss_load = K * torch.sum(gbar ** 2)

    losses = {"pde": loss_pde, "ini": loss_ini, "load": loss_load}

    if bc_fn is not None:
        losses["bc"] = bc_fn(model, x_batch.device, domain_params)

    return losses, gates


# ---------------------------------------------------------------------------
# Method 1 — SoftAdapt (adaptive loss weighting)
# ---------------------------------------------------------------------------

class SoftAdapt:
    """
    Dynamically reweights loss terms so that slower-improving terms receive
    larger weights.

    Controlled by config key: use_softadapt
    """

    def __init__(self, loss_names, base_weights=None, beta=5.0, eps=1e-8, enabled=True):
        self.loss_names  = loss_names
        self.beta        = beta
        self.eps         = eps
        self.enabled     = enabled
        self.prev_losses = None

        if base_weights is None:
            base_weights = {name: 1.0 for name in loss_names}
        self.base_weights    = base_weights
        self.current_weights = {name: float(base_weights[name]) for name in loss_names}

    def __call__(self, losses, update=True):
        def _base_tensors():
            return {
                name: torch.tensor(
                    self.base_weights[name],
                    device=losses[name].device,
                    dtype=losses[name].dtype,
                )
                for name in self.loss_names
            }

        if not self.enabled:
            return _base_tensors()

        if self.prev_losses is None:
            self.prev_losses = {name: losses[name].detach().clone() for name in self.loss_names}
            return _base_tensors()

        if not update:
            return {
                name: torch.tensor(
                    self.current_weights[name],
                    device=losses[name].device,
                    dtype=losses[name].dtype,
                )
                for name in self.loss_names
            }

        ratios = torch.stack([
            losses[name].detach() / (self.prev_losses[name] + self.eps)
            for name in self.loss_names
        ])
        ratios -= torch.max(ratios)
        adaptive_factors = len(self.loss_names) * torch.softmax(self.beta * ratios, dim=0)

        weights = {
            name: self.base_weights[name] * adaptive_factors[i]
            for i, name in enumerate(self.loss_names)
        }
        self.current_weights = {name: float(weights[name].detach().cpu()) for name in self.loss_names}
        self.prev_losses     = {name: losses[name].detach().clone() for name in self.loss_names}
        return weights


# ---------------------------------------------------------------------------
# Method 2 — Adaptive collocation refinement
# ---------------------------------------------------------------------------

@torch.no_grad()
def adaptive_refine(model, xyt_old, X_lo, X_hi, Y_lo, Y_hi, T, domain_params,
                    n_candidates=8000, replace_fraction=0.2, gamma=50.0,
                    use_fd=True, pde_fn=None):
    """
    Replaces a fraction of collocation points with high-residual candidates.

    Controlled by config key: use_adaptive_refine
    xyt_old : (N, 3) tensor of current collocation points [x, y, t].
    """
    dev       = xyt_old.device
    n_old     = xyt_old.shape[0]
    n_replace = max(1, min(int(replace_fraction * n_old), n_candidates))
    n_keep    = n_old - n_replace

    x_cand = X_lo + (X_hi - X_lo) * torch.rand(n_candidates, 1, device=dev)
    y_cand = Y_lo + (Y_hi - Y_lo) * torch.rand(n_candidates, 1, device=dev)
    t_cand = T    *                  torch.rand(n_candidates, 1, device=dev)

    if use_fd:
        _, _, _, _, _, _, r, _, _, _ = pde_residual(
            model, x_cand, y_cand, t_cand, domain_params, use_fd=True,  pde_fn=pde_fn)
    else:
        with torch.enable_grad():
            _, _, _, _, _, _, r, _, _, _ = pde_residual(
                model, x_cand, y_cand, t_cand, domain_params, use_fd=False, pde_fn=pde_fn)
    r2 = r.detach().squeeze() ** 2

    med = torch.median(r2)
    r2  = torch.clamp(r2, min=med.item(), max=(gamma * med).item()) if med.item() > 0 else r2 + 1e-12

    probs    = r2 / r2.sum()
    idx_new  = torch.multinomial(probs, num_samples=n_replace, replacement=False)
    xyt_hard = torch.cat([x_cand[idx_new], y_cand[idx_new], t_cand[idx_new]], dim=1).detach()

    perm     = torch.randperm(n_old, device=dev)
    xyt_keep = xyt_old[perm[:n_keep]].detach()

    xyt_new = torch.cat([xyt_keep, xyt_hard], dim=0)
    return xyt_new[torch.randperm(xyt_new.shape[0], device=dev)]


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def train(model, all_params, domain_params, config, eval_fn=None):
    """
    Train a PINN (vanilla or MoE) with any combination of the four methods.

    Parameters
    ----------
    model : VanillaPINN | MoEModel | callable
        Any callable with signature model(y, t) -> (gates, expert_vals, u).
        Use VanillaPINN or MoEModel for standard setups.
    all_params : list[nn.Parameter]
        All trainable parameters (passed to optimizer).
    domain_params : dict
        Y, T   -- domain extents
        c, v   -- advection speed and diffusion coefficient
        N_f    -- number of collocation points
        K      -- load-balancing scale factor (set to 0 for VanillaPINN)
    config : dict
        # Method toggles
        use_moe             : bool        -- True -> MoEModel, False -> VanillaPINN
        moe_gating          : str         -- "continuous" | "binary"  (only when use_moe=True)
        use_softadapt       : bool        -- Method 1
        use_adaptive_refine : bool        -- Method 2
        use_lbfgs           : bool        -- Method 3

        # Loss weights
        base_weights        : dict        -- {"pde": w1, "ini": w2, "load": w3}
                                             load weight is forced to 0 when use_moe=False

        # SoftAdapt params
        softadapt_beta      : float

        # Adam params
        adam_epochs         : int
        adam_lr             : float
        adam_step_size      : int
        adam_gamma          : float

        # Adaptive refinement params
        refine_every        : int
        n_candidates        : int
        replace_fraction    : float
        refine_gamma        : float

        # L-BFGS params
        lbfgs_max_iter      : int

        # Misc
        log_every           : int
        ic_fn               : callable    -- default torch.sin

    Returns
    -------
    hist : dict
        Per-epoch lists keyed by loss component name + "total".
    """
    T     = domain_params["T"]
    X_lo  = domain_params.get("X_lo", 0.0)
    X_hi  = domain_params["X_hi"]
    Y_lo  = domain_params.get("Y_lo", 0.0)
    Y_hi  = domain_params["Y_hi"]
    N_f   = domain_params["N_f"]
    K     = domain_params["K"]

    ic_fn        = domain_params.get("ic_fn",  None)
    pde_fn       = domain_params.get("pde_fn", None)
    bc_fn        = domain_params.get("bc_fn",  None)
    base_weights = dict(config.get("base_weights", {"pde": 1.0, "ini": 10.0, "load": 1e-2}))

    # No load-balancing concept for a vanilla single network
    if not config.get("use_moe", True):
        base_weights["load"] = 0.0
        K = 0.0

    # Add BC loss term if boundary conditions are defined
    if bc_fn is not None:
        base_weights["bc"] = domain_params.get("bc_weight", 10.0)

    loss_names = list(base_weights.keys())
    dev        = all_params[0].device

    # Collocation points — sampled uniformly over each dimension's [lo, hi]
    x_f = X_lo + (X_hi - X_lo) * torch.rand(N_f, 1, device=dev)
    y_f = Y_lo + (Y_hi - Y_lo) * torch.rand(N_f, 1, device=dev)
    t_f = T    *                  torch.rand(N_f, 1, device=dev)

    # Method 1 — SoftAdapt
    softadapt = SoftAdapt(
        loss_names   = loss_names,
        base_weights = base_weights,
        beta         = config.get("softadapt_beta", 5.0),
        enabled      = config.get("use_softadapt", False),
    )

    # Adam + LR scheduler
    optimizer = torch.optim.Adam(all_params, lr=config.get("adam_lr", 1e-3))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size = config.get("adam_step_size", 2500),
        gamma     = config.get("adam_gamma", 0.5),
    )

    hist = {name: [] for name in loss_names}
    hist["total"]         = []
    hist["wall_time"]     = []   # cumulative seconds at end of each epoch
    hist["eval_epochs"]   = []   # epochs at which eval_fn was called
    hist["eval_l2_rel"]   = []
    hist["eval_max_err"]  = []

    t_start      = time.time()
    eval_every   = config.get("eval_every", 0)
    adam_epochs  = config.get("adam_epochs", 10000)
    refine_every = config.get("refine_every", 400)

    # Record error at epoch 0 (before any training) so wall-clock curves start at t=0
    if eval_fn is not None and eval_every > 0:
        metrics = eval_fn(model, 0)
        hist["eval_epochs"].append(0)
        hist["eval_l2_rel"].append(metrics.get("l2_rel", float("nan")))
        hist["eval_max_err"].append(metrics.get("max_err", float("nan")))
    log_every    = config.get("log_every", 500)
    use_fd       = config.get("use_fd_deriv", True)

    # ---- Adam phase --------------------------------------------------------
    for epoch in range(1, adam_epochs + 1):
        losses, _ = compute_losses(
            model, x_f, y_f, t_f, domain_params, K, ic_fn,
            use_fd=use_fd, pde_fn=pde_fn, bc_fn=bc_fn,
        )
        weights   = softadapt(losses)

        loss_total = sum(weights[name] * losses[name] for name in loss_names)

        optimizer.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        optimizer.step()
        scheduler.step()

        hist["total"].append(loss_total.item())
        for name in loss_names:
            hist[name].append(losses[name].item())
        hist["wall_time"].append(time.time() - t_start)

        if eval_fn is not None and eval_every > 0 and epoch % eval_every == 0:
            metrics = eval_fn(model, epoch)
            hist["eval_epochs"].append(epoch)
            hist["eval_l2_rel"].append(metrics.get("l2_rel", float("nan")))
            hist["eval_max_err"].append(metrics.get("max_err", float("nan")))

        # Method 2 — adaptive collocation refinement
        if config.get("use_adaptive_refine", False) and epoch % refine_every == 0:
            xyt_f = adaptive_refine(
                model            = model,
                xyt_old          = torch.cat([x_f, y_f, t_f], dim=1),
                X_lo             = X_lo,
                X_hi             = X_hi,
                Y_lo             = Y_lo,
                Y_hi             = Y_hi,
                T                = T,
                domain_params    = domain_params,
                n_candidates     = config.get("n_candidates", 8000),
                replace_fraction = config.get("replace_fraction", 0.2),
                gamma            = config.get("refine_gamma", 50.0),
                use_fd           = use_fd,
                pde_fn           = pde_fn,
            )
            x_f = xyt_f[:, 0:1]
            y_f = xyt_f[:, 1:2]
            t_f = xyt_f[:, 2:3]

        if epoch % log_every == 0:
            parts = " | ".join(f"{n}={losses[n].item():.3e}" for n in loss_names)
            print(f"Adam {epoch:5d} | total={loss_total.item():.3e} | {parts}")

    # ---- Method 3 — L-BFGS phase ------------------------------------------
    if config.get("use_lbfgs", False):
        lbfgs_max_iter = config.get("lbfgs_max_iter", 600)
        lbfgs = torch.optim.LBFGS(
            all_params,
            lr             = 1.0,
            max_iter       = lbfgs_max_iter,
            max_eval       = lbfgs_max_iter,
            history_size   = 50,
            line_search_fn = "strong_wolfe",
        )

        def closure():
            lbfgs.zero_grad()
            losses_c, _ = compute_losses(
                model, x_f, y_f, t_f, domain_params, K, ic_fn,
                use_fd=use_fd, pde_fn=pde_fn, bc_fn=bc_fn,
            )
            weights_c   = softadapt(losses_c, update=False)
            loss_c      = sum(weights_c[name] * losses_c[name] for name in loss_names)
            loss_c.backward()
            return loss_c

        print("\nStarting L-BFGS...")
        final_loss = lbfgs.step(closure)
        print(f"L-BFGS final loss = {final_loss.item():.6e}")

    return hist


# ---------------------------------------------------------------------------
# Finite-difference reference solver
# ---------------------------------------------------------------------------

def numerical_solve(domain_params, N_y=512, N_t_plot=1000):
    """
    Explicit finite-difference solver for u_t + c*u_y = v*u_yy on periodic [0,Y].
    IC: u(y,0) = sin(y).   Exact: u(y,t) = exp(-v*t)*sin(y - c*t).

    Spatial discretisation:
      - advection : first-order upwind  (backward diff for c > 0)
      - diffusion : second-order central differences
    Time integration: explicit Euler with dt chosen for stability:
      dt = 0.4 * min( dy/|c|,  dy²/(2v) )

    Parameters
    ----------
    domain_params : dict   Y, T, c, v
    N_y           : int    spatial grid points
    N_t_plot      : int    number of time snapshots to record error history at

    Returns
    -------
    dict with keys:
        u_final, u_exact_final, y_grid, t_grid,
        l2_rel_history, max_err_history,
        l2_rel_final, max_err_final,
        solve_time_sec, N_y, N_t_plot
    """
    Y_lo = domain_params.get("Y_lo", 0.0)
    Y_hi = domain_params["Y_hi"]
    Y    = Y_hi - Y_lo                # spatial extent of the periodic 1-D domain
    T    = domain_params["T"]
    c    = domain_params.get("c", 1.0)   # advection speed along y (1-D solver only)
    v    = domain_params.get("v", 1.0)   # diffusion coefficient along y

    dy = Y / N_y
    dt = 0.4 * min(
        dy / abs(c)        if c != 0 else float("inf"),
        dy**2 / (2.0 * v)  if v != 0 else float("inf"),
    )
    N_t = max(1, int(np.ceil(T / dt)))
    dt  = T / N_t   # adjust so the last step lands exactly on T

    r_adv  = c * dt / dy       # upwind advection coefficient
    r_diff = v * dt / dy**2    # diffusion coefficient

    ic_fn_np = domain_params.get("ic_fn_np", lambda x_, y_: np.sin(y_))
    exact_fn = domain_params.get("exact_fn", None)

    y = np.linspace(0.0, Y, N_y, endpoint=False)
    u = ic_fn_np(np.zeros_like(y), y).copy()

    def analytic(t_val):
        if exact_fn is None:
            return None
        return exact_fn(np.zeros_like(y), y, t_val, domain_params)

    def errors(u_num, t_val):
        ex = analytic(t_val)
        if ex is None:
            return (float("nan"), float("nan"))
        err = u_num - ex
        return (float(np.linalg.norm(err) / np.linalg.norm(ex)),
                float(np.max(np.abs(err))))

    # Map each of the N_t_plot+1 output times to the nearest time step
    t_out       = np.linspace(0.0, T, N_t_plot + 1)
    record_step = np.clip(np.round(t_out / dt).astype(int), 0, N_t)

    step_to_out = defaultdict(list)
    for out_idx, step in enumerate(record_step):
        step_to_out[int(step)].append(out_idx)

    l2_history  = np.empty(N_t_plot + 1)
    max_history = np.empty(N_t_plot + 1)

    u_m = np.empty(N_y)  # u[j-1], pre-allocated
    u_p = np.empty(N_y)  # u[j+1], pre-allocated

    t_wall = time.time()

    for step in range(N_t + 1):
        if step in step_to_out:
            t_val = step * dt
            l2, mx = errors(u, t_val)
            for out_idx in step_to_out[step]:
                l2_history[out_idx]  = l2
                max_history[out_idx] = mx

        if step < N_t:
            u_m[1:]  = u[:-1];  u_m[0]  = u[-1]   # periodic j-1
            u_p[:-1] = u[1:];   u_p[-1] = u[0]    # periodic j+1
            # upwind advection (c > 0: backward diff) + central diffusion
            u = u - r_adv * (u - u_m) + r_diff * (u_p - 2.0 * u + u_m)

    solve_time = time.time() - t_wall

    return {
        "u_final":          u.copy(),
        "u_exact_final":    analytic(T),
        "y_grid":           y,
        "t_grid":           t_out,
        "l2_rel_history":   l2_history,
        "max_err_history":  max_history,
        "l2_rel_final":     float(l2_history[-1]),
        "max_err_final":    float(max_history[-1]),
        "solve_time_sec":   solve_time,
        "N_y":              N_y,
        "N_t_plot":         N_t_plot,
    }
