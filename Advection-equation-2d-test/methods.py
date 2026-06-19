import math
import time
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
    Maps d input coordinates to a 2m-dimensional feature vector via a fixed random
    frequency matrix. bounds: list of (lo, hi) per input dimension.
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



# ---------------------------------------------------------------------------
# Method 0a — Vanilla PINN (baseline, no gating)
# ---------------------------------------------------------------------------

class VanillaPINN(nn.Module):
    """
    Single-network PINN with the same (gates, expert_vals, u) interface as MoEModel.
    Returns trivial gates (all ones); set load-balancing weight to 0 in the config.
    """

    def __init__(self, feature_map: FeatureMap, network: nn.Module):
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

    def __init__(self, feature_map: FeatureMap, gating_net: nn.Module, experts: list):
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

    loss_pde = torch.mean(r ** 2)
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
    Adam + optional L-BFGS training loop with SoftAdapt, adaptive refinement,
    and periodic collocation resampling controlled by config toggles.
    Returns hist dict with per-epoch loss history, wall times, and eval snapshots.
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

    if bc_fn is not None:
        base_weights["bc"] = domain_params.get("bc_weight", 10.0)

    loss_names = list(base_weights.keys())
    dev        = all_params[0].device

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
    log_every      = config.get("log_every", 500)
    use_fd         = config.get("use_fd_deriv", True)
    use_ar         = config.get("use_adaptive_refine", False)
    # Skip resampling when AR is on — AR manages the collocation set.
    resample_every = config.get("resample_every", 0) if not use_ar else 0

    # ---- Adam phase --------------------------------------------------------
    for epoch in range(1, adam_epochs + 1):
        if resample_every and epoch % resample_every == 0:
            x_f = X_lo + (X_hi - X_lo) * torch.rand(N_f, 1, device=dev)
            y_f = Y_lo + (Y_hi - Y_lo) * torch.rand(N_f, 1, device=dev)
            t_f = T    *                  torch.rand(N_f, 1, device=dev)

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
        if use_ar and epoch % refine_every == 0:
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

        if eval_fn is not None and eval_every > 0:
            metrics = eval_fn(model, adam_epochs)
            hist["eval_epochs"].append(adam_epochs)
            hist["eval_l2_rel"].append(metrics.get("l2_rel", float("nan")))
            hist["eval_max_err"].append(metrics.get("max_err", float("nan")))
            hist["wall_time"].append(time.time() - t_start)

    return hist
