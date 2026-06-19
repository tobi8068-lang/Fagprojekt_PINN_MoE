import torch
import torch.nn as nn
import config
from physics import (
    compute_pde_loss,
    compute_boundary_loss,
    compute_initial_loss
)

def train(model, data, conditions, RHS):
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

    # --- Warmup: enforce IC before introducing PDE loss ---
    print("Warming up on initial condition...")
    for epoch in range(1000):
        loss_ic = compute_initial_loss(model, conditions, loss_fn)
        optimizer.zero_grad()
        loss_ic.backward()
        optimizer.step()
        if epoch % 200 == 0:
            print(f"  warmup {epoch:4d} | ic={loss_ic.item():.6f}")

    for epoch in range(config.EPOCHS):

        data_epoch = data.clone().detach().requires_grad_(True)
        RHS_epoch = RHS.clone().detach()

        loss_pde = compute_pde_loss(model, data_epoch, RHS_epoch, loss_fn)
        loss_bc = compute_boundary_loss(model, conditions, loss_fn)
        loss_ic = compute_initial_loss(model, conditions, loss_fn)

        loss_total = loss_pde + config.W_BC*loss_bc + config.W_IC*loss_ic

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(
                f"epoch {epoch:4d} | "
                f"total={loss_total.item():.6f} | "
                f"pde={loss_pde.item():.6f} | "
                f"bc={loss_bc.item():.6f} | "
                f"ic={loss_ic.item():.6f}"
            )

    return model