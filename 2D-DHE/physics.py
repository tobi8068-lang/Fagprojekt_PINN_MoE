import torch
from config import ALPHA

def compute_pde_loss(model, data, RHS, loss_fn):
    u = model(data)

    grads = torch.autograd.grad(u, data, torch.ones_like(u), create_graph=True)[0]

    u_t = grads[:, 2:3] / 1000
    u_x = grads[:, 0:1]
    u_y = grads[:, 1:2]

    grads_x = torch.autograd.grad(u_x, data, torch.ones_like(u_x), create_graph=True)[0]
    grads_y = torch.autograd.grad(u_y, data, torch.ones_like(u_y), create_graph=True)[0]

    u_xx = grads_x[:, 0:1]
    u_yy = grads_y[:, 1:2]

    RHS_pred = (1/ALPHA)*u_t -(u_xx + u_yy)

    return loss_fn(RHS_pred, RHS)

def compute_boundary_loss(model, conditions, loss_fn):
    venstre = conditions["venstre"]
    højre = conditions["højre"]
    nedre = conditions["nedre"]
    øvre = conditions["øvre"]

    neumann_v = conditions["neumann_v"]
    neumann_h = conditions["neumann_h"]
    neumann_n = conditions["neumann_n"]
    neumann_ø = conditions["neumann_ø"]

    # Predictions
    u_v = model(venstre)
    u_h = model(højre)
    u_n = model(nedre)
    u_ø = model(øvre)

    # Gradients
    grad_v = torch.autograd.grad(u_v, venstre, torch.ones_like(u_v), create_graph=True)[0]
    grad_h = torch.autograd.grad(u_h, højre, torch.ones_like(u_h), create_graph=True)[0]
    grad_n = torch.autograd.grad(u_n, nedre, torch.ones_like(u_n), create_graph=True)[0]
    grad_ø = torch.autograd.grad(u_ø, øvre, torch.ones_like(u_ø), create_graph=True)[0]

    u_x_v = grad_v[:, 0:1]
    u_x_h = grad_h[:, 0:1]
    u_y_n = grad_n[:, 1:2]
    u_y_ø = grad_ø[:, 1:2]

    loss = (
        loss_fn(u_x_v, neumann_v) +
        loss_fn(u_x_h, neumann_h) +
        loss_fn(u_y_n, neumann_n) +
        loss_fn(u_y_ø, neumann_ø)
    )

    return loss

#def compute_boundary_loss(model, conditions, loss_fn):
    venstre = conditions["venstre"]
    højre = conditions["højre"]
    nedre = conditions["nedre"]
    øvre = conditions["øvre"]

    u_v = model(venstre)
    u_h = model(højre)
    u_n = model(nedre)
    u_ø = model(øvre)

    zeros_v = torch.zeros_like(u_v)
    zeros_h = torch.zeros_like(u_h)
    zeros_n = torch.zeros_like(u_n)
    zeros_ø = torch.zeros_like(u_ø)

    loss_v = loss_fn(u_v, zeros_v)
    loss_h = loss_fn(u_h, zeros_h)
    loss_n = loss_fn(u_n, zeros_n)
    loss_ø = loss_fn(u_ø, zeros_ø)

    return loss_v + loss_h + loss_n + loss_ø

def compute_initial_loss(model, conditions, loss_fn):
    initial_points = conditions["initial_points"]
    initial = conditions["initial"]

    pred = model(initial_points)

    return loss_fn(pred, initial)