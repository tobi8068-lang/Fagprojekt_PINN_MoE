import torch
from config import device, N_GRID

def create_grid():
    x = torch.linspace(0, 1, N_GRID, device=device)
    y = torch.linspace(0, 1, N_GRID, device=device)
    t = torch.linspace(0, 1, N_GRID, device=device)

    Xg, Yg, Tg = torch.meshgrid(x, y, t, indexing="ij")

    data = torch.stack([
        Xg.reshape(-1),
        Yg.reshape(-1),
        Tg.reshape(-1)
    ], dim=1)

    return Xg, Yg, Tg, data