def create_conditions(Xg, Yg, Tg):
    import torch
    from config import device

    # --- boundaries ---
    venstre = torch.stack([
        Xg[0, :, :].reshape(-1),
        Yg[0, :, :].reshape(-1),
        Tg[0, :, :].reshape(-1)
    ], dim=1).to(device).requires_grad_(True)

    højre = torch.stack([
        Xg[-1, :, :].reshape(-1),
        Yg[-1, :, :].reshape(-1),
        Tg[-1, :, :].reshape(-1)
    ], dim=1).to(device).requires_grad_(True)

    nedre = torch.stack([
        Xg[:, 0, :].reshape(-1),
        Yg[:, 0, :].reshape(-1),
        Tg[:, 0, :].reshape(-1)
    ], dim=1).to(device).requires_grad_(True)

    øvre = torch.stack([
        Xg[:, -1, :].reshape(-1),
        Yg[:, -1, :].reshape(-1),
        Tg[:, -1, :].reshape(-1)
    ], dim=1).to(device).requires_grad_(True)

        
    # --- Neumann targets ---
    
    neumann_v = torch.zeros_like(venstre[:,0:1], device=device)
    neumann_h = torch.zeros_like(højre[:,0:1], device=device)
    neumann_n = torch.zeros_like(nedre[:,1:2], device=device)
    neumann_ø = torch.zeros_like(øvre[:,1:2], device=device)
    
    # --- initial condition ---
    pi = torch.pi

    initial = (torch.sin(pi*Xg[:,:,0]) * torch.sin(pi*Yg[:,:,0])).reshape(-1,1)

    initial_points = torch.stack([
        Xg[:,:,0].reshape(-1),
        Yg[:,:,0].reshape(-1),
        Tg[:,:,0].reshape(-1)
    ], dim=1).to(device).requires_grad_(True)

    return {
        "venstre": venstre,
        "højre": højre,
        "nedre": nedre,
        "øvre": øvre,
        "initial": initial.to(device),
        "initial_points": initial_points,
        "neumann_v": neumann_v,
        "neumann_h": neumann_h,
        "neumann_n": neumann_n,
        "neumann_ø": neumann_ø
    }

    