import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

def animate_solution(model, device, N=100):
    model.eval()

    x = torch.linspace(0, 1, N, device=device)
    y = torch.linspace(0, 1, N, device=device)

    X, Y = torch.meshgrid(x, y, indexing="ij")

    times = np.linspace(0, 1, N)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    def update(frame):
        ax.clear()
        t_val = times[frame]

        T = torch.full_like(X, float(t_val))

        data = torch.stack([
            X.reshape(-1),
            Y.reshape(-1),
            T.reshape(-1)
        ], dim=1)

        with torch.no_grad():
            u_pred = model(data)

        u_pred = u_pred.reshape(N, N).cpu().numpy()
        X_plot = X.cpu().numpy()
        Y_plot = Y.cpu().numpy()

        ax.plot_surface(X_plot, Y_plot, u_pred, cmap='hot',vmin=0,vmax=1)
        ax.set_title(f"u(x,y,t), t = {t_val:.3f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("u")
        ax.set_zlim(0, 1)

    ani = FuncAnimation(fig, update, frames=len(times), interval=80)
    return HTML(ani.to_jshtml())