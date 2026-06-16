import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path


# ---------------------------------------------------------------------------
# 1. Loss curves
# ---------------------------------------------------------------------------

def plot_loss_curves(history: dict, save_path: str = None):
    """
    Plots IC, BC, PDE, and Total loss on a log scale.
    Always plot these separately — total loss hiding a bad PDE component
    is the most common failure-mode diagnostic miss.
    """
    fig, ax = plt.subplots(figsize=(9, 4))

    for key, color in zip(
        ["total", "pde", "ic", "bc"],
        ["black", "crimson", "steelblue", "darkorange"]
    ):
        if key in history:
            ax.semilogy(history[key], label=key.upper(), color=color, linewidth=1.5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("PINN Training Loss — Burgers' Equation")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")

    return fig


def load_and_plot_loss_curves(history_path: str, save_path: str = None):
    with open(history_path) as f:
        history = json.load(f)
    return plot_loss_curves(history, save_path=save_path)


# ---------------------------------------------------------------------------
# 2. Residual heatmap  —  the PRIMARY failure-mode diagnostic
# ---------------------------------------------------------------------------

def plot_residual_heatmap(
    model,
    equation,
    nx: int = 256,
    nt: int = 100,
    device: str = "cpu",
    save_path: str = None,
):
    """
    Plots |residual(x,t)| over the full domain as a heatmap.

    This is non-negotiable for failure-mode analysis. A low total-loss
    training run can still have large localised residuals near the shock
    (t ≈ 1/pi for nu=0.01/pi). The heatmap reveals WHERE the PINN fails,
    which the scalar loss completely hides.
    """
    model.eval()

    x_vals = torch.linspace(-1, 1,  nx, device=device).view(-1, 1)
    t_vals = torch.linspace( 0, 1,  nt, device=device).view(-1, 1)

    # Build meshgrid  (nx * nt, 1) each
    X, T = torch.meshgrid(x_vals.squeeze(), t_vals.squeeze(), indexing="ij")
    x_flat = X.reshape(-1, 1)
    t_flat = T.reshape(-1, 1)

    with torch.no_grad():
        pass  # residual needs grad — use separate block below

    # Residual requires grad, so compute outside no_grad
    residual = equation.pde_residual_raw(model, x_flat, t_flat)
    R = residual.detach().abs().reshape(nx, nt).cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(
        R,
        extent=[0, 1, -1, 1],
        origin="lower",
        aspect="auto",
        cmap="hot",
    )
    fig.colorbar(im, ax=ax, label="|residual|")
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax.set_title("PDE Residual Heatmap — Burgers' Equation")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")

    return fig


# ---------------------------------------------------------------------------
# 3. Predicted u(x,t) vs Ground Truth
# ---------------------------------------------------------------------------

def plot_prediction_vs_truth(
    model,
    ground_truth: dict,
    device: str = "cpu",
    save_path: str = None,
):
    """
    Side-by-side: PINN prediction | Reference solution | Pointwise error.

    ground_truth must be the dict from load_ground_truth():
        keys: "x" (256,), "t" (100,), "u" (256, 100)
    """
    model.eval()

    x_np = ground_truth["x"]   # (256,)
    t_np = ground_truth["t"]   # (100,)
    u_ref = ground_truth["u"]  # (256, 100)

    X_np, T_np = np.meshgrid(x_np, t_np, indexing="ij")  # (256, 100) each

    x_flat = torch.FloatTensor(X_np.reshape(-1, 1)).to(device)
    t_flat = torch.FloatTensor(T_np.reshape(-1, 1)).to(device)

    with torch.no_grad():
        u_pred = model(x_flat, t_flat).cpu().numpy().reshape(256, 100)

    error = np.abs(u_pred - u_ref)

    fig = plt.figure(figsize=(15, 4))
    gs  = gridspec.GridSpec(1, 3, wspace=0.35)

    titles = ["PINN Prediction", "Reference Solution", "Absolute Error"]
    fields = [u_pred, u_ref, error]
    cmaps  = ["RdBu_r", "RdBu_r", "hot"]

    for i, (title, field, cmap) in enumerate(zip(titles, fields, cmaps)):
        ax = fig.add_subplot(gs[i])
        im = ax.imshow(
            field,
            extent=[t_np.min(), t_np.max(), x_np.min(), x_np.max()],
            origin="lower",
            aspect="auto",
            cmap=cmap,
        )
        fig.colorbar(im, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("t")
        ax.set_ylabel("x")

    fig.suptitle("Burgers' Equation — PINN vs Reference", fontsize=13)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")

    return fig