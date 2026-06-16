import torch


def burgers_residual(
    model,
    x: torch.Tensor,
    t: torch.Tensor,
    nu: float
) -> torch.Tensor:
    """
    Computes the PDE residual:  u_t + u*u_x - nu*u_xx

    BUG FIXED: original called x.requires_grad_(True) which mutates the
    tensor IN-PLACE. When the same batch tensor is reused across equations
    or loss terms, this corrupts the computation graph silently.
    Use .clone().requires_grad_(True) instead.
    """
    x = x.clone().requires_grad_(True)   # (N, 1)
    t = t.clone().requires_grad_(True)   # (N, 1)

    u = model(x, t)                      # (N, 1)

    u_t = torch.autograd.grad(
        u, t,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

    u_x = torch.autograd.grad(
        u, x,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

    u_xx = torch.autograd.grad(
        u_x, x,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True
    )[0]

    return u_t + u * u_x - nu * u_xx


class BurgersEquation:
    """
    BUG FIXED: original file only defined a bare function burgers_residual().
    StandardLoss calls equation.pde_loss(...) — a method on a class.
    Wrapping here makes it compatible with StandardLoss without touching that file.

    Args:
        nu : viscosity coefficient. Raissi et al. 2019 uses nu = 0.01/pi
    """

    def __init__(self, nu: float = 0.01 / 3.141592653589793):
        self.nu = nu

    def pde_loss(
        self,
        model,
        x: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        residual = burgers_residual(model, x, t, self.nu)
        return torch.mean(residual ** 2)

    def pde_residual_raw(
        self,
        model,
        x: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """Returns raw residual (not squared, not meaned) — used by plot.py heatmap."""
        return burgers_residual(model, x, t, self.nu)