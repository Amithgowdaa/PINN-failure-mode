import torch


def burgers_residual(
    model,
    x: torch.Tensor,
    t: torch.Tensor,
    nu: float
) -> torch.Tensor:
   
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