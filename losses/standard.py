import torch


class StandardLoss:
  

    def __init__(
        self,
        lambda_ic:  float = 1.0,
        lambda_bc:  float = 1.0,
        lambda_pde: float = 1.0,
    ):
        self.lambda_ic  = lambda_ic
        self.lambda_bc  = lambda_bc
        self.lambda_pde = lambda_pde

    def compute(
        self,
        equation,
        ic,
        bc,
        model,
        batch: dict
    ) -> dict:

        L_IC  = ic.loss(model,  batch["x_ic"],  batch["t_ic"])
        L_BC  = bc.loss(model,  batch["x_bc"],  batch["t_bc"])
        L_PDE = equation.pde_loss(model, batch["x_pde"], batch["t_pde"])

        total = (self.lambda_ic  * L_IC  + self.lambda_bc  * L_BC  + self.lambda_pde * L_PDE)

        return {
            "ic":    L_IC,
            "bc":    L_BC,
            "pde":   L_PDE,
            "total": total,
        }