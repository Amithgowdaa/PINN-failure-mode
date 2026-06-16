import torch


class StandardLoss:
    """
    Unweighted sum:  L = L_IC + L_BC + L_PDE

    BUG FIXED: original file had no `import torch` — would crash at runtime
    when .item() or any torch op is called on the returned dict.

    SUGGESTION: weights (lambda_ic, lambda_bc, lambda_pde) are set to 1.0
    by default to match the vanilla baseline. This is the first thing you
    will vary in your failure-mode study (Wang et al. 2021 NTK weighting,
    RAR, causal weighting). Expose them here so callers can sweep them
    without subclassing.
    """

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