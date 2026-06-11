class StandardLoss:

    def compute(
        self,
        equation,
        ic,
        bc,
        model,
        batch
    ):

        L_IC = ic.loss(
            model,
            batch["x_ic"],
            batch["t_ic"]
        )

        L_BC = bc.loss(
            model,
            batch["x_bc"],
            batch["t_bc"]
        )

        L_PDE = equation.pde_loss(
            model,
            batch["x_pde"],
            batch["t_pde"]
        )

        return {
            "ic": L_IC,
            "bc": L_BC,
            "pde": L_PDE,
            "total": L_IC + L_BC + L_PDE
        }