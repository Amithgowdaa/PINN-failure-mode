import torch

class DirichletZeroBC:

    def loss(self, model, x_bc, t_bc):

        u_pred = model(x_bc, t_bc)

        return torch.mean(
            u_pred ** 2
        )