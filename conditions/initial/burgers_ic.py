import torch

class BurgersIC:

    def loss(self, model, x_ic, t_ic):

        u_true = -torch.sin(torch.pi * x_ic)

        u_pred = model(x_ic, t_ic)

        return torch.mean(
            (u_pred - u_true) ** 2
        )
