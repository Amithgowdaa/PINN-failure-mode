import torch
import torch.nn as nn
from models.activation import ACTIVATIONS


class MLP(nn.Module):
    

    def __init__(self, layers: list[int], activation: str = "tanh"):
        super().__init__()

        if layers[0] != 2:
            raise ValueError(
                f"layers[0] must be 2 for (x,t) input, got {layers[0]}"
            )

        act_cls = ACTIVATIONS[activation]

        modules = []
        for i in range(len(layers) - 2):
            modules.append(nn.Linear(layers[i], layers[i + 1]))
            modules.append(act_cls())

        # final linear — no activation
        modules.append(nn.Linear(layers[-2], layers[-1]))

        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
       
        xt = torch.cat([x, t], dim=-1)   # (N, 2)
        return self.net(xt)               # (N, 1)