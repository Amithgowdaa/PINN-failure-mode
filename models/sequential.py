import torch.nn as nn
from models.activation import ACTIVATIONS

class MLP(nn.Module):

    def __init__(
        self,
        layers,
        activation="tanh"
    ):
        super().__init__()

        act = ACTIVATIONS[activation]

        modules = []

        for i in range(len(layers)-2):

            modules.append(
                nn.Linear(layers[i], layers[i+1])
            )

            modules.append(
                act()
            )

        modules.append(
            nn.Linear(layers[-2], layers[-1])
        )

        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)