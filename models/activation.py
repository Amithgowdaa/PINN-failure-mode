import torch
import torch.nn as nn


class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class Sine(nn.Module):
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


ACTIVATIONS = {
    "tanh":  nn.Tanh,
    "relu":  nn.ReLU,
    "gelu":  nn.GELU,
    "swish": Swish,
    "sine":  Sine,
}