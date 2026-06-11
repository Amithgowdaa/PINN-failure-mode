import torch
import torch.nn as nn
### Swish = x⋅σ(x)
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x) ## spectral bias -> works for wave physics

### Sine (x)
class Sine(nn.Module):
    def forward(self, x):
        return torch.sin(x)

ACTIVATIONS = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "swish": Swish,
    "sine": Sine,
}