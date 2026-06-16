import torch
import torch.nn as nn


class Swish(nn.Module):
    """
    Swish activation: x * sigmoid(x).
    Smooth, non-monotonic. Works well in PINNs because gradients don't
    vanish as sharply as tanh near saturation.

    NOTE (BUG FIXED): original comment said "spectral bias -> works for wave
    physics". That is WRONG. Swish does NOT address spectral bias.
    Spectral bias mitigation requires Fourier feature embeddings or SIREN
    (sine activations with specific weight initialisation — Sitzmann et al. 2020).
    Swish is simply a smooth nonlinearity with empirically good gradient flow.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class Sine(nn.Module):
    """
    Sine activation used in SIREN networks (Sitzmann et al. 2020).
    Addresses spectral bias by representing high-frequency components natively.
    Requires careful initialisation (see SirenMLP if you add that later).
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


ACTIVATIONS = {
    "tanh":  nn.Tanh,
    "relu":  nn.ReLU,
    "gelu":  nn.GELU,
    "swish": Swish,
    "sine":  Sine,
}