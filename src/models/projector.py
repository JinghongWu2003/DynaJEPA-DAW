from typing import List

import torch.nn as nn


def _build_mlp(input_dim: int, hidden_dims: List[int], output_dim: int) -> nn.Sequential:
    layers = []
    dims = [input_dim] + hidden_dims + [output_dim]
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class MLPProjector(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int] = None, output_dim: int = 256):
        super().__init__()
        hidden_dims = hidden_dims or [1024]
        self.net = _build_mlp(input_dim, hidden_dims, output_dim)

    def forward(self, x):
        return self.net(x)
