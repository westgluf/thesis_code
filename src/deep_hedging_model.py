from __future__ import annotations

import torch
import torch.nn as nn


class MLPHedge(nn.Module):
    def __init__(self, in_dim: int = 4, hidden: int = 64, depth: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(d, hidden))
            layers.append(nn.Tanh())
            d = hidden
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
