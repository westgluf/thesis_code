import torch
import torch.nn as nn

def cvar_loss_from_pl(pl: torch.Tensor, w: torch.Tensor, alpha: float = 0.95) -> torch.Tensor:
    loss = -pl
    a = float(alpha)
    return w + torch.relu(loss - w).mean() / (1.0 - a)

class CVaRObjective(nn.Module):
    def __init__(self, alpha: float = 0.95, w0: float = 0.0):
        super().__init__()
        self.alpha = float(alpha)
        self.w = nn.Parameter(torch.tensor(float(w0)))

    def forward(self, pl: torch.Tensor) -> torch.Tensor:
        return cvar_loss_from_pl(pl, self.w, alpha=self.alpha)
