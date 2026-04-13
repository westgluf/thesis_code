"""
Centralised configuration for the deep hedging under rough volatility project.
All hyperparameters, default model parameters, and utility functions live here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch

# ---------------------------------------------------------------------------
# Global seed
# ---------------------------------------------------------------------------
GLOBAL_SEED = 2024


def set_global_seed(seed: int = GLOBAL_SEED) -> None:
    """Set seeds for reproducibility across torch, CUDA, and cuDNN."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(dtype: torch.dtype = torch.float64) -> torch.device:
    """Return the best available device for *dtype*.

    MPS does not support float64, so we fall back to CPU when double
    precision is requested.
    """
    if dtype == torch.float64:
        # MPS cannot handle float64 — always use CPU for the simulator
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Model parameter dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RoughBergomiParams:
    """Default rBergomi parameters matching Bayer-Friz-Gatheral (2016)."""
    H: float = 0.07
    eta: float = 1.9
    rho: float = -0.7
    xi0: float = 0.235 ** 2   # ≈ 0.0553
    S0: float = 100.0
    T: float = 1.0
    n_steps: int = 100
    kappa: int = 1


@dataclass
class GBMParams:
    """Geometric Brownian Motion parameters."""
    sigma: float = 0.235
    mu: float = 0.0
    S0: float = 100.0
    T: float = 1.0
    n_steps: int = 100


@dataclass
class HestonParams:
    """Heston stochastic volatility parameters."""
    v0: float = 0.235 ** 2
    kappa: float = 1.0
    theta: float = 0.04
    sigma_v: float = 2.0
    rho: float = -0.7
    S0: float = 100.0
    T: float = 1.0
    n_steps: int = 100


# ---------------------------------------------------------------------------
# Experiment configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    """Train / validation / test split sizes."""
    n_train: int = 100_000
    n_val: int = 20_000
    n_test: int = 50_000


@dataclass
class HedgingConfig:
    """Hedging experiment parameters."""
    K: float = 100.0
    r: float = 0.0
    cost_lambda: float = 0.001
    delta_bounds: Tuple[float, float] = (0.0, 1.0)


@dataclass
class DeepHedgerConfig:
    """Deep hedging network and training parameters."""
    input_dim: int = 4          # (t_k, S_k, τ_k, δ_{k-1})
    hidden_dim: int = 128
    n_res_blocks: int = 2
    dropout: float = 0.0
    lr: float = 1e-3
    batch_size: int = 256
    epochs: int = 200
    patience: int = 20
    alpha: float = 0.95
    risk_measure: str = "es"
    entropic_lambda: float = 1.0


# ---------------------------------------------------------------------------
# H-sweep grid for Section 6.3
# ---------------------------------------------------------------------------

H_SWEEP_VALUES: List[float] = [0.01, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
