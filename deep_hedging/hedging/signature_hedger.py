"""
Signature-enriched deep hedger (treatment arm for H4).

Extends the ResidualFNN architecture from Prompt 3 to accept
path-dependent features computed by :class:`PathFeatureExtractor`.
The network architecture is unchanged; only the input dimension
varies according to the feature set.

Implements the same ``hedge_paths(S, T, S0)`` interface as
:class:`DeepHedgerFNN`, so it plugs into ``train_deep_hedger``
and ``evaluate_deep_hedger`` without modification.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from deep_hedging.hedging.deep_hedger import ResidualBlock
from deep_hedging.hedging.features import PathFeatureExtractor


class SignatureDeepHedger(nn.Module):
    """Deep hedger with path-dependent (signature) features.

    Parameters
    ----------
    feature_set : str
        Passed to :class:`PathFeatureExtractor`.
    hidden_dim : int
        Width of hidden layers.
    n_res_blocks : int
        Number of residual blocks.
    xi0, eta_ref, T : float
        Passed to :class:`PathFeatureExtractor`.
    """

    def __init__(
        self,
        feature_set: str = "sig-full",
        hidden_dim: int = 128,
        n_res_blocks: int = 2,
        xi0: float = 0.235 ** 2,
        eta_ref: float = 1.9,
        T: float = 1.0,
    ) -> None:
        super().__init__()
        self.feature_extractor = PathFeatureExtractor(
            feature_set=feature_set, xi0=xi0, eta_ref=eta_ref, T=T,
        )
        input_dim = self.feature_extractor.n_features
        self.feature_set = feature_set

        layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
        ]
        for _ in range(n_res_blocks):
            layers.append(ResidualBlock(hidden_dim))
        self.network = nn.Sequential(
            *layers,
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    # ─── Uniform interface ───────────────────────────────────

    def hedge_paths(
        self,
        S: Tensor,
        T: float = 1.0,
        S0: float = 100.0,
    ) -> Tensor:
        """Compute hedge ratios for full price paths.

        Precomputes path features once, then loops only to inject
        the recurrent delta_{k-1} at feature index 3.

        Parameters
        ----------
        S : (batch, n_steps + 1)
        T : maturity
        S0 : initial spot

        Returns
        -------
        deltas : (batch, n_steps)
        """
        batch, n_plus_1 = S.shape
        n = n_plus_1 - 1
        device = S.device

        # Precompute all features with zero deltas_prev
        zero_prev = torch.zeros(batch, n, dtype=S.dtype, device=device)
        all_feat = self.feature_extractor(S, zero_prev, S0=S0)  # (batch, n, F)
        all_feat = all_feat.float()  # network runs in float32

        delta_idx = PathFeatureExtractor.DELTA_PREV_INDEX
        deltas = torch.zeros(batch, n, dtype=torch.float32, device=device)

        for k in range(n):
            feat_k = all_feat[:, k, :].clone()
            if k > 0:
                feat_k[:, delta_idx] = deltas[:, k - 1].detach()
            delta_k = self.network(feat_k).squeeze(-1)
            deltas[:, k] = delta_k

        return deltas

    def forward(self, S: Tensor, T: float = 1.0, S0: float = 100.0) -> Tensor:
        return self.hedge_paths(S, T, S0)
