#!/usr/bin/env python
"""
Stage 1, 1.5 and 3 of the H4 validation experiment.

  Stage 1   — single-point comparison of flat / sig-3 / sig-full at H=0.05
  Stage 1.5 — diagnostic investigation if Stage 1 gate fails
  Stage 3   — analysis and figures for the full H-sweep (loads JSON)

Run:
    python -u -m deep_hedging.experiments.signature_ablation --stage 1
    python -u -m deep_hedging.experiments.signature_ablation --stage 3
    python -u -m deep_hedging.experiments.signature_ablation --stage all
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch import Tensor

from deep_hedging.core.rough_bergomi import DifferentiableRoughBergomi
from deep_hedging.hedging.delta_hedger import BlackScholesDelta
from deep_hedging.hedging.deep_hedger import (
    ResidualBlock, train_deep_hedger, evaluate_deep_hedger,
)
from deep_hedging.hedging.features import PathFeatureExtractor
from deep_hedging.hedging.signature_hedger import SignatureDeepHedger
from deep_hedging.objectives.pnl import compute_payoff, compute_hedging_pnl
from deep_hedging.objectives.risk_measures import (
    expected_shortfall, compute_all_metrics,
)

FIGURE_DIR = Path(__file__).resolve().parents[2] / "figures"
EXP_DIR = FIGURE_DIR  # alias

# ─── Colours ───────────────────────────────────────────────
C_BS = "#2196F3"
C_FLAT = "#FF9800"
C_SIG3 = "#9C27B0"
C_SIGFULL = "#4CAF50"
C_FIT = "#F44336"

# ─── Constants ─────────────────────────────────────────────
SIGMA = 0.235
XI0 = SIGMA ** 2
S0 = 100.0
K = 100.0
T = 1.0
N_STEPS = 100
ETA = 1.9
RHO = -0.7

# Common training config (shared across ALL deep hedger runs in Prompt 8)
TRAIN_CFG = dict(
    lr=1e-3,
    batch_size=2048,
    patience=30,
    alpha=0.95,
    verbose=False,
)


# =======================================================================
# Helpers
# =======================================================================

def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _simulate(
    H: float, n_paths: int, seed: int = 2024,
    n_steps: int = N_STEPS,
) -> tuple[Tensor, Tensor]:
    sim = DifferentiableRoughBergomi(
        n_steps=n_steps, T=T, H=H, eta=ETA, rho=RHO, xi0=XI0,
    )
    S, V, _ = sim.simulate(n_paths=n_paths, S0=S0, seed=seed)
    return S, V


def _bs_delta_pnl(S: Tensor, p0: float, cost_lambda: float = 0.0) -> Tensor:
    bs = BlackScholesDelta(sigma=SIGMA, K=K, T=T)
    deltas = bs.hedge_paths(S)
    payoff = compute_payoff(S, K, "call")
    return compute_hedging_pnl(S, deltas, payoff, p0, cost_lambda)


def _train_signature_hedger(
    feature_set: str,
    S_tr: Tensor, S_va: Tensor, p0: float,
    epochs: int = 200,
    init_seed: int = 3024,
    cost_lambda: float = 0.0,
    hidden_dim: int = 128,
    n_res_blocks: int = 2,
) -> tuple[SignatureDeepHedger, dict]:
    """Build and train a SignatureDeepHedger with consistent config."""
    _set_seed(init_seed)
    hedger = SignatureDeepHedger(
        feature_set=feature_set,
        hidden_dim=hidden_dim,
        n_res_blocks=n_res_blocks,
        xi0=XI0, eta_ref=ETA, T=T,
    )
    history = train_deep_hedger(
        hedger, S_tr, S_va,
        K=K, T=T, S0=S0, p0=p0,
        cost_lambda=cost_lambda, epochs=epochs,
        **TRAIN_CFG,
    )
    return hedger, history


def _eval_hedger(
    hedger: nn.Module, S_te: Tensor, p0: float, cost_lambda: float = 0.0,
) -> tuple[Tensor, dict]:
    pnl = evaluate_deep_hedger(
        hedger, S_te, K=K, T=T, S0=S0, p0=p0, cost_lambda=cost_lambda,
    )
    return pnl, compute_all_metrics(pnl)


# =======================================================================
# Two-tower architecture (for Stage 1.5 diagnostic)
# =======================================================================

class TwoTowerHedger(nn.Module):
    """Two-tower hedger: separate encoders for flat and path features."""

    def __init__(
        self,
        n_flat: int = 4,
        n_path: int = 8,  # 12 - 4 for sig-full
        hidden_dim: int = 128,
        n_res_blocks: int = 2,
        xi0: float = XI0,
        eta_ref: float = ETA,
        T_: float = T,
    ) -> None:
        super().__init__()
        self.feature_extractor = PathFeatureExtractor(
            feature_set="sig-full", xi0=xi0, eta_ref=eta_ref, T=T_,
        )
        self.n_flat = n_flat
        self.n_path = n_path
        self.flat_tower = self._build_tower(n_flat, hidden_dim, n_res_blocks)
        self.path_tower = self._build_tower(n_path, hidden_dim, n_res_blocks)
        self.combiner = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _build_tower(in_dim: int, hidden_dim: int, n_blocks: int) -> nn.Sequential:
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.LeakyReLU(0.2)]
        for _ in range(n_blocks):
            layers.append(ResidualBlock(hidden_dim))
        return nn.Sequential(*layers)

    def hedge_paths(self, S: Tensor, T_: float = 1.0, S0_: float = 100.0) -> Tensor:
        batch, n_plus_1 = S.shape
        n = n_plus_1 - 1
        device = S.device
        zero_prev = torch.zeros(batch, n, dtype=S.dtype, device=device)
        all_feat = self.feature_extractor(S, zero_prev, S0=S0_).float()

        delta_idx = PathFeatureExtractor.DELTA_PREV_INDEX
        deltas = torch.zeros(batch, n, dtype=torch.float32, device=device)
        for k in range(n):
            feats_k = all_feat[:, k, :].clone()
            if k > 0:
                feats_k[:, delta_idx] = deltas[:, k - 1].detach()
            flat_k = feats_k[:, :4]
            path_k = feats_k[:, 4:]
            flat_emb = self.flat_tower(flat_k)
            path_emb = self.path_tower(path_k)
            combined = torch.cat([flat_emb, path_emb], dim=-1)
            deltas[:, k] = self.combiner(combined).squeeze(-1)
        return deltas

    def forward(self, S: Tensor, T_: float = 1.0, S0_: float = 100.0) -> Tensor:
        return self.hedge_paths(S, T_, S0_)


# =======================================================================
# Standardised-feature hedger (for Stage 1.5 diagnostic)
# =======================================================================

class StandardisedSignatureHedger(nn.Module):
    """SignatureDeepHedger wrapped with fixed per-feature standardisation."""

    def __init__(
        self,
        hidden_dim: int = 128, n_res_blocks: int = 2,
        xi0: float = XI0, eta_ref: float = ETA, T_: float = T,
    ) -> None:
        super().__init__()
        self.feature_extractor = PathFeatureExtractor(
            feature_set="sig-full", xi0=xi0, eta_ref=eta_ref, T=T_,
        )
        in_dim = self.feature_extractor.n_features
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.LeakyReLU(0.2)]
        for _ in range(n_res_blocks):
            layers.append(ResidualBlock(hidden_dim))
        self.network = nn.Sequential(
            *layers, nn.Linear(hidden_dim, 1), nn.Sigmoid(),
        )
        # Standardisation buffers (set via .fit_standardiser)
        self.register_buffer("feat_mean", torch.zeros(in_dim))
        self.register_buffer("feat_std", torch.ones(in_dim))
        self._fitted = False

    def fit_standardiser(self, S_train: Tensor) -> None:
        """Compute per-feature mean/std on a sample of training paths."""
        batch, n_plus_1 = S_train.shape
        n = n_plus_1 - 1
        zero_prev = torch.zeros(batch, n, dtype=S_train.dtype, device=S_train.device)
        with torch.no_grad():
            feats = self.feature_extractor(S_train, zero_prev, S0=S0).float()
        # Mean and std across batch and time, per feature
        flat = feats.reshape(-1, feats.shape[-1])
        mean = flat.mean(dim=0)
        std = flat.std(dim=0).clamp(min=1e-3)
        # Don't standardise the recurrent delta_prev feature (index 3)
        mean[3] = 0.0
        std[3] = 1.0
        self.feat_mean = mean
        self.feat_std = std
        self._fitted = True

    def hedge_paths(self, S: Tensor, T_: float = 1.0, S0_: float = 100.0) -> Tensor:
        batch, n_plus_1 = S.shape
        n = n_plus_1 - 1
        device = S.device
        zero_prev = torch.zeros(batch, n, dtype=S.dtype, device=device)
        all_feat = self.feature_extractor(S, zero_prev, S0=S0_).float()

        delta_idx = PathFeatureExtractor.DELTA_PREV_INDEX
        deltas = torch.zeros(batch, n, dtype=torch.float32, device=device)
        for k in range(n):
            feats_k = all_feat[:, k, :].clone()
            if k > 0:
                feats_k[:, delta_idx] = deltas[:, k - 1].detach()
            # Standardise
            feats_k = (feats_k - self.feat_mean) / self.feat_std
            deltas[:, k] = self.network(feats_k).squeeze(-1)
        return deltas

    def forward(self, S: Tensor, T_: float = 1.0, S0_: float = 100.0) -> Tensor:
        return self.hedge_paths(S, T_, S0_)


# =======================================================================
# Long-training with warmup + cosine decay
# =======================================================================

def _train_with_warmup(
    hedger: nn.Module, S_tr: Tensor, S_va: Tensor, p0: float,
    epochs: int = 400, warmup_epochs: int = 20,
    max_lr: float = 1e-3, min_lr: float = 1e-5,
    batch_size: int = 2048, alpha: float = 0.95,
) -> dict:
    """Custom training loop with LR warmup and cosine decay."""
    device = torch.device("cpu")
    hedger = hedger.to(device)
    S_tr = S_tr.to(device)
    S_va = S_va.to(device)

    # Smooth CVaR with learnable w
    w_param = nn.Parameter(torch.tensor(0.0, device=device))
    def risk_fn(pnl):
        loss = -pnl
        return w_param + torch.relu(loss - w_param).mean() / (1.0 - alpha)

    params = list(hedger.parameters()) + [w_param]
    optimiser = torch.optim.Adam(params, lr=max_lr, weight_decay=1e-5)

    def lr_at(epoch: int) -> float:
        if epoch < warmup_epochs:
            return min_lr + (max_lr - min_lr) * epoch / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))

    n_train = S_tr.shape[0]
    best_val = float("inf")
    best_state = copy.deepcopy(hedger.state_dict())
    train_risks: list[float] = []
    val_risks: list[float] = []

    for epoch in range(1, epochs + 1):
        lr = lr_at(epoch - 1)
        for g in optimiser.param_groups:
            g["lr"] = lr

        hedger.train()
        perm = torch.randperm(n_train)
        epoch_losses: list[float] = []
        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            S_batch = S_tr[idx]
            optimiser.zero_grad(set_to_none=True)
            deltas = hedger.hedge_paths(S_batch, T, S0).to(S_batch.dtype)
            payoff = compute_payoff(S_batch, K, "call")
            pnl = compute_hedging_pnl(S_batch, deltas, payoff, p0, 0.0)
            loss = risk_fn(pnl)
            loss.backward()
            nn.utils.clip_grad_norm_(hedger.parameters(), max_norm=1.0)
            optimiser.step()
            epoch_losses.append(float(loss.detach()))
        train_risks.append(sum(epoch_losses) / len(epoch_losses))

        hedger.eval()
        with torch.no_grad():
            d_val = hedger.hedge_paths(S_va, T, S0).to(S_va.dtype)
            payoff_v = compute_payoff(S_va, K, "call")
            pnl_v = compute_hedging_pnl(S_va, d_val, payoff_v, p0, 0.0)
            v = float(risk_fn(pnl_v))
        val_risks.append(v)
        if v < best_val - 1e-6:
            best_val = v
            best_state = copy.deepcopy(hedger.state_dict())

    hedger.load_state_dict(best_state)
    return {"train_risk": train_risks, "val_risk": val_risks, "best_val_risk": best_val}


# =======================================================================
# SignatureAblationExperiment
# =======================================================================

class SignatureAblationExperiment:
    """Staged H4 validation: Stage 1, 1.5, 3."""

    def __init__(
        self,
        device: Optional[torch.device] = None,
        save_dir: str | Path = "figures",
    ) -> None:
        self.device = device or torch.device("cpu")
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────────
    # STAGE 1
    # ──────────────────────────────────────────────────────────

    def run_stage_1(
        self,
        H: float = 0.05,
        n_train: int = 80_000,
        n_val: int = 20_000,
        n_test: int = 50_000,
        epochs: int = 200,
        seed: int = 2024,
    ) -> dict:
        print("=" * 65, flush=True)
        print(f"  STAGE 1: Single-point H={H} feature ablation", flush=True)
        print("=" * 65, flush=True)
        print(f"  n_train={n_train}, epochs={epochs}, n_steps={N_STEPS}", flush=True)

        # 1. Generate data (shared)
        print(f"\n  Generating {n_train+n_val+n_test} rBergomi paths ...", flush=True)
        t0 = time.perf_counter()
        S, _ = _simulate(H, n_train + n_val + n_test, seed=seed)
        S_tr = S[:n_train]
        S_va = S[n_train:n_train + n_val]
        S_te = S[n_train + n_val:]
        del S; gc.collect()
        print(f"  done in {time.perf_counter() - t0:.1f}s", flush=True)

        # 2. p0
        payoff_tr = compute_payoff(S_tr, K, "call")
        p0 = float(payoff_tr.mean())
        print(f"  p0 = {p0:.4f}", flush=True)

        # 3. BS delta baseline
        print("\n  --- BS Delta ---", flush=True)
        bs_pnl = _bs_delta_pnl(S_te, p0)
        bs_metrics = compute_all_metrics(bs_pnl)
        print(f"  ES_95 = {bs_metrics['es_95']:.3f}", flush=True)

        # 4. Three deep hedgers (same init seed for fair comparison)
        init_seed = seed + 1000
        results = {}
        training_times = {}
        for fs, label, color in [
            ("flat", "Flat (4d)", C_FLAT),
            ("sig-3", "Sig-3 (7d)", C_SIG3),
            ("sig-full", "Sig-full (12d)", C_SIGFULL),
        ]:
            print(f"\n  --- {label} ---", flush=True)
            t0 = time.perf_counter()
            hedger, history = _train_signature_hedger(
                fs, S_tr, S_va, p0, epochs=epochs, init_seed=init_seed,
            )
            dt = time.perf_counter() - t0
            training_times[fs] = dt
            pnl, metrics = _eval_hedger(hedger, S_te, p0)
            print(f"  trained in {dt:.0f}s  |  ES_95 = {metrics['es_95']:.3f}  "
                  f"(best epoch {history['best_epoch']})", flush=True)
            results[fs.replace("-", "")] = {
                "pnl": pnl, "metrics": metrics, "history": history, "model": hedger,
                "feature_set": fs,
            }

        # 5. Compute gates
        gamma_flat = bs_metrics["es_95"] - results["flat"]["metrics"]["es_95"]
        gamma_sig3 = bs_metrics["es_95"] - results["sig3"]["metrics"]["es_95"]
        gamma_sigfull = bs_metrics["es_95"] - results["sigfull"]["metrics"]["es_95"]

        gate_threshold = 0.05 * bs_metrics["es_95"]
        gate_passed = gamma_sigfull > gamma_flat + gate_threshold

        print("\n" + "=" * 65, flush=True)
        print("  STAGE 1 RESULTS", flush=True)
        print("=" * 65, flush=True)
        print(f"  ES_95 BS:        {bs_metrics['es_95']:.3f}", flush=True)
        print(f"  ES_95 flat:      {results['flat']['metrics']['es_95']:.3f}  "
              f"(Gamma = {gamma_flat:+.3f})", flush=True)
        print(f"  ES_95 sig-3:     {results['sig3']['metrics']['es_95']:.3f}  "
              f"(Gamma = {gamma_sig3:+.3f})", flush=True)
        print(f"  ES_95 sig-full:  {results['sigfull']['metrics']['es_95']:.3f}  "
              f"(Gamma = {gamma_sigfull:+.3f})", flush=True)
        print(f"  Gate threshold:  {gate_threshold:.3f}", flush=True)
        print(f"  Roughness adv:   {gamma_sigfull - gamma_flat:+.3f}", flush=True)
        print(f"  GATE PASSED:     {gate_passed}", flush=True)

        return {
            "H": H,
            "n_train": n_train, "n_val": n_val, "n_test": n_test,
            "epochs": epochs, "p0": p0,
            "bs": {"pnl": bs_pnl, "metrics": bs_metrics},
            "flat": results["flat"],
            "sig3": results["sig3"],
            "sigfull": results["sigfull"],
            "gamma_flat": gamma_flat,
            "gamma_sig3": gamma_sig3,
            "gamma_sigfull": gamma_sigfull,
            "gate_passed": gate_passed,
            "gate_threshold": gate_threshold,
            "training_times_s": training_times,
            "S_tr": S_tr, "S_va": S_va, "S_te": S_te,
        }

    # ──────────────────────────────────────────────────────────
    # STAGE 1.5 — diagnostics
    # ──────────────────────────────────────────────────────────

    def run_stage_1_5(self, stage1_results: dict) -> dict:
        print("\n" + "=" * 65, flush=True)
        print("  STAGE 1.5: Diagnostic investigation", flush=True)
        print("=" * 65, flush=True)

        diagnostics = {}

        # (a) Training curves analysis (instant)
        print("\n  (a) Training curves analysis ...", flush=True)
        diagnostics["training_curves"] = self._diag_training_curves(stage1_results)
        print(f"      {diagnostics['training_curves']['interpretation']}", flush=True)

        # (b) Feature importance (a few minutes)
        print("\n  (b) Permutation feature importance ...", flush=True)
        diagnostics["feature_importance"] = self._diag_feature_importance(
            stage1_results["sigfull"]["model"], stage1_results["S_te"], stage1_results["p0"],
        )
        print(f"      Most important: {diagnostics['feature_importance']['most_important']}",
              flush=True)
        print(f"      Path feature signal: {diagnostics['feature_importance']['path_feature_signal']}",
              flush=True)

        # (c) Two-tower architecture
        print("\n  (c) Two-tower architecture trial ...", flush=True)
        diagnostics["two_tower"] = self._diag_two_tower(stage1_results, epochs=120)
        print(f"      Two-tower ES_95 = {diagnostics['two_tower']['metrics']['es_95']:.3f}  "
              f"(Gamma = {diagnostics['two_tower']['gamma']:+.3f})", flush=True)

        # (d) Long training
        print("\n  (d) Long training (300 epochs + warmup) ...", flush=True)
        diagnostics["long_training"] = self._diag_long_training(stage1_results, epochs=300)
        print(f"      Long-trained ES_95 = {diagnostics['long_training']['metrics']['es_95']:.3f}",
              flush=True)

        # (e) Standardised features
        print("\n  (e) Standardised features ...", flush=True)
        diagnostics["standardised"] = self._diag_standardised_features(stage1_results, epochs=150)
        print(f"      Standardised ES_95 = {diagnostics['standardised']['metrics']['es_95']:.3f}  "
              f"(Gamma = {diagnostics['standardised']['gamma']:+.3f})", flush=True)

        # Diagnose
        diagnosis, fix = self._diagnose(stage1_results, diagnostics)
        print("\n" + "=" * 65, flush=True)
        print("  DIAGNOSIS", flush=True)
        print("=" * 65, flush=True)
        print(f"  {diagnosis}", flush=True)
        print(f"  Recommended fix: {fix}", flush=True)

        return {
            "diagnostics": diagnostics,
            "diagnosis": diagnosis,
            "recommended_fix": fix,
        }

    # ─── Diagnostic sub-experiments ─────────────────────────

    def _diag_training_curves(self, stage1_results: dict) -> dict:
        flat_history = stage1_results["flat"]["history"]
        sigfull_history = stage1_results["sigfull"]["history"]

        def slope(curve, last_n=20):
            if len(curve) < last_n:
                return float("nan")
            tail = np.array(curve[-last_n:])
            x = np.arange(len(tail))
            return float(np.polyfit(x, tail, 1)[0])

        slope_flat = slope(flat_history["val_risk"])
        slope_sigfull = slope(sigfull_history["val_risk"])

        # "Converged" if slope is small (< 0.01 per epoch on the validation risk scale)
        converged_flat = abs(slope_flat) < 0.02
        converged_sigfull = abs(slope_sigfull) < 0.02

        if not converged_sigfull and slope_sigfull < -0.01:
            interp = "sig-full still descending — training budget likely insufficient"
        elif converged_sigfull:
            interp = "sig-full converged — issue is not training budget"
        else:
            interp = "sig-full plateaued without converging — may need different hyperparams"

        return {
            "converged_flat": converged_flat,
            "converged_sigfull": converged_sigfull,
            "final_slope_flat": slope_flat,
            "final_slope_sigfull": slope_sigfull,
            "interpretation": interp,
        }

    def _diag_feature_importance(
        self, model: SignatureDeepHedger, S_te: Tensor, p0: float,
    ) -> dict:
        """Permutation importance for sig-full."""
        feature_names = ["t/T", "logM", "tau/T", "delta_prev",
                         "rv5", "rv15", "rv50",
                         "R", "Q", "max", "min", "QV"]

        # Baseline ES_95
        with torch.no_grad():
            pnl_base = evaluate_deep_hedger(model, S_te, K=K, T=T, S0=S0, p0=p0)
        es_base = float(expected_shortfall(pnl_base, 0.95))

        # Precompute features once
        batch, n_plus_1 = S_te.shape
        n = n_plus_1 - 1
        zero_prev = torch.zeros(batch, n, dtype=S_te.dtype)
        with torch.no_grad():
            features = model.feature_extractor(S_te, zero_prev, S0=S0).float()  # (batch, n, F)

        n_features = features.shape[-1]
        importances: dict[str, float] = {}

        rng = torch.Generator().manual_seed(42)
        for i in range(n_features):
            if i == 3:
                # Skip the recurrent delta_prev feature — not meaningful to permute
                importances[feature_names[i]] = 0.0
                continue
            # Local permute (each time step independently)
            permuted = features.clone()
            for k in range(n):
                perm = torch.randperm(batch, generator=rng)
                permuted[:, k, i] = features[perm, k, i]
            # Run rollout with permuted features
            pnl_perm = self._rollout_with_features(model.network, permuted, S_te, p0)
            es_perm = float(expected_shortfall(pnl_perm, 0.95))
            importances[feature_names[i]] = es_perm - es_base

        # Identify most/least important
        sorted_feats = sorted(importances.items(), key=lambda kv: -kv[1])
        most_important = sorted_feats[0][0]
        least_important = sorted_feats[-1][0]

        # Path feature signal: sum of importances for features 4..11
        path_signal = sum(importances[feature_names[i]] for i in range(4, 12))
        path_feature_signal = path_signal > 0.1

        return {
            "importances": importances,
            "es_baseline": es_base,
            "most_important": most_important,
            "least_important": least_important,
            "path_signal_sum": path_signal,
            "path_feature_signal": path_feature_signal,
        }

    @staticmethod
    def _rollout_with_features(
        network: nn.Module, features: Tensor, S: Tensor, p0: float,
    ) -> Tensor:
        """Run a hedging rollout using pre-computed features (used for permutation importance)."""
        batch, n, _ = features.shape
        device = S.device
        delta_idx = PathFeatureExtractor.DELTA_PREV_INDEX
        deltas = torch.zeros(batch, n, dtype=torch.float32, device=device)
        with torch.no_grad():
            for k in range(n):
                feat_k = features[:, k, :].clone()
                if k > 0:
                    feat_k[:, delta_idx] = deltas[:, k - 1]
                deltas[:, k] = network(feat_k).squeeze(-1)
        deltas = deltas.to(S.dtype)
        payoff = compute_payoff(S, K, "call")
        return compute_hedging_pnl(S, deltas, payoff, p0, 0.0)

    def _diag_two_tower(self, stage1_results: dict, epochs: int = 120) -> dict:
        S_tr = stage1_results["S_tr"]
        S_va = stage1_results["S_va"]
        S_te = stage1_results["S_te"]
        p0 = stage1_results["p0"]

        _set_seed(stage1_results.get("seed", 2024) + 1000)
        hedger = TwoTowerHedger(n_flat=4, n_path=8, hidden_dim=128, n_res_blocks=2)
        history = train_deep_hedger(
            hedger, S_tr, S_va,
            K=K, T=T, S0=S0, p0=p0, cost_lambda=0.0,
            epochs=epochs, **TRAIN_CFG,
        )
        pnl, metrics = _eval_hedger(hedger, S_te, p0)
        es_bs = stage1_results["bs"]["metrics"]["es_95"]
        gamma = es_bs - metrics["es_95"]
        beats_flat = metrics["es_95"] < stage1_results["flat"]["metrics"]["es_95"]
        return {
            "model": hedger, "metrics": metrics, "gamma": gamma,
            "history": history, "beats_flat": beats_flat,
        }

    def _diag_long_training(self, stage1_results: dict, epochs: int = 300) -> dict:
        S_tr = stage1_results["S_tr"]
        S_va = stage1_results["S_va"]
        S_te = stage1_results["S_te"]
        p0 = stage1_results["p0"]

        _set_seed(stage1_results.get("seed", 2024) + 1000)
        hedger = SignatureDeepHedger(
            feature_set="sig-full", hidden_dim=128, n_res_blocks=2,
            xi0=XI0, eta_ref=ETA, T=T,
        )
        history = _train_with_warmup(
            hedger, S_tr, S_va, p0,
            epochs=epochs, warmup_epochs=20,
        )
        pnl, metrics = _eval_hedger(hedger, S_te, p0)
        es_bs = stage1_results["bs"]["metrics"]["es_95"]
        return {
            "model": hedger, "metrics": metrics,
            "gamma": es_bs - metrics["es_95"],
            "history": history,
            "final_val_risk": history["best_val_risk"],
        }

    def _diag_standardised_features(self, stage1_results: dict, epochs: int = 150) -> dict:
        S_tr = stage1_results["S_tr"]
        S_va = stage1_results["S_va"]
        S_te = stage1_results["S_te"]
        p0 = stage1_results["p0"]

        _set_seed(stage1_results.get("seed", 2024) + 1000)
        hedger = StandardisedSignatureHedger(
            hidden_dim=128, n_res_blocks=2, xi0=XI0, eta_ref=ETA, T_=T,
        )
        # Fit standardiser on a subset of training data
        fit_subset = S_tr[:min(20_000, S_tr.shape[0])]
        hedger.fit_standardiser(fit_subset)

        history = train_deep_hedger(
            hedger, S_tr, S_va,
            K=K, T=T, S0=S0, p0=p0, cost_lambda=0.0,
            epochs=epochs, **TRAIN_CFG,
        )
        pnl, metrics = _eval_hedger(hedger, S_te, p0)
        es_bs = stage1_results["bs"]["metrics"]["es_95"]
        return {
            "model": hedger, "metrics": metrics,
            "gamma": es_bs - metrics["es_95"],
            "history": history,
            "stats": {
                "feature_means": hedger.feat_mean.tolist(),
                "feature_stds": hedger.feat_std.tolist(),
            },
        }

    def _diagnose(self, stage1_results: dict, diagnostics: dict) -> tuple[str, str]:
        """Identify the most likely cause based on diagnostic results."""
        es_bs = stage1_results["bs"]["metrics"]["es_95"]
        es_flat = stage1_results["flat"]["metrics"]["es_95"]
        es_sigfull = stage1_results["sigfull"]["metrics"]["es_95"]
        es_long = diagnostics["long_training"]["metrics"]["es_95"]
        es_two = diagnostics["two_tower"]["metrics"]["es_95"]
        es_std = diagnostics["standardised"]["metrics"]["es_95"]

        long_helps = es_long < es_sigfull - 0.05 * es_bs
        std_helps = es_std < es_sigfull - 0.05 * es_bs
        twotower_helps = es_two < es_sigfull - 0.05 * es_bs
        path_signal = diagnostics["feature_importance"]["path_feature_signal"]

        # Decision tree
        if long_helps and not std_helps and not twotower_helps:
            return ("Training budget was the issue (long training recovered).",
                    "Use 300+ epochs for sig-full in Stage 2.")
        if std_helps and not long_helps and not twotower_helps:
            return ("Feature scaling was the issue (standardisation recovered).",
                    "Use StandardisedSignatureHedger or per-feature LayerNorm.")
        if twotower_helps and not long_helps and not std_helps:
            return ("Architecture was the issue (two-tower recovered).",
                    "Use TwoTowerHedger architecture for Stage 2.")
        if not (long_helps or std_helps or twotower_helps) and not path_signal:
            return ("Path features carry no usable signal at this scale.",
                    "Honestly report H4 as refuted in revised form. Move on.")
        if (long_helps + std_helps + twotower_helps) >= 2:
            return ("Multiple factors contribute (mixed effect).",
                    "Combine fixes: standardisation + longer training.")
        return ("No single fix dominated; effect is small or noise-dominated.",
                "Run Stage 2 anyway and interpret cautiously.")

    # ──────────────────────────────────────────────────────────
    # STAGE 3 — analysis
    # ──────────────────────────────────────────────────────────

    def run_stage_3_analysis(
        self,
        sweep_results_path: str | Path = "figures/signature_h_sweep.json",
    ) -> dict:
        sweep_results_path = Path(sweep_results_path)
        if not sweep_results_path.exists():
            print(f"  ERROR: {sweep_results_path} not found.", flush=True)
            return {}

        with open(sweep_results_path) as f:
            data = json.load(f)

        H_vals = np.array([r["H"] for r in data])
        es_bs = np.array([r["bs_metrics"]["es_95"] for r in data])
        es_flat = np.array([r["flat_metrics"]["es_95"] for r in data])
        es_sig3 = np.array([r["sig3_metrics"]["es_95"] for r in data])
        es_sigfull = np.array([r["sigfull_metrics"]["es_95"] for r in data])

        gamma_flat = es_bs - es_flat
        gamma_sig3 = es_bs - es_sig3
        gamma_sigfull = es_bs - es_sigfull
        roughness_adv = gamma_sigfull - gamma_flat

        # Power-law fit on roughness advantage (only positive points, exclude H=0.5)
        mask = (roughness_adv > 0) & (H_vals < 0.499)
        fit_result = {}
        if mask.sum() >= 3:
            x = np.log(0.5 - H_vals[mask])
            y = np.log(roughness_adv[mask])
            beta, log_c = np.polyfit(x, y, 1)
            y_hat = beta * x + log_c
            r2 = 1 - np.sum((y - y_hat) ** 2) / np.sum((y - y.mean()) ** 2)
            fit_result = {
                "beta": float(beta), "c": float(np.exp(log_c)),
                "r_squared": float(r2), "n_points": int(mask.sum()),
            }
        else:
            fit_result = {
                "beta": float("nan"), "c": float("nan"),
                "r_squared": float("nan"), "n_points": int(mask.sum()),
            }

        # Generate figures
        self._make_h4_figures(
            H_vals, es_bs, es_flat, es_sig3, es_sigfull,
            gamma_flat, gamma_sig3, gamma_sigfull, roughness_adv, fit_result,
        )

        # LaTeX table
        self._write_h4_table(H_vals, es_bs, es_flat, es_sig3, es_sigfull,
                             gamma_flat, gamma_sig3, gamma_sigfull, roughness_adv)

        return {
            "H_vals": H_vals.tolist(),
            "gamma_flat": gamma_flat.tolist(),
            "gamma_sig3": gamma_sig3.tolist(),
            "gamma_sigfull": gamma_sigfull.tolist(),
            "roughness_advantage": roughness_adv.tolist(),
            "fit": fit_result,
        }

    def _make_h4_figures(
        self, H, es_bs, es_flat, es_sig3, es_sigfull,
        g_flat, g_sig3, g_sigfull, ra, fit,
    ):
        # Figure 1: Gamma comparison
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.axhline(0, color="grey", ls="--", lw=0.7)
        ax.plot(H, g_flat, "o-", color=C_FLAT, lw=2, ms=7, label="Flat")
        ax.plot(H, g_sig3, "s-", color=C_SIG3, lw=2, ms=7, label="Sig-3")
        ax.plot(H, g_sigfull, "D-", color=C_SIGFULL, lw=2, ms=7, label="Sig-full")
        ax.axvline(0.07, color="grey", ls=":", lw=0.7)
        ax.set_xlabel("Hurst $H$")
        ax.set_ylabel("$\\Gamma(H)$")
        ax.set_title("Hedging Advantage Under rBergomi: Effect of Path Features")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.save_dir / "fig_h4_gamma_comparison.png", dpi=300)
        plt.close(fig)

        # Figure 2: Roughness advantage with power-law fit
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.axhline(0, color="grey", ls="--", lw=0.7)
        ax.plot(H, ra, "D-", color=C_SIGFULL, lw=2, ms=8)
        if not math.isnan(fit["beta"]):
            xs = np.linspace(0.01, 0.5, 100)
            ax.plot(xs, fit["c"] * (0.5 - xs) ** fit["beta"], "--", color=C_FIT, lw=2,
                    label=f"$\\beta={fit['beta']:.2f}$, $R^2={fit['r_squared']:.2f}$")
            ax.legend()
        ax.set_xlabel("Hurst $H$")
        ax.set_ylabel("$\\Gamma^{\\rm sig\\text{-}full}(H) - \\Gamma^{\\rm flat}(H)$")
        ax.set_title("Roughness-Specific Advantage of Signature Features")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.save_dir / "fig_h4_roughness_advantage.png", dpi=300)
        plt.close(fig)

        # Figure 3: Relative improvement
        rel = 100 * (es_flat - es_sigfull) / np.clip(es_flat, 1e-12, None)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.axhline(0, color="grey", ls="--", lw=0.7)
        ax.plot(H, rel, "D-", color=C_SIGFULL, lw=2, ms=8)
        ax.set_xlabel("Hurst $H$")
        ax.set_ylabel("Relative ES$_{95}$ improvement (%)")
        ax.set_title("Sig-Full vs Flat: Relative Tail Improvement")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.save_dir / "fig_h4_relative_improvement.png", dpi=300)
        plt.close(fig)

        # Figure 4: log-log fit (if positive)
        if not math.isnan(fit["beta"]):
            fig, axes = plt.subplots(2, 1, figsize=(8, 9))
            ax1, ax2 = axes
            ax1.plot(H, ra, "D-", color=C_SIGFULL, lw=2, ms=7)
            xs = np.linspace(0.01, 0.5, 100)
            ax1.plot(xs, fit["c"] * (0.5 - xs) ** fit["beta"], "--", color=C_FIT, lw=2)
            ax1.set_xlabel("$H$"); ax1.set_ylabel("$\\Gamma^{\\rm sig\\text{-}full}-\\Gamma^{\\rm flat}$")
            ax1.set_title("Linear scale"); ax1.grid(True, alpha=0.3)

            mask = (ra > 0) & (H < 0.499)
            x_log = np.log(0.5 - H[mask])
            y_log = np.log(ra[mask])
            ax2.scatter(x_log, y_log, color=C_SIGFULL, s=60, edgecolors="k", lw=0.5)
            xs_log = np.linspace(x_log.min() - 0.1, x_log.max() + 0.1, 50)
            ax2.plot(xs_log, fit["beta"] * xs_log + math.log(fit["c"]), "--", color=C_FIT, lw=2,
                     label=f"$\\beta={fit['beta']:.2f}$, $R^2={fit['r_squared']:.2f}$")
            ax2.set_xlabel("$\\ln(0.5 - H)$"); ax2.set_ylabel("$\\ln\\Gamma^{\\rm sig\\text{-}full}-\\Gamma^{\\rm flat}$")
            ax2.set_title("Log-log scale with fit"); ax2.legend(); ax2.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(self.save_dir / "fig_h4_loglog_fit.png", dpi=300)
            plt.close(fig)

        # Figure 5: 4-panel summary
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        ax = axes[0, 0]
        ax.plot(H, es_bs, "o-", color=C_BS, lw=2, ms=6, label="BS")
        ax.plot(H, es_flat, "s-", color=C_FLAT, lw=2, ms=6, label="Flat")
        ax.plot(H, es_sig3, "^-", color=C_SIG3, lw=2, ms=6, label="Sig-3")
        ax.plot(H, es_sigfull, "D-", color=C_SIGFULL, lw=2, ms=6, label="Sig-full")
        ax.set_xlabel("$H$"); ax.set_ylabel("ES$_{95}$"); ax.set_title("(A) ES$_{95}$ vs $H$")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        ax.plot(H, g_flat, "s-", color=C_FLAT, lw=2, ms=6, label="$\\Gamma^{\\rm flat}$")
        ax.plot(H, g_sigfull, "D-", color=C_SIGFULL, lw=2, ms=6, label="$\\Gamma^{\\rm sig\\text{-}full}$")
        ax.axhline(0, color="grey", ls="--", lw=0.5)
        ax.set_xlabel("$H$"); ax.set_ylabel("$\\Gamma$"); ax.set_title("(B) Advantage gap")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        ax.plot(H, ra, "D-", color=C_SIGFULL, lw=2, ms=6)
        if not math.isnan(fit["beta"]):
            xs = np.linspace(0.01, 0.5, 100)
            ax.plot(xs, fit["c"] * (0.5 - xs) ** fit["beta"], "--", color=C_FIT, lw=1.5,
                    label=f"$\\beta={fit['beta']:.2f}$")
            ax.legend(fontsize=9)
        ax.axhline(0, color="grey", ls="--", lw=0.5)
        ax.set_xlabel("$H$"); ax.set_ylabel("$\\Gamma^{\\rm sig\\text{-}full}-\\Gamma^{\\rm flat}$")
        ax.set_title("(C) Roughness-specific advantage"); ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        ax.plot(H, rel, "D-", color=C_SIGFULL, lw=2, ms=6)
        ax.axhline(0, color="grey", ls="--", lw=0.5)
        ax.set_xlabel("$H$"); ax.set_ylabel("% improvement")
        ax.set_title("(D) Relative ES improvement"); ax.grid(True, alpha=0.3)

        fig.suptitle("H4 Validation: Signature Features vs Flat Features", y=1.00)
        fig.tight_layout()
        fig.savefig(self.save_dir / "fig_h4_summary.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        for name in ["fig_h4_gamma_comparison.png", "fig_h4_roughness_advantage.png",
                     "fig_h4_relative_improvement.png", "fig_h4_summary.png"]:
            print(f"  Saved {self.save_dir / name}", flush=True)

    def _write_h4_table(self, H, es_bs, es_flat, es_sig3, es_sigfull,
                        g_flat, g_sig3, g_sigfull, ra):
        lines = [
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{H4 validation: signature features vs flat features under rBergomi.}",
            r"\begin{tabular}{c|rrrr|rr}",
            r"\hline",
            r"$H$ & ES$_{95}^{\rm BS}$ & ES$_{95}^{\rm flat}$ & ES$_{95}^{\rm sig\text{-}3}$ "
            r"& ES$_{95}^{\rm sig\text{-}full}$ & $\Gamma^{\rm sig\text{-}full}$ "
            r"& $\Gamma^{\rm sig\text{-}full}-\Gamma^{\rm flat}$ \\",
            r"\hline",
        ]
        for i in range(len(H)):
            lines.append(
                f"{H[i]:.2f} & {es_bs[i]:.2f} & {es_flat[i]:.2f} & {es_sig3[i]:.2f} & "
                f"{es_sigfull[i]:.2f} & {(es_bs[i]-es_sigfull[i]):.3f} & {ra[i]:+.3f} \\\\"
            )
        lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
        path = self.save_dir / "h4_table.tex"
        path.write_text("\n".join(lines))
        print(f"  Saved {path}", flush=True)

    # ──────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────

    def save_stage(self, name: str, results: dict) -> None:
        """Save scalar metrics to JSON, dropping tensors and modules."""
        path = self.save_dir / f"signature_ablation_{name}.json"

        def _strip(obj):
            if isinstance(obj, torch.Tensor):
                return None
            if isinstance(obj, nn.Module):
                return None
            if isinstance(obj, dict):
                return {k: _strip(v) for k, v in obj.items() if _strip(v) is not None}
            if isinstance(obj, list):
                return [_strip(v) for v in obj]
            if isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            try:
                return float(obj)
            except Exception:
                return None

        with open(path, "w") as f:
            json.dump(_strip(results), f, indent=2)
        print(f"  Saved {path}", flush=True)

    # ──────────────────────────────────────────────────────────
    # Orchestration
    # ──────────────────────────────────────────────────────────

    def run_all(
        self, skip_stage_1_5: bool = False,
    ) -> dict:
        out: dict[str, Any] = {}
        stage_1 = self.run_stage_1()
        self.save_stage("stage_1", stage_1)
        out["stage_1"] = {k: v for k, v in stage_1.items()
                          if not isinstance(v, (torch.Tensor, nn.Module, dict))}

        if (not stage_1["gate_passed"]) and (not skip_stage_1_5):
            print("\n  Stage 1 GATE FAILED — running diagnostics", flush=True)
            stage_1_5 = self.run_stage_1_5(stage_1)
            self.save_stage("stage_1_5", stage_1_5)
            out["stage_1_5"] = stage_1_5
        elif stage_1["gate_passed"]:
            print("\n  Stage 1 GATE PASSED — skipping Stage 1.5", flush=True)
        else:
            print("\n  Skipping Stage 1.5 (--skip-stage-1-5)", flush=True)

        return out


# =======================================================================
# CLI
# =======================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["1", "1.5", "3", "all"], default="1")
    parser.add_argument("--skip-stage-1-5", action="store_true")
    parser.add_argument("--n-train", type=int, default=80_000)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--H", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()

    exp = SignatureAblationExperiment(save_dir=FIGURE_DIR)

    if args.stage == "1":
        results = exp.run_stage_1(
            H=args.H, n_train=args.n_train, n_val=args.n_train // 4,
            n_test=args.n_train // 2, epochs=args.epochs, seed=args.seed,
        )
        exp.save_stage("stage_1", results)
        if not results["gate_passed"] and not args.skip_stage_1_5:
            stage_1_5 = exp.run_stage_1_5(results)
            exp.save_stage("stage_1_5", stage_1_5)
    elif args.stage == "3":
        exp.run_stage_3_analysis()
    elif args.stage == "all":
        exp.run_all(skip_stage_1_5=args.skip_stage_1_5)
    else:
        print(f"  Stage {args.stage} not directly callable", flush=True)


if __name__ == "__main__":
    main()
