# η=0 Control Experiment Results

Generated: 2026-04-21T21:33:02
Git commit: 15e840bc4980119f086b6883ec27306c0559a847-dirty
Script: `deep_hedging/experiments/eta_zero_control.py`

## Experimental setup

Rough Bergomi with η = 0, collapsing the variance process to the deterministic value v_t = ξ_0 = 0.235² = 0.055224999999999996. The price dynamics reduce to geometric Brownian motion with σ = √ξ_0 ≈ 0.2350. In this regime, the analytical Black-Scholes delta is the exact replicating strategy for a European call.

Any residual difference between Black-Scholes delta and the deep hedger trained with the ES₀.₉₅ objective comes from (a) the architectural flexibility of the neural network, or (b) the choice of ES vs pointwise replication as the training objective. We denote this residual as Γ_arch = ES₀.₉₅(BS) − ES₀.₉₅(DH).

### Parameters

- H = 0.07, η = 0.0, ρ = -0.7, ξ₀ = 0.055224999999999996
- S₀ = K = 100.0, T = 1.0, n_steps = 100
- Training: 80000 train / 20000 val / 50000 test
- Epochs: 200, patience: 30, batch_size: 2048, lr: 0.001
- Objective: ES₀.₉₅, α = 0.95
- Seeds: [4024, 4025, 4026, 4027, 4028]

## Reproducibility verification

| Metric | Original (seed 4024) | Rerun (seed 4024) | Match? |
|---|---|---|---|
| ES_0.95 BS | 1.925159 | 1.925159 | ✓ |
| ES_0.95 DH | 1.678343 | 1.678343 | ✓ |
| Γ_arch | 0.246815 | 0.246815 | ✓ |
| first_weight_sum | -10.894047 | -10.894047 | ✓ |

Verdict: **REPRODUCIBLE**

## Per-seed results

| Seed | ES_BS | ES_DH | Γ_arch | Mean P&L (DH) | Std P&L (DH) |
|---|---|---|---|---|---|
| 4024 | 1.9252 | 1.6783 | +0.2468 | -0.0387 | 0.9341 |
| 4025 | 1.7942 | 1.5650 | +0.2292 | +0.0813 | 0.9578 |
| 4026 | 1.8797 | 1.6461 | +0.2336 | -0.0167 | 0.9292 |
| 4027 | 1.8239 | 1.5957 | +0.2282 | +0.0366 | 0.9335 |
| 4028 | 1.9922 | 1.7628 | +0.2294 | -0.1190 | 0.9210 |

## Aggregated statistics

| Metric | Mean | Std | 95% CI |
|---|---|---|---|
| Γ_arch | +0.2334 | 0.0078 | [+0.2238, +0.2431] |
| ES_BS | 1.8830 | 0.0792 | — |
| ES_DH | 1.6496 | 0.0770 | — |
| p_0 (empirical) | 9.3440 | 0.0734 | — |
| p_0 (BS theoretical) | 9.3536 | — | — |

## Qualitative assessment

- Does Γ_arch > 0 in all 5 seeds? **YES** (values: +0.2468, +0.2292, +0.2336, +0.2282, +0.2294)
- Is Γ_arch statistically distinguishable from zero (95% CI excludes 0)? **YES** (CI = [+0.2238, +0.2431])
- Is empirical p_0 within 1% of analytical BS price at σ=√ξ_0? **YES** (empirical mean = 9.3440, BS = 9.3536, |Δ/BS| = 0.10%)

## Interpretation

The ES-optimal training captures a small residual advantage (Γ_arch = +0.2334 ± 0.0078, 95% CI [+0.2238, +0.2431]) even against the exact replicating BS delta in the degenerate η=0 regime. This figure represents the 'architecture + objective' floor of the advantage and must be subtracted as a baseline offset when interpreting the full Γ ≈ 1.15 from Section 6.3.1. The residual reflects the ES_{0.95} training objective's emphasis on tail losses rather than pointwise replication.

## Deliverables checklist

- [x] `deep_hedging/experiments/eta_zero_control.py`
- [x] `results/eta_zero_v2/eta_zero_5seeds.json`
- [x] `results/eta_zero_v2/eta_zero_report.md`
- [x] `results/eta_zero_v2/seed4024_rerun.json`
- [x] `figures/eta_zero_v2/gamma_arch_5seeds.png`
- [x] `figures/eta_zero_v2/pl_histogram_seed4024.png`
- [x] Git commits: pre-Phase-C, post-implementation, post-execution
