# P01.6 Rerun Results — n=400 Validation with Fixed Seeding

Generated: 2026-04-23T10:31:05
Git commit: 9cc906628e3c2db62b2eba6211e1c8c35987df01
Script: `deep_hedging/experiments/p016_rerun.py`

## Setup

- Rough Bergomi at canonical calibration (H=0.07, η=1.9, ρ=-0.7, ξ₀=0.055225)
- Grid resolution: n=400 (vs n=100 canonical)
- Training: 80000 train / 20000 val / 50000 test, 200 epochs, patience=30
- Seeds: [7401, 7402, 7403, 7404, 7405]

## Reproducibility check

_Not available._

## Per-seed results

| Seed | ES_BS | ES_DH | Γ | Mean P&L (DH) | Std P&L (DH) |
|---|---|---|---|---|---|
| 7401 | 9.5842 | 8.5004 | +1.0839 | -0.0080 | 3.5165 |
| 7402 | 9.5842 | 8.5282 | +1.0560 | -0.0055 | 3.4856 |
| 7403 | 9.5842 | 8.4868 | +1.0974 | -0.0073 | 3.5215 |
| 7404 | 9.5842 | 8.5274 | +1.0569 | -0.0109 | 3.4886 |
| 7405 | 9.5842 | 8.4934 | +1.0908 | -0.0101 | 3.5401 |

## Aggregate

- Γ(n=400) = +1.0770 ± 0.0194
- 95% CI: [+1.0529, +1.1011]

## Comparison to canonical (n=100)

- Γ(n=100) from Phase B baseline: +1.1479 ± 0.0761
- Γ(n=400) from this run: +1.0770 ± 0.0194
- Absolute difference: 0.0709
- Overlap of 95% CIs ([+0.9957, +1.3001] vs [+1.0382, +1.1157]): YES

## Verdict

**PRESERVED** — Γ(n=400) within 2σ of Γ(n=100) → coarse-grid canonical Γ is robust under grid refinement.
