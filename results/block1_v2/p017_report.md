# P01.7 Rerun Results — 4-Cell Extended Validation with Fixed Seeding

Generated: 2026-04-23T10:31:06
Git commit: 9cc906628e3c2db62b2eba6211e1c8c35987df01
Script: `deep_hedging/experiments/p017_rerun.py`

## Setup

Four validation cells at n=400 with fixed seeding:

- **Cell A:** (η=0, MSE objective); canonical Γ = -0.0204
- **Cell B:** (η=1.9, MSE objective); canonical Γ = +0.3603
- **Cell C:** H2 with λ=0.001 (transaction-cost cell); canonical Γ = +1.0801
- **Cell D:** GBM-pretrained transfer (eval only); canonical Γ = +0.3939

Seeds per cell: A=[7711, 7712, 7713], B=[7721, 7722, 7723], C=[7731, 7732, 7733], D=[7741, 7742, 7743]

## Reproducibility check (Cell A, seed 7711)

_Not available._

## Per-cell results

### Cell A — (eta=0, MSE) decomposition corner

| Seed | ES_BS | ES_DH | Γ |
|---|---|---|---|
| 7711 | 0.9042 | 0.8970 | +0.0072 |
| 7712 | 0.9395 | 0.9786 | -0.0390 |
| 7713 | 0.9562 | 0.9921 | -0.0359 |
| **Mean** | — | — | **-0.0226 ± 0.0258** |

Canonical Γ (n=100): -0.0204 → verdict: **CELL_MODESTLY_SHIFTED**

### Cell B — (eta=1.9, MSE) decomposition corner

| Seed | ES_BS | ES_DH | Γ |
|---|---|---|---|
| 7721 | 9.7293 | 9.3925 | +0.3367 |
| 7722 | 9.7664 | 9.2847 | +0.4817 |
| 7723 | 9.5601 | 9.0639 | +0.4963 |
| **Mean** | — | — | **+0.4382 ± 0.0882** |

Canonical Γ (n=100): +0.3603 → verdict: **CELL_COLLAPSED**

### Cell C — H2 representative (lambda=0.001)

Cell C uses only the first seed by design (evaluation-focused).

- **variant_C_match** (n_rebal=400): ES_BS=10.5883, ES_DH=9.0091, Γ=+1.5792, 95%CI=[+1.4599, +1.6991]
- **variant_C_subsample** (n_rebal=100): ES_BS=10.4414, ES_DH=9.2395, Γ=+1.2019, 95%CI=[+1.0642, +1.3472]

Canonical Γ (n=100): +1.0801 → verdict: **CELL_PRESERVED**

### Cell D — Transfer learning GBM-pretrained on rBergomi

ES_BS=9.6937, ES_DH=9.2992, Γ=+0.3944, 95%CI=[+0.3617, +0.4261]

Canonical Γ (n=100): +0.3939 → verdict: **CELL_PRESERVED**

## Combined verdict

- Cell A: CELL_MODESTLY_SHIFTED
- Cell B: CELL_COLLAPSED
- Cell C: CELL_PRESERVED
- Cell D: CELL_PRESERVED

Overall: **FAIL**
