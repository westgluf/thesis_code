# Canonical Re-run Results — Post Seeding Fix

Generated: 2026-04-21T11:59:51
Git commit (pre-fix): cdbd9f1 (Pre-fix snapshot)
Git commit (post-fix): a2ca83a82817b031b1b3afb2a2a840958dcf4721-dirty

## Reproducibility verification

### Baseline re-run (seed 2024)

| Metric | Run 1 | Run 2 | Match? |
|---|---|---|---|
| Γ (λ=0.0) | 1.184351 | 1.184351 | ✓ |
| ES_0.95_DH | 10.446314 | 10.446314 | ✓ |
| first_weight_sum | -5.194795 | -5.194795 | ✓ |

Verdict: **REPRODUCIBLE**

### Decomposition re-run (seed 3024)

| Metric | Run 1 | Run 2 | Match? |
|---|---|---|---|
| Γ_total | 0.877736 | 0.877736 | ✓ |
| Objective % | 78.9234 | 78.9234 | ✓ |
| ES_A_dh | 1.629356 | 1.629356 | ✓ |

Verdict: **REPRODUCIBLE**

## Baseline comparison (λ=0)

| Metric | Old (single run) | New (mean ± std, 5 seeds) | Δ abs | Δ rel |
|---|---|---|---|---|
| ES_0.95 BS | 11.3546 | 11.5921 ± 0.0316 | +0.2375 | +2.1% |
| ES_0.95 DH | 10.1622 | 10.4442 ± 0.0748 | +0.2820 | +2.8% |
| Γ | 1.1924 | 1.1479 ± 0.0761 | -0.0445 | -3.7% |
| Mean PL (DH) | 0.0440 | -0.0073 ± 0.0390 | -0.0513 | -116.5% |
| Std PL (DH) | 4.1506 | 4.1415 ± 0.0295 | -0.0091 | -0.2% |

## Baseline comparison (λ=0.001)

| Metric | Old (single run) | New (mean ± std, 5 seeds) | Δ abs | Δ rel |
|---|---|---|---|---|
| ES_0.95 BS | 11.7690 | 12.0082 ± 0.0324 | +0.2392 | +2.0% |
| ES_0.95 DH | 10.3784 | 10.6658 ± 0.0389 | +0.2874 | +2.8% |
| Γ | 1.3906 | 1.3423 ± 0.0491 | -0.0482 | -3.5% |

## Decomposition comparison

| Bucket | Old % | New mean % ± std | Δ (pp) |
|---|---|---|---|
| Objective | +46.49 | +61.49 ± 14.87 | +15.00 pp |
| Interaction | +29.09 | +17.03 ± 15.18 | -12.06 pp |
| Stoch vol | +25.06 | +8.80 ± 4.36 | -16.26 pp |
| Roughness | +1.85 | +11.24 ± 4.45 | +9.39 pp |
| Architecture | -2.49 | +1.43 ± 3.51 | +3.92 pp |

Γ_total (decomposition baseline): old = 0.8188; new mean = 0.8542 ± 0.0363

## Qualitative assessment

- Does Γ > 0 across all 5 seeds (baseline, λ=0)? **YES** (values: +1.1844, +1.1263, +1.0394, +1.2459, +1.1434)
- Is decomposition ranking preserved (objective > interaction > stoch vol > roughness > architecture)? **NO**
  - seed 3024: objective > roughness > stoch_vol > architecture > interaction
  - seed 3025: interaction > objective > roughness > stoch_vol > architecture
  - seed 3026: objective > roughness > interaction > stoch_vol > architecture
  - seed 3027: objective > roughness > stoch_vol > interaction > architecture
  - seed 3028: objective > interaction > stoch_vol > roughness > architecture
- Does old Γ (+1.1924) fall within new mean ± 2σ ([+0.9956, +1.3001])? **YES**

## Verdict

**QUALITATIVE_CHANGE** — Sign flip or ranking change; major revision required.

## Per-seed detail (baseline, λ=0)

| Seed | ES_BS | ES_DH | Γ | Mean PL DH | Std PL DH |
|---|---|---|---|---|---|
| 2024 | 11.6307 | 10.4463 | +1.1844 | +0.0139 | 4.1065 |
| 2025 | 11.5828 | 10.4565 | +1.1263 | +0.0063 | 4.1864 |
| 2026 | 11.5978 | 10.5585 | +1.0394 | -0.0660 | 4.1387 |
| 2027 | 11.6043 | 10.3584 | +1.2459 | -0.0245 | 4.1280 |
| 2028 | 11.5447 | 10.4013 | +1.1434 | +0.0340 | 4.1477 |

## Per-seed detail (decomposition)

| Seed | Γ_total | Obj% | Int% | SV% | R% | Arch% |
|---|---|---|---|---|---|---|
| 3024 | +0.8777 | +78.92 | +3.19 | +5.48 | +8.58 | +3.83 |
| 3025 | +0.9057 | +40.18 | +41.40 | +6.98 | +13.41 | -1.96 |
| 3026 | +0.8331 | +61.32 | +16.43 | +7.41 | +17.54 | -2.69 |
| 3027 | +0.8366 | +71.20 | +5.49 | +7.70 | +10.66 | +4.95 |
| 3028 | +0.8179 | +55.85 | +18.66 | +16.45 | +6.00 | +3.05 |

## Methodology

- Baseline: canonical_rerun.py with seeds [2024, 2025, 2026, 2027, 2028]
- Data: n_train=80000, n_val=20000, n_test=50000
- Training: epochs=200, patience=30, batch_size=2048, lr=0.001
- Decomposition: decomposition_rerun.py (aggregate-only) with seeds [3024, 3025, 3026, 3027, 3028]
