# Heston PDE Delta — Evaluation Report

Generated: 2026-04-24T15:30:09
Git commit: e7503d96a66aaf7ebf509e7decfdbce3352f95db
Script: `deep_hedging/experiments/heston_pde_evaluation.py`

## Calibration summary

- V_0 = θ = ξ_0 = 0.055225
- ρ = -0.70 (direct transfer from rough Bergomi)
- **κ = 1.0000**
- **σ_v = 0.5538**
- Feller slack = -0.1962 (slightly negative; documented in calibration_report.md)
- ATM call price match: 8.0157 vs target 8.0157 (rel err 0.000%) **PASS (< 2%)**

## Sanity checks (from test_heston_pde.py)

- GBM limit (σ_v = 0.001) vs BS delta at σ = 0.15, ATM: |rel err| = 0.010% **PASS (< 1%)**
- Call price sanity at canonical σ_eff ≈ 0.235: 8.82 (in [7, 13]) **PASS**
- Delta surface monotonicity in S at fixed V = V_0: monotone increasing, **PASS**
  bell-shaped Δ(S = 50) = 0.00, Δ(S = 100) = 0.61, Δ(S = 150) = 0.96

## 5-seed evaluation results

Seeds: [6024, 6025, 6026, 6027, 6028] — disjoint from other prompts.
Test set: 50,000 rough Bergomi paths per seed (n_steps = 100) at canonical
calibration (H = 0.07, η = 1.9, ρ = −0.7, ξ_0 = 0.055225).

### Per-seed ES_0.95

| Seed | ES_BS | ES_PluginDelta | ES_HestonPDE |
|---|---|---|---|
| 6024 | 11.4948 | 15.5743 | 13.5236 |
| 6025 | 11.1500 | 15.4815 | 13.3730 |
| 6026 | 11.4422 | 15.4527 | 13.4434 |
| 6027 | 11.5368 | 15.4510 | 13.5426 |
| 6028 | 11.6116 | 15.2781 | 13.3526 |

### Aggregated across 5 seeds

| Strategy | ES_0.95 (mean ± std) | ES_0.99 | Std P&L | Turnover |
|---|---|---|---|---|
| BS Delta | **11.447 ± 0.177** | 21.220 | 4.033 | 2.717 |
| PluginDelta (old BS-functional proxy) | **15.448 ± 0.107** | 25.246 | 5.072 | 8.770 |
| **HestonPDE (new, true PDE)** | **13.447 ± 0.086** | 19.316 | 4.808 | 6.263 |
| Deep Hedger (canonical from Phase B) | **10.444 ± 0.075** | — | — | — |

## Comparison: HestonPDE vs PluginDelta

| Metric | PluginDelta | HestonPDE | Δ abs | Δ rel |
|---|---|---|---|---|
| ES_0.95 | 15.448 | 13.447 | -2.000 | -12.95% |
| ES_0.99 | 25.246 | 19.316 | -5.930 | -23.49% |
| Std P&L | 5.072 | 4.808 | -0.264 | -5.21% |
| Turnover | 8.770 | 6.263 | -2.507 | -28.59% |

## Reproducibility verification (seed 6024)

| Strategy | Metric | Original | Rerun | Match? |
|---|---|---|---|---|
| bs | es_95 | 11.494837 | 11.494837 | ✓ |
| bs | first_delta_sum | 27338.403899 | 27338.403899 | ✓ |
| plugin | es_95 | 15.574277 | 15.574277 | ✓ |
| plugin | first_delta_sum | 27338.403899 | 27338.403899 | ✓ |
| heston_pde | es_95 | 13.523570 | 13.523570 | ✓ |
| heston_pde | first_delta_sum | 32909.110188 | 32909.110188 | ✓ |

Verdict: **REPRODUCIBLE**

## Interpretation

Heston PDE delta outperforms the plug-in proxy (13.447 vs 15.448, a 13.0% improvement) but remains worse than plain BS delta (13.447 vs 11.447, a 17.5% gap) and well above the deep hedger (10.444 ± 0.075). Under rough-Bergomi mis-specification, the correctly-implemented Markovian SV baseline cannot capture path dependence.

**Classification of outcome (per Phase J §10):** B

The result separates cleanly the three methodological layers in Section 6.3.1:

1. **BS Delta** (11.447) — a simple constant-σ Markovian baseline. Benefits from
   not responding to the volatile realised variance; its time-constant delta surface
   avoids the overreaction that hurts the plug-in approach.

2. **Heston PDE delta** (13.447) — the correctly-implemented Markovian SV
   baseline, calibrated so that its ATM call price matches rough Bergomi's empirical
   option price to < 0.01%. Despite correct calibration, it adapts its delta too aggressively
   to the (non-Markovian) variance path V_t and ends up hedging worse than plain BS on
   ES_{0.95}. This quantifies the cost of using a Heston surrogate that cannot see the
   long-memory structure of rough Bergomi.

3. **PluginDelta** (15.448) — the BS-functional plug-in previously identified
   in the audit as not being a true Heston delta. Its naïve dependence on instantaneous
   variance produces the worst ES in this mis-specified setting, a full 2.00
   worse than the true Heston PDE. This retroactively validates the audit's flag: the
   plug-in proxy is NOT a reasonable Heston substitute.

**The deep hedger remains strictly better** than all three Markovian strategies
(10.444 ± 0.075 vs best Markovian 11.447), confirming the Section 6.3.1 headline:
model-free hedging captures rough Bergomi structure that no Markovian SV approach can.

## Phase II implications

The Section 6.3.1 narrative (Text Task 10) should be revised to distinguish the two
Markovian stochastic-volatility approaches:
- *Plug-in proxy* (BS functional with realised V): audit-flagged as misleading;
  provides no improvement over BS in the rough-Bergomi setting (in fact WORSE).
- *True Heston PDE delta*: a faithful Markovian SV surrogate. Improves on the plug-in
  by 2.00 ES_0.95 units, but still fails to match rough
  Bergomi's path-dependence. Remains worse than plain BS delta in this setting.

Both Markovian approaches underperform the deep hedger by ≥ 3.0 ES_0.95 units,
reinforcing the core message of Section 6.3.1.

## Figures

- `figures/heston_pde/delta_surface.png` — Heston PDE delta surface heatmap + 3 V-slices
- `figures/heston_pde/strategy_comparison.png` — ES_0.95 bar chart for 4 strategies
- `figures/heston_pde/hedging_paths_example.png` — single-path deltas over time

## Deliverables

- `deep_hedging/hedging/heston_pde_delta.py` — HV ADI solver + calibration routine
- `deep_hedging/tests/test_heston_pde.py` — sanity tests (all PASS)
- `deep_hedging/experiments/heston_pde_evaluation.py` — 5-seed evaluation runner
- `results/heston_pde/calibration_report.md` — Phase 2 calibration detail
- `results/heston_pde/calibration_data.json` — calibrated parameters + moments
- `results/heston_pde/heston_pde_5seeds.json` — 5-seed aggregated results
- `results/heston_pde/seed_{6024..6028}.json` — per-seed detail
- `results/heston_pde/seed_6024_rerun.json` — reproducibility subprocess
- `results/heston_pde/heston_pde_report.md` — this file
- 3 figures under `figures/heston_pde/`