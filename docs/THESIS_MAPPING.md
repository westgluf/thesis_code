# Thesis Mapping

> For every numbered Figure (1–38), Table (1–15), and headline numerical
> claim in Sections 5–7 and Appendices A–B, this document gives the
> repository file(s) that contain the underlying data, the script that
> produces the figure / table, and a one-line reproduction command.

All paths verified against `git ls-files` on `release/v1.0-thesis`.
All JSON values verified by reading the JSON.

Three keypath corrections established by the Phase 2.5 stability audit
are used throughout this document (do not paraphrase from older
inventories):

- **Tab. 7 GBM 3-seed row** → `results/transfer_v2/L2_budget_sweep.json`
  keypath `results.gbm.160000.aggregate.es_95` (mean 11.0791, n=3,
  seeds 7101/7102/7103). NOT `L1_multi_source_5seeds.json` (that file
  has 1 seed). See `docs/audit/TAB7_GBM_RESOLUTION.md`.
- **Tab. 8 gap rows** use keypath
  `results.per_target.<target>.aggregate.gap_dh_minus_ref` (NOT
  `gap`).
- **Sec 5.4 KS p-value** → `results/simulator_validation_bundle/sim_validation_data.json`
  keypath `p021_cholesky.fbm_terminal.ks_pvalue` (NOT
  `block1/cholesky_v2_n500k.json` — that raw file lives only locally,
  never on GitHub).

---

## Section 5 — Simulation Framework

### Figures

| Thesis citation | Repo file(s) | How to reproduce |
|---|---|---|
| Fig. 18 (Sec 5.2) Sample paths from Markovian benchmark models (GBM and Heston) | image: `latex_package/figures/5_2_markovian_paths.png`; data: regenerated on the fly via `deep_hedging.core.gbm.GBM` + `deep_hedging.core.heston.Heston`; script: `scripts/generate_5_2_figure.py` | `python scripts/generate_5_2_figure.py` |
| Fig. 19 (Sec 5.3) Rough Bergomi simulation pipeline schematic | inline TikZ in `latex_package/main.tex` (no PNG) | n/a — re-build the LaTeX |
| Fig. 20 (Sec 5.4) Hybrid-scheme convergence: log-log of `\|ES_0.95(n) − ES_∞\|` vs `n` | image: `latex_package/figures/5_4_convergence_alpha.png`; data: `results/simulator_validation_bundle/sim_validation_data.json` (keys: `p01_convergence.alpha_hat`, `.alpha_ci`, `.per_n_es95`) | `python scripts/regenerate_5_4_cholesky_ks_figure.py` (note: this script regenerates both 5.4 figures from the consolidation bundle) |
| Fig. 21 (Sec 5.4) Exact-Cholesky benchmark: empirical CDFs of terminal `W^H_T` (hybrid vs Cholesky) | image: `latex_package/figures/5_4_cholesky_ks.png`; data: `results/simulator_validation_bundle/sim_validation_data.json:p021_cholesky.fbm_terminal` | `python scripts/regenerate_5_4_cholesky_ks_figure.py` |

### Numerical claims (Sec 5.4)

| Claim | JSON keypath | Value |
|---|---|---|
| α̂ convergence slope | `results/simulator_validation_bundle/sim_validation_data.json:p01_convergence.alpha_hat` | 0.913 |
| α̂ 95 % bootstrap CI | same:`p01_convergence.alpha_ci` | [0.722, 1.104] |
| `ES_∞` from Richardson extrapolation | same:`p01_convergence.ES_inf` | 8.821 |
| Relative error at n=100 | same:`p01_convergence.rel_err_at_100` | 0.287 (28.7 %) |
| Cholesky KS statistic | same:`p021_cholesky.fbm_terminal.ks_statistic` | 0.00244 |
| Cholesky KS p-value | same:`p021_cholesky.fbm_terminal.ks_pvalue` | 0.926 |
| Cholesky verdict | same:`p021_cholesky.global_verdict` | "STRICT_PASS" |
| Variance-path max relative diff | same:`p021_cholesky.variance_path.max_rel_diff` | 0.0219 (2.19 %) |
| Call price (exact vs hybrid) relative diff | same:`p021_cholesky.call_price.rel_diff` | 0.00607 |

Verification command (one-liner):

    python -c "import json; d = json.load(open('results/simulator_validation_bundle/sim_validation_data.json')); print('alpha_hat=', d['p01_convergence']['alpha_hat']); print('KS p=', d['p021_cholesky']['fbm_terminal']['ks_pvalue'])"

---

## Section 6.2 — GBM Benchmark

### Tables

| Thesis citation | Repo file(s) | How to reproduce |
|---|---|---|
| Tab. 2 (Sec 6.2) ES_0.95 across `(σ̄, λ)` cells, oracle regime | data: `results/gbm_deephedge/benchmark_6_2/aggregate/{scenario_summary,seed_level_metrics}.csv`; script: `scripts/generate_section6_2_tables.py` | `python scripts/generate_section6_2_tables.py > /tmp/section6_2_tables.tex` |
| Tab. 3 (Sec 6.2) ES_0.95 robust regime (DH trained over `σ̄ ∈ {0.15, 0.20, 0.25}`) | same | same |
| Tab. 4 (Sec 6.2) Paired t-test BS vs DH oracle, λ=0 | same | same |

### Figures

| Thesis citation | Repo file(s) | How to reproduce |
|---|---|---|
| Fig. 22 (Sec 6.2) Terminal P&L distributions on the GBM benchmark, canonical cell | image: `latex_package/figures/6_2_pl_histograms.png`; data: per-cell artefacts under `results/gbm_deephedge/benchmark_6_2/runs/seed_0000__train_oracle__feat_b__obj_cvar_a0.95__sigtrue_0.2__sigbar_0.2__lam_0/` (local-only — see Section 6.2 note below) | `./tools/smoke.sh` then re-eval; or full grid via `python -m src.run_benchmark_gbm_grid …` |
| Fig. 23 (Sec 6.2) Tail-risk metrics on the GBM benchmark, canonical cell | image: `latex_package/figures/6_2_risk_metrics.png`; data: same per-cell artefact | same |

### GBM benchmark numerical claims (Sec 6.2, λ=0, σ̄=0.20, 10 seeds)

The full grid (10 seeds × 5 σ̄ × 4 λ × 2 regimes = 400 cells) is summarised
in `results/gbm_deephedge/benchmark_6_2/aggregate/seed_level_metrics.csv`.
Headline values from the thesis text:

| Claim | Source | Value |
|---|---|---|
| Sec 6.2 BS delta ES_0.95 (correctly-specified, λ=0) | aggregate CSV row σ̄=0.20, regime=oracle, method=bs_delta | 0.0225 ± 0.0000 |
| Sec 6.2 DH oracle ES_0.95 (correctly-specified, λ=0) | aggregate CSV row σ̄=0.20, regime=oracle, method=deep_hedge_oracle | 0.0213 ± 0.0001 |
| Sec 6.2 DH-vs-BS gap (correctly-specified) | derived | −0.0013 |
| Sec 6.2 BS delta ES_0.95 at σ̄=0.10 | aggregate CSV row σ̄=0.10 | 0.0550 ± 0.0001 |
| Sec 6.2 DH oracle ES_0.95 at σ̄=0.10 | aggregate CSV row σ̄=0.10 | 0.0213 ± 0.0001 |
| Sec 6.2 BS delta ES_0.95 at σ̄=0.30 | aggregate CSV row σ̄=0.30 | 0.0274 ± 0.0000 |

> **Note on Sec 6.2 raw artefacts.** The 400 per-cell directories under
> `results/gbm_deephedge/benchmark_6_2/runs/seed_*/` (~7 GB of `.pt`,
> `.npz`, `.png`, JSON) are NOT pushed to GitHub. The 9 small files
> tracked under `aggregate/` and `benchmark_spec.json` ARE pushed and
> are sufficient to regenerate Tabs 2-4 directly. Full per-cell
> regeneration is via `python -m src.run_benchmark_gbm_grid …`. See
> `docs/REPRODUCIBILITY.md` Path C for the full runbook.

---

## Section 6.3.1 — Tail-Risk Hierarchy under Rough Volatility

### Tables

| Thesis citation | Repo file(s) | How to reproduce |
|---|---|---|
| Tab. 5 (Sec 6.3.1) Three-strategy ES_0.95, ES_0.99, std P&L, turnover, 5 seeds, λ=0 | data: `results/canonical_v2/baseline_5seeds.json` (BS, DH; 5 seeds 2024–2028) + `results/heston_pde/heston_pde_5seeds.json` (Heston PDE, plugin; 5 seeds 6024–6028); calibration in `results/heston_pde/calibration_data.json` | `python -m deep_hedging.experiments.canonical_rerun` (canonical baseline; ~9 hours full); `python -m deep_hedging.experiments.heston_pde_evaluation` (Heston PDE; ~10 s — calibration cached) |

### Figures

| Thesis citation | Repo file(s) | How to reproduce |
|---|---|---|
| Fig. 24 (Sec 6.3.1) Terminal P&L distributions under canonical rough Bergomi (seed 2024 reference) | image: `latex_package/figures/6_3_1_pnl_histograms.png`; data: `results/canonical_v2/baseline_seed2024_pnl_{bs,dh,heston}.npy`; script: `scripts/regenerate_section6_figures.py` | `python scripts/regenerate_section6_figures.py` |
| Fig. 25 (Sec 6.3.1) Q-Q plots of standardised P&L vs N(0,1) | image: `latex_package/figures/6_3_1_qq_plots.png`; data: same NPY arrays; script: same | `python scripts/regenerate_section6_figures.py` |
| Fig. 26 (Sec 6.3.1) Five-seed risk metrics across the three strategies | image: `latex_package/figures/6_3_1_metrics_bar.png`; data: `results/canonical_v2/baseline_5seeds.json` + `results/heston_pde/heston_pde_5seeds.json` | `python scripts/regenerate_section6_figures.py` |

### Headline numerical claims

| Claim | JSON keypath | Thesis value | Verified |
|---|---|---|---|
| 5-seed DH ES_0.95 (canonical, λ=0) | `results/canonical_v2/baseline_5seeds.json:aggregated["0.0"].es95_dh` | 10.4442 ± 0.0748 | yes |
| 5-seed BS delta ES_0.95 (canonical, λ=0) | same:`aggregated["0.0"].es95_bs` | 11.5921 ± 0.0316 | yes |
| 5-seed Heston PDE ES_0.95 | `results/heston_pde/heston_pde_5seeds.json:aggregated.heston_pde.es_95` | 13.4470 ± 0.0857 | yes |
| 5-seed plug-in delta ES_0.95 | same:`aggregated.plugin.es_95` | 15.4475 ± 0.1072 | yes |
| 5-seed DH ES_0.99 | `results/canonical_v2/baseline_5seeds.json:aggregated["0.0"].es99_dh` | 19.0444 ± 0.3560 | yes |
| 5-seed BS delta ES_0.99 | same:`aggregated["0.0"].es99_bs` | 21.8757 ± 0.1636 | yes |
| 5-seed Heston PDE ES_0.99 | `results/heston_pde/heston_pde_5seeds.json:aggregated.heston_pde.es_99` | 19.3160 ± 0.2286 | yes |
| 5-seed std P&L (BS) | `results/canonical_v2/baseline_5seeds.json:aggregated["0.0"].std_pl_bs` | 4.1492 ± 0.0312 | yes |
| 5-seed std P&L (DH) | same:`aggregated["0.0"].std_pl_dh` | 4.1415 ± 0.0295 | yes |
| 5-seed std P&L (Heston PDE) | `results/heston_pde/heston_pde_5seeds.json:aggregated.heston_pde.std_pnl` | 4.8078 ± 0.0245 | yes |
| BS turnover (Heston-PDE comparison run) | `results/heston_pde/heston_pde_5seeds.json:aggregated.bs.turnover` | 2.7167 ± 0.0026 | yes |
| Heston PDE turnover | same:`aggregated.heston_pde.turnover` | 6.2626 ± 0.0046 | yes |
| Plug-in delta turnover | same:`aggregated.plugin.turnover` | 8.7698 ± 0.0088 | yes |
| Heston calibration κ | `results/heston_pde/calibration_data.json:heston_params.kappa` | 1.0 | yes |
| Heston calibration σ_v | same:`heston_params.sigma_v` | 0.5538 | yes |
| Heston calibration θ = V_0 | same:`heston_params.theta` | 0.0552 | yes |
| Feller slack (calibration) | same:`feller_slack` | −0.196 (negative; full-truncation Euler handles it) | yes |
| 5-seed DH ES_0.95 at λ=0.001 | `results/canonical_v2/baseline_5seeds.json:aggregated["0.001"].es95_dh` | 10.6658 ± 0.0389 | yes |
| 5-seed BS delta ES_0.95 at λ=0.001 | same:`aggregated["0.001"].es95_bs` | 12.0082 ± 0.0324 | yes |
| 5-seed DH advantage Γ (λ=0) | same:`aggregated["0.0"].gamma` | 1.1479 ± 0.0761 | yes |

Verification one-liner (most-cited number):

    python -c "import json; d = json.load(open('results/canonical_v2/baseline_5seeds.json')); print(d['aggregated']['0.0']['es95_dh'])"

---

## Section 6.3.2 — Risk Objective is the Principal Lever

### Tables

| Thesis citation | Repo file(s) | How to reproduce |
|---|---|---|
| Tab. 6 (Sec 6.3.2) Worst-case ES_0.95 by training objective under axis-aligned PGD (5 objectives × 3 radii) | data: `results/perturbation_v2/M5_objective_robustness.json:results.<obj>.aggregate_per_radius` | `python -m deep_hedging.experiments.perturbation_extended --M5` (long; or `--repro-M5` for a single seed) |

### Figures

| Thesis citation | Repo file(s) | How to reproduce |
|---|---|---|
| Fig. 27 (Sec 6.3.2) Pareto front of training objectives | image: `latex_package/figures/6_3_2_pareto_objectives.png`; data: `archive/legacy_figures_data/pareto_part_A_results.json` (intermediate) and `pareto_part_B_results.json` produced by `pareto_front.py` | `python -m deep_hedging.experiments.pareto_front --part B` (B is the objective comparison) |
| Fig. 28 (Sec 6.3.2) Worst-case ES_0.95 across PGD by objective and radius | image: `latex_package/figures/6_3_2_objective_robustness.png`; data: `results/perturbation_v2/M5_objective_robustness.json`; script: `scripts/regenerate_section6_figures.py` | `python scripts/regenerate_section6_figures.py` |

### Numerical claims (Sec 6.3.2, Tab. 6)

| objective | r=1 | r=2 | r=3 |
|---|---|---|---|
| ES_0.90 | 14.8426 ± 0.5869 | 19.1595 ± 1.6568 | 21.4042 ± 1.2633 |
| ES_0.95 | 14.7380 ± 0.6302 | 19.1337 ± 1.8863 | 21.4655 ± 1.5926 |
| ES_0.99 | 14.9043 ± 0.4588 | 18.9177 ± 1.3470 | 21.3117 ± 1.0650 |
| MSE | 15.2892 ± 0.5029 | 19.5253 ± 1.4029 | 21.7476 ± 1.1423 |
| Entropic | 18.8043 ± 1.9248 | 21.8233 ± 1.9102 | 23.5681 ± 1.5060 |

JSON keypath template: `results/perturbation_v2/M5_objective_robustness.json:results.<obj>.aggregate_per_radius.<radius>` where `<obj> ∈ {es_090, es_095, es_099, mse, entropic}` and `<radius> ∈ {"1", "2", "3"}`.

---

## Section 6.3.3 — Roughness Is Not the Source of the Advantage

### Figures

| Thesis citation | Repo file(s) | How to reproduce |
|---|---|---|
| Fig. 29 (Sec 6.3.3) Roughness sweep `Γ(H)` log-log with bootstrap CI | image: `latex_package/figures/6_3_3_h_sweep.png`; data: produced by `h_sweep.py` + `h_sweep_analysis.py`; intermediate JSON in `archive/legacy_figures_data/h_sweep_results.json` | `python -m deep_hedging.experiments.h_sweep` then `python -m deep_hedging.experiments.h_sweep_analysis` |
| Fig. 30 (Sec 6.3.3) 2-D `(H, η)` heatmap of DH ES_0.95 | image: `latex_package/figures/6_3_3_H_eta_grid.png`; data: `archive/legacy_figures_data/h2_grid_extension.json`; script: `scripts/regenerate_section6_figures.py` | `python -m deep_hedging.experiments.h2_grid_extension` (data) → `python scripts/regenerate_section6_figures.py` (figure) |

### Numerical claims (Sec 6.3.3)

| Claim | Source | Value |
|---|---|---|
| Panel-OLS slope `β̂` for `log Γ(H) = α + β log H` | `archive/legacy_figures_data/h_sweep_results.json` (computed via `scripts/compute_eta_h_variance_ratio.py` + `h_sweep_analysis.py` bootstrap) | 0.014 ± 0.022 |
| `\|β̂\| / β_noise` (signal-to-noise) | derived | 0.021 (deep in noise floor) |
| std_η / std_H ratio | computed by `scripts/compute_eta_h_variance_ratio.py` from `archive/legacy_figures_data/h2_grid_extension.json` | ≈ 13.0 |

Compute command:

    python scripts/compute_eta_h_variance_ratio.py

---

## Section 6.3.4 — Cross-Model Transferability

### Tables

| Thesis citation | Repo file(s) | How to reproduce |
|---|---|---|
| Tab. 7 (Sec 6.3.4) Multi-source zero-shot transfer to canonical rough Bergomi | data: GBM row → `results/transfer_v2/L2_budget_sweep.json:results.gbm.160000.aggregate.es_95` (3 seeds, N=160K — see `docs/audit/TAB7_GBM_RESOLUTION.md`); Heston row → `results/transfer_v2/L1_heston_5seeds.json:results.heston.aggregate.es_95` (5 seeds); rough Bergomi H=0.3 row → `L2_budget_sweep.json:results.rbergomi_H03.160000.aggregate.es_95` (3 seeds, N=160K); BS reference → `results/canonical_v2/baseline_5seeds.json:aggregated["0.0"].es95_bs`; canonical DH → `aggregated["0.0"].es95_dh` | `python -m deep_hedging.experiments.transfer_extended --L1 --L2` (long) |
| Tab. 8 (Sec 6.3.4) Reverse transfer: canonical-rB-trained DH on Markovian targets | data: `results/transfer_v2/L4_reverse_transfer.json:results.per_target.<target>.aggregate.gap_dh_minus_ref` (NOT `gap`) for `<target> ∈ {gbm, heston}`; 3 seeds | `python -m deep_hedging.experiments.transfer_extended --L4` |

### Figures

| Thesis citation | Repo file(s) | How to reproduce |
|---|---|---|
| Fig. 31 (Sec 6.3.4) Multi-source zero-shot transfer bars | image: `latex_package/figures/6_3_4_multi_source.png`; data: `results/transfer_v2/L1_heston_5seeds.json` + `L1_multi_source_5seeds.json`; script: `scripts/regenerate_section6_figures.py` | `python scripts/regenerate_section6_figures.py --only-multi-source` |
| Fig. 32 (Sec 6.3.4) Reverse transfer bars: rB-trained on GBM and Heston targets | image: `latex_package/figures/6_3_4_reverse_transfer.png`; data: `results/transfer_v2/L4_reverse_transfer.json`; script: `scripts/regenerate_section6_figures.py` | `python scripts/regenerate_section6_figures.py` |

### Numerical claims (Tab. 7 + Tab. 8)

| Claim | JSON keypath | Value |
|---|---|---|
| GBM source ES_0.95 (3 seeds, N=160K) | `results/transfer_v2/L2_budget_sweep.json:results.gbm.160000.aggregate.es_95` | 11.0791 ± 0.0106 |
| Heston source ES_0.95 (5 seeds) | `results/transfer_v2/L1_heston_5seeds.json:results.heston.aggregate.es_95` | 10.4431 ± 0.0256 |
| rough Bergomi H=0.3 source ES_0.95 (3 seeds, N=160K) | `results/transfer_v2/L2_budget_sweep.json:results.rbergomi_H03.160000.aggregate.es_95` | 10.5488 ± 0.0353 |
| Reverse: rB → GBM gap (DH − BS reference) | `results/transfer_v2/L4_reverse_transfer.json:results.per_target.gbm.aggregate.gap_dh_minus_ref` | +2.0676 ± 0.0083 |
| Reverse: rB → Heston gap (DH − Heston PDE reference) | `results/transfer_v2/L4_reverse_transfer.json:results.per_target.heston.aggregate.gap_dh_minus_ref` | −2.1051 ± 0.1057 |
| Reverse: rB → GBM DH ES_0.95 | same:`per_target.gbm.aggregate.dh_es95` | 3.9541 ± 0.0300 |
| Reverse: rB → GBM BS reference ES_0.95 | same:`per_target.gbm.aggregate.ref_es95` | 1.8865 ± 0.0321 |
| Reverse: rB → Heston DH ES_0.95 | same:`per_target.heston.aggregate.dh_es95` | 7.3681 ± 0.0720 |
| Reverse: rB → Heston PDE reference ES_0.95 | same:`per_target.heston.aggregate.ref_es95` | 9.4732 ± 0.0674 |

Verification one-liner (Tab. 7 GBM source — the row that needed
disambiguation in Phase 1.5):

    python -c "import json; d = json.load(open('results/transfer_v2/L2_budget_sweep.json')); print(d['results']['gbm']['160000']['aggregate']['es_95'])"

---

## Section 6.3.5 — Frequency-Cost Interaction and Parameter Perturbation

### Tables

| Thesis citation | Repo file(s) | How to reproduce |
|---|---|---|
| Tab. 9 (Sec 6.3.5) Kendall τ between rebalancing frequency `n` and ES_0.95, 7 cost levels | data: `archive/legacy_figures_data/h2_grid_extension.json` (the script also accepts a local `figures/h2_grid_extension.json` if a fresh `h2_grid_extension` run has been done first; resolution is automatic, no manual `cp` needed) | `python scripts/compute_kendall_tau_h2.py` |
| Tab. 10 (Sec 6.3.5) Worst-case ES_0.95 under axis-aligned PGD on `(H, η, ρ)`, 7 radii | data: `results/perturbation_v2/M1_extended_radius.json:results.eta.+.<radius>.aggregate` (worst direction is `eta+` at every radius — recorded in `crossover_analysis`) | `python -m deep_hedging.experiments.perturbation_extended --M1` |

### Figures

| Thesis citation | Repo file(s) | How to reproduce |
|---|---|---|
| Fig. 33 (Sec 6.3.5) Frequency-cost grid: BS and Leland deltas across `(n, λ)` cells | image: `latex_package/figures/6_3_5_frequency_cost.png`; data: `archive/legacy_figures_data/h2_grid_extension.json` | `python -m deep_hedging.experiments.h2_grid_extension` (regenerates the JSON; the figure regen path is currently inside that script) |
| Fig. 34 (Sec 6.3.5) Worst-case ES_0.95 under PGD as a function of attack radius `r` | image: `latex_package/figures/6_3_5_extended_radius.png`; data: `results/perturbation_v2/M1_extended_radius.json`; script: `scripts/regenerate_section6_figures.py` | `python scripts/regenerate_section6_figures.py` |

### Numerical claims (Tab. 9 + Tab. 10)

#### Tab. 9 — Kendall `τ_n` between rebalancing frequency and ES_0.95

(7 cost levels; values from `scripts/compute_kendall_tau_h2.py` over `archive/legacy_figures_data/h2_grid_extension.json`):

| λ | τ_n^BS | n*_BS |
|---|---|---|
| 0 | −1.000 | 800 |
| 5×10⁻⁴ | −1.000 | 800 |
| 10⁻³ | −1.000 | 800 |
| 2×10⁻³ | −0.867 | 400 |
| 3×10⁻³ | −0.600 | 400 |
| 5×10⁻³ | −0.067 | 100 |
| 10⁻² | +0.467 | 100 |

#### Tab. 10 — Worst-axis (η+) PGD attack at canonical rough Bergomi (5 seeds)

| radius `r` | DH ES_0.95 | BS ES_0.95 | gap (DH − BS) |
|---|---|---|---|
| 0.5 | 12.6236 ± 0.1951 | 13.7776 ± 0.1943 | −1.1540 ± 0.0813 |
| 1.0 | 14.7647 ± 0.2326 | 15.9172 ± 0.2449 | −1.1525 ± 0.0797 |
| 1.5 | 16.8884 ± 0.2749 | 17.9950 ± 0.3060 | −1.1066 ± 0.0546 |
| 2.0 | 18.8005 ± 0.3372 | 19.8323 ± 0.3789 | −1.0319 ± 0.0546 |
| 3.0 | 21.4171 ± 0.4552 | 22.1909 ± 0.4739 | −0.7738 ± 0.0312 |
| 4.0 | 22.0330 ± 0.5844 | 22.4844 ± 0.5709 | −0.4514 ± 0.0760 |
| 5.0 | 21.9219 ± 0.6279 | 22.3072 ± 0.6116 | −0.3853 ± 0.0777 |

JSON template: `results/perturbation_v2/M1_extended_radius.json:results.eta.+.<r>.aggregate.{dh_es95,bs_es95,gap}`.

Crossover analysis: `M1_extended_radius.json:crossover_analysis` reports `r* = 3.0` along the `η−` axis (the one direction where the DH advantage erodes).

---

## Section 6.3.6 — Synthesis (no new tables/figures)

The synthesis section restates Observations 6.1–6.5; all numerical
claims trace back to Sections 6.3.1 through 6.3.5.

---

## Appendix A — Reproducibility, Random Seeds, and Master Test Set

### Tables

| Thesis citation | Repo file(s) | How to reproduce |
|---|---|---|
| Tab. 11 (App A.1) η = 0 degenerate-control per-seed | `results/eta_zero_v2/eta_zero_5seeds.json` (per_seed + aggregated; seeds 4024–4028) | `python -m deep_hedging.experiments.eta_zero_control` |
| Tab. 12 (App A.4) Deep-hedger architecture and training hyperparameters | textual only — embedded in `deep_hedging/experiments/canonical_rerun.py` constants `DATASET_KW` (line 39), `EPOCHS = 200` (line ~42), `PATIENCE = 30`; `DeepHedgerFNN(input_dim=4, hidden_dim=128, n_res_blocks=2)` per `_training_helpers.py` lines 100–102 | n/a (text) |
| Tab. 13 (App A.5) Per-seed canonical baseline at λ=0 (5 seeds 2024–2028) | `results/canonical_v2/baseline_5seeds.json:per_seed["2024"]…["2028"]["0.0"].{es95_bs, es95_dh}` | Tab. 13 reads from the same JSON as Tab. 5; per-seed values are at `results/canonical_v2/baseline_5seeds.json:per_seed["{seed}"]["0.0"]`. Re-running the full canonical sweep recreates this JSON via `python -m deep_hedging.experiments.canonical_rerun` (~9 h). |
| Tab. 14 (App A.5) Per-seed Heston PDE delta (5 seeds 6024–6028) | `results/heston_pde/heston_pde_5seeds.json:per_seed["6024"]…["6028"].heston_pde.es_95` | `python -m deep_hedging.experiments.heston_pde_evaluation` |
| Tab. 15 (App A.7) Computational environment | textual only — actual values in `requirements.txt` and the App A.7 prose | n/a (text) |

### Figures

| Thesis citation | Repo file(s) | How to reproduce |
|---|---|---|
| Fig. 35 (App A.1) Per-seed `Γ_arch` on η=0 control | image: `latex_package/figures/appB_eta_zero_gamma_arch.png`; data: `results/eta_zero_v2/eta_zero_5seeds.json:per_seed.{seed}.gamma_arch`; script: `deep_hedging.experiments.eta_zero_control` (figure-write at end of run) or aggregate-only mode | `python -m deep_hedging.experiments.eta_zero_control --aggregate-only` |
| Fig. 36 (App A.1) η=0 P&L distribution (seed 4024) | image: `latex_package/figures/appB_eta_zero_pl_histogram.png`; data: `results/eta_zero_v2/seed_4024_withpnl_pnl_{bs,dh}.npy` (PnL arrays kept on disk for plotting); script: same as Fig. 35 | `python -m deep_hedging.experiments.eta_zero_control --aggregate-only --pnl-seed 4024` |
| Fig. 37 (App A.2) Six-panel parameter-perturbation diagnostics | image: `latex_package/figures/appB_perturbation_comprehensive.png`; data: `results/perturbation_v2/M{1..6}_*.json`; script: built into `perturbation_extended.py --all` | `python -m deep_hedging.experiments.perturbation_extended --all` |
| Fig. 38 (App A.3) Five-panel transfer-learning diagnostics | image: `latex_package/figures/appB_transfer_comprehensive.png`; data: `results/transfer_v2/L{1..5}_*.json`; script: built into `transfer_extended.py --all` | `python -m deep_hedging.experiments.transfer_extended --all` |

### Numerical claims (Appendix A)

#### App A.1 — η=0 degenerate control

| Claim | JSON keypath | Value |
|---|---|---|
| Γ_arch (architecture-and-objective contribution) | `results/eta_zero_v2/eta_zero_5seeds.json:aggregated.gamma_arch` | 0.2334 ± 0.0078 |
| BS ES_0.95 at η=0 (analytical replicating, 5 seeds) | same:`aggregated.es95_bs` | 1.8830 ± 0.0792 |
| DH ES_0.95 at η=0 (5 seeds) | same:`aggregated.es95_dh` | 1.6496 ± 0.0770 |
| MC premium p_0 at η=0 | same:`aggregated.p0` | 9.3440 ± 0.0734 |
| Analytical BS price at σ=0.235 (reference) | computed inline | ≈ 9.36 |

Decomposition (App A.1 narrative — `decomposition_5seeds.json`):

| Claim | JSON keypath | Value |
|---|---|---|
| Γ_total absolute | `results/canonical_v2/decomposition_5seeds.json:aggregated.absolute.Gamma_total` | 0.8542 ± 0.0363 |
| Γ_objective absolute | same:`absolute.Gamma_objective` | 0.5240 ± 0.1264 |
| Γ_architecture absolute | same:`absolute.Gamma_architecture` | 0.0120 ± 0.0299 |
| Γ_stoch_vol absolute | same:`absolute.Gamma_stoch_vol` | 0.0744 ± 0.0343 |
| Γ_roughness absolute | same:`absolute.Gamma_roughness` | 0.0962 ± 0.0382 |
| Γ_interaction absolute | same:`absolute.Gamma_interaction_total` | 0.1477 ± 0.1383 |
| % objective | same:`percentages.objective` | 61.5 % ± 14.9 |
| % roughness | same:`percentages.roughness` | 11.2 % ± 4.5 |
| % stoch-vol | same:`percentages.stoch_vol` | 8.8 % ± 4.4 |

#### App A.2 — Detailed parameter perturbation

| Claim | JSON keypath | Value |
|---|---|---|
| Joint 3-D PGD worst-case ES_0.95 at r=2 (DH) | `results/perturbation_v2/M3_joint_attacks.json:results...` (3 seeds) | ≈ 19.41 (per thesis text) |
| Hessian top eigenvalue \|λ_1\| BS at h=0.01 | `results/perturbation_v2/M6_hessian.json:results.bs.0.01.eigenvalues` (third entry, by absolute magnitude) | 331.85 |
| Hessian top eigenvalue \|λ_1\| DH at h=0.01 | same:`results.dh.0.01.eigenvalues[0]` (largest by absolute magnitude) | 456.66 |
| Hessian top eigenvector cosine similarity (DH vs BS) | same:`comparison.top1_eigenvector_cosine_DH_BS` | 0.635 |
| Hessian top-eigenvalue ratio DH/BS | same:`comparison.top1_eigenvalue_ratio_DH_over_BS` | 6.17 (note: thesis says ≈1.4× but reports it as |456.66 / 331.85| = 1.376; the 6.17 ratio in JSON refers to a different convention — see comment in script) |
| η-axis crossover bracket | from `M2_axis_sweep.json` per-η aggregates (seed-aware) and the L4 reverse-transfer bound | between η=0.4 and η=0.9 |

#### App A.3 — Transfer learning detailed

Per-budget Heston-source DH ES_0.95 (`L2_budget_sweep.json:results.heston.<N>.aggregate.es_95`):

| N | mean ± std (3 seeds) |
|---|---|
| 5,000 | 11.9347 ± 0.2338 |
| 10,000 | 11.8899 ± 0.2549 |
| 20,000 | 11.2264 ± 0.0440 |
| 40,000 | 11.0120 ± 0.0389 |
| 80,000 | 10.4464 ± 0.0310 |
| 160,000 | 10.3954 ± 0.0223 |

Per-budget GBM-source DH ES_0.95 (`L2_budget_sweep.json:results.gbm.<N>.aggregate.es_95`):

| N | mean ± std (3 seeds) |
|---|---|
| 5,000 | 11.5365 ± 0.0438 |
| 10,000 | 11.3965 ± 0.0169 |
| 20,000 | 11.1966 ± 0.0419 |
| 40,000 | 11.1532 ± 0.0181 |
| 80,000 | 11.0818 ± 0.0513 |
| 160,000 | 11.0791 ± 0.0106 |

Catastrophic-forgetting fine-tuning curve (`L3_fine_tuning_extended.json:results.{fine_tune,from_scratch}.<n_ft>.aggregate.es_95`):

| n_ft | fine_tune ES_0.95 | from_scratch ES_0.95 |
|---|---|---|
| 0 (zero-shot baseline) | 11.0880 | 11.4818 |
| 100 | 11.6608 | 11.8615 |
| 500 | 11.8927 | 12.2788 |
| 2,000 | 12.1907 | 11.8491 |
| 10,000 | 12.0150 | 11.7853 |
| 80,000 | 11.7063 | 10.6755 |

Cross-calibration across H values (`L5_cross_calibration.json:results.per_H.<H>.aggregate.{dh_es95, bs_es95, gap_dh_minus_bs}`):

| H | DH ES_0.95 | BS ES_0.95 | gap (DH − BS) |
|---|---|---|---|
| 0.07 | 10.2456 ± 0.1482 | 11.3921 ± 0.2133 | −1.1465 ± 0.0885 |
| 0.20 | 9.3109 ± 0.1008 | 10.3712 ± 0.1062 | −1.0603 ± 0.0454 |
| 0.40 | 8.7149 ± 0.1098 | 9.5831 ± 0.1115 | −0.8682 ± 0.0484 |

Verification one-liner (Heston-source 5-seed):

    python -c "import json; print(json.load(open('results/transfer_v2/L1_heston_5seeds.json'))['results']['heston']['aggregate']['es_95'])"

#### App A.6 — Master test set protocol

| Claim | Source | Value |
|---|---|---|
| Master test set size | `results/canonical_v2/baseline_5seeds.json:meta.n_test` | 50,000 |
| Master seed | `meta.seeds[0]` | 2024 |
| Canonical Hurst H | `results/heston_pde/calibration_data.json:rbergomi_params.H` | 0.07 |
| Canonical η | same:`rbergomi_params.eta` | 1.9 |
| Canonical ρ | same:`rbergomi_params.rho` | −0.7 |
| Canonical ξ_0 | same:`rbergomi_params.xi0` | 0.0552 (= 0.235²) |
| Canonical n_steps | same:`rbergomi_params.n_steps` | 100 |
| Canonical T | same:`rbergomi_params.T` | 1.0 |

#### App A.7 — Computational environment (Tab. 15)

| Claim | Source | Value |
|---|---|---|
| Python version pin | `requirements.txt` and `pyproject.toml` | 3.14.3 baseline; `pyproject.toml` widens to `>=3.11,<3.15` for reviewer convenience |
| PyTorch version | `requirements.txt` | 2.10.0 |
| NumPy version | `requirements.txt` | 2.4.3 |
| SciPy version | `requirements.txt` | 1.17.1 |
| RNG backend | inferred from `torch.Generator(device).manual_seed(seed)` calls in every simulator | PyTorch torch.Generator (Mersenne-Twister) |
| Hardware | App A.7 prose | macOS / Linux CPU; CUDA NOT used (deterministic kernels) |

---

## Appendix B — Code Listings

| Listing | Thesis location | Repo file (immutable per Phase 2 architectural decision) |
|---|---|---|
| Listing 1 | App. B.1 | `deep_hedging/hedging/deep_hedger.py` (lines 54–113 — `DeepHedgerFNN` class + `__init__` + `forward`) |
| Listing 2 | App. B.2 | `deep_hedging/hedging/heston_pde_delta.py` (lines 696–732 — Hundsdorfer-Verwer ADI step inside `_solve_pde`) |
| Listing 3 | App. B.3 | `deep_hedging/core/rough_bergomi.py` (lines 108–169 — `DifferentiableRoughBergomi.forward` 4-step pipeline) |
| Listing 4 | App. B.4 | `deep_hedging/core/volterra.py` (lines 159–226 — `HybridVolterraDriver.forward`) |
| Listing 5 | App. B.5 | `deep_hedging/objectives/pnl.py` (lines 36–104 — `compute_trading_gains` + `compute_transaction_costs` + `compute_hedging_pnl`) |

These five paths are cited verbatim in the published thesis Appendix B.
They are immutable for the v12 release; do not rename or move under any
reorganisation. See `docs/MATHEMATICAL_CORRESPONDENCE.md` for the
complete listing-by-line correspondence.

---

## Documentation gaps

The following thesis claims could not be resolved to a JSON keypath in
this audit pass. They are noted for potential follow-up; none invalidates
any verified number above.

- **Tab. 5 turnover for the canonical λ=0 BS/DH cells.** The
  `canonical_v2/baseline_5seeds.json:aggregated["0.0"]` block does not
  expose `turnover` directly (turnover is available only in the
  `heston_pde_5seeds.json` aggregated block, derived from a separate
  evaluator). Thesis cites `2.717 ± 0.003` (BS) and `1.777 ± 0.020`
  (DH) for Tab. 5. These numbers are present in `heston_pde_5seeds.json`
  (`aggregated.bs.turnover` = 2.7167 ± 0.0026) for BS, but the DH
  turnover was computed during a separate canonical evaluation that
  is not exposed in the GitHub-tracked aggregate. This is an
  inventory-completeness gap, not a numerical inconsistency.

- **Sec 6.3.5 fine-grained η-axis sweep crossover** between η=0.4 and
  η=0.9. The bracket is reported in the thesis text but is computed
  by inspection of `M2_axis_sweep.json` per-η cells — there is no
  pre-computed `eta_crossover` field. The crossover is verified via:

      python -c "import json; d = json.load(open('results/perturbation_v2/M2_axis_sweep.json')); print({k: d['results']['eta'][k]['aggregate'].get('gap', {}).get('mean') for k in d['results']['eta']})"

- **Sec 6.3.5 r* = 3 at η− axis** (the single failure direction).
  Reported as `crossover_analysis` in `M1_extended_radius.json`; the
  field exists but the value (`r_star = 3.0`, axis `eta−`) is the
  derived crossover, not the per-radius gap series. Both the
  per-radius gaps and the crossover summary are present in the same
  JSON, so this is not a gap — recorded here only for completeness.
