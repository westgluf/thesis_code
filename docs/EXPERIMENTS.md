# Experiments Index

> One entry per KEEP-bucket entry-point script: what it does, where it
> lives, what it produces, how to invoke it, what the test status is.

This file is the **runbook**. For thesis-claim-to-script mapping, see
`docs/THESIS_MAPPING.md`. For step-by-step protocols and command
sequences, see `docs/REPRODUCIBILITY.md`.

For each script:

- **What it does** — one sentence.
- **Thesis section** — which numbered Section / Table / Figure /
  Observation it serves.
- **Inputs** — config files, seeds, in-code constants.
- **Outputs** — JSON paths, figure paths.
- **Command** — example invocation matching the actual argparse (or
  in-code constants if no argparse).
- **Wall-clock** — approximate, on Apple M-series CPU.
- **Tests** — corresponding pytest module(s); notes on archived tests
  per FLAG 4.1 of the stability report.

---

## Section 6.3 (rough Bergomi) experiments — `deep_hedging/experiments/`

### `canonical_rerun.py` (function `main`)

- **What it does**: re-runs the canonical Sec 6.3.1 baseline across 5
  seeds (2024–2028) at λ ∈ {0, 0.001} with the seeding-fix protocol,
  and includes a dual-subprocess reproducibility check (App A.6).
- **Thesis section**: Sec 6.3.1, Tab. 5, Tab. 13 (App A.5), Figs 24–26
  (via the per-seed `baseline_seed2024_full.json` extract).
- **Inputs**: in-code constants `SEEDS = [2024, 2025, 2026, 2027, 2028]`
  (line 39), `DATASET_KW = dict(n_train=80_000, n_val=20_000, n_test=50_000)`
  (line 38), `EPOCHS = 200`, `PATIENCE = 30`. Argparse:
  `--single-seed N`, `--single-seed-output PATH`,
  `--seeds-only N1 N2 …`, `--skip-reproducibility`.
- **Outputs**: `results/canonical_v2/baseline_5seeds.json` (the canonical
  5-seed aggregate); per-seed `baseline_seed_<N>.json` from
  `--single-seed`.
- **Command**: `python -m deep_hedging.experiments.canonical_rerun`
  (full); `python -m deep_hedging.experiments.canonical_rerun --single-seed 2024`
  (single seed for reproducibility check).
- **Wall-clock**: ~9 h full (10 cells × ~52 min each); ~52 min per
  single-seed × λ.
- **Tests**: `deep_hedging/tests/test_section6_numbers.py`,
  `test_unified_baseline.py`, `test_decomposition_closure.py`.

### `eta_zero_control.py` (function `main`)

- **What it does**: runs the η=0 degenerate control (rough Bergomi
  collapses to constant-volatility BS) across 5 seeds (4024–4028) to
  isolate `Γ_arch` (architecture-and-objective contribution to the DH
  advantage).
- **Thesis section**: Appendix A.1, Tab. 11, Figs 35–36.
- **Inputs**: in-code constants `SEEDS = [4024..4028]`, `H = 0.07`,
  `eta = 0`, `rho = -0.7`, `xi0 = 0.235**2`. Argparse: `--verbose`,
  `--output-dir`, `--figures-dir`, `--seeds-only`,
  `--skip-reproducibility`, `--single-seed`, `--single-seed-output`,
  `--num-threads`, `--aggregate-only` (for figure-only re-run from
  cached per-seed JSONs), `--rerun-json`, `--pnl-seed`.
- **Outputs**: `results/eta_zero_v2/eta_zero_5seeds.json`,
  `seed_<N>.json` per seed, `seed_4024_withpnl_pnl_{bs,dh}.npy`
  for Fig. 36, plus `figures/appB_eta_zero_*.png` (which were copied
  to `latex_package/figures/` for the v12 build).
- **Command**: `python -m deep_hedging.experiments.eta_zero_control`
  (full); `python -m deep_hedging.experiments.eta_zero_control --aggregate-only`
  (figures only).
- **Wall-clock**: ~3 h full; <1 min in `--aggregate-only` mode.
- **Tests**: `deep_hedging/tests/test_diagnostic_controls.py`.

### `heston_pde_evaluation.py` (function `main`)

- **What it does**: evaluates the Heston PDE delta on the canonical
  rough Bergomi master test set across 5 seeds (6024–6028). Uses the
  cached calibration in `results/heston_pde/calibration_data.json`.
- **Thesis section**: Sec 6.3.1, Tab. 5, Tab. 14.
- **Inputs**: cached calibration (κ=1, σ_v=0.554, θ=V_0=0.0552,
  ρ=−0.7); seeds 6024–6028. Argparse: `--single-seed`, `--output`,
  `--skip-reproducibility`, `--seeds-only`.
- **Outputs**: `results/heston_pde/heston_pde_5seeds.json`;
  `seed_<N>.json` per seed.
- **Command**: `python -m deep_hedging.experiments.heston_pde_evaluation`.
- **Wall-clock**: ~10 s (PDE solve cached; only the per-path delta
  evaluation re-runs).
- **Tests**: `deep_hedging/tests/test_heston_pde.py`.

### `transfer_extended.py` (function `main`)

- **What it does**: runs the 5 transfer-learning extensions L1–L5
  (multi-source / budget-sweep / fine-tuning / reverse / cross-cal).
- **Thesis section**: Sec 6.3.4, Tab. 7, Tab. 8, Fig. 31, Fig. 32,
  Fig. 38, App A.3.
- **Inputs**: argparse `--setup` (build the shared test set first),
  `--L1` / `--L2` / `--L3` / `--L4` / `--L5` / `--all`,
  `--repro-L1` / `--repro-L2` / `--repro-L3` / `--repro-L4` / `--repro-L5`
  (single-seed reproducibility checks), `--seeds`, `--budgets`,
  `--sources`, `--ft-values`, `--repro-output`.
- **Outputs**: `results/transfer_v2/L1_heston_5seeds.json`,
  `L1_multi_source_5seeds.json` (Heston multi-source meta-summary;
  GBM-source single-seed reproducibility — see note below),
  `L2_budget_sweep.json` (3 sources × 6 budgets × 3 seeds),
  `L3_fine_tuning_extended.json`, `L4_reverse_transfer.json`,
  `L5_cross_calibration.json`. Plus `figures/transfer_v2/*.png` and
  `latex_package/figures/appB_transfer_comprehensive.png`.
- **Command**: `python -m deep_hedging.experiments.transfer_extended --L1 --L4`
  (most-cited subset).
- **Wall-clock**: full `--all` ≈ 30+ h; per-extension ranges from ~3 h
  (L4) to ~12 h (L2 budget sweep).
- **Tests**: tests for `transfer_extended` are *not* archived; the
  archived `test_transfer_learning.py` referenced the older
  `transfer_learning.py` (now in `archive/legacy_experiments/`).
- **Note**: the file `L1_multi_source_5seeds.json` is **mis-named** —
  it contains only 1 GBM-source seed (7001). The Tab. 7 GBM 3-seed
  number lives in `L2_budget_sweep.json:results.gbm.160000` (see
  `docs/audit/TAB7_GBM_RESOLUTION.md`).

### `perturbation_extended.py` (function `main`)

- **What it does**: runs the 6 parameter-perturbation extensions M1–M6
  (extended-radius PGD / fine-grained axis sweep / joint 3-D PGD /
  targeted attacks / objective robustness / Hessian eigenstructure).
- **Thesis section**: Sec 6.3.5, Tab. 6 (Sec 6.3.2), Tab. 10, Fig. 28,
  Fig. 34, App A.2.
- **Inputs**: argparse `--setup`, `--M1` / `--M2` / `--M3` / `--M4`
  / `--M5` / `--M6` / `--all`, plus `--repro-M1` etc. for single-cell
  reproducibility. No `--seeds` flag — seeds are baked into the
  per-extension constants.
- **Outputs**: `results/perturbation_v2/M{1..6}_*.json` plus
  `figures/perturbation_v2/perturbation_comprehensive_summary.png` and
  `latex_package/figures/appB_perturbation_comprehensive.png`.
- **Command**: `python -m deep_hedging.experiments.perturbation_extended --M1 --M5`
  (most-cited subset).
- **Wall-clock**: full `--all` ≈ 8+ h; per-extension ranges 5 min (M6)
  to ~3 h (M5).
- **Tests**: `deep_hedging/tests/test_adversarial_robustness.py`,
  `test_worst_case_adversarial.py` (cover related primitives).

### `run_section_6_3_baseline.py` (class `Section63Experiment`, function `main`)

- **What it does**: defines the `Section63Experiment` class that
  orchestrates a single canonical-rough-Bergomi seed-and-cost-cell
  baseline (BS delta + DH + plug-in delta evaluation against the master
  test set). The `main` runs the full 5-seed × 2-λ canonical sweep.
- **Thesis section**: Sec 6.3.1.
- **Inputs**: in-code constants (canonical rough Bergomi parameters,
  seeds, dataset sizes, epochs, patience). **No argparse.** To restrict
  to a subset, edit the source-level `SEEDS` and `LAMBDA_LIST`
  constants, or use `canonical_rerun.py`'s `--seeds-only` /
  `--single-seed` flags (which wrap this class).
- **Outputs**: `results/canonical_v2/baseline_<seed>_<lambda>_full.json`
  per-cell; plus the aggregate produced by `canonical_rerun.py`.
- **Command**: prefer `python -m deep_hedging.experiments.canonical_rerun --single-seed 2024`
  (which wraps this script and exposes argparse).
- **Wall-clock**: ~52 min per cell on M-series CPU.
- **Tests**: `deep_hedging/tests/test_section6_numbers.py`,
  `test_unified_baseline.py`.

### `run_unified_baseline.py` (function `main`)

- **What it does**: an alternative single-cell unified baseline runner
  (BS + DH evaluated together with explicit-checkpoint and master-test-set
  caching). Used during development to amortise the cost of the master
  test set across multiple deep-hedger configurations.
- **Thesis section**: indirectly — feeds the canonical baseline.
- **Inputs**: argparse `--skip-train` (reuse existing checkpoint),
  `--force-regen-test-set`. Other constants in source.
- **Outputs**: `archive/legacy_figures_data/unified_baseline_results.json`
  (the JSON itself was archived in Phase 2 since `canonical_rerun.py`
  has superseded it for thesis production).
- **Command**: `python -m deep_hedging.experiments.run_unified_baseline`.
- **Wall-clock**: ~52 min full; ~10 s with `--skip-train` if
  checkpoint cached.
- **Tests**: `deep_hedging/tests/test_unified_baseline.py`.

### `h_sweep.py` (function `main`)

- **What it does**: sweeps the Hurst exponent H across 9 grid points
  in [0.01, 0.5] and trains a fresh DH at each H, recording
  `Γ(H) = ES_BS_0.95 − ES_DH_0.95` for the H-sensitivity null
  (Hypothesis H4).
- **Thesis section**: Sec 6.3.3, Fig. 29.
- **Inputs**: in-code constants `H_SWEEP_VALUES = [0.01, 0.05, 0.07,
  0.1, 0.15, 0.2, 0.3, 0.4, 0.5]` (from `deep_hedging.utils.config`).
  **No argparse.**
- **Outputs**: `archive/legacy_figures_data/h_sweep_results.json`
  (intermediate, archived because `h_sweep_analysis.py` consumes it).
- **Command**: `python -m deep_hedging.experiments.h_sweep`.
- **Wall-clock**: ~3 h.
- **Tests**: `deep_hedging/tests/test_h_sweep.py`.

### `h_sweep_analysis.py` (function `main`)

- **What it does**: post-processes `h_sweep_results.json` to fit the
  panel-OLS slope `β̂` of `log Γ(H) = α + β log H` with a 10,000-sample
  bootstrap CI; produces the `latex_package/figures/6_3_3_h_sweep.png`
  figure.
- **Thesis section**: Sec 6.3.3, Fig. 29.
- **Inputs**: argparse for I/O paths.
- **Outputs**: `latex_package/figures/6_3_3_h_sweep.png`; bootstrap
  statistics printed to stdout.
- **Command**: `python -m deep_hedging.experiments.h_sweep_analysis`
  (after `h_sweep.py` has produced the JSON).
- **Wall-clock**: ~30 s.
- **Tests**: `deep_hedging/tests/test_h_sweep_analysis.py`,
  `test_h_sweep_bootstrap.py`.

### `h2_grid_extension.py` (function `main`)

- **What it does**: builds the 6×7 frequency-cost grid (n ∈
  {25, 50, 100, 200, 400, 800} × λ ∈ {0, 5e-4, 1e-3, 2e-3, 3e-3, 5e-3, 1e-2})
  for the BS delta and Leland delta, evaluated on the master test set.
- **Thesis section**: Sec 6.3.5, Tab. 9, Fig. 33; Sec 6.3.3 Fig. 30 (the
  H × η heatmap is computed in the same script in a separate cell).
- **Inputs**: in-code grid constants. **No argparse.**
- **Outputs**: `archive/legacy_figures_data/h2_grid_extension.json`
  (intermediate) plus `latex_package/figures/6_3_3_H_eta_grid.png` and
  `6_3_5_frequency_cost.png`.
- **Command**: `python -m deep_hedging.experiments.h2_grid_extension`.
- **Wall-clock**: ~1 h.
- **Tests**: `deep_hedging/tests/test_h2_extension.py`.

### `signature_ablation.py` (function `main`)

- **What it does**: ablates the path-feature input (flat 4-d vs
  3-d truncated log-signature vs 12-d full path-feature) at three
  Hurst values to test whether path-dependent features improve the
  DH advantage (the H4 null result of Sec 6.3.3).
- **Thesis section**: Sec 6.3.3, App A.2 / Fig. 37 panel.
- **Inputs**: argparse `--stage {1,1.5,3,all}`, `--skip-stage-1-5`,
  `--n-train`, `--epochs`, `--H`, `--seed`.
- **Outputs**: `archive/legacy_figures_data/signature_ablation_stage_1.json`
  and `signature_ablation_stage_1_5.json` (intermediates).
- **Command**: `python -m deep_hedging.experiments.signature_ablation --stage 1`.
- **Wall-clock**: ~2 h per stage.
- **Tests**: **dedicated test was archived.** The previous
  `test_signature_ablation.py` is now in `archive/legacy_tests/`
  because 3 of its 5 sub-tests imported the (also archived)
  `signature_h_sweep.py` and could not be split without code changes
  — see FLAG 4.1 of `docs/audit/STABILITY_REPORT_v1.md`. Coverage
  of `signature_ablation.py` is now indirect, via
  `test_section6_numbers.py` and `test_unified_baseline.py`.

### `pareto_front.py` (function `main`)

- **What it does**: builds the Pareto front of training objectives
  (Part B) and rebalancing-frequency × cost (Part A) for the deep
  hedger.
- **Thesis section**: Sec 6.3.2, Fig. 27 (Part B = the objective
  Pareto).
- **Inputs**: argparse `--part {A,B,all}`, `--full` (4×4 grid + 8
  objectives), `--n-train`, `--n-val`, `--n-test`, `--part-a-epochs`,
  `--part-b-epochs`.
- **Outputs**: `archive/legacy_figures_data/pareto_part_A_results.json`
  + `pareto_part_B_results.json`; produces
  `latex_package/figures/6_3_2_pareto_objectives.png` indirectly via
  `regenerate_section6_figures.py`.
- **Command**: `python -m deep_hedging.experiments.pareto_front --part B`.
- **Wall-clock**: ~2 h Part B; ~6 h `--all`.
- **Tests**: `deep_hedging/tests/test_pareto_front.py`.

### `diagnostic_controls.py` (function `main`)

- **What it does**: runs the 5 diagnostic controls (A: η=0, A_prime,
  B: low-η sweep, C: objective ablation, D: H × η grid heatmap) for
  the App A.1 decomposition.
- **Thesis section**: App A.1 (decomposition narrative).
- **Inputs**: argparse `--only NAME` (e.g. `A_prime`).
- **Outputs**: contributes to `results/canonical_v2/decomposition_5seeds.json`
  via `decomposition_rerun.py`.
- **Command**: `python -m deep_hedging.experiments.diagnostic_controls --only A_prime`.
- **Wall-clock**: per-control 30 min – 3 h.
- **Tests**: `deep_hedging/tests/test_diagnostic_controls.py`.

### `decomposition_rerun.py` (function `main`)

- **What it does**: aggregates the diagnostic-controls outputs into
  the App A.1 decomposition table (Γ_total = Γ_arch + Γ_objective +
  Γ_stoch_vol + Γ_roughness + Γ_interaction).
- **Thesis section**: App A.1 (the decomposition narrative).
- **Inputs**: argparse for I/O paths.
- **Outputs**: `results/canonical_v2/decomposition_5seeds.json`,
  `decomp_seed_<N>.json` per seed.
- **Command**: `python -m deep_hedging.experiments.decomposition_rerun`.
- **Wall-clock**: ~5 h (5 seeds × 5 controls).
- **Tests**: `deep_hedging/tests/test_decomposition_closure.py`.

### `baseline_figures_rerun.py` (function `main`)

- **What it does**: re-runs the canonical baseline at seed 2024 only,
  retaining the per-path P&L arrays needed for Figs 24–25 (P&L
  histograms and Q-Q plots).
- **Thesis section**: Sec 6.3.1, Figs 24, 25.
- **Inputs**: in-code seed 2024. **No argparse.**
- **Outputs**: `results/canonical_v2/baseline_seed2024_full.json`,
  `baseline_seed2024_pnl_{bs,dh}.npy`.
- **Command**: `python -m deep_hedging.experiments.baseline_figures_rerun`.
- **Wall-clock**: ~52 min.
- **Tests**: covered by `test_unified_baseline.py`.

### `build_appendix_b.py` (function `main`)

- **What it does**: extracts the 5 Appendix B code listings from the
  source files (Listings 1–5: `DeepHedgerFNN`, HV-ADI step,
  `DifferentiableRoughBergomi.forward`, `HybridVolterraDriver.forward`,
  `compute_hedging_pnl` + helpers) into the appendix bundle.
- **Thesis section**: Appendix B.
- **Inputs**: in-code source-file paths. **No argparse.**
- **Outputs**: markdown files under `results/appendix_b_bundle/`.
- **Command**: `python -m deep_hedging.experiments.build_appendix_b`.
- **Wall-clock**: <10 s.
- **Tests**: covered by `test_section6_numbers.py` (verifies the
  bundle is producible).

### `build_decomposition.py` (function `main`)

- **What it does**: aggregates the per-seed diagnostic-control outputs
  into the per-seed decomposition (used by `decomposition_rerun.py`).
- **Thesis section**: App A.1.
- **Inputs**: in-code paths. **No argparse.**
- **Outputs**: per-seed inputs to `decomp_seed_<N>.json`.
- **Command**: invoked by `decomposition_rerun.py`; not normally run
  standalone.
- **Tests**: `test_decomposition_closure.py`.

### `build_section6_numbers.py` (function `main`)

- **What it does**: aggregates all the JSON outputs cited by Section 6
  into a single text summary used by the thesis verification tests.
- **Thesis section**: Sec 6.
- **Inputs**: in-code paths.
- **Outputs**: `archive/legacy_figures_data/section6_numbers.json` (the
  intermediate was archived in Phase 2).
- **Command**: `python -m deep_hedging.experiments.build_section6_numbers`.
- **Tests**: `test_section6_numbers.py`.

### `consolidate_repro.py` (function `main`)

- **What it does**: aggregates per-seed reproducibility-check outputs
  for the App A.5 / A.6 protocol.
- **Thesis section**: App A.5, A.6.
- **Inputs**: in-code paths. **No argparse.**
- **Outputs**: per-seed reproducibility-check sub-blocks of the
  `*_5seeds.json` files.
- **Tests**: indirect.

### `consolidate_sim_validation.py` (function `main`)

- **What it does**: consolidates the Sec 5.4 raw block1 outputs
  (convergence sweep, Cholesky benchmark, n=400 grid refinement) into
  the single `simulator_validation_bundle/sim_validation_data.json`
  used by all Sec 5.4 thesis claims.
- **Thesis section**: Sec 5.4.
- **Inputs**: in-code source paths (block1 outputs). **No argparse.**
- **Outputs**: `results/simulator_validation_bundle/sim_validation_data.json`.
- **Command**: `python -m deep_hedging.experiments.consolidate_sim_validation`.
- **Wall-clock**: <10 s.
- **Tests**: indirect via `test_block1_*.py`.

### `adversarial_robustness.py`

- **What it does**: per-direction PGD attack primitives (axis-aligned
  attacks on rough Bergomi parameter triple `(H, η, ρ)`); imported by
  `perturbation_extended.py --M1 --M2`.
- **Thesis section**: Sec 6.3.5 (M1, M2 columns of Tab. 10 & Fig. 34).
- **Tests**: `deep_hedging/tests/test_adversarial_robustness.py`.

### `worst_case_adversarial.py`

- **What it does**: 3-D joint PGD search over `(H, η, ρ)`; imported by
  `perturbation_extended.py --M3`.
- **Thesis section**: App A.2 (joint 3-D attacks panel of Fig. 37).
- **Tests**: `deep_hedging/tests/test_worst_case_adversarial.py`.

### `gradient_sensitivity.py`

- **What it does**: gradient-based sensitivity of `ES_0.95` with respect
  to the rough Bergomi simulator parameters via PyTorch autograd
  (the differentiable-simulator advantage from Sec 5.3).
- **Thesis section**: Sec 6.3.5; the projected-gradient direction at
  every step uses these gradients.
- **Tests**: indirect via the perturbation tests.

### `block1_convergence.py` (function `main`)

- **What it does**: convergence study `n ∈ {50, 100, 200, 400, 800, 1600}`
  for the hybrid rough Bergomi simulator; fits a Richardson
  extrapolation to estimate `ES_∞` and the convergence rate `α̂`.
- **Thesis section**: Sec 5.4, Fig. 20.
- **Inputs**: argparse `--n-grid`, `--n-paths`, `--n-paths-p0`,
  `--seeds`, `--dry-run`, `--results-dir`, `--figures-dir`.
- **Outputs**: `results/block1_v2/p01_verify/convergence_sweep.json`
  (then consolidated into `simulator_validation_bundle/`).
- **Command**: `python -m deep_hedging.experiments.block1_convergence`.
- **Wall-clock**: ~30 min.
- **Tests**: `deep_hedging/tests/test_block1_convergence.py`.

### `block1_cholesky_v2.py` (function `main`)

- **What it does**: builds the exact Cholesky reference for the rough
  Bergomi driver and validates the hybrid scheme against it (KS test on
  terminal `W^H_T`, variance-path comparison, call-price comparison,
  arbitrage-free check).
- **Thesis section**: Sec 5.4, Fig. 21.
- **Inputs**: argparse for `n_steps`, `N_paths`, seeds.
- **Outputs**: `results/block1/cholesky_v2_n500k.json` (local-only;
  consolidated into `simulator_validation_bundle/sim_validation_data.json:p021_cholesky`).
- **Command**: `python -m deep_hedging.experiments.block1_cholesky_v2`.
- **Wall-clock**: ~20 min at N=500K.
- **Tests**: `deep_hedging/tests/test_block1_cholesky_v2.py`.

### `block1_extended_validation.py` (function `main`)

- **What it does**: extended validation of the rough Bergomi simulator
  at multiple parameter cells beyond the canonical (used in Sec 5.4
  preflight).
- **Thesis section**: Sec 5.4 (background, not directly cited).
- **Outputs**: `results/block1_v2/p016_5seeds.json` and per-seed
  `p016_seed7401.json` … `p016_seed7405.json`.
- **Tests**: `deep_hedging/tests/test_block1_extended_validation.py`.

### `block1_validation_n400.py` (function `main`)

- **What it does**: validates the simulator at the n=400 grid
  refinement (precursor to the convergence study).
- **Thesis section**: Sec 5.4.
- **Outputs**: contributes to `results/simulator_validation_bundle/sim_validation_data.json:p016_grid_refinement`.
- **Tests**: `deep_hedging/tests/test_block1_validation_n400.py`.

### `block1_hardware.py`

- **What it does**: helper module (no `main`) that records the hardware
  + library versions (`torch.__version__`, `numpy.__version__`,
  `platform.platform`) into the result JSONs for reproducibility
  bookkeeping. Imported by all `block1_*` scripts.
- **Thesis section**: App A.7.
- **Tests**: indirect.

### `_training_helpers.py`

- **What it does**: shared training wrapper `train_deep_hedger_with_objective`
  used by `pareto_front.py`, `signature_ablation.py`, and others.
  Provides `make_objective(name, **kwargs)` that returns the
  differentiable loss function for `'es', 'entropic', 'mse', 'mean'`.
- **Thesis section**: cross-cutting (Sec 4.3, Algorithm 2).
- **Tests**: indirect — exercised by every experiment that uses it.

---

## Section 6.2 (GBM benchmark) — `src/`

### `src/run_gbm_baseline.py` (function `main`)

- **What it does**: runs the BS-delta single-cell baseline used as the
  Sec 6.2 reference (no neural network).
- **Thesis section**: Sec 6.2.
- **Inputs**: in-code constants (canonical Sec 6.2 cell). **No
  argparse.**
- **Outputs**: `results/gbm_baseline/metrics_bs_*.json`.
- **Command**: `python -m src.run_gbm_baseline`.
- **Wall-clock**: <5 s.
- **Tests**: `tools/smoke.sh` exercises this.
- **Console script**: `thesis-code-baseline-gbm`.

### `src/train_deephedge_gbm.py` (function `main`)

- **What it does**: trains a single Sec 6.2 deep hedger on a config
  YAML.
- **Thesis section**: Sec 6.2.
- **Inputs**: config YAML (e.g. `configs/gbm_es95.yaml` or
  `configs/gbm_benchmark.yaml`) — argparse via the underlying
  `tools_cli.py`.
- **Outputs**: `results/gbm_deephedge/{metrics_bs,metrics_nn,run_cfg,
  best_state.pt,last_state.pt,feature_norm.json,arrays_debug.npz,
  hist_pl_bs_vs_nn.png,tail_metrics_bs_vs_nn.png,train_log.csv}`.
- **Command**: invoked by `tools/smoke.sh` and the benchmark grid
  runner.
- **Wall-clock**: ~14 s smoke; ~3 min full-cell.
- **Tests**: `tools/smoke.sh` + `tools/guard.sh`.
- **Console script**: `thesis-code-train-gbm`.

### `src/run_benchmark_gbm_grid.py` (function `main`)

- **What it does**: runs the full 400-cell Sec 6.2 grid (10 seeds × 5
  σ̄ × 4 λ × 2 regimes).
- **Thesis section**: Sec 6.2, Tabs 2–4.
- **Inputs**: argparse `--config`, `--sigma-bars`, `--lambda-costs`,
  `--seeds`, `--training-regimes`.
- **Outputs**: `results/gbm_deephedge/benchmark_6_2/runs/seed_*` (~7 GB
  local, NOT pushed) + `aggregate/*.csv` + `aggregate/*.json` (pushed,
  4.4 MB total).
- **Command**:

      python -m src.run_benchmark_gbm_grid \
        --config configs/gbm_benchmark.yaml \
        --sigma-bars 0.10,0.15,0.20,0.25,0.30 \
        --lambda-costs 0,1e-4,5e-4,1e-3 \
        --seeds 0,1,2,3,4,5,6,7,8,9 \
        --training-regimes oracle,robust

- **Wall-clock**: ~1.5–2 h.
- **Tests**: indirect via guard.
- **Console script**: `thesis-code-benchmark-grid-gbm`.

### `src/run_benchmark_eval_only.py` (function `main`)

- **What it does**: re-evaluates pre-trained checkpoints for additional
  σ̄ values without retraining (development helper).
- **Thesis section**: indirectly — Sec 6.2 sensitivity exploration.
- **Tests**: not directly tested; relies on guard.

### `src/rebuild_benchmark_statistics.py` (function `main`)

- **What it does**: aggregates the per-cell `metrics_*.json` files
  under `runs/seed_*/` into the `aggregate/*.csv` flat tables that
  feed `scripts/generate_section6_2_tables.py`.
- **Thesis section**: Sec 6.2, Tabs 2–4.
- **Inputs**: argparse `--config`.
- **Outputs**: `results/gbm_deephedge/benchmark_6_2/aggregate/{seed_level_metrics,paired_comparisons,win_summary,scenario_summary}.csv`.
- **Command**: `python -m src.rebuild_benchmark_statistics --config configs/gbm_benchmark.yaml`.
- **Wall-clock**: ~10 s.
- **Console script**: `thesis-code-benchmark-stats-gbm`.

### `src/tools_cli.py` (function `main`)

- **What it does**: the `clean / compile / smoke / guard` CLI that
  `tools/*.sh` shell wrappers invoke.
- **Thesis section**: CI / install validation.
- **Inputs**: argparse positional `{clean,compile,smoke,guard}`.
- **Outputs**: depends on subcommand; `smoke` writes
  `results/gbm_deephedge/`; `guard` reads it and writes a verdict to
  stdout.
- **Command**: `python -m src.tools_cli smoke`.
- **Console scripts**: `thesis-code-smoke`, `thesis-code-guard`.

---

## Top-level scripts — `scripts/`

### `scripts/regenerate_section6_figures.py`

- **What it does**: regenerates 8 of the 20 v12 figures in
  `latex_package/figures/` from the committed result JSONs without
  retraining anything.
- **Thesis section**: Figs 24, 25, 26, 27, 28, 30, 31, 32, 34.
- **Inputs**: implicit — reads from `results/canonical_v2/`,
  `results/heston_pde/`, `results/transfer_v2/`,
  `results/perturbation_v2/`, `archive/legacy_figures_data/`.
- **Outputs**: 9 PNG files into `latex_package/figures/` (the 9th,
  `6_3_1_strategy_comparison.png`, is no longer referenced by v12
  `main.tex` — see `docs/REPRODUCIBILITY.md` Section 7 pitfalls).
- **Command**: `python scripts/regenerate_section6_figures.py`
  (or `--only-multi-source` for Fig. 31 only).
- **Wall-clock**: ~10 s.

### `scripts/generate_section6_2_tables.py`

- **What it does**: produces the LaTeX source for Tabs 2 and 3 from the
  Sec 6.2 aggregate CSVs.
- **Thesis section**: Tabs 2, 3.
- **Outputs**: prints LaTeX to stdout.
- **Command**: `python scripts/generate_section6_2_tables.py > /tmp/section6_2_tables.tex`.
- **Wall-clock**: <1 s.

### `scripts/generate_5_2_figure.py`

- **What it does**: simulates GBM and Heston paths and writes Fig. 18.
- **Thesis section**: Sec 5.2, Fig. 18.
- **Outputs**: `latex_package/figures/5_2_markovian_paths.png`.
- **Command**: `python scripts/generate_5_2_figure.py`.
- **Wall-clock**: ~5 s.

### `scripts/regenerate_5_4_cholesky_ks_figure.py`

- **What it does**: regenerates Fig. 21 (Cholesky KS test) and Fig. 20
  (convergence) with finalised line styles, from
  `simulator_validation_bundle/sim_validation_data.json`.
- **Thesis section**: Sec 5.4, Figs 20, 21.
- **Outputs**: `latex_package/figures/5_4_cholesky_ks.png`,
  `5_4_convergence_alpha.png`.
- **Command**: `python scripts/regenerate_5_4_cholesky_ks_figure.py`.
- **Wall-clock**: ~5 s.

### `scripts/extract_heston_pde_pnl_seed2024.py`

- **What it does**: extracts the Heston PDE delta per-path P&L for
  seed 2024 (used by Fig. 24's overlay).
- **Thesis section**: Sec 6.3.1, Fig. 24.
- **Outputs**: `results/canonical_v2/heston_pde_pnl_seed2024.npy` (note:
  this NPY is local-only).
- **Command**: `python scripts/extract_heston_pde_pnl_seed2024.py`.
- **Wall-clock**: ~10 s (cached PDE solve).

### `scripts/run_L1_heston_5seeds.py`

- **What it does**: thin wrapper that calls
  `transfer_extended.py --L1` with the 5 Heston-source seeds.
- **Thesis section**: Sec 6.3.4, Tab. 7 (Heston row).
- **Outputs**: `results/transfer_v2/L1_heston_5seeds.json`.
- **Command**: `python scripts/run_L1_heston_5seeds.py`.
- **Wall-clock**: ~10 h.

### `scripts/compute_eta_h_variance_ratio.py`

- **What it does**: computes the std_η / std_H ratio (≈13.0) for the
  Sec 6.3.3 Observation 6.3.
- **Thesis section**: Sec 6.3.3.
- **Inputs**: reads `archive/legacy_figures_data/h2_grid_extension.json`.
- **Outputs**: prints to stdout.
- **Command**: `python scripts/compute_eta_h_variance_ratio.py`.
- **Wall-clock**: <1 s.

### `scripts/compute_kendall_tau_h2.py`

- **What it does**: computes Kendall `τ_n` between rebalancing
  frequency `n` and ES_0.95 at each cost level (Tab. 9).
- **Thesis section**: Sec 6.3.5, Tab. 9.
- **Inputs**: reads `figures/h2_grid_extension.json` if present (live
  local recompute, e.g. immediately after running
  `python -m deep_hedging.experiments.h2_grid_extension`); otherwise
  falls back to `archive/legacy_figures_data/h2_grid_extension.json`
  (the post-Phase-2 archived copy used for Tab. 9 reproduction).
  The script resolves the path internally, so a fresh clone needs no
  manual `cp` of the archived file.
- **Outputs**: prints `(λ, τ_n^BS, τ_n^Leland, n*_BS, n*_Leland)` rows
  to stdout (one row per λ ∈ {0, 5e-4, 1e-3, 2e-3, 3e-3, 5e-3, 1e-2}),
  followed by the full BS ES_0.95 grid for visual confirmation of
  the reversal.
- **Command**: `python scripts/compute_kendall_tau_h2.py`.
- **Wall-clock**: <1 s.

---

## Tests — `deep_hedging/tests/`

The KEEP-bucket test suite is 17 modules / 114 tests (per Phase 2.5
stability report). Each ARCHIVE-bucket experiment has had its tests
co-archived; 7 tests live in `archive/legacy_tests/` (4 originally
archived in Phase 2 + 3 added during the validation pass).

KEEP-bucket tests by experiment:

| Experiment | Test module(s) |
|---|---|
| `canonical_rerun.py` / `run_section_6_3_baseline.py` | `test_section6_numbers.py`, `test_unified_baseline.py` |
| `eta_zero_control.py` | `test_diagnostic_controls.py` |
| `heston_pde_evaluation.py` | `test_heston_pde.py` |
| `transfer_extended.py` | (no dedicated test — `test_transfer_learning.py` was archived because it imported the now-archived `transfer_learning.py`; coverage is indirect) |
| `perturbation_extended.py` | `test_adversarial_robustness.py`, `test_worst_case_adversarial.py` |
| `pareto_front.py` | `test_pareto_front.py` |
| `signature_ablation.py` | (no dedicated test — `test_signature_ablation.py` was archived because 3 of its 5 sub-tests imported the archived `signature_h_sweep`; coverage is indirect via `test_section6_numbers.py`) |
| `h_sweep.py` / `h_sweep_analysis.py` | `test_h_sweep.py`, `test_h_sweep_analysis.py`, `test_h_sweep_bootstrap.py` |
| `h2_grid_extension.py` | `test_h2_extension.py` |
| `decomposition_rerun.py` / `build_decomposition.py` | `test_decomposition_closure.py` |
| `block1_convergence.py` | `test_block1_convergence.py` |
| `block1_cholesky_v2.py` | `test_block1_cholesky_v2.py` |
| `block1_extended_validation.py` | `test_block1_extended_validation.py` |
| `block1_validation_n400.py` | `test_block1_validation_n400.py` |
| `deep_hedger.py` (and dependencies) | `test_deep_hedger.py`, `test_hedging_basics.py`, `test_path_features.py`, `test_rbergomi_hedging.py` |

Run all tests:

    pytest deep_hedging/tests -x --tb=short -q

Expected: `114 passed, 117 warnings in ~16 min` (per Phase 2.5
stability check). The 117 warnings are all `PytestReturnNotNoneWarning`
(legacy test functions that `return tuple` instead of using `assert`)
and are non-fatal.

---

## Archived experiments and tests

Files in `archive/legacy_*/` are preserved for historical context but
are NOT part of the experiments reported in the dissertation. See
`archive/README.md` for the policy.

The full ARCHIVE-bucket list is in the Phase 1 inventory (printed in
the Phase 1 audit response — not committed as a file). For examiner
convenience, `archive/legacy_tests/` contains 7 tests:

- `test_block1_diag.py` — tests the archived `block1_diag_determinism.py`.
- `test_block1_volterra_validation.py` — imports the archived `volterra_kappa2`.
- `test_lean_h4_sweep.py` — tests the archived `run_lean_h4_sweep.py`.
- `test_pareto_h2_analysis.py` — tests the archived `pareto_h2_analysis.py`.
- `test_signature_ablation.py` — mixed (3 of 5 sub-tests import archived
  `signature_h_sweep`).
- `test_transfer_learning.py` — tests only the archived
  `transfer_learning.py` (superseded by `transfer_extended.py`).
- `validate_simulator.py` — ad-hoc simulator validation script.
