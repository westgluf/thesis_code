# Reproducibility Guide

This guide is the canonical instruction set for reproducing every
numerical claim, table, and figure in the dissertation
*"Deep Hedging under Rough Volatility: Robustness to Model Misspecification"*
(Degtiarenko 2025) from a fresh clone of `release/v1.0-thesis`.

For the per-experiment runbook (one section per script), see
`docs/EXPERIMENTS.md`. For the thesis-claim-to-JSON-keypath table, see
`docs/THESIS_MAPPING.md`.

---

## 1. Quick start (5 minutes)

    git clone https://github.com/westgluf/thesis_code.git
    cd thesis_code
    git checkout release/v1.0-thesis # until merged to main
    python -m venv .venv && source .venv/bin/activate
    python -m pip install --upgrade pip
    pip install -e ".[dev]" -r requirements.txt

    # Verify the canonical headline number (10.4442 ± 0.0748):
    python -c "import json; d = json.load(open('results/canonical_v2/baseline_5seeds.json')); print(d['aggregated']['0.0']['es95_dh'])"

    # Run the Section 6.2 byte-identical guard (~14 s):
    ./tools/smoke.sh

The smoke test reproduces a single Section 6.2 deep-hedging cell and
verifies it byte-identically against the archived baseline. A PASS
confirms that the install is correct and that no Section 6.2 hedging
math has drifted on this hardware.

---

## 2. Environment (Appendix A.7)

Required for byte-identical reproduction:

    Python 3.14.3
    PyTorch 2.10.0
    NumPy 2.4.3
    SciPy 1.17.1
    matplotlib >=3.8 (any patch in 3.x is fine)
    PyYAML >=6.0
    tqdm >=4.65
    OS / arch macOS or Linux, CPU only (CUDA NOT used)

The full pin list is in `requirements.txt`. `pyproject.toml` widens the
Python requirement to `>=3.11,<3.15` to accommodate reviewer machines
without 3.14.3 — at any other Python version + the same library pins,
results reproduce **within Monte Carlo noise** (≤ 0.001 in ES_0.95) but
**not byte-identically**.

**CUDA acceleration is deliberately disabled** because non-deterministic
CUDA kernels would defeat the byte-identical reproducibility argument
of App A.7. Use `torch.use_deterministic_algorithms(True)` is not
required for the canonical CPU-only runs.

See `docs/ENVIRONMENT.md` for verification commands and detailed
hardware notes.

---

## 3. Seed protocol (Appendix A.6)

The dissertation establishes a deterministic seeding contract:

- The **master test set** is 50,000 rough Bergomi paths under canonical
  parameters (H=0.07, η=1.9, ρ=−0.7, ξ_0=0.235², n_steps=100, T=1)
  generated with seed **2024**. Path generation is the
  `DifferentiableRoughBergomi` simulator
  (`deep_hedging/core/rough_bergomi.py` lines 22–192), seeded by
  `torch.Generator(device).manual_seed(seed)` before any random draw
  (line 187 of that file).

- The 5 canonical training seeds are **2024, 2025, 2026, 2027, 2028**
  (canonical baseline; `results/canonical_v2/baseline_5seeds.json:meta.seeds`).

- The Heston PDE comparison uses 5 seeds **6024, 6025, 6026, 6027, 6028**
  (offset by +4000 from the canonical, recorded in
  `results/heston_pde/heston_pde_5seeds.json:meta.seeds`).

- The η=0 control uses 5 seeds **4024, 4025, 4026, 4027, 4028**
  (`results/eta_zero_v2/eta_zero_5seeds.json:meta.seeds`).

- Each transfer-experiment JSON records its seeds either in
  `meta.seeds` (when a single seed band covers the whole file)
  or implicitly in the keys of the `per_seed` sub-dict of each
  result group:
    - L1 multi-source: `results.{source}.per_seed.{seed}` →
      seeds 7001–7005 for Heston source (5 seeds);
    - L2 budget sweep: `results.{source}.{budget}.per_seed.{seed}` →
      seeds 7101–7103 for budget N=160K (3 seeds), Tab. 7 GBM row;
    - L3 fine-tuning: `results.fine_tune.{n_ft}.per_seed.{seed}` →
      seeds 7201–7203;
    - L4 reverse: `results.per_target.{tgt}.per_seed.{seed}` →
      seeds 7301–7303;
    - L5 cross-cal: `results.per_H.{H}.per_seed.{seed}` →
      seeds 7401–7403.

- The perturbation experiments use bands **8001…8005** (M1, M2, M3, M4),
  **8401…8405** (M5 objective robustness), and a single seed series for
  M6 Hessian.

The `*_5seeds.json` aggregate files in
`results/{canonical_v2,heston_pde,eta_zero_v2}/` record the seed list
in `meta.seeds`. The transfer-learning and perturbation aggregates
under `results/{transfer_v2,perturbation_v2}/` record seeds implicitly
in their `per_seed` keys as listed above. Reproducibility checks are
embedded in the canonical/eta_zero/heston-PDE 5seeds aggregate files
via the `reproducibility_check` sub-block (verdict `REPRODUCIBLE`).

---

## 4. Three reproducibility paths

### Path A — Verify against committed JSON (seconds, no compute)

Every numerical claim in Section 6 and Appendix A is recorded in a
JSON file under `results/`. To verify, for example, the canonical
deep-hedging ES_0.95 (Tab. 5):

    python -c "import json; d = json.load(open('results/canonical_v2/baseline_5seeds.json')); print(d['aggregated']['0.0']['es95_dh'])"

Output: `{'mean': 10.44421100616455, 'std': 0.07484050563936308, …}`.

The full mapping of thesis claims to JSON keypaths is in
`docs/THESIS_MAPPING.md`. Path A is the **fast path** for a reviewer
who only wants to confirm that the published numbers exist in the
committed data.

To regenerate Tabs 2–4 (Section 6.2 GBM benchmark) without retraining:

    python scripts/generate_section6_2_tables.py > /tmp/section6_2_tables.tex

To regenerate the Section 6.3 figures (Figs 24, 25, 26, 27, 28, 31, 32, 34)
from the committed JSONs without retraining:

    python scripts/regenerate_section6_figures.py

(Outputs land in `latex_package/figures/` directly; do not commit changes.)

### Path B — Re-run a single experiment from scratch

For the canonical Sec 6.3.1 baseline, seed 2024 only:

    python -m deep_hedging.experiments.canonical_rerun \
        --single-seed 2024 \
        --single-seed-output /tmp/repro_seed2024.json

Wall-clock: ~52 minutes per λ-cell on Apple M-series CPU. The canonical
Tab. 5 covers λ = 0 only, so a single seed × one λ value takes
~52 min. For the full 5-seed × 2-λ canonical run (10 cells), budget
~9 hours.

For the Heston PDE delta evaluation (5 seeds; the calibration is cached):

    python -m deep_hedging.experiments.heston_pde_evaluation

Wall-clock: ≈10 s (the PDE solve and calibration are cached; the run
re-evaluates against the master test set).

For the η=0 control (5 seeds):

    python -m deep_hedging.experiments.eta_zero_control

Wall-clock: ~3 hours (5 seeds × 200 epochs).

For the Sec 6.2 single-cell smoke that validates the install:

    ./tools/smoke.sh

Wall-clock: ~14 s.

### Path C — Section 6.2 full grid

The 400-cell grid that produces Tabs 2-4 (10 seeds × 5 σ̄ × 4 λ × 2
training regimes) is **not pushed to GitHub** (~7 GB of `.pt`
checkpoints). The aggregate CSVs that already feed Tabs 2-4 ARE
committed at `results/gbm_deephedge/benchmark_6_2/aggregate/`. To
regenerate the full grid from scratch:

    python -m src.run_benchmark_gbm_grid \
        --config configs/gbm_benchmark.yaml \
        --sigma-bars 0.10,0.15,0.20,0.25,0.30 \
        --lambda-costs 0,1e-4,5e-4,1e-3 \
        --seeds 0,1,2,3,4,5,6,7,8,9 \
        --training-regimes oracle,robust

    python -m src.rebuild_benchmark_statistics \
        --config configs/gbm_benchmark.yaml

Wall-clock: ~1.5–2 hours on Apple M-series CPU. Output lands at
`results/gbm_deephedge/benchmark_6_2/runs/seed_*` and the aggregates
under `aggregate/` are rewritten.

To regenerate Tabs 2-4 LaTeX from the committed CSVs without retraining:

    python scripts/generate_section6_2_tables.py > /tmp/section6_2_tables.tex

The fast path is generally preferred — a reviewer can produce Tabs 2-4
in seconds, and the per-cell artefacts under `runs/seed_*` are not
needed for any thesis claim.

---

## 5. Per-experiment runbook

For each experiment cited by the thesis, the table below gives the
section, seeds, command, output JSON, expected runtime, and a
single-key verification. Per-script details (argparse flags, in-code
constants, test coverage) are in `docs/EXPERIMENTS.md`.

| # | Experiment | Thesis | Command | Output JSON | Runtime | Verify |
|---|---|---|---|---|---|---|
| 5.1 | Canonical 5-seed baseline | Tab. 5, Tab. 13, Figs 24–26 | `python -m deep_hedging.experiments.canonical_rerun` | `results/canonical_v2/baseline_5seeds.json` | ~9 h | `aggregated["0.0"].es95_dh.mean` ≈ 10.4442 |
| 5.2 | Heston PDE delta 5-seed | Tab. 5, Tab. 14 | `python -m deep_hedging.experiments.heston_pde_evaluation` | `results/heston_pde/heston_pde_5seeds.json` | ~10 s | `aggregated.heston_pde.es_95.mean` ≈ 13.4470 |
| 5.3 | η = 0 control | App A.1, Tab. 11, Figs 35–36 | `python -m deep_hedging.experiments.eta_zero_control` | `results/eta_zero_v2/eta_zero_5seeds.json` | ~3 h | `aggregated.gamma_arch.mean` ≈ 0.2334 |
| 5.4 | Multi-source transfer (L1) | Tab. 7 (rows: GBM/Heston/H=0.3) | `python -m deep_hedging.experiments.transfer_extended --L1` | `results/transfer_v2/L1_heston_5seeds.json` (Heston-source 5 seeds); GBM source 3 seeds = `L2_budget_sweep.json:results.gbm.160000` | ~10 h (full L1 = 5 seeds × 3 sources) | `results.heston.aggregate.es_95.mean` ≈ 10.4431 |
| 5.5 | Reverse transfer (L4) | Tab. 8 | `python -m deep_hedging.experiments.transfer_extended --L4` | `results/transfer_v2/L4_reverse_transfer.json` | ~3 h | `results.per_target.gbm.aggregate.gap_dh_minus_ref.mean` ≈ +2.0676 |
| 5.6 | Pretraining budget sweep (L2) | App A.3 | `python -m deep_hedging.experiments.transfer_extended --L2` | `results/transfer_v2/L2_budget_sweep.json` | ~12 h (3 sources × 6 budgets × 3 seeds) | `results.heston.160000.aggregate.es_95.mean` ≈ 10.3954 |
| 5.7 | Catastrophic-forgetting fine-tuning (L3) | App A.3 | `python -m deep_hedging.experiments.transfer_extended --L3` | `results/transfer_v2/L3_fine_tuning_extended.json` | ~12 h | `results.fine_tune.0.aggregate.es_95.mean` ≈ 11.0880 (zero-shot baseline) |
| 5.8 | Cross-calibration H sweep (L5) | App A.3 | `python -m deep_hedging.experiments.transfer_extended --L5` | `results/transfer_v2/L5_cross_calibration.json` | ~3 h | `results.per_H.0.07.aggregate.gap_dh_minus_bs.mean` ≈ −1.1465 |
| 5.9 | Axis-aligned PGD extended-radius (M1) | Tab. 10, Fig. 34 | `python -m deep_hedging.experiments.perturbation_extended --M1` | `results/perturbation_v2/M1_extended_radius.json` | ~30 min | `results.eta.+.2.aggregate.gap.mean` ≈ −1.0319 |
| 5.10 | Joint 3-D PGD (M3) | App A.2 | `python -m deep_hedging.experiments.perturbation_extended --M3` | `results/perturbation_v2/M3_joint_attacks.json` | ~20 min | `results...` |
| 5.11 | Targeted attacks (M4) | App A.2 | `python -m deep_hedging.experiments.perturbation_extended --M4` | `results/perturbation_v2/M4_targeted_attacks.json` | ~15 min | `results...` |
| 5.12 | Objective robustness (M5) | Tab. 6, Fig. 28 | `python -m deep_hedging.experiments.perturbation_extended --M5` | `results/perturbation_v2/M5_objective_robustness.json` | ~3 h | `results.entropic.aggregate_per_radius.1.mean` ≈ 18.8043 |
| 5.13 | Hessian eigenstructure (M6) | App A.2 | `python -m deep_hedging.experiments.perturbation_extended --M6` | `results/perturbation_v2/M6_hessian.json` | ~5 min | `results.dh.0.01.eigenvalues[0]` ≈ 456.66 |
| 5.14 | Pareto front of objectives (Part B) | Fig. 27 | `python -m deep_hedging.experiments.pareto_front --part B` | `archive/legacy_figures_data/pareto_part_B_results.json` | ~2 h | — |
| 5.15a | H sweep | Fig. 29 | `python -m deep_hedging.experiments.h_sweep` then `python -m deep_hedging.experiments.h_sweep_analysis` | `archive/legacy_figures_data/h_sweep_results.json` | ~3 h | panel-OLS slope `β̂` ≈ 0.014 |
| 5.15b | Signature ablation | Sec 6.3.3 (path-feature null) | `python -m deep_hedging.experiments.signature_ablation` | `archive/legacy_figures_data/signature_ablation_stage_1.json` | ~2 h | — |
| 5.16 | Frequency-cost grid (H2) | Tab. 9, Fig. 33 | `python -m deep_hedging.experiments.h2_grid_extension` | `archive/legacy_figures_data/h2_grid_extension.json` | ~1 h | Kendall τ at λ=10⁻² ≈ +0.467 |
| 5.17 | Section 5.4 simulator validation | Figs 20, 21 | `python -m deep_hedging.experiments.block1_convergence` and `python -m deep_hedging.experiments.block1_cholesky_v2` and `python -m deep_hedging.experiments.consolidate_sim_validation` | `results/simulator_validation_bundle/sim_validation_data.json` (the consolidated bundle) | ~30 min | `p01_convergence.alpha_hat` ≈ 0.913 |
| 5.18 | Section 6.2 GBM benchmark | Tabs 2-4, Figs 22-23 | `python -m src.run_benchmark_gbm_grid …` (full Path C above) | `results/gbm_deephedge/benchmark_6_2/aggregate/*.csv` | ~1.5-2 h | `aggregate/seed_level_metrics.csv` row σ̄=0.20, λ=0, oracle DH ≈ 0.0213 |

For the smoke test that validates the install (no thesis claim, but
required for CI):

| # | Experiment | Command | Output | Runtime |
|---|---|---|---|---|
| 5.19 | Section 6.2 single-cell smoke + guard | `./tools/smoke.sh && ./tools/guard.sh` | `results/gbm_deephedge/{metrics_bs,metrics_nn}.json` (overwritten each run); guard verdict in stdout | ~14 s + ~14 s |

---

## 6. Building the dissertation PDF

The publication-grade LaTeX source is `latex_package/main.tex`. To build:

    cd latex_package
    latexmk -pdf -interaction=nonstopmode main.tex

If `latexmk` is not available:

    cd latex_package
    pdflatex -interaction=nonstopmode main.tex
    pdflatex -interaction=nonstopmode main.tex # second pass for cross-refs
    cd ..

The thesis references 20 figures all under `latex_package/figures/`;
all 20 are present and pushed.

**Manual validation step:** open the resulting PDF and confirm Figure 30
(Sec 6.3.3) — the 2-D `(H, η)` heatmap — renders correctly. This figure
had a case-sensitivity bug (`6_3_3_h_eta_grid.png` vs
`6_3_3_H_eta_grid.png`) that was fixed in of Phase 1.5;
the fix only matters on case-sensitive filesystems (Linux / Overleaf),
not on case-insensitive macOS APFS. Build on Linux or Overleaf to
exercise this code path.

The top-level `main.tex` is the obsolete pre-v12 version; it has been
moved to `archive/legacy_latex/main_pre_v12.tex` (Phase 2 cleanup).

---

## 7. Common pitfalls

- **CUDA non-determinism.** Do not enable CUDA. The repository pins
  CPU-only execution. Setting `torch.use_deterministic_algorithms(True)`
  is not necessary for the canonical CPU runs.

- **PyTorch RNG vs NumPy RNG.** Both are seeded explicitly inside
  `_training_helpers.py:train_deep_hedger_with_objective` (lines 97–98:
  `torch.manual_seed(seed)` and `np.random.seed(seed)`). Each
  simulator additionally seeds its own `torch.Generator` for path
  generation. The canonical experiments do not depend on the global
  RNG state.

- **Float dtype.** Simulators use `float64` (lines 56–64 of
  `rough_bergomi.py`); the deep hedger's input features are cast to
  `float32` at the boundary (`build_features` line 166). This is
  intentional — the rough Bergomi paths need 64-bit precision near
  the kernel singularity, but the network itself trains in 32-bit
  for speed.

- **`regenerate_section6_figures.py` produces an extra file.** The
  script writes 9 PNGs into `latex_package/figures/`; one
  (`6_3_1_strategy_comparison.png`) is no longer referenced by the
  v12 `main.tex` — it was archived in Phase 2's Chunk 3H. The
  LaTeX build ignores unreferenced files; the orphan PNG is benign.

- **Local working tree may have untracked items.** After Chunk 4C of
  Phase 2 removed the `results/` blanket gitignore, local-only files
  (the 7 GB Sec 6.2 grid, audit-phase directories from Phase 1)
  become visible to `git status`. They are NOT on origin and a fresh
  clone does not have them. Do **not** `git add` them blindly.

- **`L1_multi_source_5seeds.json` is mis-named.** Despite the name,
  this file contains only 1 GBM-source seed (7001) and is NOT the
  source of the Tab. 7 GBM 3-seed row. The actual Tab. 7 GBM source
  is `L2_budget_sweep.json:results.gbm.160000`. See
  `docs/audit/TAB7_GBM_RESOLUTION.md`.

- **`results/block1/` is local-only.** The Sec 5.4 raw block1 JSONs
  (cholesky_v2_n500k.json, convergence_sweep.json, etc.) are not on
  GitHub. The values are reachable through the consolidation file
  `results/simulator_validation_bundle/sim_validation_data.json` —
  use that for any Sec 5.4 claim.

---

## 8. Verifying byte-identical reproduction (App A.6 protocol)

The strongest reproducibility test is byte-identical reproduction of a
single canonical seed. The protocol:

1. Verify the install matches Section 2 exactly:

       python -c "import sys, torch, numpy, scipy; print(sys.version.split[0], torch.__version__, numpy.__version__, scipy.__version__)"
       # expected: 3.14.3 2.10.0 2.4.3 1.17.1

2. Re-run a single canonical seed via Path B:

       python -m deep_hedging.experiments.canonical_rerun \
           --single-seed 2024 \
           --single-seed-output /tmp/repro_seed2024.json

3. Diff the resulting JSON's per-seed metrics block against the
   committed `results/canonical_v2/baseline_5seeds.json` per-seed entry
   for seed 2024:

       python -c "
       import json
       repro = json.load(open('/tmp/repro_seed2024.json'))
       canon = json.load(open('results/canonical_v2/baseline_5seeds.json'))['per_seed']['2024']['0.0']
       for key in ['es95_bs','es95_dh','es99_bs','es99_dh','mean_pl_dh','std_pl_dh']:
           rv = repro.get(key)
           cv = canon.get(key)
           agree = rv == cv if (rv is not None and cv is not None) else False
           print(f'{key:14s} repro={rv} canon={cv} byte_identical={agree}')
       "

4. On macOS+Apple-silicon vs Linux+x86, cross-platform byte-identical
   reproduction is **not** guaranteed by the toolchain; values match
   within Monte Carlo noise (≤ 0.001 in ES_0.95). On the same
   platform with the same library pins, byte-identical reproduction
   is expected and verified.

A separate Phase 4 verification pass will execute this on multiple
seeds and report platform-by-platform.

---

## 9. Quick reference: thesis claim → exact verification

The five most-cited headline numbers in the dissertation, with their
single-line verification commands:

```python
# DH ES_0.95 canonical (Sec 6.3.1, Tab. 5) → 10.4442 ± 0.0748
python -c "import json; d=json.load(open('results/canonical_v2/baseline_5seeds.json')); print(d['aggregated']['0.0']['es95_dh'])"

# Heston PDE ES_0.95 (Sec 6.3.1, Tab. 5) → 13.4470 ± 0.0857
python -c "import json; d=json.load(open('results/heston_pde/heston_pde_5seeds.json')); print(d['aggregated']['heston_pde']['es_95'])"

# Γ_arch (App A.1, Tab. 11) → 0.2334 ± 0.0078
python -c "import json; d=json.load(open('results/eta_zero_v2/eta_zero_5seeds.json')); print(d['aggregated']['gamma_arch'])"

# Heston-source DH (Tab. 7) → 10.4431 ± 0.0256
python -c "import json; d=json.load(open('results/transfer_v2/L1_heston_5seeds.json')); print(d['results']['heston']['aggregate']['es_95'])"

# rB → GBM gap (Tab. 8) → +2.0676 ± 0.0083
python -c "import json; d=json.load(open('results/transfer_v2/L4_reverse_transfer.json')); print(d['results']['per_target']['gbm']['aggregate']['gap_dh_minus_ref'])"
```

All five verification commands run in <1 s and require zero compute.
