# Pre-Documentation Stability Report (Phase 2.5)

**Generated:** 2026-05-06 14:35
**Branch:** `release/v1.0-thesis`
**Validation environment:** Python 3.14.3, PyTorch 2.10.0, NumPy 2.4.3, SciPy 1.17.1, macOS Darwin 25.3.0 arm64 (matches App A.7 exactly)

---

## TL;DR

- Branch is **STABLE** for Phase 3, **with one note**: the local working tree contains uncommitted modifications from a prior smoke-test run (CHECK 1.1 below). The committed branch state matches `origin/release/v1.0-thesis` byte-for-byte; a fresh clone is clean.
- **Total checks:** 22 (CHECK 1.1–1.10, 2.1–2.7, 3.1–3.5).
- **PASS:** 20. **FAIL:** 0. **WARN:** 1 (CHECK 1.1 working tree dirt). **SKIPPED:** 1 (CHECK 2.7, byte-identical Section 6.3 single-seed reproduction; deferred to Phase 4 by spec).
- **Critical issues:** none.
- **Risk flags for Phase 3:** 7 (FLAG 4.1–4.7), all minor and documented below.
- **Recommended next step:** **proceed to Phase 3.** No structural defect blocks documentation work.

---

## Step 1 — Structural sanity

| # | check | status | evidence |
|---|---|---|---|
| 1.1 | Git tree clean and on correct branch | **WARN** | branch=`release/v1.0-thesis`; HEAD=; HEAD == origin/release/v1.0-thesis; **but** working tree has 4 modified `aggregate/` files (manifest_runs.json, paired_comparisons.csv, summary_rows.json, win_summary.csv) from a prior `./tools/smoke.sh` run (Phase 2 validation) and ~30 untracked `??` items (local-only artefacts that surfaced after Chunk 4C removed the `results/` blanket gitignore). A fresh `git clone` produces a clean tree. |
| 1.2 | Top-level structure reviewer-friendly | PASS | At root: `archive/ configs/ deep_hedging/ docs/ figures/ latex_package/ results/ scripts/ src/ tools/ .github/ .gitignore pyproject.toml README.md requirements.txt requirements-dev.txt BENCHMARK_PROTOCOL_6_2.md`. Note: `figures/` is a local-only empty directory (git doesn't track empty dirs; absent in fresh clones). `INVENTORY.md` is intentionally absent (Phase 1 spec was print-only). No internal-process artefacts at root. |
| 1.3 | INVENTORY.md classifications match reality | PASS-with-note | All 30 spot-checked KEEP entries present and tracked, **except** `results/block1/cholesky_v2_n500k.json` — never tracked on GitHub (Phase 1 inventory drift, not a Phase 2 cleanup error). The Sec 5.4 Cholesky KS p-value is reachable via the tracked consolidation file `results/simulator_validation_bundle/sim_validation_data.json` → `p021_cholesky.fbm_terminal.ks_pvalue`. ARCHIVE counts: legacy_experiments=14, legacy_tests=7, legacy_simulators=1, legacy_src=1, legacy_latex=1, legacy_documentation=1, legacy_figures=99, legacy_figures_data=73 (all match Phase 2 expectations including the 3 extension-archives). All 9 spot-checked DELETE entries confirmed not-tracked. |
| 1.4 | Appendix B Listing paths intact (CRITICAL) | **PASS** | All five Listing 1–5 paths tracked at thesis-cited locations: `deep_hedging/hedging/deep_hedger.py`, `deep_hedging/hedging/heston_pde_delta.py`, `deep_hedging/core/rough_bergomi.py`, `deep_hedging/core/volterra.py`, `deep_hedging/objectives/pnl.py`. |
| 1.5 | Tab. 7 GBM source resolution verifies | PASS | `results/transfer_v2/L2_budget_sweep.json` → `results.gbm.160000.aggregate.es_95` returns mean=11.0791, std=0.0106, n=3, seeds [7101,7102,7103]. Matches thesis Tab 7 GBM-source row exactly. |
| 1.6 | Ten headline thesis numbers verify | PASS | All 10 numbers match exactly to 4 decimal places — see dedicated table below. |
| 1.7 | pyproject.toml registers both packages | PASS | `include = ['src*', 'deep_hedging*']`; 11 console scripts registered (6 thesis-code-* + 5 dh-*). |
| 1.8 | requirements.txt has App A.7 pins | PASS | `torch==2.10.0`, `numpy==2.4.3`, `scipy==1.17.1` all present. requirements-dev.txt has pytest>=8.0 + ruff>=0.5. |
| 1.9 | .gitignore no longer excludes results/ or archive/ | PASS | No `results/` or `archive/` exclusion in current .gitignore. Comment block explicitly explains the choice. |
| 1.10 | CI workflow declares both jobs | PASS | `.github/workflows/ci.yml` defines `section-6-2-gbm` (smoke) and `section-6-3-rbergomi` (pytest). |

### CHECK 1.6 — Headline thesis numbers (all match)

| # | claim | source | thesis value | actual | match |
|---|---|---|---|---|---|
| 1 | DH ES_0.95 canonical (Sec 6.3.1) | `canonical_v2/baseline_5seeds.json:aggregated."0.0".es95_dh` | 10.4442 ± 0.0748 | 10.4442 ± 0.0748 | yes |
| 2 | BS ES_0.95 canonical (Sec 6.3.1) | `canonical_v2/baseline_5seeds.json:aggregated."0.0".es95_bs` | 11.5921 ± 0.0316 | 11.5921 ± 0.0316 | yes |
| 3 | Heston PDE ES_0.95 (Sec 6.3.1) | `heston_pde/heston_pde_5seeds.json:aggregated.heston_pde.es_95` | 13.4470 ± 0.0857 | 13.4470 ± 0.0857 | yes |
| 4 | DH ES_0.95 at λ=0.001 | `canonical_v2/baseline_5seeds.json:aggregated."0.001".es95_dh` | 10.6658 ± 0.0389 | 10.6658 ± 0.0389 | yes |
| 5 | Γ_arch (App A.1) | `eta_zero_v2/eta_zero_5seeds.json:aggregated.gamma_arch` | 0.2334 ± 0.0078 | 0.2334 ± 0.0078 | yes |
| 6 | Heston-source DH ES_0.95 (Tab 7) | `transfer_v2/L1_heston_5seeds.json:results.heston.aggregate.es_95` | 10.4431 ± 0.0256 | 10.4431 ± 0.0256 | yes |
| 7 | rB→GBM gap (Tab 8) | `transfer_v2/L4_reverse_transfer.json:results.per_target.gbm.aggregate.gap_dh_minus_ref` | +2.0676 ± 0.0083 | +2.0676 ± 0.0083 | yes |
| 8 | rB→Heston gap (Tab 8) | `transfer_v2/L4_reverse_transfer.json:results.per_target.heston.aggregate.gap_dh_minus_ref` | −2.1051 ± 0.1057 | −2.1051 ± 0.1057 | yes |
| 9 | α̂ convergence (Sec 5.4) | `simulator_validation_bundle/sim_validation_data.json:p01_convergence.alpha_hat` | 0.913 | 0.9130 | yes |
| 10 | Cholesky KS p (Sec 5.4) | `simulator_validation_bundle/sim_validation_data.json:p021_cholesky.fbm_terminal.ks_pvalue` (consolidation; raw block1/cholesky_v2_n500k.json was never on GitHub) | 0.926 | 0.9264 | yes |

> Note on the spec's CHECK 1.6 keypath for row 7/8: the spec wrote `results.per_target.gbm.aggregate.gap`, but the actual key in the JSON is `gap_dh_minus_ref` (a sub-dict of `aggregate`). Numbers match thesis exactly under the corrected keypath.

> Note on the spec's CHECK 1.6 keypath for row 10: the file `results/block1/cholesky_v2_n500k.json` named in the spec was never tracked on GitHub. The consolidated value lives at the path shown above and matches the thesis.

---

## Step 2 — Functional checks

| # | check | status | evidence |
|---|---|---|---|
| 2.1 | Fresh editable install | PASS | `python -m venv /tmp/thesis_venv` + `pip install -e ".[dev]" -r requirements.txt` succeeded in ~24 s (cached wheels). Versions installed: torch 2.10.0, numpy 2.4.3, scipy 1.17.1, matplotlib 3.10.9, PyYAML 6.0.3, tqdm 4.67.3 — all match App A.7 pins. No warnings. |
| 2.2 | Both packages import cleanly | PASS | All 8 import statements (`src`, `deep_hedging`, `DifferentiableRoughBergomi`, `HybridVolterraDriver`, `DeepHedgerFNN`, `BlackScholesDelta`, `HestonPDEDelta`, `compute_hedging_pnl`, `compute_payoff`, `expected_shortfall`) resolved. |
| 2.3 | All 11 console scripts resolve | PASS | All 11 entry points (6 thesis-code-* + 5 dh-*) found in PATH after `pip install -e .`. |
| 2.4 | Entry-point modules import without error (substitute for --help; experiment scripts don't use argparse) | PASS | All 8 modules (`run_section_6_3_baseline`, `eta_zero_control`, `heston_pde_evaluation`, `transfer_extended`, `perturbation_extended`, `src.run_gbm_baseline`, `src.run_benchmark_gbm_grid`, `src.tools_cli`) imported with no ImportError. **Note:** the experiment-script entry points (`run_section_6_3_baseline`, `eta_zero_control`, etc.) intentionally do not use argparse — `python -m X --help` would launch the experiment. The deeper concern (no import-time crash) is satisfied. The `src.tools_cli` script does have argparse and shows `usage: python -m src.tools_cli [-h] {clean,compile,smoke,guard}`. |
| 2.5 | pytest deep_hedging/tests | **PASS** | `114 passed, 117 warnings in 985.63 s` (16 min 25 s). Matches Phase 2 (114 passed, 117 warnings, 1099.83 s) exactly in test-count and warning-count; 10% faster wall-clock owing to warm pip caches. The 117 warnings are non-fatal `PytestReturnNotNoneWarning` (tests that `return tuple` instead of using `assert`); code-style issue, out of scope for this prompt. |
| 2.6 | Section 6.2 smoke + guard | **PASS** | `clean OK / compile OK / smoke OK / guard OK`. Guard verdict: `PASS: metrics not worse than baseline`. BS-delta and Deep hedging metrics byte-identical to baseline: BS `mean_PL=0.000452, ES_0.95=0.0210, ES_0.99=0.0301`; DH `mean_PL=0.000571, ES_0.95=0.0235, ES_0.99=0.0298`. All 5 guard metrics (std_PL, ES_loss_0.95, VaR_loss_0.95, ES_loss_0.99, VaR_loss_0.99) equal between BASE and CUR to 16 significant figures. **Section 6.2 hedging math confirmed unchanged.** |
| 2.7 | Single-seed Sec 6.3.1 reproduction | **SKIPPED** | Per spec budget. The prior `canonical_rerun.py` 5-seed run logged `total_wall_clock_s = 31,040.68` for 5 seeds × 2 λ = 10 cells, i.e. ≈52 min per cell. A single-seed `--single-seed 2024` run on the canonical (n_train=80k, n_val=20k, n_test=50k, 200 epochs) configuration would take ≈52 min for λ=0 alone, exceeding the spec's 60-min budget. Phase 4 (verification) will run this on multiple seeds. Per FLAG 4.4 in the spec, SKIPPED is an acceptable outcome here. |

### CHECK 2.7 — note (subsection per spec)

The spec called for `python -m deep_hedging.experiments.run_section_6_3_baseline --seeds 2024 --output /tmp/repro_seed2024.json`, but `run_section_6_3_baseline.py` does not use argparse — the correct invocation is `python -m deep_hedging.experiments.canonical_rerun --single-seed 2024 --single-seed-output /tmp/repro_seed2024.json`. The script supports the protocol; the only blocker is wall-clock budget.

---

## Step 3 — Cross-reference integrity

| # | check | status | evidence |
|---|---|---|---|
| 3.1 | Every `\includegraphics` in `latex_package/main.tex` resolves | PASS | 20/20 references resolved under `latex_package/figures/`. The Phase 1.5 case-bug fix (Fig 30) is in place. |
| 3.2 | KEEP-bucket experiments don't import archived modules | PASS | 15 archived module names checked (block1_cholesky_arbitrage_kappa, block1_diag_determinism, block1_preview, comparison_report, consolidate_p016_p017_repro, lean_h4_analysis, p016_rerun, p017_rerun, pareto_h2_analysis, perturbation_synthesis, run_lean_h4_sweep, signature_h_sweep, transfer_learning, transfer_synthesis, volterra_kappa2). Zero hits across `deep_hedging/experiments/*.py`. |
| 3.3 | scripts/ doesn't import archived modules | PASS | Zero hits across `scripts/*.py`. |
| 3.4 | pytest collects cleanly (no import errors) | PASS | `114 tests collected in 0.65 s`. Zero collection errors. |
| 3.5 | README.md is the legacy version (Phase 3 will rewrite) | PASS | First line: `# thesis_code` (legacy header). No sign that Phase 3 has run. |

### Note on `regenerate_section6_figures.py`

`scripts/regenerate_section6_figures.py` produces 9 figures into `latex_package/figures/`. Eight are referenced by `latex_package/main.tex`. **One — `6_3_1_strategy_comparison.png` — is produced but no longer referenced** (it was archived in Chunk 3H because the v12 `main.tex` dropped it). This is benign: re-running the script will re-create the orphan PNG in `latex_package/figures/`, which the next LaTeX build will simply ignore. Phase 3's `docs/REPRODUCIBILITY.md` may want to mention this so a reviewer doesn't worry about an "extra" file appearing.

---

## Step 4 — Risk flags for Phase 3

### FLAG 4.1 — `archive/legacy_tests/` exceeds inventory's expected count (4 → 7)

**Observation:** the Phase 1 inventory listed 4 archive-tests; Phase 2 added 3 extensions in commits (test_signature_ablation, test_transfer_learning) and (test_block1_volterra_validation). Each was added after a pytest collection or run failure due to imports of archived modules.

| archived test | reason for archive | structural consequence |
|---|---|---|
| `test_block1_diag.py` | imports `block1_diag_determinism` | tests an audit-phase determinism check, not a thesis number |
| `test_block1_volterra_validation.py` | imports `volterra_kappa2` (also tests volterra_exact, KEPT) | mixed — see below |
| `test_lean_h4_sweep.py` | imports `run_lean_h4_sweep` | tests an exploratory sweep that didn't make the paper |
| `test_pareto_h2_analysis.py` | imports `pareto_h2_analysis` | tests an exploratory analysis |
| `test_signature_ablation.py` | imports `signature_h_sweep` (also tests signature_ablation, KEPT) | mixed — see below |
| `test_transfer_learning.py` | imports archived `transfer_learning` only | tests only archived experiment |
| `validate_simulator.py` | not a pytest, no archived imports | ad-hoc simulator validation script |

**Mixed cases (test_signature_ablation.py and test_block1_volterra_validation.py):**
- **`test_signature_ablation.py`** has 5 sub-tests defined as `def test_* -> Tuple[bool, str]` (not pytest-discovered; called via the file's own `main`):
  - `test_stage_1_structure` — tests KEPT `signature_ablation` only.
  - `test_feature_importance` — tests KEPT `signature_ablation` only.
  - `test_two_tower` — uses ARCHIVED `SignatureHSweepExperiment`.
  - `test_h_sweep_single` — uses ARCHIVED `SignatureHSweepExperiment`.
  - `test_json_roundtrip` — uses ARCHIVED `SignatureHSweepExperiment`.
  → 2 of 5 sub-tests cover KEPT machinery; 3 of 5 cover ARCHIVED. Coverage of `signature_ablation.py` is now indirect (via `test_section6_numbers.py`, `test_unified_baseline.py`).
- **`test_block1_volterra_validation.py`** has 4 sub-tests; `test_kappa2_api` directly imports the archived module, while others exercise `volterra_exact` (KEPT) — but the file fails at import time because of the top-level `volterra_kappa2` import in another test function. Splitting would require code changes.

**Recommendation for Phase 3 `docs/EXPERIMENTS.md`:**
- (a) just note "test archived" for the 5 unambiguous cases; AND
- (b) for `test_signature_ablation.py` specifically, mention that `signature_ablation` (KEPT, Sec 6.3.3 source) has lost its dedicated test file. If desirable, a future cleanup pass could extract the 2 KEEP-only sub-tests (`test_stage_1_structure`, `test_feature_importance`) into a new `test_signature_ablation_keep.py`. **Not required for Phase 3.**

### FLAG 4.2 — Section 6.2 raw `runs/seed_*/` (~7 GB) is local-only

**Observation:** locally there are 401 cell directories under `results/gbm_deephedge/benchmark_6_2/runs/`, totalling ~7 GB. Only 9 small files are tracked: 6 CSV + 2 JSON aggregates + `benchmark_spec.json` (totalling 4.4 MB). A reviewer with a fresh GitHub clone has the aggregates, not the per-cell artefacts.

**Recovery options for the reviewer:**
1. **From aggregates only:** `python scripts/generate_section6_2_tables.py` reads `results/gbm_deephedge/benchmark_6_2/aggregate/{scenario_summary,seed_level_metrics}.csv` and prints LaTeX for Tabs 2-4. **No retraining required.** This is the fast path.
2. **Full regeneration:** `python -m src.run_benchmark_gbm_grid --config configs/gbm_benchmark.yaml --sigma-bars 0.10,0.15,0.20,0.25,0.30 --lambda-costs 0,1e-4,5e-4,1e-3 --seeds 0,1,2,3,4,5,6,7,8,9 --training-regimes oracle,robust` — produces the full 400-cell grid. Approximate wall-clock per Phase 2's smoke timings: 13 s/cell training × 400 + eval ≈ 1.5–2 hours on CPU.

**Recommendation for Phase 3 `docs/REPRODUCIBILITY.md`:** make Option 1 the **primary documented path** ("the dissertation Tabs 2–4 can be regenerated in seconds from the committed CSVs"). Document Option 2 as an alternative for full reproducibility verification.

### FLAG 4.3 — LaTeX build was not validated in Phase 2 or Phase 2.5

**Observation:** no `pdflatex` / `latexmk` / `xelatex` in this environment. The Phase 1.5 case-bug fix (Fig 30, lowercase→uppercase H) was textual and verified via `grep`, but the v12 PDF has not been re-built since.

**Recommendation for Phase 3 `docs/REPRODUCIBILITY.md`:** include explicit LaTeX-build instructions:

```
cd latex_package
latexmk -pdf -interaction=nonstopmode main.tex
# OR (if no latexmk):
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex # second pass for cross-refs
```

Plus a manual-validation step: open the resulting PDF and confirm Figure 30 (Sec 6.3.3, "Two-dimensional heatmap of the deep hedger's ES_0.95 across the (H, η) grid") renders correctly. This is the figure that the case-bug fix was needed for.

### FLAG 4.4 — Byte-identical reproduction (App A.6) status

**Observation:** CHECK 2.7 was SKIPPED per budget (see CHECK 2.7 note above). The strongest reproducibility evidence we have is CHECK 2.6 PASS (Section 6.2 byte-identical guard).

**Recommendation for Phase 3 `docs/REPRODUCIBILITY.md`:** document the App A.6 protocol — `python -m deep_hedging.experiments.canonical_rerun --single-seed 2024` for byte-identical seed-2024 reproduction — and note that Phase 4 will execute it on multiple seeds for verification. No documentation gap remains.

### FLAG 4.5 — `_training_helpers.py` and `experiments/__init__.py` present

**Status:** both present and tracked. `_training_helpers.py` is imported by `pareto_front.py` (KEPT). `__init__.py` is the package marker.

### FLAG 4.6 — `deep_hedging/utils/config.py` present

**Status:** present and tracked. Contains `class RoughBergomiParams`, `class DatasetConfig`, `H_SWEEP_VALUES = [0.01, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]`. All three are imported by KEEP-bucket experiments.

### FLAG 4.7 — `BENCHMARK_PROTOCOL_6_2.md` at root

**Status:** present and tracked at `BENCHMARK_PROTOCOL_6_2.md`. **Recommendation for Phase 3:** integrate into `docs/EXPERIMENTS.md` as the canonical Section 6.2 methodology spec. The root-level file can stay as a redirect or be archived.

---

## Appendix A — Branch commit log

```
ad9c658 chore: archive test_block1_volterra_validation (imports archived volterra_kappa2)
51bf7ff chore: archive tests that import archived signature_h_sweep / transfer_learning
b588dfb ci: add Section 6.3 (rough Bergomi) test job alongside existing 6.2
914a893 chore: rewrite .gitignore — keep results/ and archive/ tracked
fcd822e build: pin reproducibility-critical dependencies (Appendix A.7)
b5131e2 build: register both src and deep_hedging packages, expand metadata
ac7163b chore: archive latex_package/figures orphans (not in v12 main.tex)
f53cb83 chore: archive src/ablation_regularization (not thesis-published)
c9833ba chore: archive tests for archived experiments
4b3e79c chore: archive superseded experiment scripts
300f89a chore: archive κ=2 hybrid Volterra (deferred per kappa2_deferred.md)
1eb527f chore: archive top-level figures/ working artefacts and remove duplicates
33702a5 chore: archive Section_6_Data_Bundle staging doc (replaced by THESIS_MAPPING.md)
32afbe5 chore: archive pre-v12 LaTeX manuscript
86efb6d chore: scaffold archive directory
58df2f4 chore: remove duplicate dissertation/figures/section_6_3 (orphaned by latex_package/main.tex)
e03605a chore: remove explicitly-deprecated reduced-budget signature sweep
a92aad0 chore: remove internal audit reports and revision-phase outputs
```

(18 commits; matches Phase 2's expected count.)

## Appendix B — File-count manifest

**Total tracked files: 468.**

| top-level | tracked file count |
|---|---|
| archive/ | 198 |
| results/ | 133 |
| deep_hedging/ | 69 |
| src/ | 24 |
| latex_package/ | 22 |
| scripts/ | 8 |
| (root) | 6 |
| tools/ | 4 |
| configs/ | 2 |
| docs/ | 1 |
| .github/ | 1 |
| **total** | **468** |

Root files: `.gitignore`, `BENCHMARK_PROTOCOL_6_2.md`, `README.md`, `pyproject.toml`, `requirements-dev.txt`, `requirements.txt`. (`docs/audit/TAB7_GBM_RESOLUTION.md` is the single tracked file in `docs/`; this report becomes the second.)

## Appendix C — Raw validation outputs

### CHECK 2.5 — pytest summary line

```
114 passed, 117 warnings in 985.63s (0:16:25)
```

(Full 56 KB log at `/tmp/pytest_v25.log` in the validation environment; not committed. The 117 warnings are all `PytestReturnNotNoneWarning`, e.g. from `test_block1_volterra_validation.py`-style return-tuple test functions in `test_block1_extended_validation.py`, `test_block1_validation_n400.py`, `test_unified_baseline.py`, `test_worst_case_adversarial.py`. Code-style only; tests still pass.)

### CHECK 2.6 — smoke + guard summary

```
clean OK
compile OK
BS-delta: {'mean_PL': 0.00045150602047356236, 'std_PL': 0.009659714036677,
              'entropic': -0.0004048751673427081, 'VaR_loss_0.95': 0.015172094870377217,
              'ES_loss_0.95': 0.020993625421322094, 'VaR_loss_0.99': 0.02322837904541456,
              'ES_loss_0.99': 0.0300604936641639}
Deep hedging: {'mean_PL': 0.0005713649443350732, 'std_PL': 0.014343766495585442,
              'entropic': -0.0004688464105129242, 'VaR_loss_0.95': 0.019596224650740623,
              'ES_loss_0.95': 0.0234979297965765, 'VaR_loss_0.99': 0.02567189931869507,
              'ES_loss_0.99': 0.02980222925543785}
Note: p0 = BS price (sigma_true=0.2): 0.079656; BS(sigma_bar=0.2): 0.079656
Using baseline: results/archive/gbm_baseline_metrics_20260401_222235.json
[60-epoch training in ~14 s]
BASE: {'std_PL': 0.014343766495585442, 'ES_loss_0.95': 0.0234979297965765,
       'VaR_loss_0.95': 0.019596224650740623, 'ES_loss_0.99': 0.02980222925543785,
       'VaR_loss_0.99': 0.02567189931869507}
CUR: {'std_PL': 0.014343766495585442, 'ES_loss_0.95': 0.0234979297965765,
       'VaR_loss_0.95': 0.019596224650740623, 'ES_loss_0.99': 0.02980222925543785,
       'VaR_loss_0.99': 0.02567189931869507}

PASS: metrics not worse than baseline.
guard OK
smoke OK
```

(BASE == CUR for all 5 metrics to full precision. Section 6.2 hedging math is byte-identically reproducible after the cleanup.)

### CHECK 2.7 — SKIPPED (rationale)

Single-seed canonical Sec 6.3.1 reproduction was not run. Per prior wall-clock data (`canonical_rerun.py` 5-seed × 2-λ run took 31,040 s ≈ 52 min/cell), a single-seed two-λ run on the canonical configuration would take ≈104 min, well over the spec's 60-min budget for this check. The spec explicitly allows SKIPPED here, and FLAG 4.4 confirms Phase 4 (verification) will execute the byte-identical reproduction on multiple seeds.
