# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — 2026-05-07

The thesis-defence release. Empirically reproducible to
byte-identical precision on the canonical App A.7 environment
(Python 3.14.3 + PyTorch 2.10.0 + NumPy 2.4.3 + SciPy 1.17.1, CPU only).

### Added

- `LICENSE` (MIT)
- `CITATION.cff` (Citation File Format 1.2.0)
- `CHANGELOG.md` (this file)
- `CONTRIBUTING.md`
- `.editorconfig`
- `.pre-commit-config.yaml`
- `docs/MATHEMATICAL_CORRESPONDENCE.md` — every Definition /
  Proposition / Theorem / Algorithm / Listing → code function
  with line ranges (Phase 3)
- `docs/THESIS_MAPPING.md` — every Figure / Table / headline number
  → JSON keypath with reproduction command (Phase 3)
- `docs/REPRODUCIBILITY.md` — three reproducibility paths +
  19-experiment runbook (Phase 3)
- `docs/EXPERIMENTS.md` — one-page entry per script (Phase 3)
- `docs/ENVIRONMENT.md` — Python / PyTorch versions, hardware,
  RNG (Phase 3)
- `docs/audit/INVENTORY.md` (Phase 1 print-only)
- `docs/audit/STABILITY_REPORT_v1.md` (Phase 2.5 audit)
- `docs/audit/REPRODUCIBILITY_VERIFICATION_v1.md` (Phase 4 verification audit)
- `docs/audit/TAB7_GBM_RESOLUTION.md` (Phase 1.5)
- `archive/` directory with `legacy_latex/`, `legacy_documentation/`,
  `legacy_figures/`, `legacy_simulators/`, `legacy_experiments/`,
  `legacy_tests/`, `legacy_src/` subdirectories
- Section 6.2 aggregate CSVs at
  `results/gbm_deephedge/benchmark_6_2/aggregate/` (Phase 1.5)

### Changed

- `pyproject.toml` registers both `src/` and `deep_hedging/` packages,
  pins App A.7 reproducibility-critical dependencies, declares 11
  console scripts (Prompts 2 + 5)
- `.gitignore` no longer excludes `results/` and `archive/` (Phase 2)
- `.github/workflows/ci.yml` exercises Section 6.2 smoke + Section 6.3
  pytest (Phase 2)
- `requirements.txt` documents the Python version contract (Phase 5)
- LaTeX figure 30 file-reference fix: `6_3_3_h_eta_grid.png` →
  `6_3_3_H_eta_grid.png` (Phase 1.5)
- Tab. 13 reproduction command corrected — no `--aggregate-only`
  on `canonical_rerun` (Phase 4)
- `scripts/compute_kendall_tau_h2.py` reads from `figures/` first,
  falls back to `archive/legacy_figures_data/` (Phase 4)
- Listing 2 verbatim line range corrected to 696–735 (Phase 4)
- `_cvar_loss` closure line range corrected to 268–272 (Phase 4)
- Transfer-experiment seed-band locations clarified (Phase 4)

### Removed

- 14 internal audit reports + revision-phase outputs (Phase 2)
- `DO_NOT_USE` marker (Phase 2)
- 18 obsolete `dissertation/figures/section_6_3/` files (Phase 2)
- 8 byte-identical figure duplicates (consolidated into
  `latex_package/figures/`; Phase 2)

### Verified

- **App A.6 byte-identical reproduction:** PASS across 4 single-seed
  re-runs (canonical 2024, canonical 2025, η=0 4024, Heston PDE 6024).
  51 numeric metric-keys × 4 experiments = 204 byte-identical
  comparisons, all `max abs diff = 0.00e+00`. (Phase 4 verification,
  audit at `docs/audit/REPRODUCIBILITY_VERIFICATION_v1.md`.)
- **Section 6.2 byte-identical guard:** PASS. (`./tools/smoke.sh`
  against archived baseline, all 5 metrics 0.00e+00 diff.)
- **pytest:** 114 passed, 117 warnings, ~16 min wall-clock.
- **Tab. 9 reproduction (post-fix):** PASS — 7 rows × 4 columns =
  28 values match thesis verbatim.

### Documentation gaps (acknowledged, non-blocking)

- Tab. 5 turnover for canonical λ=0 BS/DH not in
  `canonical_v2/baseline_5seeds.json` (BS turnover available via
  `heston_pde_5seeds.json:aggregated.bs.turnover`).
- Sec 6.3.5 fine-grained η-axis crossover [η=0.4, η=0.9] is
  derived by inspection of `M2_axis_sweep.json` (no pre-computed
  `eta_crossover` field).
- PluginDelta vs HestonPDEDelta split: thesis Sec 4.2.2 conflates;
  code separates (`HestonPDEDelta` for Definition 4.14; `PluginDelta`
  as a comparator).

## [Pre-1.0]

Prior to v1.0 the repository was a working development tree with
intermediate audit reports, deprecated figure regenerators, and an
unpushed Sec 6.2 7-GB benchmark grid. Pre-1.0 history is preserved
in the git log; the cleanup commits are
`a92aad0..ad9c658` (Phase 2). For a full file-by-file
classification of pre-1.0 → v1.0 changes, see
`docs/audit/INVENTORY.md` (Phase 1) and
`docs/audit/STABILITY_REPORT_v1.md` (Phase 2.5).

[1.0.0]: https://github.com/westgluf/thesis_code/releases/tag/v1.0-thesis
[Pre-1.0]: https://github.com/westgluf/thesis_code/commits/main
