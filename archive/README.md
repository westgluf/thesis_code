# Archive

Files in this directory are preserved for historical context but are
NOT part of the experiments reported in the dissertation. The active
code path lives in the top-level directories.

## Subdirectories
- `legacy_latex/` — pre-v12 LaTeX source for diff against the
  published manuscript.
- `legacy_documentation/` — staging documents from the May 2026
  Section 6 rewrite. These are replaced by `docs/THESIS_MAPPING.md`
  and `docs/REPRODUCIBILITY.md`, both of which are now committed
  under `docs/` and serve as the canonical mappings for thesis
  numerical claims and reproduction protocols respectively.
- `legacy_figures/` — figures from earlier manuscript phases or
  working research outputs that did not appear in the v12 PDF.
- `legacy_figures_data/` — `.pt` checkpoints, `.json` intermediates,
  `.tex` table fragments produced by earlier experiments.
- `legacy_simulators/` — alternative simulator implementations
  (κ=2 hybrid scheme) used during Section 5.4 validation.
- `legacy_experiments/` — earlier or superseded versions of
  experiment scripts; the canonical versions remain in
  `deep_hedging/experiments/`.
- `legacy_tests/` — tests for archived experiments.
- `legacy_src/` — modules from the GBM benchmark code (`src/`)
  that were exploratory and did not produce thesis-published numbers.

Files here are kept under git history; if you need to inspect a
development trajectory, `git log --follow archive/...` works as
expected.
