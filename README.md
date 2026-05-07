# Deep Hedging under Rough Volatility

> Code accompanying the MMath dissertation
> **"Deep Hedging under Rough Volatility: Robustness to Model Misspecification"**
> Nikita Degtiarenko, supervised by Dr. Huy Chau.
> University of Manchester, School of Mathematics, November 2025.

[![CI](https://github.com/westgluf/thesis_code/actions/workflows/ci.yml/badge.svg?branch=release%2Fv1.0-thesis)](https://github.com/westgluf/thesis_code/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-%3E%3D3.11%2C%3C3.15-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.10.0-EE4C2C)

---

## Abstract

This dissertation examines how well hedging European options works when
real market dynamics show rough volatility and the hedger uses models
that are not fully accurate. We compare classical delta hedging
(Black–Scholes and Heston PDE) against deep hedging — neural-network
trading strategies trained to minimise convex risk measures (Expected
Shortfall, entropic risk) — on a discrete-time market with proportional
transaction costs. The data-generating process is the rough Bergomi
model (Bayer, Friz & Gatheral 2016); experiments are organised around
four model-risk axes (structural, parameter, discretisation+cost,
path-dependence). The headline result is a **10.2 % reduction in
ES_0.95** of the deep hedger over the Black–Scholes delta on canonical
rough Bergomi paths, plus four converging lines of evidence that the
advantage is driven by the **tail-aware objective and architecture
flexibility**, not by the exploitation of rough-path structure.

## Headline result

On the canonical rough Bergomi calibration (H = 0.07, η = 1.9, ρ = −0.7,
ξ_0 = 0.235², n = 100 steps, T = 1) with master test set of 50,000 paths
under seed 2024 and zero transaction cost (5 seeds 2024–2028):

| Strategy | ES_0.95 | ES_0.99 | std P&L |
|---|---|---|---|
| Black–Scholes delta | 11.5921 ± 0.0316 | 21.8757 ± 0.1636 | 4.1492 ± 0.0312 |
| Heston PDE delta (privileged variance access) | 13.4470 ± 0.0857 | 19.3160 ± 0.2286 | 4.8078 ± 0.0245 |
| **Deep hedger** | **10.4442 ± 0.0748** | **19.0444 ± 0.3560** | **4.1415 ± 0.0295** |

A correctly-implemented Markovian Heston PDE delta — calibrated by
ATM call-price matching to the rough Bergomi process and solved via the
Hundsdorfer–Verwer ADI scheme — places **above** (worse than) the
Black–Scholes delta. Privileged information about an unobservable state
variable is not automatically beneficial when the assumed dynamics of
that state are wrong.

A deep hedger pretrained exclusively on **calibrated Heston paths** (no
rough Bergomi exposure) achieves canonical performance on rough Bergomi
to within Monte Carlo noise (10.4431 ± 0.0256 vs canonical 10.4442
± 0.0748). The advantage is **bounded** along the η → 0 axis: three
independent measurements locate the basin boundary near η ∈ [0.4, 0.9].

## Repository structure

```
thesis_code/
├── deep_hedging/ Section 6.3 + Appendices A, B (rough volatility)
│ ├── core/ Differentiable rBergomi, GBM, Heston, hybrid Volterra
│ ├── hedging/ DH FNN + BS / Heston PDE / Plug-in / Leland deltas
│ ├── objectives/ PnL, expected shortfall, entropic risk
│ ├── experiments/ KEEP-bucket Sec 6.3 experiment scripts
│ ├── tests/ 114-test pytest suite (run by CI)
│ └── utils/ Config dataclasses (RoughBergomiParams etc.)
├── src/ Section 6.2 GBM benchmark (self-contained)
├── scripts/ Figure / table regeneration utilities
├── tools/ clean / compile / smoke / guard shell wrappers
├── configs/ YAML configs for the GBM benchmark
├── latex_package/ Publication-grade LaTeX source for the v12 PDF
├── results/ Numerical outputs (JSONs, NPYs, aggregate CSVs)
├── archive/ Pre-cleanup historical / superseded files
├── docs/ Documentation suite (this README + 5 docs/ files)
└── .github/workflows/ CI (Sec 6.2 smoke + Sec 6.3 pytest)
```

The dual-package layout (`src/` for Sec 6.2, `deep_hedging/` for Sec 6.3)
is intentional and **immutable**: the thesis Appendix B Listings 1–5
cite paths under `deep_hedging/{core,hedging,objectives}/` verbatim;
moving them would invalidate the published thesis text.

## Quickstart

### 5-minute path (verify install + canonical headline number)

```bash
git clone https://github.com/westgluf/thesis_code.git
cd thesis_code
git checkout release/v1.0-thesis # until merged to main
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]" -r requirements.txt

# Verify the canonical headline number directly from the committed JSON:
python -c "import json; d = json.load(open('results/canonical_v2/baseline_5seeds.json')); print(d['aggregated']['0.0']['es95_dh'])"
# Expected: {'mean': 10.44421100616455, 'std': 0.07484050563936308, …}

# Section 6.2 byte-identical guard (~14 s):
./tools/smoke.sh
# Expected: "PASS: metrics not worse than baseline. guard OK"
```

### 30-minute path (re-run one canonical seed end-to-end)

```bash
python -m deep_hedging.experiments.canonical_rerun \
    --single-seed 2024 \
    --single-seed-output /tmp/repro_seed2024.json
# ~52 minutes on Apple M-series CPU
```

Compare the result against `results/canonical_v2/baseline_5seeds.json`'s
seed-2024 entry — under the App A.7 environment pins, the per-seed
metrics block reproduces byte-identically.

### Multi-hour path (full canonical 5-seed reproduction = Tab. 5 + Tab. 13)

```bash
python -m deep_hedging.experiments.canonical_rerun
# ~9 hours: 5 seeds × 2 λ values × ~52 min/cell
```

### Section 6.2 full grid (Tabs 2-4) — alternative routes

```bash
# Fast path: regenerate Tabs 2-4 from the committed CSV aggregates (no retraining):
python scripts/generate_section6_2_tables.py > /tmp/section6_2_tables.tex

# Slow path: re-run the full 400-cell grid from scratch (~1.5–2 h):
python -m src.run_benchmark_gbm_grid \
    --config configs/gbm_benchmark.yaml \
    --sigma-bars 0.10,0.15,0.20,0.25,0.30 \
    --lambda-costs 0,1e-4,5e-4,1e-3 \
    --seeds 0,1,2,3,4,5,6,7,8,9 \
    --training-regimes oracle,robust
python -m src.rebuild_benchmark_statistics --config configs/gbm_benchmark.yaml
```

## Reproducing the dissertation

The complete, indexed reproducibility guide is in
**[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)**. Highlights:

- **Path A** (verify against committed JSONs, seconds, no compute) — the
  recommended path for a time-pressed reviewer.
- **Path B** (re-run a single experiment from scratch, ~30 min – ~9 h
  depending on the experiment).
- **Path C** (full Section 6.2 grid, ~1.5-2 h).

For each thesis Figure / Table / numerical claim with its source JSON
keypath and reproduction command, see
**[`docs/THESIS_MAPPING.md`](docs/THESIS_MAPPING.md)**.

For each KEEP-bucket experiment script with its argparse interface and
expected outputs, see **[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)**.

## Two-package layout (intentional)

| Package | Purpose | Thesis sections |
|---|---|---|
| **`src/`** | Section 6.2 GBM benchmark (self-contained NumPy + PyTorch) | 6.2 (Tabs 2-4, Figs 22-23) |
| **`deep_hedging/`** | Section 6.3 + Appendices A, B (rough volatility, differentiable simulator) | 6.3 (Tabs 5-10, Figs 24-34), App A (Tabs 11-15, Figs 35-38), App B (Listings 1-5) |

The two packages are kept separate. The Section 6.2 benchmark code path
predates the rough-volatility code path; both are referenced verbatim
by the thesis text. They share no code; the only "common API" is the
`(S, V) = simulator.forward(...)` shape contract that all three
simulators (`GBM`, `Heston`, `DifferentiableRoughBergomi`) satisfy. Both
are registered in `pyproject.toml`:

```toml
[tool.setuptools.packages.find]
include = ["src*", "deep_hedging*"]
```

## Documentation map

| File | Purpose |
|---|---|
| **[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)** | Step-by-step instructions for every experiment in the thesis (3 reproducibility paths + per-experiment runbook + LaTeX build + common pitfalls + App A.6 byte-identical protocol) |
| **[`docs/THESIS_MAPPING.md`](docs/THESIS_MAPPING.md)** | Every figure / table / headline number → repo file with JSON keypath and reproduction command |
| **[`docs/MATHEMATICAL_CORRESPONDENCE.md`](docs/MATHEMATICAL_CORRESPONDENCE.md)** | Every Definition / Proposition / Theorem / Algorithm / Listing → code function with line ranges and 1-3 sentence implementation note |
| **[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)** | One-page entry per experiment script (what it does, argparse, outputs, runtime, tests) |
| **[`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md)** | Python / PyTorch versions, CPU-only requirement, RNG backend, validation checklist |
| `docs/audit/INVENTORY.md` | Print-only output of Phase 1 (full file classification table; not committed) |
| [`docs/audit/STABILITY_REPORT_v1.md`](docs/audit/STABILITY_REPORT_v1.md) | Pre-documentation stability audit (PASS — branch ready for documentation) |
| [`docs/audit/TAB7_GBM_RESOLUTION.md`](docs/audit/TAB7_GBM_RESOLUTION.md) | Resolution doc for the Tab. 7 GBM 3-seed source-location confusion |
| [`archive/README.md`](archive/README.md) | Policy for the `archive/` directory (pre-cleanup historical files) |

## Citation

```bibtex
@mastersthesis{Degtiarenko2025DeepHedgingRoughVolatility,
  author = {Degtiarenko, Nikita},
  title = {Deep Hedging under Rough Volatility:
                  Robustness to Model Misspecification},
  school = {University of Manchester, School of Mathematics},
  type = {{MMath} dissertation},
  year = {2025},
  month = {November},
  url = {https://github.com/westgluf/thesis_code},
  note = {Code release tag: v1.0}
}
```

## License

The code is released under the MIT License — see `pyproject.toml`
`[project].license`. The dissertation text and figures are
© Nikita Degtiarenko 2025; reused with permission.

## Acknowledgements

I am grateful to Dr. Huy Chau (University of Manchester) for the
supervision throughout this project, and to the authors of the rough
Bergomi model (Bayer, Friz & Gatheral 2016) and the deep hedging
framework (Bühler, Gonon, Teichmann & Wood 2019) on whose work this
dissertation builds.
