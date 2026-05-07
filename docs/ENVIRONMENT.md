# Environment Specification

> Versions, hardware, RNG backend, and verification commands for
> reproducing the dissertation's numerical results.

This document complements `docs/REPRODUCIBILITY.md` (the runbook) with
environment-specific detail. It is a faithful rendering of Appendix A.7
of the dissertation, plus practical guidance for reviewer machines that
do not exactly match the canonical environment.

---

## Verbatim from Appendix A.7

| Component | Value |
|---|---|
| Python | 3.14.3 |
| PyTorch | 2.10.0 |
| NumPy | 2.4.3 |
| SciPy | 1.17.1 |
| Operating system | macOS or Linux |
| Hardware | CPU (CUDA acceleration not used) |
| RNG backend | PyTorch `torch.Generator` (Mersenne-Twister) |

The unpinned dependencies (`matplotlib >= 3.8`, `PyYAML >= 6.0`,
`tqdm >= 4.65`) do not affect numerical reproducibility — they are
plotting / I/O / progress utilities only.

The full pin list is in `requirements.txt`:

    torch==2.10.0
    numpy==2.4.3
    scipy==1.17.1
    matplotlib>=3.8
    PyYAML>=6.0
    tqdm>=4.65

The `pyproject.toml` `[project]` section widens `requires-python` to
`>=3.11,<3.15` so reviewer machines can install on Python 3.11, 3.12,
3.13, or 3.14.

---

## Verifying your environment

After `pip install -e ".[dev]" -r requirements.txt` in a fresh venv,
run:

    python -c "import sys; print(sys.version)"
    python -c "import torch; print(torch.__version__)"
    python -c "import numpy; print(numpy.__version__)"
    python -c "import scipy; print(scipy.__version__)"

Expected output (matching App A.7 exactly):

    3.14.3 …
    2.10.0
    2.4.3
    1.17.1

For the RNG backend:

    python -c "import torch; g = torch.Generator; g.manual_seed(0); print(g.initial_seed)"

Expected: `0` (the seed is round-tripped through the generator).

For a one-shot install + Section 6.2 byte-identical guard:

    python -m venv .venv && source .venv/bin/activate
    python -m pip install --upgrade pip
    pip install -e ".[dev]" -r requirements.txt
    ./tools/smoke.sh # ~14 s; PASS confirms install + math match

The `smoke.sh` step trains a single Sec 6.2 deep hedger and compares
its 5 risk metrics against an archived baseline. A `PASS: metrics not
worse than baseline` verdict confirms the install and the on-disk
hedging math are intact.

---

## Running on a different version

If the canonical Python 3.14.3 is not available on your machine:

| Reviewer environment | Reproducibility level |
|---|---|
| Python 3.14.3 + PyTorch 2.10.0 + NumPy 2.4.3 + SciPy 1.17.1 + same arch | **byte-identical** (App A.6 protocol passes) |
| Python 3.11 / 3.12 / 3.13 + same library pins, same arch | **within Monte Carlo noise** (≤ 0.001 in ES_0.95) |
| Python 3.14.3 + same pins, **different CPU arch** (Apple silicon ↔ x86_64 Intel/AMD) | **within Monte Carlo noise** (small floating-point drift in cumulative operations) |
| Newer PyTorch / NumPy than the pin (e.g. PyTorch 2.11) | **within Monte Carlo noise**, modulo any kernel changes Upstream |
| GPU (CUDA enabled) | **NOT supported** for reproducibility — disabled by design (see below) |

For a stricter reproducibility test even at non-canonical Python
versions:

    pytest deep_hedging/tests -x --tb=short -q

Expected: `114 passed` (114 tests, 117 non-fatal warnings). Any failure
indicates a real environment incompatibility.

---

## Hardware

The dissertation experiments were performed on Apple M-series silicon
(macOS) and verified to also reproduce on Linux x86_64. App A.7 of the
thesis records:

> All numerical experiments were performed on a single machine with the
> configuration of Tab. 15. […] CUDA acceleration is not used because
> the canonical experiments fit comfortably in CPU memory and because
> non-deterministic CUDA kernels would compromise the byte-identical
> reproducibility required by the seeding protocol.

Memory budget for the canonical canonical_v2 baseline (5 seeds × 2 λ × ~52 min/cell):

- Per-cell peak memory: ~1.5 GB (80,000 training paths × 100 steps ×
  4 features in float32 + the master test set in float64).
- Disk: ~50 MB per per-cell artefact (`.pt` checkpoints + `arrays_debug.npz`).

The Sec 6.2 GBM benchmark grid (400 cells, ~7 GB total) is not pushed
to GitHub — it is regenerated on demand via
`python -m src.run_benchmark_gbm_grid` (see `docs/REPRODUCIBILITY.md`
Path C). The aggregate CSVs (~4.4 MB) that feed Tabs 2-4 ARE pushed.

---

## Why CUDA is disabled

PyTorch's CUDA kernels (cuBLAS in particular) are not bit-deterministic
across runs even with `torch.manual_seed(...)`, because the parallel
reduction order in matrix multiplication varies with the CUDA stream
scheduling. App A.6 of the thesis requires byte-identical reproduction
of every numerical claim across fresh Python subprocesses — this is
incompatible with non-deterministic CUDA.

The canonical CPU runs use the deterministic single-threaded
Mersenne-Twister via `torch.Generator(device).manual_seed(seed)`. Each
simulator (rough Bergomi, GBM, Heston) creates its own `torch.Generator`
in `simulate(seed=...)` (see e.g.
`deep_hedging/core/rough_bergomi.py:simulate` lines 173–192) and seeds
it before any random draw.

If you nonetheless want to enable CUDA for speed at the cost of
byte-identical reproduction, the smoke test will likely still PASS
(metric drift is below the guard threshold), but the App A.6 byte-level
diff will fail. In that case set:

    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    python -c "import torch; torch.use_deterministic_algorithms(True, warn_only=True)"

These settings reduce CUDA non-determinism but do not eliminate it.

---

## Float-precision policy

The simulators use `float64` (per `register_buffer(..., dtype=torch.float64)`
in `gbm.py` line 35, `heston.py` line 56, `rough_bergomi.py` line 59,
`volterra.py` line 121) because the rough Bergomi kernel
`(t-s)^{H-1/2}` has H ≈ 0.07 and is highly singular near s = t —
single-precision accumulation introduces visible bias in the variance
process at the smallest grid cell.

The deep hedger network (`DeepHedgerFNN`) trains in `float32` for speed
and memory; the cast happens at the feature-construction boundary
(`build_features` line 166: `return feat.float`). This split is
intentional and verified by the App A.6 byte-identical reproducibility
checks built into `canonical_rerun.py`.

---

## Validation checklist

Before declaring a fresh install good, run all four of these in order:

1. **Imports smoke** (~1 s):

       python -c "
       import deep_hedging.core.rough_bergomi
       import deep_hedging.core.volterra
       import deep_hedging.hedging.deep_hedger
       import deep_hedging.hedging.delta_hedger
       import deep_hedging.hedging.heston_pde_delta
       import deep_hedging.objectives.pnl
       import deep_hedging.objectives.risk_measures
       import src
       print('all KEEP-bucket imports OK')
       "

2. **Test suite** (~16 min):

       pytest deep_hedging/tests -x --tb=short -q

   Expected: `114 passed, 117 warnings`.

3. **Section 6.2 smoke + guard** (~30 s combined):

       ./tools/clean.sh && ./tools/compile.sh && ./tools/smoke.sh && ./tools/guard.sh

   Expected: `PASS: metrics not worse than baseline. guard OK`.

4. **One verified canonical number** (<1 s):

       python -c "import json; d = json.load(open('results/canonical_v2/baseline_5seeds.json')); print(d['aggregated']['0.0']['es95_dh'])"

   Expected: `{'mean': 10.44421100616455, 'std': 0.07484050563936308, …}`.

A fresh install passes all four. If steps 1–3 pass but step 4 returns
a different number, the install is fine but the JSON has been
overwritten — check `git status results/canonical_v2/`.
