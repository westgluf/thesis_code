# P01 Convergence Sweep Verification

Generated: 2026-04-23T15:22:19
Git commit: d6e1317fdef3a9ef5523cbc4e77cefffc757d9c9

## Summary

P01 convergence sweep was re-executed with the current post-fix code to verify
that grid-resolution scaling results (used in Section 5.3.X) are unaffected by
the Phase B seeding fix. P01 does not involve neural-network training; only
path simulation. The seeding fix affects only `DeepHedgerFNN` initialisation
and minibatch shuffling, neither of which is invoked by P01.

## Comparison

### Per-n aggregate ES_0.95

| n | Original ES_0.95 (mean) | Rerun ES_0.95 (mean) | Match? |
|---|---|---|---|
| 50 | 13.631480 | 13.631480 | ✓ |
| 100 | 11.190148 | 11.190148 | ✓ |
| 200 | 10.302130 | 10.302130 | ✓ |
| 400 | 9.610986 | 9.610986 | ✓ |
| 800 | 9.152398 | 9.152398 | ✓ |
| 1600 | 8.978350 | 8.978350 | ✓ |

### Richardson fit on ES_0.95

| Metric | Original | Rerun | Match? |
|---|---|---|---|
| α̂ | 0.912977 | 0.912977 | ✓ |
| ES_∞ | 8.820811 | 8.820811 | ✓ |
| C (prefactor) | 169.621788 | 169.621788 | ✓ |
| rel_err at n=100 | 0.287091 | 0.287091 | ✓ |
| α̂ CI (low) | 0.722362 | 0.722362 | ✓ |
| α̂ CI (high) | 1.103592 | 1.103592 | ✓ |

### Per-seed per-n sample checks

| Seed | n | Original | Rerun | Match? |
|---|---|---|---|---|
| 7001 | 100 | 10.988647 | 10.988647 | ✓ |
| 7001 | 400 | 9.660616 | 9.660616 | ✓ |
| 7001 | 1600 | 8.917392 | 8.917392 | ✓ |
| 7002 | 100 | 10.775249 | 10.775249 | ✓ |
| 7002 | 400 | 9.611070 | 9.611070 | ✓ |
| 7002 | 1600 | 8.717855 | 8.717855 | ✓ |
| 7003 | 100 | 11.806550 | 11.806550 | ✓ |
| 7003 | 400 | 9.561273 | 9.561273 | ✓ |
| 7003 | 1600 | 9.299803 | 9.299803 | ✓ |

## Verdict

**UNCHANGED** — all 6 grid resolutions × 3 seeds + Richardson fit parameters reproduce byte-identically across two fresh invocations of the code.

## Interpretation

P01 path simulation is deterministic given the seed passed explicitly to
`DifferentiableRoughBergomi.simulate(..., seed=...)`. The Phase B seeding fix
added `torch.manual_seed` + `np.random.seed` calls only before
`DeepHedgerFNN(...)` instantiation; P01 does not construct any neural network,
so no surface is exposed to the fix. The byte-identical match confirms this
expectation and validates the use of P01 numbers (α̂ ≈ 0.91, ES_∞ ≈ 8.82) in
Section 5.3.X without any caveat about the pre-fix/post-fix state.

## Source files compared

- Original: `results/block1/convergence_sweep.json` (script commit 07d52a7, wall 10.7 s)
- Rerun: `results/block1_v2/p01_verify/convergence_sweep.json` (script commit d6e1317, wall 11.6 s)