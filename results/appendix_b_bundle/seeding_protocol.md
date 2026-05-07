# Seeding Protocol

All neural-network training in the deep-hedging revision programme follows the
protocol below. This is the version installed by the Phase B seeding fix and
empirically verified for byte-identical cross-subprocess reproducibility in
Prompts B, C, D, E.

## Rule

Before every `DeepHedgerFNN(...)` instantiation, call:

```python
torch.manual_seed(seed)
np.random.seed(seed)
```

Path simulation uses a separate explicit seed via the simulator's
`torch.Generator`:

```python
sim.simulate(n_paths=..., S0=..., seed=seed)
```

In any loop that trains multiple models (multi-seed sweep, ablation, cell
runner), the two reseed lines are re-invoked at the start of each iteration
before the next `DeepHedgerFNN(...)` is constructed.

## Scope

Applied to 6 scripts / 12 call-sites:

- `deep_hedging/experiments/run_section_6_3_baseline.py` — 1 call-site
- `deep_hedging/experiments/diagnostic_controls.py` — 2 call-sites (A/A' plus Experiment C per-variant sub-seed)
- `deep_hedging/experiments/run_unified_baseline.py` — 3 call-sites
- `deep_hedging/experiments/h_sweep.py` — 1 call-site
- `deep_hedging/experiments/block1_validation_n400.py` — 1 call-site
- `deep_hedging/experiments/block1_extended_validation.py` — 4 call-sites (cells A/B/C/D)

## What the protocol guarantees

- **Byte-identical cross-subprocess reproducibility** given the same seed.
- **RNG independence**: path-simulator RNG state is separated from
  neural-network-initialisation and minibatch-shuffle RNG states; seeding
  the simulator does not leak into model init and vice versa.
- **No dependency on system entropy or process start time.**

## History

An April 2026 read-only audit of the codebase (`audit_master_report.md`)
identified that earlier experiments relied on the system-entropy initialisation
of the global `torch.default_generator` for neural-network weight
initialisation and minibatch shuffling. Two consequences:

1. **Per-seed irreproducibility** — running the same nominal `seed` twice in
   fresh Python subprocesses gave different Γ values.
2. **Order dependence** in `diagnostic_controls.py::run_experiment_C`, where
   three sub-models trained in a `for` loop over `dict.items` share the
   global RNG state and produce results that depend on Python dict iteration
   order.

The fix installed by Phase B — applied identically at all 12 call-sites —
eliminates both problems. The post-fix protocol has since been verified
across five independent experimental programmes (Prompts B, C, D, E, F/G).
See `results/appendix_b_bundle/reproducibility_verification.md` for the
cumulative evidence table.