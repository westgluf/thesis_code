# Resolution: Tab. 7 GBM 3-seed source location

The dissertation Sec 6.3.4 / Tab. 7 row
`GBM (σ = 0.235, 3 seeds)` cites `11.0791 ± 0.0106`.

**Source:** `results/transfer_v2/L2_budget_sweep.json`
**Key path:** `results.gbm.160000.aggregate.es_95`

- `mean`: 11.0791
- `std`: 0.0106
- 3 seeds: `7101`, `7102`, `7103`
- `N` (training set size): 160,000

This corresponds to the largest budget point in the pretraining
budget sweep documented in Appendix A.3. The thesis treats it as
the canonical 3-seed GBM-source zero-shot transfer baseline.

## Common confusion

The intermediate file `results/transfer_v2/L1_multi_source_5seeds.json`
contains only 1 seed (7001) for the GBM source and value 11.0634.
It is NOT the source of the Tab. 7 number.

`THESIS_MAPPING.md` (built in Phase 3) must point to
`L2_budget_sweep.json` for Tab. 7 row 1.

## Verification command

    python3 -c "import json; \
        d = json.load(open('results/transfer_v2/L2_budget_sweep.json')); \
        print(d['results']['gbm']['160000']['aggregate']['es_95'])"
