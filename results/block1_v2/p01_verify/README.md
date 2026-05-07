# Block 1.1 — Convergence Sweep Summary

Generated: 2026-04-23T14:19:07.634580+00:00
Git commit: d6e1317fdef3a9ef5523cbc4e77cefffc757d9c9

## Key findings

- Richardson rate `α̂ = 0.9130` (95% CI: [0.7224, 1.1036])
- BLP theoretical: `α_BLP = 0.57`
- Extrapolated `ES_0.95(∞) = 8.8208`
- Relative error at `n = 100`: 0.2871 (28.7%)
- Convergence flag (rel change 200 → 400 < 5%): **FAIL**

## Per-level relative changes

| From n | To n | rel_change ES_0.95 | rel_change ES_0.99 |
|--------|------|--------------------|--------------------|
|     50 |  100 | 0.179095           | 0.250307           |
|    100 |  200 | 0.079357           | 0.103425           |
|    200 |  400 | 0.067087           | 0.083182           |
|    400 |  800 | 0.047715           | 0.084758           |
|    800 | 1600 | 0.019017           | 0.002406           |

## Interpretation

The Richardson fit yields α̂ = 0.913. The relative error at n = 100 is 28.7%, and the consecutive change from n = 200 to n = 400 exceeds the 5% threshold. Convergence at n = 100 is NOT confirmed. Downstream experiments should be re-evaluated at n = 400.