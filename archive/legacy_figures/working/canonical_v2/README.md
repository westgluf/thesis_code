# Canonical baseline figures (Section 6.3.1)

Generated: 2026-04-22T00:58:55
Seed: 2024
Script: `deep_hedging/experiments/baseline_figures_rerun.py`
Git commit: 6887b3b7a489ca862c166ac40c4ca20d8898e8e0

## Figure mapping

- `6_3_1_pnl_histograms_seed2024.png` → Figure 20 (terminal P&L histograms; BS, Heston plug-in, Deep Hedger)
- `6_3_1_qq_plots_seed2024.png` → Figure 21 (Q-Q plots vs Gaussian)
- `6_3_1_metrics_bar_seed2024.png` → Figure 22 (risk metrics bar chart + turnover panel)

## Per-seed values for this figure

- ES_0.95 BS: 11.6307
- ES_0.95 Heston (plug-in): 15.7381
- ES_0.95 DH: 10.4463
- Γ (seed 2024): +1.1844

## Aggregate across 5 seeds (from Prompt B `baseline_5seeds.json`)

- Γ = 1.1479 ± 0.0761 (mean ± std, 5 seeds)
- 95% CI: [0.9957, 1.3001]

This figure shows seed 2024 as a representative realisation. Per-seed
variation across 5 seeds is documented in Appendix B Table B.1.

## Raw P&L arrays

- `../../results/canonical_v2/baseline_seed2024_pnl_bs.npy`
- `../../results/canonical_v2/baseline_seed2024_pnl_heston.npy`
- `../../results/canonical_v2/baseline_seed2024_pnl_dh.npy`

Each is 50000 float32 values (1 per test path).