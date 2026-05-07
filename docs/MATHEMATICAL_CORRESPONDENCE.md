# Mathematical Correspondence

> For every numbered theoretical object in the dissertation —
> Definition, Proposition, Theorem, Hypothesis, Algorithm, Listing —
> this document gives the corresponding code symbol with line ranges,
> a 1-3 sentence note on the implementation choice, and any intentional
> deviation from the text.

All line ranges refer to HEAD of `release/v1.0-thesis`. The function /
class names are the stable anchors — should the line numbers drift in a
later edit, the named symbol is what to re-locate. Where a thesis
object is purely abstract (proof-only or motivational), the row is
marked `no code (abstract; …)`.

Conventions:
- "App. C.X" refers to Appendix C of the dissertation (proofs).
- "Listing N" refers to Appendix B Listings 1–5.
- All paths under `deep_hedging/{core,hedging,objectives}/` are cited
  verbatim in the thesis Appendix B and **do not change** under any
  reorganisation.

---

## Section 2 — Mathematical Background

### 2.1 Discrete-Time Market Model

| Thesis | Code | Notes |
|---|---|---|
| Definition 2.1 (discrete-time market `(Ω, F, P)`, grid, info process, price process) | no direct code (abstract data class) | The setup is realised implicitly by every experiment script: each generates a Monte Carlo sample under a simulator (Section 5.2 / 5.3 in code) on a fixed grid. |
| Definition 2.2 (trading strategy `δ`, predictability, transaction-cost function) | `deep_hedging/hedging/deep_hedger.py::DeepHedgerFNN` (lines 54–125) for the deep-hedging case; `deep_hedging/hedging/delta_hedger.py::BlackScholesDelta` (lines 44–125) for the analytical case | `DeepHedgerFNN.forward` produces `δ_k ∈ [0,1]` via sigmoid (Remark 4.23). The single network is reused at every time step (weight sharing). |
| Definition 2.3 (terminal P&L `PL_T = -Z + p_0 + Σ δ_k (S_{k+1} - S_k) - C_T(δ)`) | `deep_hedging/objectives/pnl.py::compute_hedging_pnl` (lines 81–104) | The cost summation runs `k = 0 … n-1`, with the `δ_{-1} = 0` convention enforced by `compute_transaction_costs` lines 50–78. This matches the Bühler 2019 deep-hedging convention referenced in thesis footnote 2 of Sec 2.1. |

### 2.2 Convex and Coherent Risk Measures

| Thesis | Code | Notes |
|---|---|---|
| Definition 2.4 (convex risk measure: monotone, convex, cash-invariant) | no code (abstract) | Both KEEP-bucket implementations (`expected_shortfall`, `entropic_risk`) satisfy convexity; the proof is in App. C.2. |
| Definition 2.5 (coherent risk measure adds positive homogeneity) | no code (abstract) | `expected_shortfall` is coherent (CVaR is the standard example); `entropic_risk` is convex but not coherent. |
| Definition 2.6 (OCE / Rockafellar-Uryasev representation) | `deep_hedging/hedging/deep_hedger.py::train_deep_hedger` (lines 214–359) at lines 268–272 (the `_cvar_loss` closure) | The training loop materialises the OCE as a learnable scalar `w` (the auxiliary quantile parameter) optimised jointly with the network weights via Adam — the thesis Remark 4.20 explanation of the joint OCE optimisation. The evaluation-time ES (`risk_measures.py::expected_shortfall` lines 16–36) uses the sort-based estimator. |
| Theorem 2.1 (Universal Approximation, Hornik 1991) | no code (motivational) | Justifies the `DeepHedgerFNN` parametric family. Proof sketch in App. C.1. |
| Proposition 2.3 (parameter-class density: π_M(X) → π(X) as M → ∞) | no code (motivational) | Motivates the residual-block depth; proof in App. C.2. |
| Proposition 2.4 (consistency of empirical risk minimisation) | no code (motivational) | Proof in App. C.3. |

### 2.3 Neural Networks and Universal Approximation

| Thesis | Code | Notes |
|---|---|---|
| Definition 2.7 (feedforward neural network architecture) | `deep_hedging/hedging/deep_hedger.py::DeepHedgerFNN.__init__` (lines 82–102) and `ResidualBlock` (lines 31–51) | Layer-norm + LeakyReLU residual blocks; the canonical configuration is `hidden_dim=128, n_res_blocks=2`, ≈67k parameters (Tab. 12 of thesis). |
| Algorithm (SGD / backprop summary, Sec 2.3.4) | `deep_hedging/experiments/_training_helpers.py::train_deep_hedger_with_objective` (lines 67–120) calling `deep_hedger.py::train_deep_hedger` (lines 214–359) | Adam (lr 1e-3, weight_decay 1e-5), batch 2048, early-stopping on validation risk with patience 30. Joint optimisation of `w` and `θ` per Remark 4.20. |

---

## Section 3 — Stochastic Volatility and Rough Models

### 3.1 / 3.2 Stylised Facts and Failure of Classical Models

| Thesis | Code | Notes |
|---|---|---|
| Definition 3.1 (realised variance) | no code (data-summary statistic) | Used implicitly by the convergence study (Sec 5.4). |
| Definition 3.2 (log-volatility increments) | no code (abstract) | — |
| Definition 3.3 (second-order structure function) | no code (abstract) | — |
| Proposition 3.1 (log-log slope estimation of H) | no code (the actual scaling fit happens in `block1_convergence.py::fit_richardson` lines 187–250) | Proof in App. C.4. |
| Proposition 3.2 (BS variance is constant → m_2(h) = 0) | no code (analytical statement) | Proof in App. C.5. |
| Proposition 3.3 (Heston m_2(h) ∝ h, i.e. H_eff = 1/2) | no code (analytical statement) | Proof in App. C.6. |

### 3.3 Rough Volterra and Rough Bergomi

| Thesis | Code | Notes |
|---|---|---|
| Definition 3.6 (Volterra fractional Brownian driver `W^H_t = ∫_0^t (t-s)^{H-1/2} dW_s`) | `deep_hedging/core/volterra.py::HybridVolterraDriver.forward` (lines 159–226), specifically the singular kernel `(t-s)^{H-1/2}` realised by the Cholesky near-field at lines 191–197 + FFT far-field at lines 200–216 | This is Listing 4 of App. B.4. |
| Theorem 3.1 (Hölder regularity / `α`-Hölder for `α ∈ (0, H)`) | no code (theoretical justification for the simulator's behaviour at small H) | Proof sketch in App. C — relies on Kolmogorov-Centsov; not implemented. |
| Proposition 3.4 (Gaussianity of `W^H`) | exploited in `deep_hedging/core/volterra_exact.py::ExactCholeskyVolterra` (lines 58–166), the exact-Cholesky reference used in the Sec 5.4 KS-test validation (Fig. 21) | The exact reference uses the full fBm covariance Cholesky factor; the production simulator uses the hybrid scheme. |
| Definition 3.7 (rough variance `v_t = ξ_0 exp(η W^H_t − ½ η² t^{2H})`) | `deep_hedging/core/rough_bergomi.py::DifferentiableRoughBergomi.forward` (lines 108–169), specifically line 148 `V = xi0 * torch.exp(eta * WH - 0.5 * eta**2 * t_2H.unsqueeze(0))` | Step 2 of the 4-step forward pass; this is the variance line in Listing 3. |
| Definition 3.8 (rough Bergomi price `dS_t = S_t √v_t dB_t` with `dB = ρ dW + √(1-ρ²) dW^⊥`) | same `forward` (lines 108–169), specifically lines 154–168 (Steps 3 + 4: correlated `dB` then log-Euler price update) | Log-Euler discretisation with a `clamp(min=1e-12)` on V to avoid log of zero (line 152). |
| Proposition 3.5 (forward-variance preservation `E[v_t] = ξ_0(t)`) | no direct code (a property of Definition 3.7 baked into the simulator) | Verified empirically by the Sec 5.4 Cholesky comparison (`E[v]` columns in `results/simulator_validation_bundle/sim_validation_data.json:p021_cholesky.variance_path`). |
| Proposition 3.6 (conditional lognormality of `v_t`) | no direct code (consequence of Definition 3.7) | — |
| Proposition 3.8 (replication impossible in incomplete markets → risk-minimising hedging) | no code (motivates the framework) | Proof in App. C.7. |

### 3.4 Numerical Simulation of Rough Volatility

| Thesis | Code | Notes |
|---|---|---|
| Direct Cholesky simulation (Class I) | `deep_hedging/core/volterra_exact.py::ExactCholeskyVolterra` (lines 58–166) | Uses the full `(n+1)×(n+1)` fBm covariance matrix; computationally prohibitive for training but used as the Sec 5.4 reference (KS p ≈ 0.926). |
| Hybrid scheme (Class II, Bennedsen-Lunde-Pakkanen 2017) | `deep_hedging/core/volterra.py::HybridVolterraDriver.forward` (lines 159–226) | Listing 4 of App. B.4. Near-field Cholesky for the singular small-lag part + FFT convolution for the smooth long-memory part. |
| Higher-order Markovian approximations (Class III) | not implemented; the κ=2 variant was deferred (see `archive/legacy_simulators/volterra_kappa2.py`) | The thesis discusses these as future work; the deferred κ=2 variant is preserved in `archive/` for git-history reproducibility. |
| Convergence rate `α̂ ≈ 0.913` (Fig. 20) | measured by `deep_hedging/experiments/block1_convergence.py::fit_richardson` (lines 187–250) | Result lives in `results/simulator_validation_bundle/sim_validation_data.json:p01_convergence.alpha_hat`. |

---

## Section 4 — Hedging Frameworks

### 4.0 Comparison Protocol

| Thesis | Code | Notes |
|---|---|---|
| Definition 4.1 (traded universe, hedged liability) | `deep_hedging/objectives/pnl.py::compute_payoff` (lines 14–33) | European call payoff `(S_T − K)^+`; ATM by default (`K = 100`, `S_0 = 100`). |
| Definition 4.2 (trading grid) | every simulator's `t_grid` buffer; canonical n=100, T=1. See `deep_hedging/core/rough_bergomi.py` line 63 (the `t_grid` initialiser). | — |
| Definition 4.3 (transaction costs / proportional λ) | `deep_hedging/objectives/pnl.py::compute_transaction_costs` (lines 50–78) | `δ_{-1} = 0` enforced by the prepend-zero column at line 71. |
| Definition 4.4 (information set `I_k = (t_k, S_k, τ_k, δ_{k-1})`) | `deep_hedging/hedging/deep_hedger.py::build_features` (lines 127–166) | Returns the 4-vector `(t_k/T, log(S_k/S_0), τ_k/T, δ_{k-1})` per Definition 4.5 of the thesis. |
| Definition 4.5 (feature normalisation) | same `build_features` (lines 127–166); see lines 159–164 for the normalised stack | The canonical `feature_set_b` is implemented inline; richer feature sets live in `deep_hedging/hedging/features.py::PathFeatureExtractor` (lines 34–193). |
| Definition 4.6 (performance criteria: PnL distribution + ES/VaR/entropic) | `deep_hedging/objectives/risk_measures.py::compute_all_metrics` (lines 82–107) | Returns the full evaluation vector M(δ): mean_pnl, std_pnl, var_95, es_95, es_99, entropic_1, max_loss, min_pnl, skewness, kurtosis. |

### 4.1 Sources of Model Risk and Experimental Axes

| Thesis | Code | Notes |
|---|---|---|
| Definitions 4.7–4.11 (three "worlds": true / assumed / training; four model-risk axes) | no direct code; the experimental design embodied across all KEEP-bucket experiment scripts. Mapping: |
| Axis I (structural mis-specification) | `deep_hedging/experiments/run_section_6_3_baseline.py::Section63Experiment` (lines 52–423) + `deep_hedging/experiments/heston_pde_evaluation.py::main` |
| Axis II (parameter mis-specification) | `deep_hedging/experiments/perturbation_extended.py` + `deep_hedging/experiments/transfer_extended.py` |
| Axis III (discretisation + cost) | `deep_hedging/experiments/pareto_front.py` + `deep_hedging/experiments/h2_grid_extension.py` |
| Axis IV (path dependence / state compression) | `deep_hedging/experiments/signature_ablation.py` + `deep_hedging/experiments/h_sweep.py` + `deep_hedging/experiments/h_sweep_analysis.py` |

### 4.2 Delta Hedging

| Thesis | Code | Notes |
|---|---|---|
| Definition 4.12 (discrete-time delta strategy `δ_k = ∂_s u_assumed(t_k, S_k; θ̂)`) | `deep_hedging/hedging/delta_hedger.py::BlackScholesDelta.hedge_paths` (lines 85–108) | Vectorised across all paths and all time steps in one tensor op (no Python loop). |
| Definition 4.13 (BS delta `Φ(d_1)`) | `deep_hedging/hedging/delta_hedger.py::BlackScholesDelta.compute_delta` (lines 69–83) and `bs_call_price` (lines 110–125) | The standard `d_1 = (log(S/K) + ½σ²τ)/(σ√τ)` plus normal CDF; clamped to [0,1] for sigmoid-comparable codomain. |
| Definition 4.14 (Heston PDE delta `∂_s u_Hes(t, S, V; θ̂_Hes)`) | `deep_hedging/hedging/heston_pde_delta.py::HestonPDEDelta` (lines 360–897) | Implements the 2D Crank-Nicolson Heston PDE solver on a non-uniform `(S, V, t)` grid; the per-step Hundsdorfer–Verwer ADI step is at lines 671–732 (Listing 2 of App. B.2). |
| Algorithm 1 (discrete-time delta hedging) | full hedge loop is `BlackScholesDelta.hedge_paths` (lines 85–108) for BS, `HestonPDEDelta.hedge_paths` (lines 817–897) for Heston PDE. P&L assembled by `compute_hedging_pnl` (lines 81–104 of `pnl.py`). | The implementation closes the position at maturity implicitly (`δ_{n} = 0` plus trading-cost summation k=0…n-1) per Sec 2.1 footnote 2 / Bühler 2019. |
| Definition 4.15 (baseline parameter policy) | parameters set inline in each experiment script's `params` dict; e.g. `eta_zero_control.py::HESTON_FALLBACK_PARAMS` | The baseline (controlled inputs) approach; calibration is studied separately in Sec 6.3.1's Heston PDE calibration (`results/heston_pde/calibration_data.json`). |
| Definition 4.16 (two-regime baseline premium `p_0`) | **GBM benchmark (Sec 6.2):** `src/run_benchmark_gbm_grid.py` uses `BlackScholesDelta.bs_call_price` (delta_hedger.py lines 110–125). **Rough Bergomi (Sec 6.3):** `_training_helpers.py::train_deep_hedger_with_objective` (lines 67–120) takes `p0` from the caller; canonical experiments compute it as `compute_payoff(S_train, K).mean`. | The two-regime convention is documented in the (now-archived) `archive/legacy_documentation/Section_6_Data_Bundle.md`; the thesis footnote in Sec 6.1 makes it explicit. |

### 4.3 Deep Hedging

| Thesis | Code | Notes |
|---|---|---|
| Definition 4.17 (deep-hedging strategy `δ_k = F^θ(I_k)`, weight sharing across time) | `deep_hedging/hedging/deep_hedger.py::DeepHedgerFNN.forward` (lines 104–113) | A single network `F^θ` is applied at every grid time `t_k`. The 4-d input set is built by `build_features` (lines 127–166). |
| Definition 4.18 (ES baseline objective at level α) | `deep_hedging/objectives/risk_measures.py::expected_shortfall` (lines 16–36); training-time CVaR via `_cvar_loss` closure in `train_deep_hedger` (lines 268–272) | Evaluation uses sort-based ES; training uses the smoother Rockafellar-Uryasev formulation with a learnable scalar `w` jointly optimised with the network parameters (per Remark 4.20). |
| Remark 4.18 (alternative entropic objective) | `deep_hedging/objectives/risk_measures.py::entropic_risk` (lines 39–54) | Uses `torch.logsumexp` for numerical stability. |
| Remark 4.20 (joint optimisation of `w` and `θ` via single Adam) | `deep_hedging/hedging/deep_hedger.py::train_deep_hedger` lines 264–281 — `_w_param` is a `torch.nn.Parameter` appended to the optimiser parameter list at line 280. | The Rockafellar-Uryasev convexity argument means `w → VaR_α` at the optimum. |
| Definition 4.19 (training data: `M_train = M_true` baseline) | every experiment script generates training paths via the corresponding simulator's `simulate(seed=…)`; canonical seed 2024 + n_train=80,000 + n_val=20,000 + n_test=50,000 (Tab. 12). |
| Remark 4.23 (sigmoid squashing for admissibility `δ ∈ [0, 1]`) | `DeepHedgerFNN.__init__` line 101 `nn.Sigmoid` in the output head | Differentiable everywhere — preferred over hard projection (Remark 4.22) for SGD stability. |
| Algorithm 2 (deep hedging training loop) | `deep_hedging/experiments/_training_helpers.py::train_deep_hedger_with_objective` (lines 67–120) wrapping `deep_hedger.py::train_deep_hedger` (lines 214–359) | Mini-batch Adam, early stopping; full forward pass for each mini-batch. |
| Algorithm 3 (deep hedging online execution) | `deep_hedging/hedging/deep_hedger.py::hedge_paths_deep` (lines 173–212) | Vectorised across paths and time. After training the same function is used at evaluation time on the master test set. |

### 4.4 Testable Hypotheses

| Thesis | Code | Notes |
|---|---|---|
| H1 (heavier left tails for diffusion deltas under rough) | tested by `run_section_6_3_baseline.py` (BS, DH) + `heston_pde_evaluation.py` (Heston PDE) | Result in Sec 6.3.1 / Tab. 5: `results/canonical_v2/baseline_5seeds.json` + `results/heston_pde/heston_pde_5seeds.json`. |
| H2 (frequency-cost reversal) | `pareto_front.py` + `h2_grid_extension.py` | Result in Sec 6.3.5 / Tab. 9. |
| H3 (parameter-perturbation preservation) | `perturbation_extended.py` + `worst_case_adversarial.py` + `gradient_sensitivity.py` + `adversarial_robustness.py` | Result in Sec 6.3.5 / Tab. 10 + App. A.2: `results/perturbation_v2/M{1..6}_*.json`. |
| H4 (flat-feature sufficiency) | `signature_ablation.py` + `h_sweep.py` + `h_sweep_analysis.py` | Result in Sec 6.3.3 (statistically flat panel-OLS slope). |

---

## Section 5 — Simulation Framework

### 5.1 Time Discretisation and Monte Carlo Protocol

| Thesis | Code | Notes |
|---|---|---|
| Definition 5.1 (trading + simulation grid) | `t_grid` buffer set in every simulator's `__init__` (e.g. `rough_bergomi.py` line 63) | Canonical n=100, T=1. |
| Definition 5.2 (Monte Carlo sample) | `DifferentiableRoughBergomi.simulate(seed)` (lines 173–192) and equivalent on `GBM`, `Heston` | Seeded by `torch.Generator(device).manual_seed(seed)` (line 187) before any random draw. App. A.6 of the thesis documents the seeding contract. |
| Definition 5.3 (recorded info I_k) | `deep_hedging/hedging/deep_hedger.py::build_features` (lines 127–166) | Same 4-d feature set across BS, Heston PDE, and DH evaluations. |
| Definition 5.4 (train/val/test split) | `deep_hedging/experiments/canonical_rerun.py` lines 38–41 (`DATASET_KW = dict(n_train=80_000, n_val=20_000, n_test=50_000)`) — and equivalent in every other experiment script. |
| Definition 5.5 (pathwise terminal P&L) | `deep_hedging/objectives/pnl.py::compute_hedging_pnl` (lines 81–104) | Listing 5 of App. B.5. |

### 5.2 Markovian Benchmark Models

| Thesis | Code | Notes |
|---|---|---|
| Definition 5.6 (GBM dynamics) | `deep_hedging/core/gbm.py::GBM.forward` (lines 59–102) | Exact log-Euler scheme (no time-stepping bias under constant `σ`). |
| Definition 5.6 (Heston dynamics + Definition 5.6(b) full-truncation Euler) | `deep_hedging/core/heston.py::Heston.forward` (lines 77–140) | Full-truncation Euler scheme of Lord-Koekkoek-Van Dijk (2010); explicit `torch.clamp(V, min=0.0)` at lines 113 + 121 handles negative-variance drift when the Feller condition is violated (the calibration point used in Sec 6.3.1 has Feller slack ≈ −0.196, see `results/heston_pde/calibration_data.json:feller_slack`). |
| Algorithm — GBM exact discretisation (Sec 5.2 (a)) | `gbm.py::forward` lines 88–96 (the log-Euler line `log_inc = -0.5 σ² dt + σ √dt Z`) | Vectorised cumsum then exp; no path loop. |
| Algorithm — Heston full-truncation Euler (Sec 5.2 (b)) | `heston.py::forward` lines 111–128 (the path loop) | The only KEEP-bucket simulator with a Python time loop — variance positivity needs the truncate-then-update sequence. |

### 5.3 Rough Bergomi Simulator

| Thesis | Code | Notes |
|---|---|---|
| Listing 3 (App. B.3) — `DifferentiableRoughBergomi.forward` 4-step structure | `deep_hedging/core/rough_bergomi.py::DifferentiableRoughBergomi.forward` (lines 108–169) | Step 1: WH via Volterra (line 143); Step 2: variance (line 148); Step 3: correlated dB (line 156); Step 4: log-Euler S (lines 159–168). The verbatim listing in the thesis is a faithful extract of these lines. |
| Listing 4 (App. B.4) — hybrid Volterra driver | `deep_hedging/core/volterra.py::HybridVolterraDriver.forward` (lines 159–226) | Cholesky near-field at lines 191–197; FFT far-field at lines 200–216; `√(2a+1)` rescaling at line 221. |

### 5.4 Numerical Validation

| Thesis | Code | Notes |
|---|---|---|
| Convergence sweep `n ∈ {50…1600}`, slope `α̂ ≈ 0.913` (Fig. 20) | `deep_hedging/experiments/block1_convergence.py::main` (lines 734–905) → `fit_richardson` (lines 187–250) → `plot_convergence_curves` (lines 360–440) | Outputs `results/block1_v2/p01_verify/convergence_sweep.json`; consolidated into `results/simulator_validation_bundle/sim_validation_data.json:p01_convergence`. |
| Cholesky benchmark, KS p ≈ 0.926 (Fig. 21) | `deep_hedging/experiments/block1_cholesky_v2.py` (full module; main at the bottom) — uses `ExactCholeskyVolterra` (volterra_exact.py) as the reference | Outputs are local-only (`results/block1/cholesky_v2_n500k.json` is **not** on GitHub); the thesis-cited number lives in `results/simulator_validation_bundle/sim_validation_data.json:p021_cholesky.fbm_terminal.ks_pvalue`. |
| `ExactCholeskyVolterra` (Sec 5.4 reference) | `deep_hedging/core/volterra_exact.py::ExactCholeskyVolterra.forward` (lines 105–137) | Builds the full `(n+1)×(n+1)` fBm covariance + Cholesky in `__init__`; per-batch `forward` is then matrix multiplication. |

---

## Section 6 — Numerical Experiments

Section 6 has no new theoretical objects; everything in it is an
empirical observation backed by an experiment script. The mapping
from observations to code lives in `docs/EXPERIMENTS.md`; the mapping
from numerical claims to JSON keypaths lives in `docs/THESIS_MAPPING.md`.

| Thesis | Code |
|---|---|
| Observation 6.1 (tail-risk hierarchy under rough volatility) | `run_section_6_3_baseline.py` + `heston_pde_evaluation.py` |
| Observation 6.2 (risk objective is the principal lever) | `pareto_front.py` (Sec 6.3.2) + `perturbation_extended.py --M5` |
| Observation 6.3 (roughness `H` is not the source of advantage) | `h_sweep.py` + `h_sweep_analysis.py` + `signature_ablation.py` |
| Observation 6.4 (asymmetric cross-model transfer with bounded basin) | `transfer_extended.py --L1 --L4 --L5` |
| Observation 6.5 (frequency-cost reversal, parameter-perturbation basin) | `h2_grid_extension.py` + `perturbation_extended.py --M1` |

---

## Listings (Appendix B) — direct line-range cross-reference

Each listing is an extract from the production code. The thesis quotes
the listing verbatim with the introductory phrase "The following extract
from <PATH>". The on-disk function spans the listed lines below; the
verbatim portion within the listing is a contiguous sub-range.

| Listing | Function (file) | Function lines | Verbatim extract |
|---|---|---|---|
| 1 (App. B.1) | `DeepHedgerFNN` (`deep_hedging/hedging/deep_hedger.py`) | 54–125 | The class definition + `__init__` + `forward` → lines 54–113. |
| 2 (App. B.2) | `HestonPDEDelta._solve_pde` HV-ADI step (`deep_hedging/hedging/heston_pde_delta.py`) | 456–758 (full method); the HV-ADI step itself is lines 671–735 | The verbatim `for step in range(self.n_t - 1, -1, -1):` loop (lines 696–735). |
| 3 (App. B.3) | `DifferentiableRoughBergomi.forward` (`deep_hedging/core/rough_bergomi.py`) | 108–169 | The full method body (lines 108–169). |
| 4 (App. B.4) | `HybridVolterraDriver.forward` (`deep_hedging/core/volterra.py`) | 159–226 | The full method body (lines 159–226). |
| 5 (App. B.5) | `compute_trading_gains` + `compute_transaction_costs` + `compute_hedging_pnl` (`deep_hedging/objectives/pnl.py`) | 36–104 | All three functions verbatim. |

---

## Proofs (Appendix C) — no code dependency

| Thesis | Notes |
|---|---|
| C.1 Theorem 2.1 (Universal Approximation) | no code (motivates Definition 2.7 / `DeepHedgerFNN`) |
| C.2 Proposition 2.3 (parameter-class density) | no code (motivates the increasing residual-block depth family) |
| C.3 Proposition 2.4 (consistency of empirical risk minimisation) | no code (motivates the Monte Carlo sample sizes in Tab. 12) |
| C.4 Proposition 3.1 (Hurst estimator log-log slope consistency) | no code (motivates `block1_convergence.py::fit_richardson`) |
| C.5 Proposition 3.2 (BS m_2(h) = 0) | no code (analytical argument) |
| C.6 Proposition 3.3 (Heston m_2(h) ∝ h) | no code (analytical argument) |
| C.7 Proposition 3.8 (replication impossible → risk-minimising hedging) | no code (motivates the convex-risk hedging framework) |

---

## Summary

Total numbered objects mapped: 38 Definitions/Propositions/Theorems +
4 Hypotheses + 5 Observations + 3 Algorithms + 5 Listings + 7 Proofs = **62 rows**.

Of these, **42 have direct code correspondence** with line ranges given
above; the remaining 20 are abstract / proof-only / motivational and
are explicitly marked.
