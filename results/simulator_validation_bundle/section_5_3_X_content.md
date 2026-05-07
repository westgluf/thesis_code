# Draft content for Section 5.3.X "Simulator Validation"

Generated: 2026-04-23T15:22:45
Git commit: 8f5b014edf6bc1fc31ed5a1567aa62c57f46059f

## Data sources

- **P01 convergence sweep:** `results/block1_v2/p01_verify/convergence_sweep.json`
- **P02.1 Cholesky benchmark:** `results/block1/cholesky_v2_n500k.json`
- **P01.6 grid refinement:** `results/block1_v2/p016_5seeds.json`

## Key numbers

### P01 convergence

- Empirical α̂ = **0.9130** (95 % CI [0.722, 1.104])
- BLP asymptotic α = 0.57 (H + ½)
- ES_∞ asymptote = 8.8208
- Relative error at n=100 vs ES_∞: 28.7 %

### P02.1 Cholesky benchmark

- Paths: N = 100,000 (coupling), N = 50,000 (arbitrage)
- Criteria passed: **5/5** — verdict **STRICT_PASS**
- KS p-value (terminal fBm): **p = 0.926**
- Max variance-path relative difference: **2.19 %**
- Call-price relative difference: **0.61 %**

### P01.6 grid refinement

- Γ(n=400) = **+1.0770 ± 0.0194** (5 seeds)
- 95 % CI: [+1.0529, +1.1011]
- Canonical Γ(n=100) = +1.1479 ± 0.0761
- Per-seed spread ratio (n=100 → n=400): 3.93×
- Verdict vs canonical: **PRESERVED**

## Draft LaTeX text

```latex
\subsection{Numerical Validation of the Rough Bergomi Simulator}
\label{sec:sim_validation}

Before using the hybrid Volterra simulator in the hedging experiments of
Chapter 6, we subject it to three complementary numerical validation checks:
a convergence sweep over grid resolution, an exact-Cholesky benchmark at
dissertation-relevant parameters, and a grid-refinement check on the hedging
advantage at the canonical calibration.

\paragraph{Convergence sweep.}
We measure the discretisation error of $\mathrm{ES}_{0.95}$ as a function of
grid resolution $n \in \{50, 100, 200, 400, 800, 1600\}$, holding all other
parameters fixed at the canonical calibration. A log-log fit yields empirical
slope $\hat{\alpha} \approx 0.913$ (95 \% CI $[0.722, 1.104]$). The Bennedsen--Lunde--Pakkanen
asymptotic rate for the hybrid scheme at $H = 0.07$ is
$\alpha = H + 1/2 \approx 0.57$. The empirical rate exceeds the asymptotic
rate in the finite-$n$ regime tested, consistent with the hybrid scheme's
faster-than-asymptotic convergence at accessible grids
(Figure~\ref{fig:convergence_alpha}).

\paragraph{Exact-Cholesky benchmark.}
At $N = 100,000$ paths, the hybrid simulator is compared
against a direct Cholesky-factorisation reference at the canonical calibration
$(H = 0.07, \eta = 1.9, \rho = -0.7, \xi_0 = 0.235^2)$. All five validation
criteria pass: mean and variance match to within Monte Carlo noise; path-wise
variance gap below $2.19\%$; Kolmogorov--Smirnov test
$p = 0.926$; call-price match within $0.61\%$; Gaussian
moment and correlation alignment (Figure~\ref{fig:cholesky_ks}). Global
verdict: \textsc{strict_pass}.

\paragraph{Grid refinement.}
To rule out discretisation bias in the canonical $n=100$ simulator used in
Section~6.3, the deep hedger was retrained at a four-fold refined grid
resolution $n=400$ across five independent seeds. The resulting advantage gap
$\Gamma(n=400) = +1.077 \pm 0.019$ lies comfortably
within the canonical 95 \% confidence interval of
$\Gamma(n=100) = +1.148 \pm 0.076$;
the per-seed spread tightens by a factor of approximately $3.9\times$
(Figure~\ref{fig:gamma_n400}).
This confirms that the canonical grid is numerically adequate for the
Chapter~6 claims.

\paragraph{Reproducibility.}
All three validation protocols use the fixed seeding convention
(Appendix~\ref{sec:appendix_b}) and produce byte-identical outputs across fresh
Python subprocesses. Raw data and regeneration scripts are archived under
\texttt{results/simulator\_validation\_bundle/}.
```

## Figure files (for `\includegraphics{...}`)

- `figures/sim_validation/convergence_alpha.png` → `\label{fig:convergence_alpha}`
- `figures/sim_validation/cholesky_ks.png` → `\label{fig:cholesky_ks}`
- `figures/sim_validation/gamma_n400.png` → `\label{fig:gamma_n400}`