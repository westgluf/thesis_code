# Appendix B — Draft content

Generated: 2026-04-23T15:26:16
Git commit: 0099245e2d4c5bbd2ede4ae184a9eff52b85009d

This is a LaTeX-ready draft combining seeding protocol, per-seed tables,
reproducibility verification, and the decomposition-removal note into a
single appendix section.

## LaTeX source

```latex
\appendix

\chapter{Reproducibility and Seeding Protocol}
\label{sec:appendix_b}

\section{Seeding protocol}
\label{sec:app_b_seeding}

All neural-network training follows the protocol:
\begin{enumerate}
    \item Before every \texttt{DeepHedgerFNN(...)} instantiation, call:
    \begin{verbatim}
    torch.manual_seed(seed)
    np.random.seed(seed)
    \end{verbatim}
    \item Path simulation uses a separate explicit seed via the simulator's
    \texttt{torch.Generator}:
    \begin{verbatim}
    sim.simulate(n_paths=..., S0=..., seed=seed)
    \end{verbatim}
    \item In any loop that trains multiple models, the two reseed lines are
    re-invoked at the start of each iteration before the next
    \texttt{DeepHedgerFNN(...)} is constructed.
\end{enumerate}

The protocol is applied at 12 call-sites across 6 scripts. It guarantees
byte-identical cross-subprocess reproducibility given the same seed, RNG
independence between path simulation and neural-network initialisation, and
no dependence on system entropy.

\section{Per-seed numerical tables}
\label{sec:app_b_tables}

\begin{table}[h]
\centering
\caption{Per-seed canonical baseline results at $\lambda = 0$.}
\label{tab:B_1}
\begin{tabular}{c|rrr|rr}
\hline
Seed & $\mathrm{ES}_{0.95}^{\mathrm{BS}}$ & $\mathrm{ES}_{0.95}^{\mathrm{DH}}$ & $\Gamma$ & $\mu_{P\&L}^{\mathrm{DH}}$ & $\sigma_{P\&L}^{\mathrm{DH}}$ \\
\hline
2024 & 11.6307 & 10.4463 & +1.1844 & +0.0139 & 4.1065 \\
2025 & 11.5828 & 10.4565 & +1.1263 & +0.0063 & 4.1864 \\
2026 & 11.5978 & 10.5585 & +1.0394 & -0.0660 & 4.1387 \\
2027 & 11.6043 & 10.3584 & +1.2459 & -0.0245 & 4.1280 \\
2028 & 11.5447 & 10.4013 & +1.1434 & +0.0340 & 4.1477 \\
\hline
Mean & 11.5921 & 10.4442 & +1.1479 & -0.0073 & 4.1415 \\
Std & 0.0316 & 0.0748 & 0.0761 & 0.0390 & 0.0295 \\
95\% CI ($\Gamma$) & & & $[+1.0811, +1.2146]$ & & \\
\hline
\end{tabular}
\end{table}

\begin{table}[h]
\centering
\caption{Per-seed $\eta=0$ control results.}
\label{tab:B_2}
\begin{tabular}{c|rrr|rr}
\hline
Seed & $\mathrm{ES}_{0.95}^{\mathrm{BS}}$ & $\mathrm{ES}_{0.95}^{\mathrm{DH}}$ & $\Gamma_{\mathrm{arch}}$ & $\mu_{P\&L}^{\mathrm{DH}}$ & $\sigma_{P\&L}^{\mathrm{DH}}$ \\
\hline
4024 & 1.9252 & 1.6783 & +0.2468 & -0.0387 & 0.9341 \\
4025 & 1.7942 & 1.5650 & +0.2292 & +0.0813 & 0.9578 \\
4026 & 1.8797 & 1.6461 & +0.2336 & -0.0167 & 0.9292 \\
4027 & 1.8239 & 1.5957 & +0.2282 & +0.0366 & 0.9335 \\
4028 & 1.9922 & 1.7628 & +0.2294 & -0.1190 & 0.9210 \\
\hline
Mean & 1.8830 & 1.6496 & +0.2334 & -0.0113 & 0.9351 \\
Std & 0.0792 & 0.0770 & 0.0078 & 0.0763 & 0.0137 \\
95\% CI ($\Gamma_{\mathrm{arch}}$) & & & $[+0.2238, +0.2431]$ & & \\
\hline
\end{tabular}
\end{table}

\begin{table}[h]
\centering
\caption{Per-seed grid refinement validation at $n=400$.}
\label{tab:B_3}
\begin{tabular}{c|rrr|r}
\hline
Seed & $\mathrm{ES}_{0.95}^{\mathrm{BS}}$ & $\mathrm{ES}_{0.95}^{\mathrm{DH}}$ & $\Gamma(n{=}400)$ & Best epoch \\
\hline
7401 & 9.5842 & 8.5004 & +1.0839 & 200 \\
7402 & 9.5842 & 8.5282 & +1.0560 & 200 \\
7403 & 9.5842 & 8.4868 & +1.0974 & 199 \\
7404 & 9.5842 & 8.5274 & +1.0569 & 197 \\
7405 & 9.5842 & 8.4934 & +1.0908 & 200 \\
\hline
Mean & 9.5842 & 8.5072 & +1.0770 & --- \\
Std & 0.0000 & 0.0194 & 0.0194 & --- \\
95\% CI ($\Gamma$) & & & $[+1.0529, +1.1011]$ & --- \\
\hline
\end{tabular}
\end{table}

\section{Reproducibility verification}
\label{sec:app_b_repro}

Every experiment reported in this dissertation was rerun in a fresh Python
subprocess to verify that the seeding protocol produces byte-identical results.
All checks passed:
\begin{itemize}
    \item Canonical baseline, seed 2024 (n=100, 200 epochs): \textsc{reproducible}.
    \item $\eta=0$ control, seed 4024: \textsc{reproducible}.
    \item Phase-D 6.3.1 figures, seed 2024: $\Gamma = +1.1844$ exact match.
    \item P01.6 grid refinement, seed 7401 (n=400): \textsc{reproducible}.
    \item P01.7 Cell A, seed 7711: \textsc{reproducible}.
\end{itemize}

The protocol therefore works across three grid resolutions (n=20 mini-test,
n=100 canonical, n=400 refined) and three training budgets (diagnostic,
canonical, H2).

\section{Note on the removed decomposition}
\label{sec:app_b_decomposition}

An earlier draft of this dissertation included a five-bucket factorial
decomposition of the advantage gap $\Gamma$ into contributions from the
training objective, interaction terms, stochastic volatility level, roughness,
and architecture. Sensitivity analysis across five seeds subsequently revealed
that the objective and interaction components have Pearson cross-seed
correlation $\approx -0.97$: their sum is stable at approximately $78\%$, but
their split between the two categories is not separately identifiable through
the $2 \times 2$ factorial arithmetic and varies substantially across seeds.
The decomposition was therefore removed from the main text. Raw per-seed
values remain archived at
\texttt{results/canonical\_v2/decomposition\_5seeds.json} for completeness.
The $\eta = 0$ control experiment in Section~\ref{sec:eta_zero_control}
(Section~6.3.3) provides a statistically identifiable alternative: the
architecture + objective contribution is isolated through a direct physical
intervention (switching off stochastic volatility) rather than residual
arithmetic.

\section{Commit history}
\label{sec:app_b_git}

Key commits marking milestones of the revision programme:

\begin{itemize}
    \item \texttt{cdbd9f1} --- Phase B pre-fix snapshot
    \item \texttt{5070800} --- Phase B seeding fix applied to 12 call-sites
    \item \texttt{765713a} --- Phase C $\eta=0$ control complete
    \item \texttt{9cc9066} --- Phase D 6.3.1 figures regenerated on seed 2024
    \item \texttt{88f54f2} --- Phase E P01.6 + P01.7 rerun complete
\end{itemize}

Full history is available via \texttt{git log --all} on the revision branch.
```

## Figure reference

The tables above refer to data generated by scripts in
`deep_hedging/experiments/`. The figure files associated with each
experiment are:

- Canonical baseline: `figures/canonical_v2/{gamma_5seeds,decomposition_5seeds,es_distribution_comparison}.png`
- η=0 control: `figures/eta_zero_v2/{gamma_arch_5seeds,pl_histogram_seed4024}.png`
- Grid refinement: `figures/sim_validation/gamma_n400.png`
- Section 6.3.1 seed 2024: `figures/canonical_v2/6_3_1_{pnl_histograms,qq_plots,metrics_bar}_seed2024.png`