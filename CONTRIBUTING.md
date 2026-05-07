# Contributing

This repository is the code release accompanying the MMath
dissertation *"Deep Hedging under Rough Volatility: Robustness
to Model Misspecification"* (Degtiarenko, University of
Manchester, November 2025). It is intended primarily as a
reproducibility artefact for the published thesis; the
`release/v1.0-thesis` branch and its `v1.0-thesis` annotated tag
are frozen as the official version.

## Reporting issues

If you find a numerical discrepancy, a documentation error, or
a reproducibility problem, please open an issue at
<https://github.com/westgluf/thesis_code/issues>. Helpful issues
include:

- Your environment fingerprint (Python / PyTorch / NumPy / SciPy
  versions and OS / arch).
- The exact command you ran.
- The expected output (preferably with a thesis-section reference).
- The observed output.

For numerical-discrepancy reports specifically, please run the
relevant Path A verification first
(see `docs/REPRODUCIBILITY.md` §4) — the answer is most often
that you're querying a different keypath than the one cited.

## Contributing code

Code contributions are not accepted at this time. The thesis
Appendix B Listings 1–5 cite specific file paths verbatim, and
code reorganisation would invalidate the published text. Bug
fixes — by exception — may be accepted via pull request to
`main` if (a) they preserve the App A.6 byte-identical
reproducibility on the canonical environment, and (b) they do
not modify the file paths under `deep_hedging/{core, hedging,
objectives}/`.

## Reproducibility-respect contract

Any change that perturbs the byte-identical reproduction of the
canonical baseline (verified at
`docs/audit/REPRODUCIBILITY_VERIFICATION_v1.md`) must be
discussed in an issue first. The default position is "no" — the
canonical numbers are the ones in the thesis text and tables.
