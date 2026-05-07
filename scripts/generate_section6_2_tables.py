"""Generate Tables 2 and 3 for Section 6.2 from the GBM benchmark CSV.

Reads:
  results/gbm_deephedge/benchmark_6_2/aggregate/scenario_summary.csv
  results/gbm_deephedge/benchmark_6_2/aggregate/seed_level_metrics.csv

Writes (to stdout):
  - Table 2 LaTeX: ES_0.95 across (sigma_bar, lambda) cells, oracle + robust
  - Table 3 LaTeX: paired t-test BS vs DH at the canonical sigma_bar=0.20 cells

Run from repo root::

    python scripts/generate_section6_2_tables.py > /tmp/section6_2_tables.tex
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy import stats as sstats

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_DIR = REPO_ROOT / "results" / "gbm_deephedge" / "benchmark_6_2" / "aggregate"

def load_scenario_summary() -> list[dict]:
    rows = []
    with open(CSV_DIR / "scenario_summary.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def load_seed_level() -> list[dict]:
    rows = []
    with open(CSV_DIR / "seed_level_metrics.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def fmt_pm(mean: float, sd: float, n: int = 10) -> str:
    """Format mean +/- standard error."""
    se = sd / np.sqrt(n)
    return f"{mean:.4f} $\\pm$ {se:.4f}"

def make_table2(rows: list[dict]) -> str:
    """Table 2: ES_0.95 across (sigma_bar, lambda, regime, method).

    Structure: rows = sigma_bar values, columns split into two regime blocks
    (oracle, robust) each with BS, DH columns × four lambda values.
    """
    # Filter to fixed sigma_true = 0.20
    rows = [r for r in rows if abs(float(r["sigma_true"]) - 0.20) < 1e-9]

    # Index: cells[(sigma_bar, lambda_cost, regime, method)] = row
    cells = {}
    for r in rows:
        key = (
            float(r["sigma_bar"]),
            float(r["lambda_cost"]),
            r["training_regime"],
            r["method"],
        )
        cells[key] = r

    sigma_bars = sorted({float(r["sigma_bar"]) for r in rows})
    lambdas = sorted({float(r["lambda_cost"]) for r in rows})

    # Build table — too wide for 5 sigma_bar columns plus 8 lambda columns.
    # Better layout: rows are (sigma_bar), columns are lambda; one sub-table
    # per (regime, method) configuration. Use compact form: per (sigma_bar,
    # lambda) cell give the BS / DH-oracle / DH-robust ES_0.95 means stacked
    # vertically. Here we split into two tables: oracle and robust.

    out = []
    out.append("% Table 2 (oracle): ES_0.95 across sigma_bar x lambda (BS vs DH-oracle)")
    out.append("\\begin{table}[H]")
    out.append("\\centering")
    out.append("\\caption{Out-of-sample $\\mathrm{ES}_{0.95}$ on the GBM "
               "benchmark with $\\sigma_{\\mathrm{true}} = 0.20$, oracle "
               "training regime ($\\bar\\sigma$ = $\\sigma_{\\mathrm{true}}$). "
               "Each cell reports the 10-seed mean $\\pm$ standard error. "
               "Lower is better.}")
    out.append("\\label{tab:gbm_table2_oracle}")
    out.append("\\begin{tabular}{r r c c c c}")
    out.append("\\toprule")
    out.append("$\\bar\\sigma$ & method & $\\lambda = 0$ & $\\lambda = 10^{-4}$ & "
               "$\\lambda = 5{\\cdot}10^{-4}$ & $\\lambda = 10^{-3}$ \\\\")
    out.append("\\midrule")

    for sb in sigma_bars:
        for method, label in [("bs_delta", "BS delta"),
                                ("deep_hedge_oracle", "DH oracle")]:
            row_cells = [f"{sb:.2f}" if method == "bs_delta" else "",
                         label]
            for lam in lambdas:
                key = (sb, lam, "oracle", method)
                if key in cells:
                    cell = cells[key]
                    m = float(cell["ES_loss_0.95_mean"])
                    sd = float(cell["ES_loss_0.95_sd"])
                    n = int(cell["n_seeds"])
                    row_cells.append(fmt_pm(m, sd, n))
                else:
                    row_cells.append("---")
            out.append(" & ".join(row_cells) + " \\\\")
        if sb != sigma_bars[-1]:
            out.append("\\midrule")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    out.append("\\end{table}")

    # Robust regime table
    out.append("")
    out.append("% Table 2 (robust): same with DH-robust trained on "
               "{0.15, 0.20, 0.25}")
    out.append("\\begin{table}[H]")
    out.append("\\centering")
    out.append("\\caption{Out-of-sample $\\mathrm{ES}_{0.95}$ on the GBM "
               "benchmark with $\\sigma_{\\mathrm{true}} = 0.20$, robust "
               "training regime (DH trained over $\\bar\\sigma \\in "
               "\\{0.15, 0.20, 0.25\\}$, evaluated at the indicated "
               "$\\bar\\sigma$). Each cell reports the 10-seed mean $\\pm$ "
               "standard error. Lower is better.}")
    out.append("\\label{tab:gbm_table2_robust}")
    out.append("\\begin{tabular}{r r c c c c}")
    out.append("\\toprule")
    out.append("$\\bar\\sigma$ & method & $\\lambda = 0$ & $\\lambda = 10^{-4}$ & "
               "$\\lambda = 5{\\cdot}10^{-4}$ & $\\lambda = 10^{-3}$ \\\\")
    out.append("\\midrule")
    for sb in sigma_bars:
        for method, label in [("bs_delta", "BS delta"),
                                ("deep_hedge_robust", "DH robust")]:
            row_cells = [f"{sb:.2f}" if method == "bs_delta" else "",
                         label]
            for lam in lambdas:
                key = (sb, lam, "robust", method)
                if key in cells:
                    cell = cells[key]
                    m = float(cell["ES_loss_0.95_mean"])
                    sd = float(cell["ES_loss_0.95_sd"])
                    n = int(cell["n_seeds"])
                    row_cells.append(fmt_pm(m, sd, n))
                else:
                    row_cells.append("---")
            out.append(" & ".join(row_cells) + " \\\\")
        if sb != sigma_bars[-1]:
            out.append("\\midrule")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    out.append("\\end{table}")

    return "\n".join(out)

def make_table3(seed_rows: list[dict], scenario_rows: list[dict]) -> str:
    """Table 3: per-(sigma_bar) paired t-test BS vs DH at the
    canonical lambda=0 oracle cell.

    Tests whether DH-oracle has a lower per-seed ES_0.95 than BS at each
    sigma_bar (paired by seed within the cell).
    """
    # Filter seed-level data to sigma_true=0.20, lambda=0, regime=oracle
    seed_rows = [
        r for r in seed_rows
        if abs(float(r["sigma_true"]) - 0.20) < 1e-9
        and abs(float(r["lambda_cost"]) - 0.0) < 1e-9
        and r["training_regime"] == "oracle"
    ]
    sigma_bars = sorted({float(r["sigma_bar"]) for r in seed_rows})

    out = []
    out.append("% Table 3: paired t-test BS vs DH-oracle per sigma_bar")
    out.append("\\begin{table}[H]")
    out.append("\\centering")
    out.append("\\caption{Paired t-test of $\\mathrm{ES}_{0.95}$, BS delta "
               "versus DH oracle, at the canonical $\\lambda = 0$ "
               "frictionless setting and $\\sigma_{\\mathrm{true}} = 0.20$. "
               "Pairing is by seed within each $\\bar\\sigma$ cell ($n = 10$ "
               "seeds). Negative $\\Delta = \\mathrm{ES}_{0.95}^{\\mathrm{DH}} "
               "- \\mathrm{ES}_{0.95}^{\\mathrm{BS}}$ favours DH.}")
    out.append("\\label{tab:gbm_table3_paired}")
    out.append("\\begin{tabular}{r c c c c c}")
    out.append("\\toprule")
    out.append("$\\bar\\sigma$ & BS mean & DH mean & $\\Delta$ mean & "
               "$t$-stat & $p$-value \\\\")
    out.append("\\midrule")
    for sb in sigma_bars:
        bs_vals = sorted(
            [r for r in seed_rows
             if abs(float(r["sigma_bar"]) - sb) < 1e-9
             and r["method"] == "bs_delta"],
            key=lambda r: r["seed"],
        )
        dh_vals = sorted(
            [r for r in seed_rows
             if abs(float(r["sigma_bar"]) - sb) < 1e-9
             and r["method"] == "deep_hedge_oracle"],
            key=lambda r: r["seed"],
        )
        bs_arr = np.array([float(r["ES_loss_0.95"]) for r in bs_vals])
        dh_arr = np.array([float(r["ES_loss_0.95"]) for r in dh_vals])
        if len(bs_arr) == 0 or len(dh_arr) == 0:
            continue
        diff = dh_arr - bs_arr
        t_stat, p_val = sstats.ttest_rel(dh_arr, bs_arr)
        out.append(
            f"{sb:.2f} & {bs_arr.mean():.4f} & {dh_arr.mean():.4f} & "
            f"{diff.mean():+.4f} & {t_stat:.2f} & {p_val:.3g} \\\\"
        )
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    out.append("\\end{table}")
    return "\n".join(out)

def main() -> None:
    scenario = load_scenario_summary()
    seed_lvl = load_seed_level()
    print(f"% Generated from {CSV_DIR}/scenario_summary.csv ({len(scenario)} rows)")
    print(f"% Generated from {CSV_DIR}/seed_level_metrics.csv ({len(seed_lvl)} rows)")
    print()
    print(make_table2(scenario))
    print()
    print(make_table3(seed_lvl, scenario))

if __name__ == "__main__":
    main()
