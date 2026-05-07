"""5-panel synthesis figure + comprehensive report for Prompt L.

Reads:
  results/transfer_v2/L1_multi_source_5seeds.json (single seed; aggregates from commit msg)
  results/transfer_v2/L2_budget_sweep.json
  results/transfer_v2/L3_fine_tuning_extended.json
  results/transfer_v2/L4_reverse_transfer.json (3 seeds × 2 targets, regenerated)
  results/transfer_v2/L5_cross_calibration.json (3 seeds × 3 H, regenerated)

Writes:
  figures/transfer_v2/transfer_comprehensive_summary.png
  results/transfer_v2/transfer_comprehensive_report.md
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results" / "transfer_v2"
FIGURES = REPO_ROOT / "figures" / "transfer_v2"
FIGURES.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Load data
# --------------------------------------------------------------------------

def _load(name):
    with open(RESULTS / name) as f:
        return json.load(f)

L2 = _load("L2_budget_sweep.json")
L3 = _load("L3_fine_tuning_extended.json")
L4 = _load("L4_reverse_transfer.json")
L5 = _load("L5_cross_calibration.json")

# L.1 main file lost per-seed data (overwrite bug); aggregates are from commit
# message dc2ac00 (verified, recorded numbers from full multi-seed run):
L1_AGG = {
    "gbm":          {"mean": 11.0877, "std": 0.0257, "n": 5},
    "heston":       {"mean": 10.4431, "std": 0.0256, "n": 5},
    "rbergomi_H03": {"mean": 10.7289, "std": 0.1148, "n": 5},
}

# References (from Prompt B canonical 5-seed runs)
BS_5SEED_MEAN = 11.5921
BS_5SEED_STD = 0.0316
DH_5SEED_MEAN = 10.4442
DH_5SEED_STD = 0.0748

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _se_from(std, n):
    return std / np.sqrt(n) if n > 0 else 0.0

# --------------------------------------------------------------------------
# Build figure
# --------------------------------------------------------------------------

fig, axes = plt.subplots(1, 5, figsize=(22, 4.6))
plt.subplots_adjust(left=0.04, right=0.99, top=0.82, bottom=0.16, wspace=0.30)

# -- Panel A: L.1 Multi-source zero-shot --
ax = axes[0]
labels = ["GBM", "Heston", "rB H=0.3"]
src_keys = ["gbm", "heston", "rbergomi_H03"]
means = [L1_AGG[k]["mean"] for k in src_keys]
sems = [_se_from(L1_AGG[k]["std"], L1_AGG[k]["n"]) for k in src_keys]
colors = ["#5B8FF9", "#2EBC8C", "#9C5CFF"]
xs = np.arange(3)
ax.bar(xs, means, yerr=sems, capsize=5, color=colors, edgecolor="black", linewidth=0.8)
ax.axhline(BS_5SEED_MEAN, color="red", ls="--", lw=1.4, label=f"BS = {BS_5SEED_MEAN:.2f}")
ax.axhline(DH_5SEED_MEAN, color="black", ls=":", lw=1.4, label=f"DH = {DH_5SEED_MEAN:.2f}")
ax.set_xticks(xs)
ax.set_xticklabels(labels)
ax.set_ylabel("ES$_{0.95}$ on rB H=0.07 test set")
ax.set_title("(A) L.1: Zero-shot transfer\n(3 sources × 5 seeds)")
ax.set_ylim(10.0, 12.0)
ax.legend(loc="upper right", fontsize=8)
ax.grid(True, axis="y", alpha=0.3)
for x, m, s in zip(xs, means, sems):
    ax.text(x, m + s + 0.03, f"{m:.3f}", ha="center", fontsize=8)

# -- Panel B: L.2 Pretraining budget sweep --
ax = axes[1]
src_colors = {"gbm": "#5B8FF9", "heston": "#2EBC8C", "rbergomi_H03": "#9C5CFF"}
src_labels = {"gbm": "GBM", "heston": "Heston", "rbergomi_H03": "rB H=0.3"}
for src, content in L2["results"].items():
    Ns, mns, sems_ = [], [], []
    for N, cell in sorted(content.items(), key=lambda kv: int(kv[0])):
        ag = cell.get("aggregate", {}).get("es_95")
        if ag is None or ag.get("n", 0) == 0:
            continue
        Ns.append(int(N))
        mns.append(ag["mean"])
        sems_.append(ag.get("se", 0.0))
    if Ns:
        ax.errorbar(Ns, mns, yerr=sems_, marker="o", lw=1.6, capsize=4,
                    color=src_colors.get(src, "gray"), label=src_labels.get(src, src))
ax.axhline(BS_5SEED_MEAN, color="red", ls="--", lw=1.4, label="BS")
ax.axhline(DH_5SEED_MEAN, color="black", ls=":", lw=1.4, label="DH (full)")
ax.set_xscale("log")
ax.set_xlabel("$N_{\\mathrm{train}}$")
ax.set_ylabel("ES$_{0.95}$")
ax.set_title("(B) L.2: Pretraining budget\n(3 sources × 6 budgets × 3 seeds)")
ax.legend(loc="upper right", fontsize=8)
ax.grid(True, alpha=0.3)

# -- Panel C: L.3 Fine-tuning vs from-scratch --
ax = axes[2]
res3 = L3["results"]
def _curve(reg):
    cells = res3.get(reg, {})
    ns, mns, ses = [], [], []
    for n_ft, cell in sorted(cells.items(), key=lambda kv: int(kv[0])):
        per = cell.get("per_seed", {})
        vals = [v["es_95"] for v in per.values() if isinstance(v, dict) and "es_95" in v]
        if vals:
            ns.append(int(n_ft))
            mns.append(np.mean(vals))
            ses.append(np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)
    return np.array(ns), np.array(mns), np.array(ses)

n_ft_ft, mn_ft, se_ft = _curve("fine_tune")
n_ft_fs, mn_fs, se_fs = _curve("from_scratch")
mask_ft = n_ft_ft > 0
ax.errorbar(n_ft_ft[mask_ft], mn_ft[mask_ft], yerr=se_ft[mask_ft],
            marker="o", color="#FF7A5C", lw=1.6, capsize=4, label="fine-tune (GBM-pre)")
mask_fs = n_ft_fs > 0
ax.errorbar(n_ft_fs[mask_fs], mn_fs[mask_fs], yerr=se_fs[mask_fs],
            marker="s", color="#2EBC8C", lw=1.6, capsize=4, label="from-scratch")
zs = res3.get("base_zero_shot", {}).get("es_95")
if zs is not None:
    ax.axhline(zs, color="#5B8FF9", ls=":", lw=1.4, label=f"zero-shot = {zs:.3f}")
ax.axhline(BS_5SEED_MEAN, color="red", ls="--", lw=1.4, label="BS")
ax.axhline(DH_5SEED_MEAN, color="black", ls=":", lw=1.0, label="DH (full)")
ax.set_xscale("log")
ax.set_xlabel("$N_{\\mathrm{ft}}$ (rough Bergomi paths)")
ax.set_ylabel("ES$_{0.95}$")
ax.set_title("(C) L.3: Fine-tune vs from-scratch\n(11 budgets × 3 seeds × 2 regimes)")
ax.legend(loc="upper right", fontsize=8)
ax.grid(True, alpha=0.3)

# -- Panel D: L.4 Reverse transfer --
ax = axes[3]
targets = ["gbm", "heston"]
target_labels = ["GBM (BS ref)", "Heston (PDE ref)"]
xs = np.arange(2)
w = 0.36
dh_means = []
dh_sems = []
ref_means = []
ref_sems = []
gaps = []
for t in targets:
    ag = L4["results"]["per_target"][t]["aggregate"]
    dh_means.append(ag["dh_es95"]["mean"])
    dh_sems.append(ag["dh_es95"]["se"])
    ref_means.append(ag["ref_es95"]["mean"])
    ref_sems.append(ag["ref_es95"]["se"])
    gaps.append(ag["gap_dh_minus_ref"]["mean"])
ax.bar(xs - w/2, dh_means, w, yerr=dh_sems, capsize=4, color="#5B8FF9",
       label="rB-trained DH", edgecolor="black", linewidth=0.8)
ax.bar(xs + w/2, ref_means, w, yerr=ref_sems, capsize=4, color="#FF7A5C",
       label="oracle (BS / PDE)", edgecolor="black", linewidth=0.8)
ax.set_xticks(xs)
ax.set_xticklabels(target_labels, fontsize=9)
ax.set_ylabel("ES$_{0.95}$ on target test set")
ax.set_title("(D) L.4: Reverse transfer\n(rB-trained → GBM, Heston)")
ax.legend(loc="upper left", fontsize=8)
ax.grid(True, axis="y", alpha=0.3)
for x, m, s in zip(xs - w/2, dh_means, dh_sems):
    ax.text(x, m + s + 0.15, f"{m:.2f}", ha="center", fontsize=8)
for x, m, s in zip(xs + w/2, ref_means, ref_sems):
    ax.text(x, m + s + 0.15, f"{m:.2f}", ha="center", fontsize=8)
ax.text(0.05, 0.92, f"GBM gap = {gaps[0]:+.2f} (DH worse)\nHeston gap = {gaps[1]:+.2f} (DH better)",
        transform=ax.transAxes, fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

# -- Panel E: L.5 Cross-calibration --
ax = axes[4]
H_keys = sorted(L5["results"]["per_H"].keys(), key=float)
Hs = [float(k) for k in H_keys]
dh_means = [L5["results"]["per_H"][k]["aggregate"]["dh_es95"]["mean"] for k in H_keys]
dh_sems = [L5["results"]["per_H"][k]["aggregate"]["dh_es95"]["se"] for k in H_keys]
bs_means = [L5["results"]["per_H"][k]["aggregate"]["bs_es95"]["mean"] for k in H_keys]
bs_sems = [L5["results"]["per_H"][k]["aggregate"]["bs_es95"]["se"] for k in H_keys]
gap_means = [L5["results"]["per_H"][k]["aggregate"]["gap_dh_minus_bs"]["mean"] for k in H_keys]

ax.errorbar(Hs, dh_means, yerr=dh_sems, marker="o", color="#5B8FF9", lw=1.8,
            capsize=5, label="rB-trained DH")
ax.errorbar(Hs, bs_means, yerr=bs_sems, marker="s", color="red", ls="--", lw=1.4,
            capsize=5, label="BS delta (σ=√ξ₀)")
ax.set_xlabel("target $H$")
ax.set_ylabel("ES$_{0.95}$")
ax.set_title("(E) L.5: Cross-calibration\n(rB-trained DH at H ∈ {0.07, 0.20, 0.40})")
ax.legend(loc="upper right", fontsize=8)
ax.grid(True, alpha=0.3)
for x, g, dh in zip(Hs, gap_means, dh_means):
    ax.annotate(f"gap = {g:+.2f}", xy=(x, dh),
                xytext=(0, -15), textcoords="offset points",
                ha="center", fontsize=7)
ax.axvline(0.07, color="black", ls=":", lw=0.8, alpha=0.5)
ax.text(0.07, ax.get_ylim()[1] * 0.99, "canonical", rotation=90, va="top",
        ha="right", fontsize=7, alpha=0.7)

fig.suptitle("Prompt L — Transfer learning comprehensive extension (5-panel synthesis)",
             fontsize=13, y=0.995)

out_png = FIGURES / "transfer_comprehensive_summary.png"
fig.savefig(out_png, dpi=170, bbox_inches="tight")
print(f"  wrote {out_png}")

# --------------------------------------------------------------------------
# Build report
# --------------------------------------------------------------------------

report_path = RESULTS / "transfer_comprehensive_report.md"

def _l2_summary():
    lines = []
    for src in ("gbm", "heston", "rbergomi_H03"):
        content = L2["results"].get(src, {})
        if not content:
            lines.append(f"\n### {src} (PENDING)\n")
            continue
        lines.append(f"\n### {src}\n")
        lines.append("| N_train | epochs | ES_0.95 mean ± SE | min | max | beats BS? |")
        lines.append("|---|---|---|---|---|---|")
        for N in sorted(content.keys(), key=int):
            cell = content[N]
            ag = cell.get("aggregate", {}).get("es_95", {})
            n = ag.get("n", 0)
            if n == 0:
                continue
            mn = ag["mean"]; se = ag.get("se", 0.0); mn_v = ag["min"]; mx_v = ag["max"]
            ep = cell.get("epochs", "?")
            tag = " ✓" if mn < BS_5SEED_MEAN else " "
            lines.append(f"| {int(N):>6,} | {ep} | {mn:.4f} ± {se:.4f} | {mn_v:.4f} | {mx_v:.4f} |{tag}|")
    return "\n".join(lines)

def _l3_summary():
    lines = []
    zs = L3["results"].get("base_zero_shot", {}).get("es_95")
    lines.append(f"Zero-shot baseline (GBM-pretrained, evaluated on rB test): **{zs:.4f}**\n")
    lines.append("| n_ft | fine-tune ES (mean ± SE) | from-scratch ES (mean ± SE) |")
    lines.append("|---|---|---|")
    ft = L3["results"].get("fine_tune", {})
    fs = L3["results"].get("from_scratch", {})
    for n_ft in sorted(set(int(k) for k in ft.keys()) | set(int(k) for k in fs.keys())):
        s_ft = "—"
        if str(n_ft) in ft:
            per = ft[str(n_ft)].get("per_seed", {})
            vals = [v["es_95"] for v in per.values() if isinstance(v, dict) and "es_95" in v]
            if vals:
                m = np.mean(vals); se = np.std(vals, ddof=1)/np.sqrt(len(vals)) if len(vals)>1 else 0.0
                s_ft = f"{m:.4f} ± {se:.4f}"
        s_fs = "—"
        if str(n_ft) in fs:
            per = fs[str(n_ft)].get("per_seed", {})
            vals = [v["es_95"] for v in per.values() if isinstance(v, dict) and "es_95" in v]
            if vals:
                m = np.mean(vals); se = np.std(vals, ddof=1)/np.sqrt(len(vals)) if len(vals)>1 else 0.0
                s_fs = f"{m:.4f} ± {se:.4f}"
        lines.append(f"| {n_ft:>6,} | {s_ft} | {s_fs} |")
    return "\n".join(lines)

def _l4_summary():
    lines = []
    lines.append("| target | DH ES_0.95 (3 seeds) | reference ES_0.95 | gap (DH − ref) |")
    lines.append("|---|---|---|---|")
    for t in ["gbm", "heston"]:
        ag = L4["results"]["per_target"][t]["aggregate"]
        dh = ag["dh_es95"]; ref = ag["ref_es95"]; gap = ag["gap_dh_minus_ref"]
        ref_label = "BS delta" if t == "gbm" else "Heston PDE"
        verdict = "DH WORSE" if gap["mean"] > 0 else "DH BETTER"
        lines.append(f"| {t:6s} ({ref_label}) | {dh['mean']:.4f} ± {dh['se']:.4f} | "
                     f"{ref['mean']:.4f} ± {ref['se']:.4f} | "
                     f"**{gap['mean']:+.4f}** ({verdict}) |")
    return "\n".join(lines)

def _l5_summary():
    lines = []
    lines.append("| target H | DH ES_0.95 | BS ES_0.95 | gap |")
    lines.append("|---|---|---|---|")
    for k in sorted(L5["results"]["per_H"].keys(), key=float):
        ag = L5["results"]["per_H"][k]["aggregate"]
        dh = ag["dh_es95"]; bs = ag["bs_es95"]; gap = ag["gap_dh_minus_bs"]
        tag = "(canonical)" if float(k) == 0.07 else ""
        lines.append(f"| H={float(k):.2f} {tag} | {dh['mean']:.4f} ± {dh['se']:.4f} | "
                     f"{bs['mean']:.4f} ± {bs['se']:.4f} | **{gap['mean']:+.4f}** |")
    return "\n".join(lines)


# Number of L.2 cells completed
total_cells = 0
for src, budgets in L2["results"].items():
    for N, cell in budgets.items():
        if cell.get("aggregate", {}).get("es_95", {}).get("n", 0) > 0:
            total_cells += 1
l2_status = f"{total_cells}/18 cells done (across all sources)"

report = f"""# Prompt L — Transfer learning comprehensive extension (results bundle)

This is the consolidated results bundle for Block 4 (5 sub-experiments). For
the executive overview cross-referenced with figures and the
dissertation-revision narrative, see `results/PROMPT_L_FINAL_REPORT.md`.

## Reference benchmarks (canonical rough Bergomi H=0.07)

* BS delta:        **{BS_5SEED_MEAN:.4f} ± {BS_5SEED_STD:.4f}** (5 seeds, Prompt B canonical)
* Canonical DH:    **{DH_5SEED_MEAN:.4f} ± {DH_5SEED_STD:.4f}** (5 seeds, Prompt B canonical)

All sub-experiments evaluate on the cached canonical test set
`results/transfer_v2/shared_test_set.pt` (50,000 paths, seed=2024,
p0=8.0319) for L.1, L.2, L.3 (target-side L.4/L.5 use freshly simulated
target-family test paths to reflect the target distribution).

---

## L.1 — Multi-source zero-shot (3 sources × 5 seeds)

| source | ES_0.95 mean ± SE | gap vs BS | beats BS? |
|---|---|---|---|
| GBM           | **{L1_AGG["gbm"]["mean"]:.4f} ± {_se_from(L1_AGG["gbm"]["std"], L1_AGG["gbm"]["n"]):.4f}** | {L1_AGG["gbm"]["mean"] - BS_5SEED_MEAN:+.4f} | yes |
| Heston        | **{L1_AGG["heston"]["mean"]:.4f} ± {_se_from(L1_AGG["heston"]["std"], L1_AGG["heston"]["n"]):.4f}** | {L1_AGG["heston"]["mean"] - BS_5SEED_MEAN:+.4f} | yes (matches canonical DH) |
| rBergomi H=0.3| **{L1_AGG["rbergomi_H03"]["mean"]:.4f} ± {_se_from(L1_AGG["rbergomi_H03"]["std"], L1_AGG["rbergomi_H03"]["n"]):.4f}** | {L1_AGG["rbergomi_H03"]["mean"] - BS_5SEED_MEAN:+.4f} | yes |

**Headline.** All three sources beat BS (gap negative). Heston-pretrained
hedger matches the canonical-trained DH (10.44 ≈ 10.44) within MC noise —
zero-shot Heston→rB transfer is as good as training on rough Bergomi
directly. GBM pretraining alone yields a meaningful −0.50 gap over BS,
demonstrating that even the simplest dynamical model captures enough
delta-hedging structure for transfer.

NOTE: per-seed values for L.1 were lost when the original reproducibility
subprocess overwrote the main results file (bug now fixed: each
`--repro-LX` mode passes its own `out_path`). Aggregate statistics above
are reproduced verbatim from the L.1 commit message (`dc2ac00`).

---

## L.2 — Pretraining budget sweep (3 sources × 6 budgets × 3 seeds)

L.2 status at this report: **{l2_status}**.

{_l2_summary()}

**Headline.** Three distinct convergence regimes by source:

* **GBM** beats BS at N=5k already (11.54 < 11.59) and plateaus at
  ~11.08 by N=80k. Fast and stable but the plateau is well above
  canonical DH (10.44).
* **Heston** is WORSE than BS at N≤10k (≥11.89), beats BS at N=20k
  (11.23), reaches canonical-DH at N=80k (10.45 ≈ 10.44), and continues
  to improve to 10.40 at N=160k.
* **rBergomi H=0.3** is the slowest learner: WORSE than BS at N≤20k,
  beats BS only at N=40k (11.47), and reaches the DH-canonical
  neighborhood only at N=160k (10.55).

**Implication: source matters less than expected at sufficient N, but
data efficiency varies dramatically.** Heston (a Markovian
stochastic-volatility model) is the best source — it converges nearly as
fast as GBM and reaches a lower plateau than either GBM or rBergomi at
modest N. The rough-Bergomi-source-on-rough-Bergomi-target case is
counter-intuitively the LEAST data-efficient, suggesting that the
non-Markovian noise in the H=0.3 simulator slows down stable
representation learning even when the target is a similar (rougher)
rough Bergomi process.

---

## L.3 — Extended fine-tuning curve (11 n_ft × 3 seeds × 2 regimes)

{_l3_summary()}

**Headline (catastrophic forgetting).** Fine-tuning the GBM-pretrained
hedger on rough Bergomi paths produces ES values WORSE than the zero-shot
baseline at every n_ft tested. The fine-tune curve never returns below
the zero-shot baseline. Training a brand-new hedger from scratch on the
same n_ft does eventually catch up: at N=80k from-scratch reaches
~10.68, close to canonical-DH 10.44.

This **reverses** the dissertation Section 6.3.5 fine-tuning claim and
provides a concrete rebuttal: with the existing optimisation regime
(lr 5e-4, 30 epochs, patience 5), adapting a transferred hedger to the
target dynamics destroys its transferred representation faster than it
learns the target-specific representation.

---

## L.4 — Reverse transfer (3 seeds × 2 targets)

{_l4_summary()}

**Headline (asymmetric transfer).** The rough-Bergomi-trained hedger
fails on simple GBM by ~+2.07 ES units (over-fitting to non-Markovian
structure absent from GBM) but BEATS the Heston PDE delta on Heston by
~2.10 ES units. This is consistent with the L.1 finding that Heston ≈
rBergomi for hedging purposes, while GBM is structurally simpler than
either.

---

## L.5 — Cross-calibration transfer (3 H values × 3 seeds)

{_l5_summary()}

**Headline (graceful degradation).** The DH retains a negative gap (beats
BS) at every tested H. The advantage shrinks gradually as H increases
(smoother dynamics). Supports the dynamics-agnostic claim WITHIN the
rough-Bergomi family.

---

## Reproducibility verification

| sub-experiment | seed | reproduced ES_0.95 | match |
|---|---|---|---|
| L.1 (gbm)        | 7001 | 11.063391 | byte-identical |
| L.2 (gbm N=80k)  | 7101 | (will check after L.2 completes) | — |
| L.3 (n_ft=2000)  | 7201 | 12.448100 | byte-identical |
| L.4 (target=gbm) | 7301 | 3.986717 | byte-identical |
| L.5 (H=0.20)     | 7401 | 9.195396 | byte-identical |

All single-seed reproducibility runs in fresh subprocesses produced
byte-identical metrics, confirming the seeding protocol is deterministic.

---

## Synthesis: dynamics-agnostic hypothesis

The combined evidence supports a **partially** dynamics-agnostic deep
hedging policy:

* **Forward transfer (L.1, L.2, L.5):** any source dynamic with stochastic
  variance + leverage (Heston, rBergomi H=0.3) is sufficient. Simple GBM
  is also sufficient with enough data. Transfer survives recalibration of
  the target Hurst parameter.
* **Reverse transfer (L.4):** asymmetric. rB-trained transfers to Heston
  (where it BEATS the Heston PDE delta) but degrades on GBM. The
  rough-trained policy memorises non-Markovian structure that GBM lacks.
* **Adaptation (L.3):** with the current optimisation regime, fine-tuning
  is harmful — catastrophic forgetting destroys the transferred
  representation. From-scratch training on enough target data is the
  correct adaptation strategy.

These findings argue for a single source-trained DH that is robust enough
to deploy directly on similar (Heston/rBergomi-family) targets, rather
than a per-target retrained hedger. The "any source works" property is
particularly powerful: a practitioner can train on whatever model fits
their existing infrastructure (GBM is cheapest) and still deploy on the
true rough-Bergomi market.

---

## Bug discovered and fixed: `--repro-LX` file overwrite

During L.1, L.4, L.5 commit preparation, a bug was identified in the
orchestrator script. The `run_LX_*` functions wrote incremental saves to
a hardcoded path equal to the main results path. When the `--repro-LX`
subprocess re-ran the same function with a single seed (in a fresh
process), the incremental save **overwrote the main results file** with
the single-seed data, then the explicit `_save_json(LX_repro.json)`
saved a duplicate.

**Consequence.** L.1's per-seed values were lost. L.4 and L.5 were
re-generated as part of Prompt L's final pass (~10 min total) and now
contain full per-seed data. The L.1 aggregate statistics reported in
this document are reproduced verbatim from the L.1 commit message
(`dc2ac00`), which was generated from the full data before the
overwrite.

**Fix.** Each `run_LX_*` function now accepts an `out_path` parameter,
and the `--repro-LX` dispatchers pass `out_path=Path("L*_repro.json")`
so the main file is never touched. A new `--repro-L2` mode was added
with the same safe plumbing.

L.2 and L.3 are unaffected (their full data was preserved on disk
because no repro mode was run during their commit). The reproducibility
certificates are valid because byte-identity was verified between the
in-process metric (printed to stdout) and the subprocess metric (printed
to stdout); both were computed correctly from the seeded source.
"""

report_path.write_text(report)
print(f"  wrote {report_path}")

print("\nDONE.")
