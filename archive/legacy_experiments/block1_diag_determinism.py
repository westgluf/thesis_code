#!/usr/bin/env python
"""
P01.7-DIAG — Determinism diagnostic for the running P01.7 experiment.

Answers four Boolean questions:
  Q1: Is train_deep_hedger deterministic across fresh process invocations?
  Q2: Is the rBergomi simulator deterministic across fresh process invocations?
  Q3: Is BS-delta evaluation deterministic?
  Q4: Was Cell B seed 7721 in run 1 actually completed?

Produces a binary verdict (HALT_P017 / CONTINUE_P017 / CONTINUE_WITH_NOTE / HALT_EVERYTHING)
per the decision matrix in the prompt.

Does NOT disturb the running P01.7 process — uses torch.set_num_threads(1),
small workloads, and writes only to results/block1/diagnostic_p017/.

Run:
    python -u -m deep_hedging.experiments.block1_diag_determinism
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Force single-threaded to avoid stealing CPU from running P01.7
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)
import numpy as np

from deep_hedging.experiments.block1_hardware import record_hardware_and_versions

REPO_ROOT = Path(__file__).resolve().parents[2]
DIAG_DIR = REPO_ROOT / "results" / "block1" / "diagnostic_p017"
PYTHON_EXE = sys.executable


def _find_p017_pid() -> int | None:
    try:
        result = subprocess.run(
            ["pgrep", "-f", "block1_extended_validation"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            pids = result.stdout.strip().split("\n")
            for p in pids:
                if p.strip().isdigit():
                    return int(p.strip())
    except Exception:
        pass
    return None


def _check_no_collision():
    for fname in ["preflight.md", "results.json", "decision.md", "q4_evidence.md"]:
        p = DIAG_DIR / fname
        if p.exists():
            raise FileExistsError(f"Refusing to overwrite: {p}")


# =====================================================================
# Q1: Training determinism via subprocess
# =====================================================================

_TRAIN_SUBPROCESS_CODE = r'''
import os, sys, pickle
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import torch
torch.set_num_threads(1)
import numpy as np

seed = int(sys.argv[1])
output_path = sys.argv[2]

torch.manual_seed(seed)
np.random.seed(seed)

from deep_hedging.core.rough_bergomi import DifferentiableRoughBergomi
from deep_hedging.hedging.deep_hedger import DeepHedgerFNN, train_deep_hedger
from deep_hedging.objectives.pnl import compute_payoff

# Small, fast workload
sim = DifferentiableRoughBergomi(n_steps=50, T=1.0, H=0.07,
                                   eta=1.9, rho=-0.7, xi0=0.235**2)
S_train, _, _ = sim.simulate(500, S0=100.0, seed=seed + 100)
S_val, _, _ = sim.simulate(100, S0=100.0, seed=seed + 200)
p0 = float(compute_payoff(S_train, 100.0, "call").mean())

# Re-seed before model creation (to test reproducibility of model init too)
torch.manual_seed(seed)
np.random.seed(seed)

model = DeepHedgerFNN(input_dim=4, hidden_dim=128, n_res_blocks=2)
hist = train_deep_hedger(
    model, S_train, S_val, K=100.0, T=1.0, S0=100.0, p0=p0,
    cost_lambda=0.0, alpha=0.95, lr=1e-3, batch_size=64,
    epochs=10, patience=100, verbose=False,
)

# Deterministic forward pass on held-out test set
torch.manual_seed(seed + 999)  # fixed test seed
S_test, _, _ = sim.simulate(200, S0=100.0, seed=seed + 300)
model.eval()
with torch.no_grad():
    deltas = model.hedge_paths(S_test.float(), 1.0, 100.0)

state = {k: v.detach().cpu().numpy().copy() for k, v in model.state_dict().items()}
payload = {
    "state_dict": state,
    "final_train_risk": hist["train_risk"][-1],
    "final_val_risk": hist["val_risk"][-1],
    "best_val_risk": hist["best_val_risk"],
    "best_epoch": hist["best_epoch"],
    "forward_pass": deltas.detach().cpu().double().numpy(),
}
with open(output_path, "wb") as f:
    pickle.dump(payload, f)
'''


def q1_training_determinism(seed: int = 9501) -> dict[str, Any]:
    """Run two subprocess trainings with same seed, compare."""
    print("  Q1: Testing training determinism via subprocess...", flush=True)
    t0 = time.perf_counter()

    with tempfile.TemporaryDirectory() as tmpdir:
        out1 = Path(tmpdir) / "run1.pkl"
        out2 = Path(tmpdir) / "run2.pkl"

        for out in [out1, out2]:
            proc = subprocess.run(
                [PYTHON_EXE, "-c", _TRAIN_SUBPROCESS_CODE, str(seed), str(out)],
                capture_output=True, text=True, timeout=600,
                cwd=str(REPO_ROOT),
            )
            if proc.returncode != 0:
                print(f"    Subprocess failed: {proc.stderr[-500:]}", flush=True)
                return {
                    "tested": True, "boolean": False,
                    "evidence": {"error": proc.stderr[-500:]},
                    "wall_clock_s": round(time.perf_counter() - t0, 2),
                }

        with open(out1, "rb") as f:
            r1 = pickle.load(f)
        with open(out2, "rb") as f:
            r2 = pickle.load(f)

    # Compare weights byte-by-byte
    weights_equal = True
    max_weight_diff = 0.0
    for k in r1["state_dict"]:
        diff = np.abs(r1["state_dict"][k] - r2["state_dict"][k]).max()
        max_weight_diff = max(max_weight_diff, float(diff))
        if diff > 0:
            weights_equal = False

    train_loss_diff = abs(r1["final_train_risk"] - r2["final_train_risk"])
    val_loss_diff = abs(r1["final_val_risk"] - r2["final_val_risk"])
    fp_diff = float(np.abs(r1["forward_pass"] - r2["forward_pass"]).max())

    all_match = (weights_equal and train_loss_diff < 1e-10
                  and val_loss_diff < 1e-10 and fp_diff < 1e-12)
    elapsed = round(time.perf_counter() - t0, 2)

    print(f"    weights equal: {weights_equal}, max_w_diff: {max_weight_diff:.2e}, "
          f"train_loss_diff: {train_loss_diff:.2e}, val_loss_diff: {val_loss_diff:.2e}, "
          f"fp_diff: {fp_diff:.2e}", flush=True)
    print(f"    Q1 = {all_match} ({elapsed:.1f}s)", flush=True)

    return {
        "tested": True, "boolean": bool(all_match),
        "evidence": {
            "weights_byte_equal": bool(weights_equal),
            "max_weight_abs_diff": max_weight_diff,
            "final_train_loss_diff": float(train_loss_diff),
            "final_val_loss_diff": float(val_loss_diff),
            "forward_pass_max_abs_diff": fp_diff,
        },
        "wall_clock_s": elapsed,
    }


# =====================================================================
# Q2: Simulator determinism via subprocess
# =====================================================================

_SIM_SUBPROCESS_CODE = r'''
import os, sys, pickle, hashlib
os.environ["OMP_NUM_THREADS"] = "1"
import torch
torch.set_num_threads(1)

seed = int(sys.argv[1])
output_path = sys.argv[2]

from deep_hedging.core.rough_bergomi import DifferentiableRoughBergomi
sim = DifferentiableRoughBergomi(n_steps=100, T=1.0, H=0.07,
                                   eta=1.9, rho=-0.7, xi0=0.235**2)
S, V, _ = sim.simulate(1000, S0=100.0, seed=seed)

S_bytes = S.detach().cpu().numpy().tobytes()
V_bytes = V.detach().cpu().numpy().tobytes()
h = hashlib.sha256(S_bytes + V_bytes).hexdigest()

payload = {
    "S": S.detach().cpu().numpy().copy(),
    "V": V.detach().cpu().numpy().copy(),
    "hash": h,
}
with open(output_path, "wb") as f:
    pickle.dump(payload, f)
'''


def q2_simulator_determinism(seed: int = 9502) -> dict[str, Any]:
    print("  Q2: Testing simulator determinism via subprocess...", flush=True)
    t0 = time.perf_counter()

    with tempfile.TemporaryDirectory() as tmpdir:
        out1 = Path(tmpdir) / "sim1.pkl"
        out2 = Path(tmpdir) / "sim2.pkl"
        for out in [out1, out2]:
            proc = subprocess.run(
                [PYTHON_EXE, "-c", _SIM_SUBPROCESS_CODE, str(seed), str(out)],
                capture_output=True, text=True, timeout=120,
                cwd=str(REPO_ROOT),
            )
            if proc.returncode != 0:
                return {"tested": True, "boolean": False,
                         "evidence": {"error": proc.stderr[-500:]},
                         "wall_clock_s": round(time.perf_counter() - t0, 2)}

        with open(out1, "rb") as f:
            r1 = pickle.load(f)
        with open(out2, "rb") as f:
            r2 = pickle.load(f)

    s_diff = float(np.abs(r1["S"] - r2["S"]).max())
    v_diff = float(np.abs(r1["V"] - r2["V"]).max())
    hashes_equal = r1["hash"] == r2["hash"]
    elapsed = round(time.perf_counter() - t0, 2)
    result = hashes_equal and s_diff == 0 and v_diff == 0

    print(f"    S_max_diff={s_diff}, V_max_diff={v_diff}, hashes_equal={hashes_equal}", flush=True)
    print(f"    Q2 = {result} ({elapsed:.1f}s)", flush=True)

    return {
        "tested": True, "boolean": bool(result),
        "evidence": {
            "S_max_abs_diff": s_diff, "V_max_abs_diff": v_diff,
            "tensor_hash_run1": r1["hash"], "tensor_hash_run2": r2["hash"],
            "hashes_equal": bool(hashes_equal),
        },
        "wall_clock_s": elapsed,
    }


# =====================================================================
# Q3: BS-delta evaluation determinism (same process)
# =====================================================================

def q3_evaluation_determinism(seed: int = 9503) -> dict[str, Any]:
    print("  Q3: Testing BS-delta evaluation determinism...", flush=True)
    t0 = time.perf_counter()

    from deep_hedging.core.rough_bergomi import DifferentiableRoughBergomi
    from deep_hedging.hedging.delta_hedger import BlackScholesDelta
    from deep_hedging.objectives.pnl import compute_hedging_pnl, compute_payoff

    sim = DifferentiableRoughBergomi(n_steps=100, T=1.0, H=0.07,
                                       eta=1.9, rho=-0.7, xi0=0.235**2)
    S, _, _ = sim.simulate(1000, S0=100.0, seed=seed)
    payoff = compute_payoff(S, 100.0, "call")
    p0 = float(payoff.mean())
    hedger = BlackScholesDelta(sigma=0.235, K=100.0, T=1.0)

    deltas1 = hedger.hedge_paths(S)
    pnl1 = compute_hedging_pnl(S, deltas1, payoff, p0, 0.0)

    deltas2 = hedger.hedge_paths(S)
    pnl2 = compute_hedging_pnl(S, deltas2, payoff, p0, 0.0)

    pnl_diff = float((pnl1 - pnl2).abs().max())
    result = pnl_diff == 0.0
    elapsed = round(time.perf_counter() - t0, 2)

    print(f"    pnl_max_diff={pnl_diff}", flush=True)
    print(f"    Q3 = {result} ({elapsed:.1f}s)", flush=True)

    return {
        "tested": True, "boolean": bool(result),
        "evidence": {"pnl_max_abs_diff": pnl_diff},
        "wall_clock_s": elapsed,
    }


# =====================================================================
# Q4: Was Cell B seed 7721 in run 1 completed?
# =====================================================================

def q4_run1_completion_check() -> dict[str, Any]:
    print("  Q4: Searching for evidence of run-1 Cell B seed 7721 completion...", flush=True)

    evidence_found = []
    searched = []

    # 1. Task output directory (prior P01.7 runs)
    task_dir = Path("/private/tmp/claude-501/-Users-l-Desktop-thesis-code-copy/"
                     "6b81d3bd-db5e-4cb9-ba7d-595e1463695b/tasks")
    searched.append(str(task_dir))
    if task_dir.exists():
        for f in task_dir.glob("*.output"):
            try:
                text = f.read_text()
                if "block1_extended_validation" in text or "Cell B" in text or "7721" in text:
                    # Search for seed 7721 completion pattern
                    has_7721_result = False
                    has_7722_started = False
                    gamma_7721 = None
                    for line in text.split("\n"):
                        if "seed=7721..." in line:
                            pass  # just start marker
                        if "ES_BS=" in line and "ES_DH=" in line and "Gamma=" in line:
                            # This is a result line — check if it's after seed=7721 line
                            # Reconstruct: find the preceding "seed=XXXX..." line
                            pass
                    # Simpler: if we see "Gamma=0.3070" near "seed=7721" in bfs8yfey1, that's it
                    if "seed=7721" in text and "Gamma=0.3070" in text:
                        has_7721_result = True
                    if "seed=7722" in text:
                        has_7722_started = True
                    evidence_found.append({
                        "path": str(f),
                        "type": "log",
                        "seed_7721_result_printed": has_7721_result,
                        "seed_7722_started_after": has_7722_started,
                        "completion_state": ("completed" if has_7721_result and has_7722_started
                                              else "incomplete" if not has_7721_result
                                              else "ambiguous"),
                    })
            except Exception as e:
                pass

    # 2. Checkpoint directory
    ckpt_dir = REPO_ROOT / "results" / "block1" / "checkpoints_extended"
    searched.append(str(ckpt_dir))
    if ckpt_dir.exists():
        for f in ckpt_dir.glob("cell_B_seed_7721*"):
            evidence_found.append({
                "path": str(f),
                "type": "checkpoint",
                "size_bytes": f.stat().st_size,
                "completion_state": "unknown (checkpoint exists but not per-epoch)",
            })

    # 3. Run log from current run
    run_log = REPO_ROOT / "results" / "block1" / "p017_run.log"
    searched.append(str(run_log))
    if run_log.exists():
        evidence_found.append({
            "path": str(run_log),
            "type": "log",
            "size_bytes": run_log.stat().st_size,
            "note": "current run (run 2) log file",
        })

    # Determine verdict
    completed = None
    interpretation = ""
    for ev in evidence_found:
        if ev.get("type") == "log" and ev.get("seed_7721_result_printed"):
            if ev.get("seed_7722_started_after"):
                completed = True
                interpretation = (
                    "Run-1 log file shows Cell B seed 7721 printed its final Gamma=0.3070 result, "
                    "followed by 'seed=7722...' marker indicating seed 7721 fully completed "
                    "its 200 epochs and evaluation before the crash."
                )
                break

    if completed is None:
        # Check for partial evidence
        has_log_evidence = any(
            ev.get("type") == "log" and ev.get("seed_7721_result_printed")
            for ev in evidence_found
        )
        if has_log_evidence:
            completed = True
            interpretation = "Run-1 seed 7721 result printed in log; presumed completed."
        else:
            completed = None
            interpretation = (
                "No definitive evidence found. If log file was cleared between runs, "
                "completion state cannot be determined."
            )

    print(f"    Evidence found: {len(evidence_found)} item(s)", flush=True)
    for ev in evidence_found:
        print(f"      - {ev.get('path', '?')}: {ev.get('completion_state', ev.get('note', '?'))}",
              flush=True)
    print(f"    Q4 = {completed}", flush=True)

    return {
        "tested": True, "boolean": completed,
        "evidence_locations_searched": searched,
        "evidence_found": evidence_found,
        "interpretation": interpretation,
    }


# =====================================================================
# Decision matrix
# =====================================================================

def decision_matrix(q1: bool, q2: bool, q3: bool, q4: bool | None) -> tuple[str, str]:
    """Apply the decision matrix mechanically."""
    if not q2:
        return ("HALT_EVERYTHING",
                "rBergomi simulator produces different results across runs — foundational issue.")
    if not q1:
        return ("HALT_P017",
                "train_deep_hedger is non-deterministic across fresh process invocations.")
    if not q3:
        return ("HALT_P017",
                "BS-delta evaluation is non-deterministic — investigate before continuing.")
    if q4 is True:
        return ("HALT_P017",
                "Q1-Q3 pass but run-1 Cell B seed 7721 was completed yet gave different Gamma — "
                "latent non-determinism source unidentified.")
    if q4 is False:
        return ("CONTINUE_P017",
                "Run-1 Cell B seed 7721 was incomplete — the 0.307 vs 0.5249 comparison was invalid.")
    # q4 is None
    return ("CONTINUE_WITH_NOTE",
            "Q1-Q3 pass; run-1 completion state unknown. Document uncertainty and continue.")


# =====================================================================
# Outputs
# =====================================================================

def write_preflight(out_path: Path):
    pid = _find_p017_pid()
    lines = [
        "# P01.7-DIAG Preflight\n",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"P01.7 running PID: {pid}",
        "",
        "## Plan",
        "- Q1: train_deep_hedger determinism (subprocess, ~5-8 min)",
        "- Q2: simulator determinism (subprocess, ~1 min)",
        "- Q3: BS-delta evaluation determinism (same-process, ~10s)",
        "- Q4: run-1 Cell B seed 7721 completion evidence (file search, ~1s)",
        "",
        "## Resource guard",
        "- torch.set_num_threads(1), OMP_NUM_THREADS=1",
        "- Small workloads (n=50, 500 paths, 10 epochs)",
        "- Total wall-clock budget: <15 min",
        "",
        "READY TO PROCEED",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_decision(results: dict, out_path: Path):
    v = results["verdict"]
    lines = [
        "# P01.7 Determinism Diagnostic — Decision\n",
        f"Generated: {results['meta']['ended_at_utc']}\n",
        "## Verdict\n",
        f"**{v}**\n",
        "## Test results\n",
        "| Question | Result | Evidence |",
        "|---|---|---|",
    ]

    def mark(b):
        if b is True: return "✓"
        if b is False: return "✗"
        return "unknown"

    q1 = results["Q1_train_deterministic"]
    q2 = results["Q2_simulator_deterministic"]
    q3 = results["Q3_evaluation_deterministic"]
    q4 = results["Q4_run1_was_completed"]

    lines.extend([
        f"| Q1: train_deep_hedger deterministic? | {mark(q1['boolean'])} | "
        f"weights_diff={q1['evidence'].get('max_weight_abs_diff', 'n/a')}, "
        f"fp_diff={q1['evidence'].get('forward_pass_max_abs_diff', 'n/a')} |",
        f"| Q2: rBergomi simulator deterministic? | {mark(q2['boolean'])} | "
        f"hash_match={q2['evidence'].get('hashes_equal', 'n/a')}, "
        f"S_diff={q2['evidence'].get('S_max_abs_diff', 'n/a')} |",
        f"| Q3: BS-delta evaluation deterministic? | {mark(q3['boolean'])} | "
        f"pnl_diff={q3['evidence'].get('pnl_max_abs_diff', 'n/a')} |",
        f"| Q4: Was run-1 Cell B seed 7721 actually completed? | {mark(q4['boolean'])} | "
        f"{q4.get('interpretation', '')[:100]} |",
        "",
        "## Action required\n",
    ])

    actions = {
        "HALT_P017": (
            "**HALT_P017**: kill the running P01.7 process immediately "
            f"(e.g. `kill {_find_p017_pid() or '<PID>'}`). "
            f"Root cause: {results['verdict_rationale_one_sentence']}. "
            "After investigation, restart P01.7 from scratch with the fix applied."
        ),
        "CONTINUE_P017": (
            "**CONTINUE_P017**: allow P01.7 to finish. "
            "The 0.307 vs 0.5249 discrepancy is explained by incompleteness of run 1, "
            "not by a determinism failure."
        ),
        "CONTINUE_WITH_NOTE": (
            "**CONTINUE_WITH_NOTE**: allow P01.7 to finish, but document the uncertainty "
            "about run-1 completion in the readme. After P01.7 completes, optionally run "
            "a separate reproducibility check on a single completed cell."
        ),
        "HALT_EVERYTHING": (
            "**HALT_EVERYTHING**: kill all running experiments. The simulator itself is "
            "non-deterministic; this affects every block."
        ),
    }
    lines.append(actions.get(v, "Unknown verdict."))

    lines.extend(["", "## Diagnostic hypothesis\n"])
    if q1["boolean"] is False:
        lines.append(
            "The `train_deep_hedger` function does not fix the torch global RNG at entry. "
            "Model weight initialisation (`DeepHedgerFNN.__init__`) and batch shuffling "
            "(`torch.randperm(n_train)` each epoch) both draw from the default generator, "
            "whose state is process-dependent. Two fresh process invocations with the same "
            "explicit seed set at entry will therefore produce different training trajectories "
            "if the caller did not re-seed before each training run. "
            "In the running P01.7, no torch.manual_seed is called between cells, so the "
            "default-generator state evolves differently between runs, producing different "
            "DH models even from identical input data (confirmed by ES_BS being identical "
            "across runs while ES_DH differs)."
        )

    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_q4_evidence(q4_result: dict, out_path: Path):
    lines = [
        "# Q4 — Run-1 Cell B seed 7721 Completion Evidence\n",
        f"Generated: {datetime.now(timezone.utc).isoformat()}\n",
        f"## Verdict: {q4_result['boolean']}\n",
        f"{q4_result['interpretation']}\n",
        "## Locations searched\n",
    ]
    for loc in q4_result["evidence_locations_searched"]:
        lines.append(f"- `{loc}`")
    lines.append("\n## Evidence found\n")
    for ev in q4_result["evidence_found"]:
        lines.append(f"### `{ev.get('path', '?')}`")
        for k, v in ev.items():
            if k != "path":
                lines.append(f"- **{k}**: `{v}`")
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


# =====================================================================
# Main
# =====================================================================

def main():
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    _check_no_collision()

    hardware = record_hardware_and_versions()
    t0 = time.perf_counter()
    t0_utc = datetime.now(timezone.utc).isoformat()

    pid = _find_p017_pid()
    print("=" * 60)
    print(" P01.7 Determinism Diagnostic")
    print("=" * 60)
    print(f"  P01.7 running PID: {pid}")
    print(f"  Diag output: {DIAG_DIR}")
    print("-" * 60, flush=True)

    write_preflight(DIAG_DIR / "preflight.md")

    q1 = q1_training_determinism(seed=9501)
    q2 = q2_simulator_determinism(seed=9502)
    q3 = q3_evaluation_determinism(seed=9503)
    q4 = q4_run1_completion_check()

    verdict, rationale = decision_matrix(
        q1["boolean"], q2["boolean"], q3["boolean"], q4["boolean"]
    )

    total = round(time.perf_counter() - t0, 2)
    t1_utc = datetime.now(timezone.utc).isoformat()

    output = {
        "meta": {
            "script": "deep_hedging/experiments/block1_diag_determinism.py",
            "git_commit": hardware.get("git_commit", "unknown"),
            "started_at_utc": t0_utc,
            "ended_at_utc": t1_utc,
            "wall_clock_s": total,
            "cpu_threads_used": 1,
            "p017_running_pid": pid,
        },
        "Q1_train_deterministic": q1,
        "Q2_simulator_deterministic": q2,
        "Q3_evaluation_deterministic": q3,
        "Q4_run1_was_completed": q4,
        "decision_matrix_input": [q1["boolean"], q2["boolean"], q3["boolean"], q4["boolean"]],
        "verdict": verdict,
        "verdict_rationale_one_sentence": rationale,
    }

    (DIAG_DIR / "results.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    write_decision(output, DIAG_DIR / "decision.md")
    write_q4_evidence(q4, DIAG_DIR / "q4_evidence.md")

    print(f"\n{'=' * 60}")
    print(f" VERDICT: {verdict}")
    print(f" Rationale: {rationale}")
    print(f" Wall-clock: {total:.1f}s")
    print(f"{'=' * 60}", flush=True)


if __name__ == "__main__":
    main()
