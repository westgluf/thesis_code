"""
Phase-1 sanity tests for HestonPDEDelta (Phase J).

Three tests:
  1. GBM limit (σ_v → 0) matches BS delta at ATM to within 1 %.
  2. Call price at (t=0, S_0, V_0) is in the expected range for canonical
     calibration (approximately [10, 14] for σ_eff ≈ 0.235, T=1, S=K=100).
  3. Delta surface monotone non-decreasing in S at fixed V = V_0.
"""
from __future__ import annotations

import math
import sys
import time

import torch

from deep_hedging.hedging.delta_hedger import BlackScholesDelta
from deep_hedging.hedging.heston_pde_delta import HestonPDEDelta

def test_gbm_limit() -> tuple[bool, str]:
    """PDE delta with σ_v → 0 should match BS delta at ATM."""
    print("  Test 1: GBM limit (σ_v = 0.001)...", flush=True)
    t0 = time.time()
    sigma_bs = 0.15
    V0 = sigma_bs ** 2  # 0.0225

    # Small σ_v to minimise stochastic-variance effects.
    pde = HestonPDEDelta(
        kappa=5.0, theta=V0, sigma_v=0.001,
        rho=0.0, V0=V0,
        K=100.0, T=1.0,
        S_max=400.0, V_max=0.5,
        n_S=200, n_V=60, n_t=400,
    )
    solve_time = time.time() - t0

    # BS delta at t=0, S=K=100
    bs = BlackScholesDelta(sigma=sigma_bs, K=100.0, T=1.0)
    S_atm = torch.tensor([100.0], dtype=torch.float64)
    bs_delta_atm = float(bs.compute_delta(torch.tensor(0.0, dtype=torch.float64), S_atm)[0])

    # PDE delta at t=0, S=100, V=V0
    pde_delta_atm = pde.delta(t=0.0, S=100.0, V=V0)

    abs_err = abs(pde_delta_atm - bs_delta_atm)
    rel_err = abs_err / max(abs(bs_delta_atm), 1e-9)
    ok = rel_err < 0.01
    status = "PASS" if ok else "FAIL"
    print(f"    BS delta (σ={sigma_bs}):   {bs_delta_atm:.6f}", flush=True)
    print(f"    PDE delta (σ_v=0.001): {pde_delta_atm:.6f}", flush=True)
    print(f"    |abs err| = {abs_err:.6f}  |rel err| = {rel_err*100:.3f}%  "
          f"[{status}]  (solve: {solve_time:.1f}s)", flush=True)
    msg = (f"|rel err| = {rel_err*100:.3f}%; BS={bs_delta_atm:.4f}, "
           f"PDE={pde_delta_atm:.4f}")
    return ok, msg

def test_call_price_sanity() -> tuple[bool, str]:
    """Heston call price at canonical calibration is in a sensible range."""
    print("  Test 2: Call price sanity (canonical calibration)...", flush=True)
    t0 = time.time()
    xi0 = 0.235 ** 2  # ≈ 0.055225
    # Canonical-ish Heston parameters (kappa, sigma_v TBD by Phase 2 calibration)
    # Here we just test ballpark magnitude with a moderate (kappa, sigma_v).
    pde = HestonPDEDelta(
        kappa=3.0, theta=xi0, sigma_v=0.5,
        rho=-0.7, V0=xi0,
        K=100.0, T=1.0,
        S_max=400.0, V_max=1.0,
        n_S=200, n_V=80, n_t=400,
    )
    price = pde.price(t=0.0, S=100.0, V=xi0)
    solve_time = time.time() - t0

    # At sigma_eff ≈ 0.235, T=1, S=K=100, BS call ≈ 9.35; Heston with
    # negative rho skews down slightly. Acceptable range [7, 13].
    ok = 7.0 <= price <= 13.0
    status = "PASS" if ok else "FAIL"
    print(f"    Heston PDE price = {price:.4f}  (expected [7, 13])  "
          f"[{status}]  (solve: {solve_time:.1f}s)", flush=True)
    msg = f"Heston call = {price:.4f} ATM; expected [7, 13]"
    return ok, msg

def test_monotonicity() -> tuple[bool, str]:
    """Delta surface Δ(t=0, S, V=V_0) should be monotone non-decreasing in S."""
    print("  Test 3: Delta monotonicity in S at fixed V...", flush=True)
    t0 = time.time()
    xi0 = 0.235 ** 2
    pde = HestonPDEDelta(
        kappa=3.0, theta=xi0, sigma_v=0.5,
        rho=-0.7, V0=xi0,
        K=100.0, T=1.0,
        S_max=400.0, V_max=1.0,
        n_S=200, n_V=80, n_t=400,
    )
    solve_time = time.time() - t0
    # Sample deltas at V=V0 for S in [50, 150]
    S_samples = torch.linspace(50.0, 150.0, 21)
    deltas = [pde.delta(t=0.0, S=float(s), V=xi0) for s in S_samples]
    # Check monotone non-decreasing
    monotone = all(d_next >= d_prev - 1e-4 for d_prev, d_next in zip(deltas, deltas[1:]))
    # Basic smoothness check: no wild swings
    max_diff = max(abs(d_next - d_prev)
                    for d_prev, d_next in zip(deltas, deltas[1:]))
    # At S=50 delta should be small; at S=150 delta should be near 1.
    atm = pde.delta(t=0.0, S=100.0, V=xi0)
    deep_otm = pde.delta(t=0.0, S=50.0, V=xi0)
    deep_itm = pde.delta(t=0.0, S=150.0, V=xi0)
    bell_shape = (deep_otm < atm < deep_itm) and deep_itm > 0.9 and deep_otm < 0.15
    ok = monotone and bell_shape
    status = "PASS" if ok else "FAIL"
    print(f"    monotone in S: {monotone}, max step: {max_diff:.4f}", flush=True)
    print(f"    Δ(S=50) = {deep_otm:.4f}, Δ(S=100) = {atm:.4f}, Δ(S=150) = {deep_itm:.4f}", flush=True)
    print(f"    bell_shape OK: {bell_shape}  [{status}]  (solve: {solve_time:.1f}s)",
          flush=True)
    msg = f"monotone={monotone}, bell_shape={bell_shape}"
    return ok, msg

def main() -> int:
    print("=" * 70, flush=True)
    print("  HestonPDEDelta sanity tests (Phase J, Phase 1)", flush=True)
    print("=" * 70, flush=True)
    tests = [
        ("GBM limit", test_gbm_limit),
        ("Call price sanity", test_call_price_sanity),
        ("Delta monotonicity", test_monotonicity),
    ]
    results = []
    for name, fn in tests:
        try:
            ok, msg = fn()
        except Exception as exc:
            ok = False
            msg = f"EXCEPTION: {exc}"
            import traceback
            traceback.print_exc()
        results.append((name, ok, msg))
        print()
    print("-" * 70, flush=True)
    for name, ok, msg in results:
        badge = "PASS" if ok else "FAIL"
        print(f"  [{badge}] {name}: {msg}", flush=True)
    print("-" * 70, flush=True)
    all_ok = all(r[1] for r in results)
    print(f"  Summary: {'ALL PASS' if all_ok else 'SOME FAIL'}", flush=True)
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
