"""
Heston PDE delta (Block 2, Phase J).

A faithful Markovian stochastic-volatility baseline for Section 6.3.1.
Solves the 2D Heston PDE via Crank-Nicolson finite-difference on a
non-uniform (log-spaced in S, uniform in V, uniform in t) grid and
caches the delta surface for bilinear interpolation during hedging.

This module is a NEW addition; the existing `PluginDelta` / `HestonDelta`
alias in `delta_hedger.py` remains untouched as a historical reference
for the dissertation text.

Heston (1993) risk-neutral dynamics:
    dS_t = √V_t S_t dW^S_t
    dV_t = κ(θ − V_t) dt + σ_v √V_t dW^V_t
    d⟨W^S, W^V⟩_t = ρ dt

European call PDE (r = q = 0):
    ∂u/∂t + ½ V S² ∂²u/∂S² + ρ σ_v V S ∂²u/∂S∂V
          + ½ σ_v² V ∂²u/∂V² + κ(θ − V) ∂u/∂V = 0
    u(T, S, V) = (S − K)⁺
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Analytical CIR variance moments (used for calibration)
# ---------------------------------------------------------------------------

def cir_mean_variance(
    kappa: float, theta: float, sigma_v: float, V0: float, T: float,
) -> tuple[float, float]:
    """Analytical first two moments of the CIR variance process V_T given V_0.

    For dV_t = κ(θ − V_t) dt + σ_v √V_t dW_t with V_0 given:

      E[V_T | V_0] = V_0 · e^(−κT) + θ (1 − e^(−κT))
      Var[V_T | V_0] = V_0 · (σ_v²/κ) · (e^(−κT) − e^(−2κT))
                      + θ · (σ_v²/(2κ)) · (1 − e^(−κT))²

    Returns (E[V_T], Var[V_T]).
    """
    if kappa <= 0 or sigma_v < 0 or theta < 0 or V0 < 0 or T <= 0:
        return float("nan"), float("nan")
    e_kT = math.exp(-kappa * T)
    e_2kT = math.exp(-2 * kappa * T)
    mean = V0 * e_kT + theta * (1.0 - e_kT)
    var = (V0 * (sigma_v ** 2 / kappa) * (e_kT - e_2kT)
           + theta * (sigma_v ** 2 / (2.0 * kappa)) * (1.0 - e_kT) ** 2)
    return float(mean), float(var)

# ---------------------------------------------------------------------------
# Calibration: match rough Bergomi V_T moments via 2-parameter Heston fit
# ---------------------------------------------------------------------------

def calibrate_heston_price_match(
    target_V0: float,
    target_EVT: float,
    target_call_ATM: float,
    rho_rbergomi: float,
    K: float,
    S0: float,
    T: float,
    enforce_feller: bool = True,
    kappa_grid: tuple[float, ...] = (1.0, 2.0, 3.0, 5.0, 8.0),
    n_S: int = 200,
    n_V: int = 80,
    n_t: int = 400,
    S_max: float = 400.0,
    V_max: float = 1.0,
) -> dict[str, Any]:
    """Calibrate (kappa, sigma_v) by matching the ATM Heston PDE call price.

    Strategy:
      - Fix V_0 = theta = target_V0 and rho = rho_rbergomi (direct transfer).
      - Grid-search over kappa in `kappa_grid`.
      - For each kappa, solve for sigma_v that matches target_call_ATM via
        a bisection on (sigma_v_min, sigma_v_max) constrained to Feller:
            2 * kappa * theta >= sigma_v^2  =>  sigma_v <= sqrt(2 kappa theta)
      - Pick the (kappa, sigma_v) giving the smallest ATM price error.

    This approach is preferred over analytical-moment matching because rough
    Bergomi's heavy-tailed variance process (Var[V_T] ≈ 0.10 at η=1.9)
    requires sigma_v values that violate Feller under Heston, producing
    physically-unrealistic PDE solutions. Matching the call price subject
    to Feller gives a physically-meaningful Heston surrogate whose hedging
    behaviour is well-defined.

    Returns dict with kappa, sigma_v, theta, V0, rho, achieved call price,
    rel error, and all search details for audit.
    """
    theta = target_V0

    best = None
    grid = []

    for kappa in kappa_grid:
        # Feller upper bound for sigma_v at this kappa/theta
        sigma_v_max_feller = math.sqrt(2.0 * kappa * theta)
        if enforce_feller:
            lo, hi = 0.05, sigma_v_max_feller
        else:
            lo, hi = 0.05, 2.0

        # If sigma_v lower bound already gives price > target, bisection is
        # useless — price is monotone DECREASING in sigma_v for ρ < 0 (negative
        # skew), so we need to go HIGHER in sigma_v. If upper gives price >
        # target, hi needs to increase.
        # (The caller relies on Feller OR wider range; see calibrate_main_range.)

        # Bisection to match call price
        def call_price_at(sigma_v: float) -> float:
            pde = HestonPDEDelta(
                kappa=kappa, theta=theta, sigma_v=sigma_v,
                rho=rho_rbergomi, V0=target_V0,
                K=K, T=T,
                S_max=S_max, V_max=V_max,
                n_S=n_S, n_V=n_V, n_t=n_t,
            )
            return pde.price(t=0.0, S=S0, V=target_V0)

        # Call price dependence on sigma_v depends on sign of ρ:
        #   ρ > 0: call price increases with sigma_v (positive skew effect)
        #   ρ < 0: call price decreases with sigma_v (negative skew → lower ATM)
        #   ρ = 0: call price increases with sigma_v (pure convexity / vega)
        # Detect direction from endpoints, then bisect accordingly.
        p_lo = call_price_at(lo)
        p_hi = call_price_at(hi)
        decreasing = p_hi < p_lo

        if (decreasing and p_lo < target_call_ATM) or (not decreasing and p_hi < target_call_ATM):
            # Target exceeds the achievable maximum; use the endpoint with
            # higher price.
            if p_hi > p_lo:
                sigma_v_star = hi
                price_star = p_hi
            else:
                sigma_v_star = lo
                price_star = p_lo
        elif (decreasing and p_hi > target_call_ATM) or (not decreasing and p_lo > target_call_ATM):
            # Target is below the achievable minimum; use the endpoint with
            # lower price.
            if p_hi < p_lo:
                sigma_v_star = hi
                price_star = p_hi
            else:
                sigma_v_star = lo
                price_star = p_lo
        else:
            # Target is bracketed → bisect
            for _ in range(25):
                mid = 0.5 * (lo + hi)
                p_mid = call_price_at(mid)
                if decreasing:
                    # Higher sigma_v -> lower price
                    if p_mid > target_call_ATM:
                        lo = mid  # need higher sigma_v
                        p_lo = p_mid
                    else:
                        hi = mid  # need lower sigma_v
                        p_hi = p_mid
                else:
                    if p_mid > target_call_ATM:
                        hi = mid
                        p_hi = p_mid
                    else:
                        lo = mid
                        p_lo = p_mid
                if abs(p_hi - p_lo) < 1e-4:
                    break
            sigma_v_star = 0.5 * (lo + hi)
            price_star = call_price_at(sigma_v_star)

        rel_err = abs(price_star - target_call_ATM) / target_call_ATM
        evt_fit, varvt_fit = cir_mean_variance(
            kappa, theta, sigma_v_star, target_V0, T,
        )
        entry = {
            "kappa": kappa,
            "sigma_v": sigma_v_star,
            "call_price": price_star,
            "rel_err_call": rel_err,
            "E_VT_fit": evt_fit,
            "Var_VT_fit": varvt_fit,
            "feller_slack": 2.0 * kappa * theta - sigma_v_star ** 2,
        }
        grid.append(entry)

        if best is None or rel_err < best["rel_err_call"]:
            best = entry

    assert best is not None

    verdict = "PASS" if best["rel_err_call"] < 0.02 else (
        "MARGINAL" if best["rel_err_call"] < 0.05 else "FAIL"
    )

    return {
        "kappa": best["kappa"],
        "sigma_v": best["sigma_v"],
        "V0": target_V0,
        "theta": theta,
        "rho": rho_rbergomi,
        "target_call_ATM": target_call_ATM,
        "achieved_call_ATM": best["call_price"],
        "rel_err_call": best["rel_err_call"],
        "E_VT_target": target_EVT,
        "E_VT_achieved": best["E_VT_fit"],
        "Var_VT_rbergomi_empirical": None,  # filled in by caller if available
        "Var_VT_heston_analytical": best["Var_VT_fit"],
        "grid_search": grid,
        "feller_slack": best["feller_slack"],
        "enforce_feller": enforce_feller,
        "verdict": verdict,
    }

def calibrate_heston_to_rbergomi(
    target_V0: float = 0.055225,
    target_EVT: float | None = None,
    target_VarVT: float | None = None,
    rho_rbergomi: float = -0.7,
    T: float = 1.0,
    kappa_init: float = 5.0,
    sigma_v_init: float = 0.8,
) -> dict[str, Any]:
    """Fit (κ, σ_v) by matching E[V_T] and Var[V_T] moments.

    V_0 = θ = target_V0 are fixed analytically. ρ = rho_rbergomi. If the
    target moments are not provided, the caller must supply them from a
    rough Bergomi MC sample (see `heston_pde_evaluation.py`).

    Returns a dict with {kappa, sigma_v, V0, theta, rho, target_moments,
    achieved_moments, rel_err_EVT, rel_err_VarVT, verdict}.
    """
    from scipy.optimize import minimize

    if target_EVT is None or target_VarVT is None:
        raise ValueError("target_EVT and target_VarVT must be supplied; use "
                         "rough Bergomi MC sample to compute.")

    theta = target_V0

    def objective(x):
        k, s = float(x[0]), float(x[1])
        if k <= 0.05 or s <= 0.0:
            return 1e10
        evt, varvt = cir_mean_variance(k, theta, s, target_V0, T)
        if not math.isfinite(evt) or not math.isfinite(varvt):
            return 1e10
        err_mean = (evt - target_EVT) ** 2 / target_EVT ** 2
        err_var = (varvt - target_VarVT) ** 2 / target_VarVT ** 2
        return err_mean + err_var

    x0 = [kappa_init, sigma_v_init]
    result = minimize(
        objective, x0, method="Nelder-Mead",
        options={"xatol": 1e-6, "fatol": 1e-10, "maxiter": 5000, "adaptive": True},
    )
    kappa_fit = float(result.x[0])
    sigma_v_fit = float(result.x[1])
    evt_fit, varvt_fit = cir_mean_variance(
        kappa_fit, theta, sigma_v_fit, target_V0, T,
    )

    rel_err_mean = abs(evt_fit - target_EVT) / target_EVT if target_EVT else float("nan")
    rel_err_var = abs(varvt_fit - target_VarVT) / target_VarVT if target_VarVT else float("nan")
    verdict = ("PASS" if rel_err_mean < 0.05 and rel_err_var < 0.05
               else ("MARGINAL" if rel_err_mean < 0.10 and rel_err_var < 0.10
                     else "FAIL"))

    return {
        "kappa": kappa_fit,
        "sigma_v": sigma_v_fit,
        "V0": target_V0,
        "theta": theta,
        "rho": rho_rbergomi,
        "target_moments": {"E_VT": target_EVT, "Var_VT": target_VarVT},
        "achieved_moments": {"E_VT": evt_fit, "Var_VT": varvt_fit},
        "rel_err_EVT": rel_err_mean,
        "rel_err_VarVT": rel_err_var,
        "calibration_objective": float(result.fun),
        "optimiser_success": bool(result.success),
        "verdict": verdict,
    }

# ---------------------------------------------------------------------------
# PDE grid setup
# ---------------------------------------------------------------------------

def _build_S_grid(S_max: float, K: float, n_S: int) -> np.ndarray:
    """Non-uniform log-spaced S grid, refined near the strike K.

    We use a two-region grid: linear spacing in [0, S_max] is simple
    but inefficient near K. Here we use a sinh-concentrated grid:
        S_i = K + c · sinh(u_i), u_i linear in [u_min, u_max]
    with c = 0.25 K chosen so that ~half the points lie in [K/2, 2K].
    Clamp to [1e-4, S_max].
    """
    c = 0.25 * K
    u_min = math.asinh((0.0 - K) / c)
    u_max = math.asinh((S_max - K) / c)
    u = np.linspace(u_min, u_max, n_S)
    S = K + c * np.sinh(u)
    # Clamp tiny negative / below-eps values to a small positive spot
    S = np.maximum(S, 1e-4)
    S[-1] = S_max
    return S

def _build_V_grid(V_max: float, V0: float, n_V: int) -> np.ndarray:
    """Uniform V grid.

    We deliberately use a uniform spacing (rather than a sinh-concentrated
    grid near V0) because the explicit treatment of the mixed derivative
    ρ σ_v V S ∂²u/∂S∂V in the Crank-Nicolson–Lie-splitting scheme has a
    CFL-like stability condition ∝ 1/(h_S · h_V). Very fine h_V near V_0
    destabilises the cross term for σ_v ≳ 0.3.

    A uniform grid with h_V ≈ V_max / n_V gives ~0.01 for V_max=1, n_V=80,
    which keeps the mixed-term eigenvalue within the stability bound for
    the grid sizes used here.
    """
    V = np.linspace(0.0, V_max, n_V)
    return V

# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

@dataclass
class _GridMetadata:
    S_grid: np.ndarray
    V_grid: np.ndarray
    t_grid: np.ndarray
    n_S: int
    n_V: int
    n_t: int
    S_max: float
    V_max: float

class HestonPDEDelta:
    """Heston PDE delta via 2D Crank-Nicolson finite-difference.

    Solves the Heston PDE on a non-uniform (S, V, t) grid and caches the
    delta surface ∂u/∂S(t, S, V) for bilinear interpolation during hedging.

    The solver uses Crank-Nicolson for the S-diffusion and V-diffusion+drift
    operators and handles the mixed ρ σ_v V S ∂²u/∂S∂V term explicitly
    (at the current time level). This explicit treatment is standard
    for "partial-implicit" schemes and is stable at the grid sizes used here.

    Boundary conditions:
      S = 0:     u = 0 (call worthless)
      S = S_max: u = S_max − K (r = 0 assumed)
      V = 0:     natural BC (PDE degenerate — one-sided V-derivative)
      V = V_max: ∂u/∂V = 0 (Neumann)

    Parameters
    ----------
    kappa, theta, sigma_v, rho, V0 : float
        Heston parameters.
    K, T : float
        Strike and maturity.
    S_max, V_max : float
        Truncation bounds.
    n_S, n_V, n_t : int
        Grid sizes.
    device : str | torch.device
        Kept for interface consistency; PDE is solved in NumPy/SciPy.
    """

    def __init__(
        self,
        kappa: float,
        theta: float,
        sigma_v: float,
        rho: float,
        V0: float,
        K: float,
        T: float,
        S_max: float = 400.0,
        V_max: float = 1.0,
        n_S: int = 200,
        n_V: int = 80,
        n_t: int = 400,
        device: str | torch.device = "cpu",
    ) -> None:
        if S_max <= 0 or V_max <= 0 or n_S < 10 or n_V < 10 or n_t < 10:
            raise ValueError("invalid grid parameters")
        if kappa < 0 or sigma_v < 0 or theta < 0 or V0 < 0:
            raise ValueError("Heston parameters must be non-negative (except rho)")

        self.kappa = float(kappa)
        self.theta = float(theta)
        self.sigma_v = float(sigma_v)
        self.rho = float(rho)
        self.V0 = float(V0)
        self.K = float(K)
        self.T = float(T)
        self.S_max = float(S_max)
        self.V_max = float(V_max)
        self.n_S = int(n_S)
        self.n_V = int(n_V)
        self.n_t = int(n_t)
        self.device = torch.device(device)

        # Build grid and solve PDE
        self._grid = self._build_grid()
        self._u_surface, self._delta_surface = self._solve_pde()

        # Torch tensors for fast path-wise interpolation
        self._t_grid_t = torch.as_tensor(self._grid.t_grid, dtype=torch.float32)
        self._S_grid_t = torch.as_tensor(self._grid.S_grid, dtype=torch.float32)
        self._V_grid_t = torch.as_tensor(self._grid.V_grid, dtype=torch.float32)
        self._delta_surface_t = torch.as_tensor(
            self._delta_surface, dtype=torch.float32,
        )

    # ------------------------------------------------------------------
    # Grid construction
    # ------------------------------------------------------------------

    def _build_grid(self) -> _GridMetadata:
        S = _build_S_grid(self.S_max, self.K, self.n_S)
        V = _build_V_grid(self.V_max, self.V0, self.n_V)
        t = np.linspace(0.0, self.T, self.n_t + 1)
        return _GridMetadata(
            S_grid=S, V_grid=V, t_grid=t,
            n_S=self.n_S, n_V=self.n_V, n_t=self.n_t + 1,
            S_max=self.S_max, V_max=self.V_max,
        )

    # ------------------------------------------------------------------
    # PDE solver — Crank-Nicolson with explicit cross term
    # ------------------------------------------------------------------

    def _solve_pde(self) -> tuple[np.ndarray, np.ndarray]:
        """Solve the Heston PDE backward in time from t = T to t = 0.

        Returns (u_surface, delta_surface) each of shape (n_t+1, n_S, n_V).
        """
        S = self._grid.S_grid   # (n_S,)
        V = self._grid.V_grid   # (n_V,)
        t = self._grid.t_grid   # (n_t+1,)
        nS, nV = self.n_S, self.n_V

        # --- Precompute grid spacings (non-uniform in S and V) ---
        # For interior i = 1..nS-2:
        #   dS_plus[i]  = S[i+1] - S[i]
        #   dS_minus[i] = S[i] - S[i-1]
        dS_plus = np.diff(S)           # length nS-1
        dS_minus = np.roll(dS_plus, 1) # length nS-1 (shifted)
        dS_plus_int = dS_plus[1:]      # for i=1..nS-2: S[i+1] - S[i]
        dS_minus_int = dS_plus[:-1]    # for i=1..nS-2: S[i] - S[i-1]
        dS_sum_int = dS_plus_int + dS_minus_int

        dV_plus = np.diff(V)
        dV_plus_int = dV_plus[1:]
        dV_minus_int = dV_plus[:-1]
        dV_sum_int = dV_plus_int + dV_minus_int

        # Non-uniform central difference coefficients for ∂²/∂S²:
        #   a_S[i] =  2 / (dS_minus (dS_minus+dS_plus))        (i-1 term)
        #   b_S[i] = -2 / (dS_minus dS_plus)                    (i term)
        #   c_S[i] =  2 / (dS_plus  (dS_minus+dS_plus))         (i+1 term)
        a_SS = 2.0 / (dS_minus_int * dS_sum_int)
        c_SS = 2.0 / (dS_plus_int * dS_sum_int)
        b_SS = -(a_SS + c_SS)

        # ∂²/∂V² coefficients
        a_VV = 2.0 / (dV_minus_int * dV_sum_int)
        c_VV = 2.0 / (dV_plus_int * dV_sum_int)
        b_VV = -(a_VV + c_VV)

        # ∂/∂V (central) coefficients
        a_V_c = -dV_plus_int / (dV_minus_int * dV_sum_int)
        c_V_c = dV_minus_int / (dV_plus_int * dV_sum_int)
        b_V_c = (dV_plus_int - dV_minus_int) / (dV_plus_int * dV_minus_int)

        # ∂/∂S (central) coefficients — used only for cross term
        a_S_c = -dS_plus_int / (dS_minus_int * dS_sum_int)
        c_S_c = dS_minus_int / (dS_plus_int * dS_sum_int)
        b_S_c = (dS_plus_int - dS_minus_int) / (dS_plus_int * dS_minus_int)

        # --- Build the S-direction operator L_S (1D per V-slice) ---
        # L_S[u] at interior i: 0.5 V S_i^2 ∂²u/∂S²
        # Shape (nS, nS). Coefficients depend on V_j so we build L_S[j] per slice.

        # --- Build the V-direction operator L_V (1D per S-slice) ---
        # L_V[u] at interior j: 0.5 σ_v² V_j ∂²u/∂V² + κ(θ − V_j) ∂u/∂V
        # This doesn't depend on S, so we build it once.

        # Operator L_V as sparse (nV, nV):
        main = np.zeros(nV)
        upper = np.zeros(nV - 1)
        lower = np.zeros(nV - 1)
        # Interior j=1..nV-2
        V_int = V[1:-1]
        diff_coef = 0.5 * self.sigma_v ** 2 * V_int  # (nV-2,)
        drift_coef = self.kappa * (self.theta - V_int)  # (nV-2,)
        # Main diagonal (j)
        main[1:-1] = diff_coef * b_VV + drift_coef * b_V_c
        # Lower diagonal (j-1): stored in lower[j-1]
        lower[:-1] = diff_coef * a_VV + drift_coef * a_V_c
        # Upper diagonal (j+1): stored in upper[j]
        upper[1:] = diff_coef * c_VV + drift_coef * c_V_c

        # Boundary rows:
        # At V = 0 (j=0): use Feller-upwind: PDE degenerates to
        #   ∂u/∂t + 0.5 V S^2 ∂^2u/∂S^2 + κθ ∂u/∂V = 0
        # (the ρ σ_v V S ∂²/∂S∂V and σ_v² V ∂²/∂V² vanish because V=0)
        # We implement ∂u/∂V with a forward difference:
        #   (u[1] - u[0]) / (V[1] - V[0])
        if nV > 1:
            dV0 = V[1] - V[0]
            # Contribution to L_V at j=0: κ(θ − 0) · (u[1] − u[0])/dV0
            # = (κθ / dV0) · (u[1] − u[0])
            main[0] = -self.kappa * self.theta / dV0
            upper[0] = self.kappa * self.theta / dV0

        # At V = V_max (j=nV-1): Neumann ∂u/∂V = 0 => u[nV-1] = u[nV-2]
        # Implement as: main[nV-1] = 0, lower[nV-2] = 0 (handled below when solving)
        # We'll enforce u[nV-1] = u[nV-2] after each time step by copy.

        # Build sparse matrix
        L_V = sp.diags([lower, main, upper], offsets=[-1, 0, 1],
                        shape=(nV, nV), format="csc")

        # --- Terminal condition ---
        # u(T, S, V) = max(S - K, 0) — independent of V
        u = np.maximum(S[:, None] - self.K, 0.0)  # (nS, nV)
        u = np.broadcast_to(u, (nS, nV)).copy()

        # Storage for the full solution surface u(t, S, V)
        u_surface = np.zeros((self._grid.n_t, nS, nV), dtype=np.float64)
        u_surface[-1] = u

        # Uniform time step size
        dt = self.T / self.n_t

        # Crank-Nicolson coefficients: we use θ = 0.5 (Crank-Nicolson)
        # For V direction, we solve (I − (dt/2) L_V) u^{n} = (I + (dt/2) L_V) u^{n+1}
        # For S direction, we solve per-V-slice (I − (dt/2) L_S^j) u^{n} = (I + (dt/2) L_S^j) u^{n+1}
        # Cross term is handled explicitly at u^{n+1}.
        # We use a fractional (sequential) implicit approach:
        #   1) Apply explicit V+cross term to get u_star
        #   2) Apply implicit S to get u_starstar
        #   3) Apply implicit V to get u^{n}
        # This is an ADI-like scheme. To keep it simple and stable,
        # we instead do a full Crank-Nicolson on the combined L_S + L_V
        # at each (V slice, S slice) independently (operator splitting,
        # θ = 0.5 for each sub-step).

        # Build S-direction operator for each V-slice
        L_S_per_V: list[sp.csc_matrix] = []
        for j in range(nV):
            V_j = V[j]
            coef_SS = 0.5 * V_j * S[1:-1] ** 2   # (nS-2,)
            main_S = np.zeros(nS)
            upper_S = np.zeros(nS - 1)
            lower_S = np.zeros(nS - 1)
            main_S[1:-1] = coef_SS * b_SS
            lower_S[:-1] = coef_SS * a_SS
            upper_S[1:] = coef_SS * c_SS
            # Boundary rows for S:
            # S = 0 (i=0): u = 0 always, row is all zero (treat as Dirichlet pin)
            # S = S_max (i=nS-1): u = S_max - K; same treatment
            L = sp.diags([lower_S, main_S, upper_S], offsets=[-1, 0, 1],
                          shape=(nS, nS), format="csc")
            L_S_per_V.append(L)

        # Pre-factor CN matrices (I - dt/2 L) for both directions
        # For each V slice, we LU-decompose (I - dt/2 L_S_j) once
        I_S = sp.eye(nS, format="csc")
        S_implicit_solvers = []
        S_explicit_mats = []
        for j in range(nV):
            A = (I_S - 0.5 * dt * L_S_per_V[j]).tocsc()
            B = (I_S + 0.5 * dt * L_S_per_V[j]).tocsc()
            # Enforce Dirichlet rows for S=0 and S=S_max in A (u at those indices fixed)
            A = A.tolil()
            A[0, :] = 0.0
            A[0, 0] = 1.0
            A[-1, :] = 0.0
            A[-1, -1] = 1.0
            A = A.tocsc()
            solver = spla.splu(A)
            S_implicit_solvers.append(solver)
            S_explicit_mats.append(B)

        I_V = sp.eye(nV, format="csc")
        A_V = (I_V - 0.5 * dt * L_V).tocsc()
        # Enforce V=V_max Neumann by equating last row to (u[nV-1] - u[nV-2] = 0)
        A_V = A_V.tolil()
        A_V[-1, :] = 0.0
        A_V[-1, -1] = 1.0
        A_V[-1, -2] = -1.0
        A_V = A_V.tocsc()
        V_implicit_solver = spla.splu(A_V)
        B_V = (I_V + 0.5 * dt * L_V).tocsc()

        # Cross-term coefficient at each (i, j): ρ σ_v V_j S_i
        cross_coef = self.rho * self.sigma_v * np.outer(S, V)  # (nS, nV)

        def apply_F0(u_arr: np.ndarray) -> np.ndarray:
            """F_0(u) = ρ σ_v V S ∂²u/∂S∂V  — the mixed-derivative operator."""
            result = np.zeros_like(u_arr)
            dS_total = (S[2:] - S[:-2])[:, None]
            dV_total = (V[2:] - V[:-2])[None, :]
            d2 = (
                u_arr[2:, 2:]   - u_arr[2:, :-2]
              - u_arr[:-2, 2:]  + u_arr[:-2, :-2]
            ) / (dS_total * dV_total)
            result[1:-1, 1:-1] = cross_coef[1:-1, 1:-1] * d2
            return result

        def apply_F1(u_arr: np.ndarray) -> np.ndarray:
            """F_1(u) = 0.5 σ_v² V ∂²u/∂V² + κ(θ − V) ∂u/∂V  — the V operator."""
            return (L_V @ u_arr.T).T  # (nS, nV) returned

        def apply_F2(u_arr: np.ndarray) -> np.ndarray:
            """F_2(u) = 0.5 V S² ∂²u/∂S²  — the S operator, V-slice dependent."""
            out = np.zeros_like(u_arr)
            for j in range(nV):
                out[:, j] = L_S_per_V[j] @ u_arr[:, j]
            return out

        def apply_boundaries(u_arr: np.ndarray) -> None:
            u_arr[0, :] = 0.0
            u_arr[-1, :] = self.S_max - self.K
            u_arr[:, -1] = u_arr[:, -2]

        def solve_implicit_V(rhs: np.ndarray) -> np.ndarray:
            """Solve (I - θ dt L_V) u = rhs  row-wise (per S-slice)."""
            rhs_mod = rhs.copy()
            rhs_mod[:, -1] = 0.0  # Neumann enforcement
            out = np.empty_like(rhs)
            for i in range(nS):
                out[i, :] = V_implicit_solver.solve(rhs_mod[i, :])
            return out

        def solve_implicit_S(rhs: np.ndarray) -> np.ndarray:
            """Solve (I - θ dt L_S) u = rhs  column-wise (per V-slice)."""
            out = np.empty_like(rhs)
            for j in range(nV):
                rhs_j = rhs[:, j].copy()
                rhs_j[0] = 0.0
                rhs_j[-1] = self.S_max - self.K
                out[:, j] = S_implicit_solvers[j].solve(rhs_j)
            return out

        # --- Hundsdorfer-Verwer (HV) ADI scheme, θ = 0.5 ---
        # For each backward step, F = F_0 + F_1 + F_2 where F_1, F_2 are
        # the V and S diffusion operators (implicit-friendly) and F_0 is
        # the mixed-derivative operator (must be explicit).
        #
        # HV scheme:
        #   Y0 = u_n + dt * F(u_n)
        #   Y1 = Y0 + θ·dt · (F_1(Y1) - F_1(u_n))   [implicit V correction]
        #   Y2 = Y1 + θ·dt · (F_2(Y2) - F_2(u_n))   [implicit S correction]
        #   Y_tilde = Y0 + 0.5·dt · F_0(Y2 - u_n)   [mixed-term rebalance]
        #   Y3 = Y_tilde + θ·dt · (F_1(Y3) - F_1(u_n))
        #   Y4 = Y3 + θ·dt · (F_2(Y4) - F_2(u_n))
        #   u_{n+1} = Y4
        #
        # This scheme is second-order and unconditionally stable for the
        # Heston PDE at θ = 0.5. The implicit V and S matrices are already
        # pre-factored as A_V (with 0.5·dt factor) and A_S_j.
        #
        # Note that the pre-factored matrices A_V = (I - 0.5 dt L_V) and
        # A_S_j = (I - 0.5 dt L_S_j) use the timestep coefficient θ·dt = 0.5·dt
        # which matches the HV θ factor exactly (so the pre-factored solvers
        # are directly reusable).

        theta_hv = 0.5

        for step in range(self.n_t - 1, -1, -1):
            u_n = u.copy()
            F0_un = apply_F0(u_n)
            F1_un = apply_F1(u_n)
            F2_un = apply_F2(u_n)
            F_un = F0_un + F1_un + F2_un

            # Predictor Y0 = u_n + dt * F(u_n)
            Y0 = u_n + dt * F_un
            apply_boundaries(Y0)

            # Corrector 1: Y1 = Y0 + θ*dt*(F_1(Y1) - F_1(u_n))
            # => (I - θ*dt*L_V) Y1 = Y0 - θ*dt*F_1(u_n)
            rhs1 = Y0 - theta_hv * dt * F1_un
            Y1 = solve_implicit_V(rhs1)
            apply_boundaries(Y1)

            # Corrector 2: Y2 = Y1 + θ*dt*(F_2(Y2) - F_2(u_n))
            # => (I - θ*dt*L_S) Y2 = Y1 - θ*dt*F_2(u_n)
            rhs2 = Y1 - theta_hv * dt * F2_un
            Y2 = solve_implicit_S(rhs2)
            apply_boundaries(Y2)

            # Mixed-term rebalance:
            # Y_tilde = Y0 + 0.5*dt * F_0(Y2 - u_n)
            F0_Y2_minus_un = apply_F0(Y2 - u_n)
            Y_tilde = Y0 + 0.5 * dt * F0_Y2_minus_un
            apply_boundaries(Y_tilde)

            # Final correctors: V, then S
            rhs3 = Y_tilde - theta_hv * dt * F1_un
            Y3 = solve_implicit_V(rhs3)
            apply_boundaries(Y3)

            rhs4 = Y3 - theta_hv * dt * F2_un
            Y4 = solve_implicit_S(rhs4)
            apply_boundaries(Y4)

            u = Y4
            u_surface[step] = u

        # --- Compute delta surface Δ(t, S, V) by central difference in S ---
        delta_surface = np.zeros_like(u_surface)
        # Interior i
        dS_plus_full = np.diff(S)
        dS_sum_full = dS_plus_full[:-1] + dS_plus_full[1:]  # (nS-2,)
        delta_surface[:, 1:-1, :] = (
            u_surface[:, 2:, :] - u_surface[:, :-2, :]
        ) / dS_sum_full[None, :, None]
        # One-sided at boundaries
        delta_surface[:, 0, :] = (u_surface[:, 1, :] - u_surface[:, 0, :]) / (S[1] - S[0])
        delta_surface[:, -1, :] = (u_surface[:, -1, :] - u_surface[:, -2, :]) / (S[-1] - S[-2])

        # Clip to [0, 1] (European call bounds)
        delta_surface = np.clip(delta_surface, 0.0, 1.0)

        return u_surface, delta_surface

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    @property
    def S_grid(self) -> np.ndarray:
        return self._grid.S_grid

    @property
    def V_grid(self) -> np.ndarray:
        return self._grid.V_grid

    @property
    def t_grid(self) -> np.ndarray:
        return self._grid.t_grid

    def price(self, t: float, S: float, V: float) -> float:
        """Return Heston call price u(t, S, V) by bilinear interpolation."""
        return float(self._interp_scalar(self._u_surface, t, S, V))

    def delta(self, t: float, S: float, V: float) -> float:
        """Return Heston delta Δ(t, S, V) by bilinear interpolation."""
        return float(self._interp_scalar(self._delta_surface, t, S, V))

    # ------------------------------------------------------------------
    # Interpolation
    # ------------------------------------------------------------------

    def _interp_scalar(self, surface: np.ndarray, t: float, S: float, V: float) -> float:
        """Trilinear interpolation in (t, S, V) using numpy."""
        t = float(np.clip(t, self.t_grid[0], self.t_grid[-1]))
        S = float(np.clip(S, self.S_grid[0], self.S_grid[-1]))
        V = float(np.clip(V, self.V_grid[0], self.V_grid[-1]))

        ti = int(np.searchsorted(self.t_grid, t, side="right") - 1)
        ti = max(0, min(ti, surface.shape[0] - 2))
        Si = int(np.searchsorted(self.S_grid, S, side="right") - 1)
        Si = max(0, min(Si, surface.shape[1] - 2))
        Vi = int(np.searchsorted(self.V_grid, V, side="right") - 1)
        Vi = max(0, min(Vi, surface.shape[2] - 2))

        t_alpha = (t - self.t_grid[ti]) / (self.t_grid[ti + 1] - self.t_grid[ti])
        S_alpha = (S - self.S_grid[Si]) / (self.S_grid[Si + 1] - self.S_grid[Si])
        V_alpha = (V - self.V_grid[Vi]) / (self.V_grid[Vi + 1] - self.V_grid[Vi])

        def _2d(ti):
            c00 = surface[ti, Si, Vi]
            c01 = surface[ti, Si, Vi + 1]
            c10 = surface[ti, Si + 1, Vi]
            c11 = surface[ti, Si + 1, Vi + 1]
            return ((1 - S_alpha) * (1 - V_alpha) * c00
                    + (1 - S_alpha) * V_alpha * c01
                    + S_alpha * (1 - V_alpha) * c10
                    + S_alpha * V_alpha * c11)

        v0 = _2d(ti)
        v1 = _2d(ti + 1)
        return (1 - t_alpha) * v0 + t_alpha * v1

    # ------------------------------------------------------------------
    # Hedge paths (vectorised over batch)
    # ------------------------------------------------------------------

    def hedge_paths(self, S: Tensor, V: Tensor, t_grid: Tensor | None = None) -> Tensor:
        """Evaluate Heston PDE delta at every hedging step.

        Args:
            S: (batch, n_steps + 1) price paths
            V: (batch, n_steps + 1) variance paths
            t_grid: optional (n_steps + 1,) time grid. If None, use uniform
                [0, T] in n_steps steps.

        Returns:
            deltas: (batch, n_steps) hedge ratios (clamped to [0, 1]).
        """
        batch, n_plus_1 = S.shape
        n = n_plus_1 - 1

        if t_grid is None:
            t_grid = torch.arange(n, dtype=S.dtype, device=S.device) * (self.T / n)
        else:
            t_grid = t_grid[:n].to(S.dtype)

        # Slices for times 0..n-1 (delta is decided at start of each step)
        S_k = S[:, :-1].float()  # (batch, n)
        V_k = V[:, :-1].float()  # (batch, n)
        V_k = torch.clamp(V_k, min=0.0, max=float(self.V_max - 1e-6))
        S_k_c = torch.clamp(S_k, min=float(self.S_grid[0] + 1e-6),
                             max=float(self.S_grid[-1] - 1e-6))
        t_k = t_grid[:n].float()  # (n,)
        t_k = torch.clamp(t_k, min=float(self.t_grid[0]),
                          max=float(self.t_grid[-1] - 1e-9))

        # Trilinear interpolation, vectorised. We find integer indices and
        # fractional offsets in each axis, then blend 8 corners.
        T_grid = self._t_grid_t  # (n_t+1,)
        S_grid = self._S_grid_t  # (n_S,)
        V_grid = self._V_grid_t  # (n_V,)
        surface = self._delta_surface_t  # (n_t+1, n_S, n_V)

        # Indices (floored). Use searchsorted with right side minus 1.
        ti = torch.searchsorted(T_grid, t_k.contiguous(), right=True) - 1
        ti = torch.clamp(ti, 0, T_grid.numel() - 2)
        # t_alpha per (n,)
        t_alpha = (t_k - T_grid[ti]) / (T_grid[ti + 1] - T_grid[ti])
        # Broadcast t_alpha over batch
        t_alpha = t_alpha.unsqueeze(0).expand(batch, -1)  # (batch, n)
        ti_b = ti.unsqueeze(0).expand(batch, -1)           # (batch, n)

        # S/V indices per (batch, n)
        Si = torch.searchsorted(S_grid, S_k_c.contiguous(), right=True) - 1
        Si = torch.clamp(Si, 0, S_grid.numel() - 2)
        Vi = torch.searchsorted(V_grid, V_k.contiguous(), right=True) - 1
        Vi = torch.clamp(Vi, 0, V_grid.numel() - 2)

        S_alpha = (S_k_c - S_grid[Si]) / (S_grid[Si + 1] - S_grid[Si])
        V_alpha = (V_k - V_grid[Vi]) / (V_grid[Vi + 1] - V_grid[Vi])

        # Gather 8 corners: surface[ti+d_t, Si+d_s, Vi+d_v] for d ∈ {0,1}³
        def _gather(dt: int, ds: int, dv: int) -> Tensor:
            return surface[ti_b + dt, Si + ds, Vi + dv]

        c000 = _gather(0, 0, 0)
        c001 = _gather(0, 0, 1)
        c010 = _gather(0, 1, 0)
        c011 = _gather(0, 1, 1)
        c100 = _gather(1, 0, 0)
        c101 = _gather(1, 0, 1)
        c110 = _gather(1, 1, 0)
        c111 = _gather(1, 1, 1)

        # Blend along V
        c00 = c000 * (1 - V_alpha) + c001 * V_alpha
        c01 = c010 * (1 - V_alpha) + c011 * V_alpha
        c10 = c100 * (1 - V_alpha) + c101 * V_alpha
        c11 = c110 * (1 - V_alpha) + c111 * V_alpha
        # Blend along S
        c0 = c00 * (1 - S_alpha) + c01 * S_alpha
        c1 = c10 * (1 - S_alpha) + c11 * S_alpha
        # Blend along t
        delta = c0 * (1 - t_alpha) + c1 * t_alpha

        delta = torch.clamp(delta, 0.0, 1.0)
        return delta.to(S.dtype)
