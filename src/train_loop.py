from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple, Any
import os
import csv
import torch


def _extract_from_data(data: Dict[str, Any]):
    return (
        data["S_tr_t"],
        data["Z_tr_t"],
        data["F_tr_t"],
        data["S_va_t"],
        data["Z_va_t"],
        data["F_va_t"],
        float(data["p0_true_mc"]),
        float(data["lam_cost"]),
    )


def train_loop(
    model,
    objective: Callable[[torch.Tensor], torch.Tensor],
    rollout_fn: Callable,
    pl_fn: Callable,
    S_tr_t: Optional[torch.Tensor] = None,
    Z_tr_t: Optional[torch.Tensor] = None,
    F_tr_t: Optional[torch.Tensor] = None,
    S_va_t: Optional[torch.Tensor] = None,
    Z_va_t: Optional[torch.Tensor] = None,
    F_va_t: Optional[torch.Tensor] = None,
    p0_true_mc: Optional[float] = None,
    lam_cost: float = 0.0,
    opt: Optional[torch.optim.Optimizer] = None,
    epochs: int = 60,
    batch_size: int = 2048,
    patience: int = 10,
    grad_clip: float = 1.0,
    out_dir: str = "results/gbm_deephedge",
    w_value_fn: Optional[Callable[[], float]] = None,
    trange=None,
    data: Optional[Dict[str, Any]] = None,
    **_ignore_kwargs,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], List[Dict[str, float]]]:
    if trange is None:
        from tqdm import trange as _trange
        trange = _trange

    if data is not None:
        S_tr_t, Z_tr_t, F_tr_t, S_va_t, Z_va_t, F_va_t, p0_true_mc, lam_cost = _extract_from_data(data)

    if any(x is None for x in [S_tr_t, Z_tr_t, F_tr_t, S_va_t, Z_va_t, F_va_t, p0_true_mc, opt]):
        raise TypeError("train_loop: missing required inputs (tensors/opt/p0_true_mc). Provide them or pass data=dict(...).")

    best_val = float("inf")
    best_state = None
    bad = 0
    train_log: List[Dict[str, float]] = []

    Ntr = int(F_tr_t.shape[0])

    for ep in trange(int(epochs), desc="Training"):
        model.train()
        perm = torch.randperm(Ntr, device=F_tr_t.device)
        total_loss = 0.0
        nb = 0

        for start in range(0, Ntr, int(batch_size)):
            idx = perm[start:start + int(batch_size)]
            F_b = F_tr_t[idx]
            S_b = S_tr_t[idx]
            Z_b = Z_tr_t[idx]

            opt.zero_grad()
            deltas = rollout_fn(model, F_b)
            pl = pl_fn(S_b, deltas, Z_b, float(p0_true_mc), float(lam_cost))

            loss = objective(pl)
            loss = loss + (1e-4 * (deltas ** 2).mean()) + (0.0 * ((deltas[:, 1:] - deltas[:, :-1]) ** 2).mean())

            loss.backward()
            if grad_clip is not None and float(grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
            opt.step()

            total_loss += float(loss.detach().cpu())
            nb += 1

        model.eval()
        with torch.no_grad():
            deltas_va = rollout_fn(model, F_va_t)
            pl_va = pl_fn(S_va_t, deltas_va, Z_va_t, float(p0_true_mc), float(lam_cost))
            val_loss = float(objective(pl_va).detach().cpu().item())

        train_loss_epoch = float(total_loss) / float(max(nb, 1))
        lr_now = float(opt.param_groups[0].get("lr", 0.0))
        w_now = float(w_value_fn()) if w_value_fn is not None else float("nan")

        train_log.append(
            {
                "epoch": float(ep),
                "train_loss": float(train_loss_epoch),
                "val_loss": float(val_loss),
                "lr": float(lr_now),
                "w": float(w_now),
            }
        )

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= int(patience):
                break

    last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is None:
        best_state = last_state

    os.makedirs(out_dir, exist_ok=True)
    try:
        torch.save(best_state, os.path.join(out_dir, "best_state.pt"))
        torch.save(last_state, os.path.join(out_dir, "last_state.pt"))
    except Exception as e:
        print("Warning: could not save checkpoints:", e)

    try:
        with open(os.path.join(out_dir, "train_log.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "lr", "w"])
            writer.writeheader()
            for row in train_log:
                writer.writerow(row)
    except Exception as e:
        print("Warning: could not save train_log.csv:", e)

    return best_state, last_state, train_log
