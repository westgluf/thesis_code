from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

import torch

from src.hedge_core import compute_pl_torch, rollout_strategy


TrainLog = List[Dict[str, int | float]]

_DATA_KEYS = (
    "F_tr",
    "S_tr",
    "Z_tr",
    "F_va",
    "S_va",
    "Z_va",
    "p0_true_mc",
    "lam_cost",
)
_DEFAULT_GRAD_CLIP = 1.0


def _validate_data(data: Dict[str, Any]) -> Dict[str, Any]:
    missing = [key for key in _DATA_KEYS if key not in data]
    extra = sorted(set(data) - set(_DATA_KEYS))
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"missing keys: {missing}")
        if extra:
            parts.append(f"unexpected keys: {extra}")
        raise TypeError(f"train_loop data payload invalid ({'; '.join(parts)})")
    return data


def _extract_w_value(objective_fn: Callable[[torch.Tensor], torch.Tensor]) -> float:
    monitor_value = getattr(objective_fn, "monitor_value", None)
    if callable(monitor_value):
        return float(monitor_value())
    w = getattr(objective_fn, "w", None)
    if isinstance(w, torch.Tensor):
        return float(w.detach().cpu().item())
    return float("nan")


def train_loop(
    *,
    model,
    optimizer: torch.optim.Optimizer,
    objective_fn: Callable[[torch.Tensor], torch.Tensor],
    data: Dict[str, Any],
    epochs: int,
    batch_size: int,
    patience: int,
    device: str | torch.device,
    trange=None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], TrainLog]:
    if trange is None:
        from tqdm import trange as _trange

        trange = _trange

    data = _validate_data(data)
    device = torch.device(device)

    F_tr = data["F_tr"]
    S_tr = data["S_tr"]
    Z_tr = data["Z_tr"]
    F_va = data["F_va"]
    S_va = data["S_va"]
    Z_va = data["Z_va"]
    p0_true_mc = float(data["p0_true_mc"])
    lam_cost = float(data["lam_cost"])

    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    train_log: TrainLog = []

    num_train = int(F_tr.shape[0])

    for ep in trange(int(epochs), desc="Training"):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        perm = torch.randperm(num_train, device=device)
        total_loss = 0.0
        num_batches = 0

        for start in range(0, num_train, int(batch_size)):
            idx = perm[start : start + int(batch_size)]
            F_batch = F_tr[idx]
            S_batch = S_tr[idx]
            Z_batch = Z_tr[idx]

            optimizer.zero_grad(set_to_none=True)
            deltas = rollout_strategy(model, F_batch)
            pl = compute_pl_torch(S_batch, deltas, Z_batch, p0_true_mc, lam_cost)
            loss = objective_fn(pl)
            loss = loss + (1e-4 * (deltas**2).mean()) + (0.0 * ((deltas[:, 1:] - deltas[:, :-1]) ** 2).mean())

            loss.backward()
            if _DEFAULT_GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=_DEFAULT_GRAD_CLIP)
            optimizer.step()

            total_loss += float(loss.detach().cpu())
            num_batches += 1

        model.eval()
        with torch.no_grad():
            deltas_va = rollout_strategy(model, F_va)
            pl_va = compute_pl_torch(S_va, deltas_va, Z_va, p0_true_mc, lam_cost)
            val_loss = float(objective_fn(pl_va).detach().cpu().item())

        train_log.append(
            {
                "epoch": int(ep),
                "train_loss": float(total_loss) / float(max(num_batches, 1)),
                "val_loss": float(val_loss),
                "lr": float(optimizer.param_groups[0].get("lr", 0.0)),
                "w": _extract_w_value(objective_fn),
            }
        )

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= int(patience):
                break

    last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is None:
        best_state = last_state

    return best_state, last_state, train_log
