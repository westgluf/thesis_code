import sys
import re
from pathlib import Path

p = Path("src/train_deephedge_gbm.py")
s = p.read_text(encoding="utf-8")

if "from src.train_loop import train_loop" not in s:
    s = s.replace(
        "from src.eval import save_eval_artifacts\n",
        "from src.eval import save_eval_artifacts\nfrom src.train_loop import train_loop\n",
        1
    )

lines = s.splitlines(True)

start = None
end = None

for i, ln in enumerate(lines):
    if start is None and re.search(r"^\s*for\s+ep\s+in\s+trange\(", ln):
        start = i
        continue
    if start is not None and re.search(r"^\s*if\s+best_state\s+is\s+not\s+None\s*:", ln):
        end = i
        break

if start is None or end is None:
    raise SystemExit("Could not find training loop block to replace.")

indent = re.match(r"^(\s*)", lines[start]).group(1)

replacement = []
replacement.append(f"{indent}best_state, train_log, best_val = train_loop(\n")
replacement.append(f"{indent}    model=model,\n")
replacement.append(f"{indent}    opt=opt,\n")
replacement.append(f"{indent}    w_es=w_es,\n")
replacement.append(f"{indent}    F_tr_t=F_tr_t,\n")
replacement.append(f"{indent}    S_tr_t=S_tr_t,\n")
replacement.append(f"{indent}    Z_tr_t=Z_tr_t,\n")
replacement.append(f"{indent}    F_va_t=F_va_t,\n")
replacement.append(f"{indent}    S_va_t=S_va_t,\n")
replacement.append(f"{indent}    Z_va_t=Z_va_t,\n")
replacement.append(f"{indent}    p0_true_mc=p0_true_mc,\n")
replacement.append(f"{indent}    lam_cost=lam_cost,\n")
replacement.append(f"{indent}    alpha_es=alpha_es,\n")
replacement.append(f"{indent}    epochs=epochs,\n")
replacement.append(f"{indent}    batch_size=batch_size,\n")
replacement.append(f"{indent}    patience=patience,\n")
replacement.append(f"{indent})\n\n")

new_s = "".join(lines[:start] + replacement + lines[end:])
p.write_text(new_s, encoding="utf-8")
print("A6 patched:", p)
