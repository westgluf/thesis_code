import sys
from pathlib import Path
import re

train_path = Path(sys.argv[1])
s = train_path.read_text(encoding="utf-8")

if "from src.eval import save_eval_artifacts" not in s:
    s = s.replace(
        "from src.plots import plot_hist, plot_es_var_bars\n",
        "from src.plots import plot_hist, plot_es_var_bars\nfrom src.eval import save_eval_artifacts\n",
        1
    )

lines = s.splitlines(True)

start_i = None
end_i = None

for i, ln in enumerate(lines):
    if start_i is None and re.search(r"^\s*alpha_list\s*=\s*\(", ln):
        start_i = i
        continue
    if start_i is not None and re.search(r'^\s*if\s+__name__\s*==\s*["\']__main__["\']\s*:\s*$', ln):
        end_i = i
        break

if start_i is None:
    raise SystemExit("Could not find alpha_list block to replace (expected near evaluation section).")
if end_i is None:
    raise SystemExit("Could not find __main__ guard.")

indent = re.match(r"^(\s*)", lines[start_i]).group(1)

replacement = []
replacement.append(f"{indent}alpha_list = (0.95, 0.99)\n")
replacement.append(f"{indent}out_dir = \"results/gbm_deephedge\"\n")
replacement.append(f"{indent}arrays = dict(\n")
replacement.append(f"{indent}    S_test=S_te,\n")
replacement.append(f"{indent}    Z_test=Z_te,\n")
replacement.append(f"{indent}    deltas_nn=deltas_te,\n")
replacement.append(f"{indent}    pl_nn=pl_te,\n")
replacement.append(f"{indent}    pl_bs=PL_bs,\n")
replacement.append(f"{indent})\n")
replacement.append(f"{indent}m_bs, m_nn = save_eval_artifacts(\n")
replacement.append(f"{indent}    out_dir=out_dir,\n")
replacement.append(f"{indent}    pl_bs=PL_bs,\n")
replacement.append(f"{indent}    pl_nn=pl_te,\n")
replacement.append(f"{indent}    label_bs=\"BS-delta\",\n")
replacement.append(f"{indent}    label_nn=\"Deep hedging\",\n")
replacement.append(f"{indent}    alpha_list=alpha_list,\n")
replacement.append(f"{indent}    lam_entropic=1.0,\n")
replacement.append(f"{indent}    arrays_debug=arrays,\n")
replacement.append(f"{indent})\n")
replacement.append(f"{indent}print(\"Saved results to: results/gbm_deephedge\")\n")
replacement.append(f"{indent}print(\"BS-delta:\", m_bs)\n")
replacement.append(f"{indent}print(\"Deep hedging:\", m_nn)\n")
replacement.append(f"{indent}print(\"Note: p0 used = MC estimate from train set; BS price also available:\", p0_bs)\n")

new_s = "".join(lines[:start_i] + replacement + lines[end_i:])
train_path.write_text(new_s, encoding="utf-8")
