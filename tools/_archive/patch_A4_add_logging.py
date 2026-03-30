from pathlib import Path
import re
import time
import shutil
import sys

p = Path("src/train_deephedge_gbm.py")
bak = Path(f"archive/train_deephedge_gbm_A4_backup_{time.strftime('%Y%m%d_%H%M%S')}.py")
bak.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(p, bak)

txt = p.read_text(encoding="utf-8").splitlines(True)

full = "".join(txt)
if "train_log.csv" in full and "best_state.pt" in full and "last_state.pt" in full:
    print("A4 already present; nothing to do. Backup:", bak)
    sys.exit(0)

def find_line_index(pattern):
    for i, ln in enumerate(txt):
        if re.search(pattern, ln):
            return i
    return None

def leading_ws(s):
    return re.match(r"^(\s*)", s).group(1)

i_for = find_line_index(r"^\s*for\s+ep\s+in\s+trange\(")
if i_for is None:
    raise SystemExit("Could not find training loop: 'for ep in trange(...)'")

i_epochs = None
for i in range(max(0, i_for-50), i_for):
    if re.search(r"^\s*epochs\s*=", txt[i]):
        i_epochs = i
        break
if i_epochs is None:
    raise SystemExit("Could not find 'epochs = ...' before training loop.")

indent_main = leading_ws(txt[i_epochs])

if "out_dir = \"results/gbm_deephedge\"" in full:
    pass
else:
    insert_block = [
        indent_main + 'out_dir = "results/gbm_deephedge"\n',
        indent_main + "os.makedirs(out_dir, exist_ok=True)\n",
        indent_main + "train_log = []\n",
        "\n",
    ]
    txt[i_epochs+1:i_epochs+1] = insert_block

full2 = "".join(txt)

i_if_val = None
for i in range(i_for, len(txt)):
    if re.search(r"^\s*if\s+val_loss\s*<\s*best_val", txt[i]):
        i_if_val = i
        break
if i_if_val is None:
    raise SystemExit("Could not find 'if val_loss < best_val ...' block (for inserting epoch logging).")

indent_epoch = leading_ws(txt[i_if_val])

epoch_log_block = [
    indent_epoch + "train_loss_epoch = float(total_loss) / float(max(nb, 1))\n",
    indent_epoch + "w_now = float(w_es.detach().cpu().item()) if 'w_es' in locals() else float('nan')\n",
    indent_epoch + "train_log.append({'epoch': int(ep), 'train_loss': train_loss_epoch, 'val_loss': float(val_loss), 'w': w_now})\n",
    "\n",
]

window = "".join(txt[max(0, i_if_val-20):i_if_val+5])
if "train_log.append" not in window:
    txt[i_if_val:i_if_val] = epoch_log_block

full3 = "".join(txt)

i_load = find_line_index(r"model\.load_state_dict\(\s*best_state\s*\)")
if i_load is None:
    raise SystemExit("Could not find 'model.load_state_dict(best_state)' for inserting checkpoint/log save.")

indent_after = leading_ws(txt[i_load])

save_block = [
    "\n",
    indent_after + "try:\n",
    indent_after + "    import csv\n",
    indent_after + "    torch.save(model.state_dict(), os.path.join(out_dir, 'last_state.pt'))\n",
    indent_after + "    if best_state is not None:\n",
    indent_after + "        torch.save(best_state, os.path.join(out_dir, 'best_state.pt'))\n",
    indent_after + "    with open(os.path.join(out_dir, 'train_log.csv'), 'w', newline='') as f:\n",
    indent_after + "        w = csv.DictWriter(f, fieldnames=['epoch','train_loss','val_loss','w'])\n",
    indent_after + "        w.writeheader()\n",
    indent_after + "        for row in train_log:\n",
    indent_after + "            w.writerow(row)\n",
    indent_after + "except Exception as e:\n",
    indent_after + "    print('Warning: could not save train_log/checkpoints:', e)\n",
    "\n",
]

lookahead = "".join(txt[i_load:i_load+60])
if "last_state.pt" not in lookahead and "train_log.csv" not in lookahead:
    txt[i_load+1:i_load+1] = save_block

p.write_text("".join(txt), encoding="utf-8")
print("A4 patch applied to:", p)
print("Backup:", bak)
