import re
from pathlib import Path

p = Path("src/train_deephedge_gbm.py")
txt = p.read_text(encoding="utf-8")

# 0) Ensure csv import exists (safe even if unused elsewhere)
if not re.search(r"^\s*import\s+csv\s*$", txt, flags=re.M):
    # insert after imports block
    lines = txt.splitlines(True)
    ins = 0
    for i, ln in enumerate(lines):
        if ln.startswith("import ") or ln.startswith("from "):
            ins = i + 1
    lines.insert(ins, "import csv\n")
    txt = "".join(lines)

# 1) Remove any "train_log_path" / per-epoch file append blocks (they caused messy state)
txt = re.sub(r"^\s*train_log_path\s*=.*\n", "", txt, flags=re.M)
txt = re.sub(
    r"^\s*try:\s*\n\s*import\s+csv\s*\n\s*.*?with\s+open\(train_log_path.*?^\s*except\s+Exception.*?^\s*pass\s*\n",
    "",
    txt,
    flags=re.M | re.S
)

# 2) Ensure we have a single train_log list initialised once inside main()
# If exists, keep it; if not, add it right after out_dir is defined (or after data loaded).
if "train_log = []" not in txt:
    m = re.search(r"^\s*out_dir\s*=\s*.*\n", txt, flags=re.M)
    if m:
        ins = m.end()
        txt = txt[:ins] + "    train_log = []\n" + txt[ins:]
    else:
        # fallback: insert after def main():
        m2 = re.search(r"^\s*def\s+main\s*\(\s*\)\s*:\s*\n", txt, flags=re.M)
        if not m2:
            raise SystemExit("Could not find main() to insert train_log.")
        ins = m2.end()
        txt = txt[:ins] + "    train_log = []\n" + txt[ins:]

# 3) In the epoch loop: ensure exactly one append per epoch with lr and w
# We'll anchor on val_loss assignment and insert a clean logging block AFTER it,
# while deleting any existing train_log.append(...) blocks in that vicinity.
# First, delete all train_log.append(...) occurrences inside main() to avoid duplicates.
txt = re.sub(r"^\s*train_log\.append\(.*\)\s*\n", "", txt, flags=re.M)

# Now insert a single block after "val_loss = ..."
m = re.search(r"^\s*val_loss\s*=.*\n", txt, flags=re.M)
if not m:
    raise SystemExit("Could not find val_loss assignment to attach logging.")

indent = re.match(r"^(\s*)", txt[m.start():].splitlines()[0]).group(1)

log_block = (
    f"{indent}train_loss = float(total_loss) / float(max(nb, 1))\n"
    f"{indent}lr_now = float(opt.param_groups[0].get('lr', 0.0))\n"
    f"{indent}w_now = float('nan')\n"
    f"{indent}try:\n"
    f"{indent}    if 'obj' in locals() and hasattr(obj, 'w'):\n"
    f"{indent}        w_now = float(obj.w.detach().cpu().item())\n"
    f"{indent}    elif 'w_es' in locals():\n"
    f"{indent}        w_now = float(w_es.detach().cpu().item())\n"
    f"{indent}except Exception:\n"
    f"{indent}    w_now = float('nan')\n"
    f"{indent}train_log.append({{'epoch': int(ep), 'train_loss': train_loss, 'val_loss': float(val_loss), 'lr': lr_now, 'w': w_now}})\n"
)

txt = txt[:m.end()] + log_block + txt[m.end():]

# 4) Ensure final CSV write exists once (after training loop and before evaluation)
# If already exists, replace it with a clean deterministic writer.
txt = re.sub(
    r"^\s*with\s+open\(os\.path\.join\(out_dir,\s*['\"]train_log\.csv['\"]\).*$",
    "",
    txt,
    flags=re.M
)
txt = re.sub(r"^\s*print\('Warning: could not save train_log\.csv:'.*\n", "", txt, flags=re.M)

# Insert writer near the place you save checkpoints (or before model.eval()).
m2 = re.search(r"^\s*model\.eval\(\)\s*$", txt, flags=re.M)
if not m2:
    raise SystemExit("Could not find model.eval() to place train_log.csv writer before evaluation.")

indent2 = re.match(r"^(\s*)", txt[m2.start():].splitlines()[0]).group(1)

writer = (
    f"{indent2}try:\n"
    f"{indent2}    os.makedirs(out_dir, exist_ok=True)\n"
    f"{indent2}    with open(os.path.join(out_dir, 'train_log.csv'), 'w', newline='') as f:\n"
    f"{indent2}        writer = csv.DictWriter(f, fieldnames=['epoch','train_loss','val_loss','lr','w'])\n"
    f"{indent2}        writer.writeheader()\n"
    f"{indent2}        for row in train_log:\n"
    f"{indent2}            writer.writerow(row)\n"
    f"{indent2}except Exception as e:\n"
    f"{indent2}    print('Warning: could not save train_log.csv:', e)\n\n"
)

txt = txt[:m2.start()] + writer + txt[m2.start():]

p.write_text(txt, encoding="utf-8")
print("A4 patched clean logging into:", p)
