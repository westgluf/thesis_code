import sys, re, shutil, time
from pathlib import Path

root = Path(__file__).resolve().parents[1]
train = root / "src" / "train_deephedge_gbm.py"
objf  = root / "src" / "objectives.py"
arch  = root / "archive"
arch.mkdir(parents=True, exist_ok=True)

bak = arch / f"train_deephedge_gbm_PRE_A5_{time.strftime('%Y%m%d_%H%M%S')}.py"
shutil.copyfile(train, bak)

s = train.read_text(encoding="utf-8")

if not objf.exists():
    objf.write_text(
        "import torch\n"
        "import torch.nn as nn\n\n"
        "def es_loss_from_pl(pl: torch.Tensor, w: torch.Tensor, alpha: float = 0.95) -> torch.Tensor:\n"
        "    loss = -pl\n"
        "    return w + (torch.relu(loss - w).mean() / (1.0 - alpha))\n\n"
        "class CVaRObjective(nn.Module):\n"
        "    def __init__(self, alpha: float = 0.95, w0: float = 0.0):\n"
        "        super().__init__()\n"
        "        self.alpha = float(alpha)\n"
        "        self.w = nn.Parameter(torch.tensor(float(w0)))\n"
        "    def forward(self, pl: torch.Tensor) -> torch.Tensor:\n"
        "        loss = -pl\n"
        "        w = self.w\n"
        "        return w + torch.relu(loss - w).mean() / (1.0 - self.alpha)\n",
        encoding="utf-8"
    )
else:
    txt = objf.read_text(encoding="utf-8")
    if "def es_loss_from_pl" not in txt:
        ins = (
            "\n\ndef es_loss_from_pl(pl: torch.Tensor, w: torch.Tensor, alpha: float = 0.95) -> torch.Tensor:\n"
            "    loss = -pl\n"
            "    return w + (torch.relu(loss - w).mean() / (1.0 - alpha))\n"
        )
        objf.write_text(txt.rstrip() + ins + "\n", encoding="utf-8")

if "from src.objectives import es_loss_from_pl" not in s:
    lines = s.splitlines(True)
    ins_i = 0
    for i, ln in enumerate(lines):
        if ln.startswith("import ") or ln.startswith("from "):
            ins_i = i + 1
    lines.insert(ins_i, "from src.objectives import es_loss_from_pl\n")
    s = "".join(lines)

s2, n = re.subn(
    r"\n*def\s+es_loss_from_pl\([^\)]*\)\s*->\s*torch\.Tensor:\s*\n(?:[ \t].*\n)+",
    "\n",
    s,
    count=1
)
s = s2

train.write_text(s, encoding="utf-8")
print("Backup:", bak.as_posix())
print("Wrote/updated:", objf.as_posix())
print("Patched:", train.as_posix())
