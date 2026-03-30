from __future__ import annotations

import ast
import inspect
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "src" / "train_deephedge_gbm.py"
BAK = ROOT / "archive" / f"train_deephedge_gbm_FIX_trainloop_kwargs_{time.strftime('%Y%m%d_%H%M%S')}.py"

def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)

def get_allowed_kwargs() -> set[str]:
    import importlib
    m = importlib.import_module("src.train_loop")
    if not hasattr(m, "train_loop"):
        raise SystemExit("src.train_loop has no train_loop symbol")
    sig = inspect.signature(m.train_loop)
    allowed = set()
    for name, p in sig.parameters.items():
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
            allowed.add(name)
    return allowed

class TrainLoopKwargFilter(ast.NodeTransformer):
    def __init__(self, allowed: set[str]):
        super().__init__()
        self.allowed = allowed
        self.changed = 0

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        is_train_loop = False

        if isinstance(node.func, ast.Name) and node.func.id == "train_loop":
            is_train_loop = True
        elif isinstance(node.func, ast.Attribute) and node.func.attr == "train_loop":
            is_train_loop = True

        if not is_train_loop:
            return node

        new_keywords = []
        for kw in node.keywords:
            if kw.arg is None:
                new_keywords.append(kw)
                continue
            if kw.arg in self.allowed:
                new_keywords.append(kw)
            else:
                self.changed += 1

        node.keywords = new_keywords
        return node

def main() -> None:
    txt = TRAIN.read_text(encoding="utf-8")
    shutil.copyfile(TRAIN, BAK)

    try:
        allowed = get_allowed_kwargs()
    except Exception as e:
        shutil.copyfile(BAK, TRAIN)
        raise SystemExit(f"Could not import src.train_loop / inspect signature: {e}")

    tree = ast.parse(txt)
    filt = TrainLoopKwargFilter(allowed=allowed)
    tree2 = filt.visit(tree)
    ast.fix_missing_locations(tree2)

    if filt.changed == 0:
        print("No unknown kwargs found in train_loop(...) call. Nothing changed.")
        return

    new_txt = ast.unparse(tree2) + "\n"
    TRAIN.write_text(new_txt, encoding="utf-8")
    print("Backup:", str(BAK))
    print("Removed unknown kwargs from train_loop(...) call:", filt.changed)

    try:
        run([sys.executable, "-m", "py_compile", str(TRAIN)])
        run([sys.executable, "tools/guard_train_gbm.py"])
        print("OK: compile + guard passed")
    except subprocess.CalledProcessError:
        shutil.copyfile(BAK, TRAIN)
        print("ROLLBACK:", str(BAK))
        raise

if __name__ == "__main__":
    main()
