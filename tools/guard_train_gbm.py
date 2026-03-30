import json, os, subprocess, sys, glob, math, shutil, time

KEYS = ["std_PL", "ES_loss_0.95", "VaR_loss_0.95", "ES_loss_0.99", "VaR_loss_0.99"]
TOL = 1e-10

def load_json(p):
    with open(p, "r") as f:
        return json.load(f)

def worse(base, cur):
    bad = []
    for k in KEYS:
        if k in base and k in cur:
            if (cur[k] - base[k]) > TOL:
                bad.append((k, base[k], cur[k]))
    return bad

def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(root)

    baselines = sorted(glob.glob("results/archive/gbm_baseline_metrics_*.json"))
    if not baselines:
        print("ERROR: no baseline in results/archive. Create one first.")
        sys.exit(1)
    baseline_path = baselines[-1]
    print("Using baseline:", baseline_path)

    if os.path.exists("results/gbm_deephedge"):
        shutil.rmtree("results/gbm_deephedge")
    os.makedirs("results/gbm_deephedge", exist_ok=True)

    cmd = [sys.executable, "-m", "src.train_deephedge_gbm"]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    cur_path = "results/gbm_deephedge/metrics_nn.json"
    if not os.path.exists(cur_path):
        print("ERROR: current run did not produce", cur_path)
        sys.exit(1)

    base = load_json(baseline_path)
    cur = load_json(cur_path)

    print("BASE:", {k: base.get(k) for k in KEYS})
    print("CUR: ", {k: cur.get(k) for k in KEYS})

    bad = worse(base, cur)
    if bad:
        print("\nFAIL: metrics worsened:")
        for k,a,b in bad:
            print(f"  {k}: {a} -> {b}")
        sys.exit(2)

    print("\nPASS: metrics not worse than baseline.")
    sys.exit(0)

if __name__ == "__main__":
    main()
