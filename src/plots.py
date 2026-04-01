import os
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_hist(pl_a: np.ndarray, pl_b: np.ndarray, label_a: str, label_b: str, outpath: str):
    plt.figure()
    plt.hist(pl_a, bins=60, density=True, alpha=0.6, label=label_a)
    plt.hist(pl_b, bins=60, density=True, alpha=0.6, label=label_b)
    plt.xlabel("Terminal P&L")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_es_var_bars(metrics_a: dict, metrics_b: dict, alpha_list, outpath: str, title: str):
    labels = []
    a_vals = []
    b_vals = []
    for a in alpha_list:
        labels.append(f"VaR@{a:.2f}")
        a_vals.append(metrics_a[f"VaR_loss_{a:.2f}"])
        b_vals.append(metrics_b[f"VaR_loss_{a:.2f}"])
        labels.append(f"ES@{a:.2f}")
        a_vals.append(metrics_a[f"ES_loss_{a:.2f}"])
        b_vals.append(metrics_b[f"ES_loss_{a:.2f}"])

    x = np.arange(len(labels))
    width = 0.35

    plt.figure()
    plt.bar(x - width/2, a_vals, width, label="BS-delta")
    plt.bar(x + width/2, b_vals, width, label="Deep hedging")
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Loss metric value")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_train_val_curves(train_log, outpath: str):
    epochs = [int(row["epoch"]) for row in train_log]
    train_loss = [float(row["train_loss"]) for row in train_log]
    val_loss = [float(row["val_loss"]) for row in train_log]

    plt.figure()
    plt.plot(epochs, train_loss, label="train")
    plt.plot(epochs, val_loss, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Objective")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=200)
    plt.close()
