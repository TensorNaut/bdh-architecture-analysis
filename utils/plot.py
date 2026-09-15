import json
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


# ---- setup ----
SAVE_DIR = "results/plots"
os.makedirs(SAVE_DIR, exist_ok=True)


# ---- load logs ----
def load_logs(path="results/log.jsonl"):
    data = defaultdict(list)

    with open(path) as f:
        for line in f:
            d = json.loads(line)
            data[d["model"]].append(d)

    return data


# ---- smoothing ----
def moving_avg(x, k=20):
    if len(x) < k:
        return x
    return np.convolve(x, np.ones(k) / k, mode="valid")


# 1. Raw loss curves
def plot_loss():
    data = load_logs()

    plt.figure()

    for model, entries in data.items():
        steps = [e["step"] for e in entries]
        losses = [e["loss"] for e in entries]

        plt.plot(steps, losses, label=model)

    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()

    plt.savefig(f"{SAVE_DIR}/loss.png")
    plt.close()


# 2. Smoothed curves
def plot_smooth():
    data = load_logs()

    plt.figure()

    for model, entries in data.items():
        losses = [e["loss"] for e in entries]
        sm = moving_avg(losses)

        plt.plot(sm, label=model)

    plt.title("Smoothed Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()

    plt.savefig(f"{SAVE_DIR}/smooth_loss.png")
    plt.close()


# 3. Bar chart (final comparison)
def plot_bar():
    from utils.metrics import summarize

    s = summarize()

    names = list(s.keys())
    vals = [s[k]["avg_last_50"] for k in names]

    plt.figure()

    plt.bar(names, vals)
    plt.xticks(rotation=30)
    plt.ylabel("Avg Last 50 Loss")
    plt.title("Model Comparison")

    plt.savefig(f"{SAVE_DIR}/comparison.png")
    plt.close()


# 4. Component impact
def plot_component():
    from utils.metrics import summarize

    s = summarize()
    base = s["bdh_base"]["avg_last_50"]

    labels = ["multiplication", "latent", "activation"]
    vals = [
        s["bdh_nomul"]["avg_last_50"] - base,
        s["bdh_lowdim"]["avg_last_50"] - base,
        s["bdh_improved"]["avg_last_50"] - base,
    ]

    plt.figure()

    plt.bar(labels, vals)
    plt.ylabel("Loss Increase")
    plt.title("Component Impact")

    plt.savefig(f"{SAVE_DIR}/component_impact.png")
    plt.close()


# 5. Train vs val loss (overfitting check) -- new
def plot_overfit_gap():
    data = load_logs()

    plt.figure()

    for model, entries in data.items():
        val_entries = [e for e in entries if "val_loss" in e]
        if not val_entries:
            continue

        steps = [e["step"] for e in val_entries]
        train_at_val_steps = [e["loss"] for e in val_entries]
        val_losses = [e["val_loss"] for e in val_entries]

        line, = plt.plot(steps, val_losses, linestyle="-", label=f"{model} val")
        plt.plot(steps, train_at_val_steps, linestyle="--", color=line.get_color(),
                  alpha=0.6, label=f"{model} train")

    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Train vs Validation Loss (dashed = train, solid = val)")
    plt.legend(fontsize=7)

    plt.savefig(f"{SAVE_DIR}/overfit_gap.png")
    plt.close()