import json
from collections import defaultdict
import numpy as np


def load_logs(path="results/log.jsonl"):
    data = defaultdict(list)

    with open(path) as f:
        for line in f:
            d = json.loads(line)
            data[d["model"]].append(d)

    return data


def summarize():
    data = load_logs()

    summary = {}

    for model, entries in data.items():
        losses = [e["loss"] for e in entries]
        val_losses = [e["val_loss"] for e in entries if "val_loss" in e]

        s = {
            "final_loss": losses[-1],
            "avg_last_50": np.mean(losses[-50:]),
            "std_last_50": np.std(losses[-50:]),
        }

        if val_losses:
            s["final_val_loss"] = val_losses[-1]
            s["avg_val_last_50"] = np.mean(val_losses[-50:])
            # positive gap = val loss running higher than train loss -> overfitting signal
            s["overfit_gap"] = s["avg_val_last_50"] - s["avg_last_50"]

        summary[model] = s

    return summary


def compare():
    summary = summarize()

    print("\n===== RESULTS =====\n")

    for m, s in summary.items():
        line = (
            f"{m:20} | final {s['final_loss']:.4f} | avg50 {s['avg_last_50']:.4f} "
            f"| std {s['std_last_50']:.4f}"
        )
        if "avg_val_last_50" in s:
            line += f" | val50 {s['avg_val_last_50']:.4f} | gap {s['overfit_gap']:+.4f}"
        print(line)

    return summary


def component_analysis():
    s = summarize()

    base = s["bdh_base"]

    print("\n===== COMPONENT IMPACT (train loss) =====\n")

    def impact(name):
        return s[name]["avg_last_50"] - base["avg_last_50"]

    print(f"Multiplication  : {impact('bdh_nomul'):.4f}")
    print(f"Latent dim      : {impact('bdh_lowdim'):.4f}")
    print(f"Activation      : {impact('bdh_improved'):.4f}")

    if "avg_val_last_50" in base:
        print("\n===== COMPONENT IMPACT (val loss) =====\n")

        def val_impact(name):
            return s[name]["avg_val_last_50"] - base["avg_val_last_50"]

        print(f"Multiplication  : {val_impact('bdh_nomul'):.4f}")
        print(f"Latent dim      : {val_impact('bdh_lowdim'):.4f}")
        print(f"Activation      : {val_impact('bdh_improved'):.4f}")