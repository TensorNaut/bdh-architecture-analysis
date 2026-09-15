import json
import os


LOG_PATH = "results/log.jsonl"


def log_step(model, step, loss, val_loss=None):
    os.makedirs("results", exist_ok=True)

    entry = {"model": model, "step": step, "loss": loss}
    if val_loss is not None:
        entry["val_loss"] = val_loss

    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")