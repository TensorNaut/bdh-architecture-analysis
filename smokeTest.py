"""
Quick smoke test for the full-scale config.

Runs a small number of steps at the REAL full-scale settings (n_layer=24,
batch_size, block_size from config.py) on bdh_base, and reports:
  - iterations/sec, so you can convert max_iters into real wall-clock hours
  - peak GPU memory, so you can confirm depth-24 actually fits your 6GB card
    before kicking off the full runner.py sweep (which trains 5 models).

Run this from your project root (same place you'd run `python -m experiments.runner`):
    python smoke_test.py
"""
import time
import torch

from configs.config import ModelConfig, TrainConfig
from data.dataset import load_data, get_batch
from models.bdh_base import BDH

N_WARMUP_STEPS = 3
N_TIMED_STEPS = 100


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    model_cfg = ModelConfig()
    train_cfg = TrainConfig()
    torch.manual_seed(train_cfg.seed)

    print(
        f"device: {device} | n_layer: {model_cfg.n_layer} | "
        f"batch_size: {train_cfg.batch_size} | block_size: {train_cfg.block_size}\n"
    )

    train_data, _ = load_data()
    model = BDH(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"bdh_base parameter count: {n_params:,}\n")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )

    def step():
        x, y = get_batch(train_data, train_cfg.block_size, train_cfg.batch_size, device)
        with torch.autocast(
            device_type=device.type, dtype=autocast_dtype, enabled=(device.type == "cuda")
        ):
            _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()
        return loss.item()

    # warmup: first CUDA calls / cuDNN autotune are slower and would skew the timing
    model.train()
    for _ in range(N_WARMUP_STEPS):
        step()

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    start = time.time()
    last_loss = None
    for _ in range(N_TIMED_STEPS):
        last_loss = step()

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start

    it_per_sec = N_TIMED_STEPS / elapsed
    hours_one_model = train_cfg.max_iters / it_per_sec / 3600
    hours_all_five = 5 * hours_one_model

    print(f"loss after warmup+timed steps: {last_loss:.4f}")
    print(f"{N_TIMED_STEPS} steps in {elapsed:.1f}s -> {it_per_sec:.2f} it/s\n")
    print(f"at max_iters={train_cfg.max_iters}:")
    print(f"  ~{hours_one_model:.2f} hours for this one model")
    print(f"  ~{hours_all_five:.2f} hours for all 5 models run sequentially (runner.py)\n")

    if device.type == "cuda":
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        reserved_gb = torch.cuda.max_memory_reserved() / 1e9
        print(f"peak allocated GPU memory: {peak_gb:.2f} GB")
        print(f"peak reserved GPU memory:  {reserved_gb:.2f} GB")
        print("(compare against your card's total VRAM; if this is already")
        print(" close to your limit, drop batch_size before the full run)")
    else:
        print("running on CPU - GPU memory stats not applicable.")


if __name__ == "__main__":
    main()