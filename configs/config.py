from dataclasses import dataclass


@dataclass
class ModelConfig:
    # shared
    n_layer: int = 24  # was 6 -- deliberate scale-up variable
    n_embd: int = 256
    n_head: int = 4
    vocab_size: int = 256
    dropout: float = 0.1

    # BDH
    mlp_internal_dim_multiplier: int = 128

    # Transformer (FIX)
    ffn_dim: int = 1024  # 4 * n_embd


@dataclass
class TrainConfig:
    block_size: int = 128
    batch_size: int = 8

    # full-scale run
    max_iters: int = 10000  # was 1000 -- placeholder, tune after the smoke test
    log_freq: int = 100     # was 10

    # training
    learning_rate: float = 1e-3
    weight_decay: float = 0.1
    seed: int = 42

    # LR schedule (linear warmup -> cosine decay) -- new, needed for depth 24
    warmup_iters: int = 500
    lr_decay_iters: int = 50000
    min_lr: float = 1e-4

    # gradient clipping -- new, needed for depth 24
    grad_clip: float = 1.0