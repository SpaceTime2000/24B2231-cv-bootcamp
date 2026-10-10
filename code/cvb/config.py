"""Frozen training recipe and output paths.

The bootcamp rules require the SAME optimizer / schedule / augmentation across the baseline and every
method (except the change being tested). Everything lives in one frozen `Recipe`; method-specific knobs
live in `methods.py`. The recipe is dumped to results so the exact settings are on record.
"""
from dataclasses import dataclass, asdict
from pathlib import Path

MAX_EPOCHS = 60  # hard cap from the problem statement (per network)


@dataclass(frozen=True)
class Recipe:
    seed: int = 0
    epochs: int = 60
    batch_size: int = 128
    optimizer: str = "sgd"        # "sgd" | "adamw"
    lr: float = 0.1               # sgd peak lr
    adamw_lr: float = 1e-3        # used only if optimizer == "adamw"
    momentum: float = 0.9
    weight_decay: float = 5e-4
    nesterov: bool = True
    schedule: str = "cosine"      # "cosine" | "step"
    warmup_epochs: float = 2.0
    crop_pad: int = 4             # random crop with this padding + horizontal flip
    amp: bool = True              # mixed precision (CUDA only)

    def to_dict(self):
        return asdict(self)

    def lr_at(self, t: float, total_epochs: int) -> float:
        """Learning rate at fractional epoch t (0-based) of a run lasting total_epochs."""
        import math
        base = self.lr if self.optimizer == "sgd" else self.adamw_lr
        if t < self.warmup_epochs:
            return base * (t + 1e-3) / self.warmup_epochs
        if self.schedule == "cosine":
            p = (t - self.warmup_epochs) / max(total_epochs - self.warmup_epochs, 1e-8)
            return 0.5 * base * (1 + math.cos(math.pi * min(p, 1.0)))
        if self.schedule == "step":  # x0.1 at 50% and 75% of training
            return base * (0.1 ** ((t >= 0.5 * total_epochs) + (t >= 0.75 * total_epochs)))
        raise ValueError(self.schedule)


class Paths:
    """Everything a script writes goes under one root: <root>/runs/<run>, <root>/predictions, <root>/reports."""

    def __init__(self, out_dir="outputs"):
        self.root = Path(out_dir)
        self.runs = self.root / "runs"
        self.preds = self.root / "predictions"
        self.reports = self.root / "reports"
        for p in (self.runs, self.preds, self.reports):
            p.mkdir(parents=True, exist_ok=True)

    def run(self, name: str) -> Path:
        return self.runs / name
