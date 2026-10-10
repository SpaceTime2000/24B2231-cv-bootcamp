"""Resumable training loop shared by every task.

Design points
  * The engine never sees clean labels: it gets (train_x, train_y) and optional `eval_labels` used only to log
    accuracy. Whatever labels the caller passes are the labels it logs against.
  * Every epoch, logits for each requested eval set are saved to <run_dir>/logits/<name>_epNNN.npy (float32).
    Task 2 dynamics, early-stop predictions, final predictions and Task 3 held-out scores are all derived from
    these files afterwards, so no extra training and no weights per epoch are needed.
  * A checkpoint is written after each epoch (atomic replace). Re-running the same command resumes; a finished
    run (DONE marker) is loaded instead of retrained.
Epochs are 1-based in files/logs (ep001..epE).
"""
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import Recipe, MAX_EPOCHS


def _make_opt(params, r: Recipe):
    if r.optimizer == "sgd":
        return torch.optim.SGD(params, lr=r.lr, momentum=r.momentum, weight_decay=r.weight_decay,
                               nesterov=r.nesterov)
    if r.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=r.adamw_lr, weight_decay=r.weight_decay)
    raise ValueError(r.optimizer)


@torch.no_grad()
def predict_logits(nets, method, data, x_u8, bs=1000) -> np.ndarray:
    for n in nets:
        n.eval()
    amp = data.device.type == "cuda"
    out = []
    for i in range(0, len(x_u8), bs):
        x = data.prep(x_u8[i:i + bs])
        with torch.autocast(device_type=data.device.type, dtype=torch.float16, enabled=amp):
            out.append(method.predict(nets, x).float().cpu())
    return torch.cat(out).numpy()


def load_logits(run_dir, name) -> np.ndarray:
    """Stack all saved epochs -> (E, N, C) float32."""
    files = sorted((Path(run_dir) / "logits").glob(f"{name}_ep*.npy"))
    assert files, f"no logits for '{name}' in {run_dir}"
    return np.stack([np.load(f) for f in files]).astype(np.float32)


def run_training(run_dir, data, recipe: Recipe, method, train_x, train_y, eval_sets=None, eval_labels=None,
                 epochs=None, log=print):
    """Train `method` on (train_x uint8 NCHW tensor, train_y int64 array/tensor). Returns the list of nets."""
    run_dir = Path(run_dir)
    (run_dir / "logits").mkdir(parents=True, exist_ok=True)
    E = int(epochs or recipe.epochs)
    assert E <= MAX_EPOCHS, f"budget: at most {MAX_EPOCHS} epochs per network"
    eval_sets, eval_labels = eval_sets or {}, eval_labels or {}
    dev = data.device
    amp = recipe.amp and dev.type == "cuda"

    torch.manual_seed(recipe.seed)
    nets = method.build(dev)
    final = run_dir / "final.pt"
    if (run_dir / "DONE").exists():
        for n, sd in zip(nets, torch.load(final, map_location=dev)):
            n.load_state_dict(sd)
        log(f"[{run_dir.name}] already finished - loaded final weights")
        return nets

    params = [p for n in nets for p in n.parameters()]
    opt = _make_opt(params, recipe)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    train_y = data.t(train_y).long()
    N = len(train_y)
    bs = min(recipe.batch_size, N)
    steps = max(1, N // bs)

    start, history = 0, []
    ckpt = run_dir / "ckpt.pt"
    if ckpt.exists():
        ck = torch.load(ckpt, map_location=dev)
        for n, sd in zip(nets, ck["nets"]):
            n.load_state_dict(sd)
        opt.load_state_dict(ck["opt"]); scaler.load_state_dict(ck["scaler"])
        start, history = ck["epoch"], ck["history"]
        torch.manual_seed(recipe.seed + start)
        log(f"[{run_dir.name}] resuming from epoch {start}/{E}")
    pd.Series({**recipe.to_dict(), **method.describe(), "epochs": E, "n_train": N}).to_json(run_dir / "config.json")

    for ep in range(start, E):
        t0 = time.time()
        for n in nets:
            n.train()
        perm = torch.randperm(N, device=dev)
        tot = torch.zeros((), device=dev)
        for b in range(steps):
            t = ep + b / steps
            lr = recipe.lr_at(t, E)
            for g in opt.param_groups:
                g["lr"] = lr
            idx = perm[b * bs:(b + 1) * bs]
            x = data.prep(train_x[idx], augment=True, pad=recipe.crop_pad)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
                losses = method.step_losses(nets, x, train_y[idx], t)
            loss = sum(losses)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.detach()
        row = {"epoch": ep + 1, "lr": lr, "train_loss": (tot / steps).item() / method.n_nets}

        for name, x_eval in eval_sets.items():
            lg = predict_logits(nets, method, data, x_eval)
            np.save(run_dir / "logits" / f"{name}_ep{ep + 1:03d}.npy", lg.astype(np.float32))
            if name in eval_labels:
                row[f"acc_{name}"] = float((lg.argmax(1) == np.asarray(eval_labels[name])).mean())
        row["sec"] = time.time() - t0
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "log.csv", index=False)

        torch.save({"nets": [n.state_dict() for n in nets], "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": ep + 1, "history": history}, str(ckpt) + ".tmp")
        os.replace(str(ckpt) + ".tmp", ckpt)
        log(f"[{run_dir.name}] ep {ep + 1}/{E} " + " ".join(
            f"{k}={v:.4f}" for k, v in row.items() if k not in ("epoch", "sec")) + f" ({row['sec']:.0f}s)")

    torch.save([n.state_dict() for n in nets], final)
    (run_dir / "DONE").write_text("ok")
    if ckpt.exists():
        ckpt.unlink()   # weights live in final.pt; keeps the run folder small
    return nets
