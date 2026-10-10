"""Shared command-line options so every task script has the same flags."""
import argparse

from .config import Recipe, Paths
from .data import Data


def base_parser(desc=""):
    p = argparse.ArgumentParser(description=desc, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-dir", default="data", help="folder with train_images.npy, labels.csv, ...")
    p.add_argument("--out-dir", default="outputs", help="everything is written under this folder")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=60, help="epochs per run (<=60); keep identical across tasks")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--optimizer", default="sgd", choices=["sgd", "adamw"])
    p.add_argument("--lr", type=float, default=None, help="peak lr (default 0.1 for sgd, 1e-3 for adamw)")
    p.add_argument("--schedule", default="cosine", choices=["cosine", "step"])
    return p


def setup(args):
    """-> (data, recipe, paths)"""
    kw = {}
    if args.lr is not None:
        kw["lr" if args.optimizer == "sgd" else "adamw_lr"] = args.lr
    recipe = Recipe(seed=args.seed, epochs=args.epochs, batch_size=args.batch_size, optimizer=args.optimizer,
                    schedule=args.schedule, **kw)
    return Data(args.data_dir, args.device), recipe, Paths(args.out_dir)


def class_names(data_dir):
    from pathlib import Path
    return [l.strip() for l in (Path(data_dir) / "classes.txt").read_text().splitlines() if l.strip()]
