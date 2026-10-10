"""One function that trains any Task 4 / Task 5 method on any label set. Both task4_robust.py and
task5_synthetic.py call it, so 'identical recipe, only the method differs' is true by construction."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import selection
from .engine import run_training, load_logits
from .methods import make_method

FILTER_METHODS = ("filter", "relabel")
ALL_METHODS = ("ce", "ls", "gce", "sce", "coteach") + FILTER_METHODS


@dataclass
class MethodOpts:
    forget_rate: float = 0.4      # co-teaching; taken from the ~40% stated in the problem statement
    drop_mode: str = "gmm"        # filter/relabel: "gmm" (label-free) or "frac"
    drop_frac: float = 0.35       # used when drop_mode == "frac"
    per_class: bool = True        # drop within each given-label class (class-dependent noise)
    relabel_conf: float = 0.8     # relabel only if out-of-fold confidence >= this


def scores_path(paths, label_set):
    return paths.reports / f"t3_{label_set}_scores.csv"


def oof_path(paths, label_set):
    return paths.reports / f"t3_{label_set}_oof_probs.npy"


def run_method(name, label_set, data, recipe, paths, run_prefix, opts: MethodOpts = None, log=print):
    """Train `name` on `label_set` ('noisy'|'synthetic') labels; model selection uses that label set's val labels.
    Returns dict(val_acc, test_pred, info)."""
    opts = opts or MethodOpts()
    y_tr, y_va = data.labels(label_set, "train"), data.labels(label_set, "val")
    x_tr, info = data.x_train, {}

    if name in FILTER_METHODS:
        sp = scores_path(paths, label_set)
        assert sp.exists(), (f"{sp} missing - run: python task3_scores.py --label-set {label_set} "
                             f"(filter/relabel need Task 3 scores computed on THESE labels)")
        df = pd.read_csv(sp)
        assert (df["index"].to_numpy() == data.train_idx).all(), "score file not aligned with train_idx"
        scores = df["score"].to_numpy()
        if opts.drop_mode == "gmm":
            keep = selection.keep_mask_gmm(scores, y_tr, per_class=opts.per_class, seed=recipe.seed)
        else:
            keep = selection.keep_mask_frac(scores, y_tr, opts.drop_frac, opts.per_class)
        info["dropped"] = int((~keep).sum())
        if name == "relabel":
            oof = np.load(oof_path(paths, label_set))
            y_new, keep = selection.relabel(scores, y_tr, oof, keep, opts.relabel_conf)
            info["relabeled"] = int((y_new != y_tr).sum())
            y_tr = y_new
        info["kept"], info["kept_frac"] = int(keep.sum()), float(keep.mean())
        sel = data.t(np.where(keep)[0])
        x_tr, y_tr = x_tr[sel], y_tr[keep]
        method = make_method("ce")
    elif name == "coteach":
        method = make_method("coteach", forget_rate=opts.forget_rate)
    else:
        method = make_method(name)

    run_dir = paths.run(f"{run_prefix}_{name}")
    run_training(run_dir, data, recipe, method, x_tr, y_tr,
                 eval_sets={"val": data.x_val, "test": data.x_test}, eval_labels={"val": y_va}, log=log)
    val_logits, test_logits = load_logits(run_dir, "val")[-1], load_logits(run_dir, "test")[-1]
    return {"val_acc": float((val_logits.argmax(1) == y_va).mean()), "test_pred": test_logits.argmax(1),
            "info": {**method.describe(), **info}}
