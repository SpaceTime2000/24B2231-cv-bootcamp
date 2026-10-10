"""Measurements. Functions that take clean labels are measurement-only (see data.py)."""
import numpy as np


def accuracy(logits, y) -> float:
    return float((np.asarray(logits).argmax(1) == np.asarray(y)).mean())


def noise_stats(noisy, clean, n_classes=10):
    """Overall noise rate, per-class noise rate (by clean class) and the confusion matrix (rows=clean, cols=noisy)."""
    noisy, clean = np.asarray(noisy), np.asarray(clean)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (clean, noisy), 1)
    wrong = noisy != clean
    per_class = [float(wrong[clean == k].mean()) for k in range(n_classes)]
    return {"noise_rate": float(wrong.mean()), "per_class_noise_rate": per_class, "confusion": cm}


def top_noise_pairs(cm, names, k=10):
    """Largest off-diagonal cells as (clean class -> noisy class, count, share of all wrong labels)."""
    off = cm.copy().astype(float)
    np.fill_diagonal(off, 0)
    total = off.sum()
    order = np.dstack(np.unravel_index(np.argsort(-off, axis=None), off.shape))[0][:k]
    return [{"clean": names[i], "noisy": names[j], "count": int(off[i, j]), "share": float(off[i, j] / total)}
            for i, j in order]


def dynamics(train_logits, val_logits, y_train_noisy, y_train_clean, y_val_noisy, y_val_clean):
    """Per-epoch Task 2 table from saved logits. Inputs are (E,N,C) arrays.
    train_acc_correct   : accuracy on training samples whose noisy label is right
    train_acc_mislabeled: accuracy w.r.t. the GIVEN (wrong) label on mislabeled samples  -> memorization
    also reports acc of the mislabeled subset w.r.t. its TRUE class (how much the net still 'knows' the truth)."""
    wrong = y_train_noisy != y_train_clean
    rows = []
    for e in range(train_logits.shape[0]):
        p = train_logits[e].argmax(1)
        rows.append({
            "epoch": e + 1,
            "train_acc_correct": float((p[~wrong] == y_train_noisy[~wrong]).mean()),
            "train_acc_mislabeled": float((p[wrong] == y_train_noisy[wrong]).mean()),
            "val_acc_noisy": float((val_logits[e].argmax(1) == y_val_noisy).mean()),
            "val_acc_clean": float((val_logits[e].argmax(1) == y_val_clean).mean()),
            "train_acc_mislabeled_vs_true": float((p[wrong] == y_train_clean[wrong]).mean()),
        })
    return rows


def pick_early_stop(val_acc_noisy, window=3):
    """Epoch (1-based) maximizing the moving average of NOISY val accuracy; ties -> earlier epoch."""
    a = np.asarray(val_acc_noisy, dtype=float)
    k = min(window, len(a))
    ma = np.array([a[max(0, i - k + 1):i + 1].mean() for i in range(len(a))])  # trailing window
    return int(np.argmax(ma)) + 1   # argmax returns the first (earliest) maximum


def auroc(scores, is_wrong) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(np.asarray(is_wrong).astype(int), scores))


def precision_at_k(scores, is_wrong, k=None) -> float:
    """k defaults to the true number of mislabeled images."""
    is_wrong = np.asarray(is_wrong).astype(bool)
    k = int(k or is_wrong.sum())
    top = np.argsort(-np.asarray(scores), kind="stable")[:k]
    return float(is_wrong[top].mean())
