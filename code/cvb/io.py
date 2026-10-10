"""Writers with format checks for the files the grader reads."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def write_predictions(path, preds, n_expected=None):
    """id,pred with one row per test image; pred in 0..9."""
    preds = np.asarray(preds).astype(int)
    assert preds.ndim == 1 and preds.min() >= 0 and preds.max() <= 9, "pred must be in 0..9"
    if n_expected is not None:
        assert len(preds) == n_expected, f"expected {n_expected} rows, got {len(preds)}"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": np.arange(len(preds)), "pred": preds}).to_csv(path, index=False)


def write_scores(path, index, score):
    index = np.asarray(index)
    score = np.asarray(score, dtype=float)
    assert len(index) == len(score) and np.isfinite(score).all(), "scores must be finite, one per index"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"index": index, "score": score}).to_csv(path, index=False)


def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=float))


def load_json(path, default=None):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else default
