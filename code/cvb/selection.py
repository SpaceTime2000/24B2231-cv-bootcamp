"""Label-free data selection for the filter / relabel methods. Inputs are Task 3 scores (computed from NOISY
labels only) and, for relabeling, out-of-fold predicted probabilities. No clean labels, no known noise rate."""
import numpy as np


def keep_mask_frac(scores, labels, drop_frac=0.35, per_class=True):
    """Drop the `drop_frac` highest-scoring samples (overall, or within each given-label class)."""
    scores, labels = np.asarray(scores), np.asarray(labels)
    keep = np.ones(len(scores), dtype=bool)
    groups = [np.arange(len(scores))] if not per_class else [np.where(labels == c)[0] for c in np.unique(labels)]
    for g in groups:
        k = int(round(drop_frac * len(g)))
        if k:
            keep[g[np.argsort(-scores[g], kind="stable")[:k]]] = False
    return keep


def keep_mask_gmm(scores, labels=None, per_class=False, seed=0):
    """Two-component 1-D Gaussian mixture on the scores; drop samples more likely in the high-score component.
    Fully label-free choice of how many to drop. Falls back to the median split if the GMM is degenerate."""
    from sklearn.mixture import GaussianMixture
    scores = np.asarray(scores, dtype=float)

    def fit(s):
        z = ((s - s.mean()) / (s.std() + 1e-12)).reshape(-1, 1)
        g = GaussianMixture(2, random_state=seed, n_init=3).fit(z)
        hi = int(np.argmax(g.means_.ravel()))
        return g.predict_proba(z)[:, hi] < 0.5   # keep if NOT likely in high-score (noisy) component

    if not per_class or labels is None:
        return fit(scores)
    keep = np.ones(len(scores), dtype=bool)
    for c in np.unique(labels):
        g = np.where(labels == c)[0]
        keep[g] = fit(scores[g])
    return keep


def relabel(scores, labels, oof_probs, keep, conf=0.8):
    """For samples dropped by `keep`: replace the label with the out-of-fold argmax if its confidence >= conf
    (relabel), otherwise leave the sample out. Returns (new_labels, use_mask)."""
    labels = np.asarray(labels).copy()
    use = keep.copy()
    drop = np.where(~keep)[0]
    pred, p = oof_probs[drop].argmax(1), oof_probs[drop].max(1)
    confident = p >= conf
    labels[drop[confident]] = pred[confident]
    use[drop[confident]] = True
    return labels, use
