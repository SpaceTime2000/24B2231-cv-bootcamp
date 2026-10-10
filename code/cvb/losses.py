"""Per-sample losses (reduction='none'). Each takes (logits, targets) and returns a (B,) tensor."""
import torch
import torch.nn.functional as F


def ce(logits, y):
    return F.cross_entropy(logits, y, reduction="none")


def label_smoothing(eps=0.1):
    def fn(logits, y):
        return F.cross_entropy(logits, y, reduction="none", label_smoothing=eps)
    return fn


def gce(q=0.7):
    """Generalized cross-entropy (Zhang & Sabuncu 2018): (1 - p_y^q) / q. q->0 is CE, q=1 is MAE."""
    def fn(logits, y):
        p_y = F.softmax(logits, dim=1).gather(1, y[:, None]).squeeze(1).clamp_min(1e-7)
        return (1.0 - p_y ** q) / q
    return fn


def sce(alpha=0.1, beta=1.0, A=-4.0):
    """Symmetric cross-entropy (Wang et al. 2019): alpha*CE + beta*RCE, RCE = -A * (1 - p_y)."""
    def fn(logits, y):
        c = F.cross_entropy(logits, y, reduction="none")
        p_y = F.softmax(logits, dim=1).gather(1, y[:, None]).squeeze(1)
        return alpha * c + beta * (-A) * (1.0 - p_y)
    return fn


def make_loss(name: str, **kw):
    if name == "ce":
        return ce
    if name == "ls":
        return label_smoothing(**kw)
    if name == "gce":
        return gce(**kw)
    if name == "sce":
        return sce(**kw)
    raise ValueError(f"unknown loss {name}")
