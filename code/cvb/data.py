"""Dataset loading + GPU-side preprocessing.

Label hygiene: training code only ever receives a label array handed to it explicitly. The clean column is
reachable only through `clean_labels_for_measurement` (Task 1 reference model + measurements), so any use of
clean labels in the code is greppable.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

_COLS = {"noisy": "noisy_label", "synthetic": "synthetic_label"}


class Data:
    def __init__(self, data_dir, device="cuda"):
        d = Path(data_dir)
        want = torch.device(device)
        self.device = want if (want.type != "cuda" or torch.cuda.is_available()) else torch.device("cpu")

        imgs = np.load(d / "train_images.npy")          # (50000, 32, 32, 3) uint8, row i == index i
        test = np.load(d / "test_images.npy")           # (10000, 32, 32, 3) uint8
        df = pd.read_csv(d / "labels.csv").sort_values("index").reset_index(drop=True)
        assert (df["index"].to_numpy() == np.arange(len(df))).all(), "labels.csv index must be 0..N-1"
        assert len(df) == len(imgs), "labels.csv and train_images.npy disagree on length"
        self._df = df

        self.train_idx = np.loadtxt(d / "train_idx.txt", dtype=np.int64)
        self.val_idx = np.loadtxt(d / "val_idx.txt", dtype=np.int64)
        assert not set(self.train_idx.tolist()) & set(self.val_idx.tolist()), "train/val overlap"

        tr = imgs[self.train_idx]
        # per-channel mean/std from the TRAIN subset only (never val/test)
        s = np.zeros(3); s2 = np.zeros(3); n = 0
        for i in range(0, len(tr), 5000):
            c = tr[i:i + 5000].astype(np.float64) / 255.0
            s += c.sum((0, 1, 2)); s2 += (c ** 2).sum((0, 1, 2)); n += c.shape[0] * 32 * 32
        mean = s / n
        std = np.sqrt(s2 / n - mean ** 2)
        self.mean = torch.tensor(mean, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)

        to_t = lambda a: torch.from_numpy(a).permute(0, 3, 1, 2).contiguous().to(self.device)
        self.x_train = to_t(tr)                         # uint8 on device, positions 0..len(train_idx)-1
        self.x_val = to_t(imgs[self.val_idx])
        self.x_test = to_t(test)
        self.n_train, self.n_val, self.n_test = len(tr), len(self.val_idx), len(test)

    # ---- labels -------------------------------------------------------------------------------------
    def labels(self, which: str, split: str) -> np.ndarray:
        """'noisy' | 'synthetic' labels for split 'train' (45k) or 'val' (5k), as a numpy array."""
        idx = self.train_idx if split == "train" else self.val_idx
        return self._df[_COLS[which]].to_numpy()[idx].astype(np.int64)

    def clean_labels_for_measurement(self, split: str) -> np.ndarray:
        """CLEAN labels. Allowed ONLY for: Task 1 reference model + measuring (confusion matrix, Task 2 curves,
        Task 3 self-check). Never train / filter / relabel / select with this."""
        idx = self.train_idx if split == "train" else self.val_idx
        return self._df["clean_label"].to_numpy()[idx].astype(np.int64)

    def t(self, arr) -> torch.Tensor:
        return torch.as_tensor(arr, device=self.device)

    # ---- preprocessing ------------------------------------------------------------------------------
    def prep(self, x_u8: torch.Tensor, augment: bool = False, pad: int = 4) -> torch.Tensor:
        x = (x_u8.float() / 255.0 - self.mean) / self.std
        if augment:
            x = random_crop_flip(x, pad)
        if self.device.type == "cuda":
            x = x.contiguous(memory_format=torch.channels_last)
        return x


def random_crop_flip(x: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """Batched random crop (zero-pad `pad`, crop back to HxW) + random horizontal flip, all on device."""
    n, c, h, w = x.shape
    dev = x.device
    xp = F.pad(x, (pad, pad, pad, pad))
    i = torch.randint(0, 2 * pad + 1, (n,), device=dev)
    j = torch.randint(0, 2 * pad + 1, (n,), device=dev)
    rows = (i[:, None] + torch.arange(h, device=dev)[None])
    cols = (j[:, None] + torch.arange(w, device=dev)[None])
    flip = torch.rand(n, device=dev) < 0.5
    cols = torch.where(flip[:, None], cols.flip(1), cols)
    out = xp[torch.arange(n, device=dev)[:, None, None, None],
             torch.arange(c, device=dev)[None, :, None, None],
             rows[:, None, :, None], cols[:, None, None, :]]
    return out
