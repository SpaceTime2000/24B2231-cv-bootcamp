"""Task 3 - per-image noise scores via K-fold out-of-fold (OOF) predictions.

Each training image is scored by a model that never trained on it, so a net that memorized a wrong label cannot
hide the error. Budget: at most 5 training runs here (= --folds, max 5), <=60 epochs each. Uses NOISY labels only.

Signals (higher = more likely mislabeled), all computed from the held-out logits saved every epoch:
  nll_mean       mean over epochs of  -log p(given label)         <- submitted by default (pre-committed)
  nll_final      same at the last epoch
  nll_classnorm  nll_mean z-scored within each given-label class
  disagree_frac  fraction of epochs where argmax != given label
The submitted signal is fixed by --submit-signal BEFORE looking at any clean-label evaluation.

    python task3_scores.py --data-dir data --out-dir outputs                 # real (human) noise
    python task3_scores.py --label-set synthetic ...                          # only if Task 5 uses filter/relabel
"""
import numpy as np
import pandas as pd
from scipy.special import log_softmax, softmax
from sklearn.model_selection import StratifiedKFold

from cvb import cli, io
from cvb.engine import run_training, load_logits
from cvb.methods import make_method
from cvb.runners import scores_path, oof_path

SIGNALS = ["nll_mean", "nll_final", "nll_classnorm", "disagree_frac"]


def compute_signals(L, y):
    """L: (E, n, C) held-out logits over epochs; y: (n,) given labels."""
    logp = log_softmax(L, axis=-1)
    nll = -np.take_along_axis(logp, y[None, :, None].repeat(L.shape[0], 0), axis=2)[..., 0]  # (E, n)
    return {"nll_mean": nll.mean(0), "nll_final": nll[-1], "disagree_frac": (L.argmax(-1) != y[None]).mean(0)}, \
        softmax(L[-1], axis=-1)


def main():
    ap = cli.base_parser(__doc__)
    ap.add_argument("--label-set", default="noisy", choices=["noisy", "synthetic"])
    ap.add_argument("--folds", type=int, default=5, help="= number of training runs; budget cap is 5")
    ap.add_argument("--fold-epochs", type=int, default=30)
    ap.add_argument("--submit-signal", default="nll_mean", choices=SIGNALS)
    args = ap.parse_args()
    assert args.folds <= 5, "budget: at most 5 training runs in Task 3"
    data, recipe, paths = cli.setup(args)

    y = data.labels(args.label_set, "train")
    oof_logits = None
    skf = StratifiedKFold(args.folds, shuffle=True, random_state=args.seed)
    sig = {k: np.zeros(len(y)) for k in ("nll_mean", "nll_final", "disagree_frac")}
    oof_probs = np.zeros((len(y), 10), dtype=np.float32)

    for k, (tr, ho) in enumerate(skf.split(np.zeros(len(y)), y)):
        d = paths.run(f"t3_{args.label_set}_fold{k}")
        tr_t, ho_t = data.t(tr), data.t(ho)
        run_training(d, data, recipe, make_method("ce"), data.x_train[tr_t], y[tr],
                     eval_sets={"heldout": data.x_train[ho_t]}, epochs=args.fold_epochs)
        s, p = compute_signals(load_logits(d, "heldout"), y[ho])
        for name in sig:
            sig[name][ho] = s[name]
        oof_probs[ho] = p

    # class-normalized NLL: uses only the given labels
    z = np.zeros(len(y))
    for c in np.unique(y):
        m = y == c
        z[m] = (sig["nll_mean"][m] - sig["nll_mean"][m].mean()) / (sig["nll_mean"][m].std() + 1e-12)
    sig["nll_classnorm"] = z

    df = pd.DataFrame({"index": data.train_idx, "given_label": y, **sig})
    df.to_csv(paths.reports / f"t3_{args.label_set}_signals.csv", index=False)
    np.save(oof_path(paths, args.label_set), oof_probs)
    chosen = df[["index"]].assign(score=df[args.submit_signal])
    chosen.to_csv(scores_path(paths, args.label_set), index=False)       # consumed by filter / relabel
    if args.label_set == "noisy":
        io.write_scores(paths.preds / "t3_noise_scores.csv", chosen["index"], chosen["score"])
    io.save_json(paths.reports / f"t3_{args.label_set}_meta.json",
                 {"submit_signal": args.submit_signal, "folds": args.folds, "fold_epochs": args.fold_epochs})
    print(f"Task 3 done; submitted signal = {args.submit_signal}")


if __name__ == "__main__":
    main()
