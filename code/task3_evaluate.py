"""Task 3 self-check (measurement only: this is where clean labels are allowed for Task 3).
AUROC and precision@k (k = true number of mislabeled) for every signal, plus a per-class breakdown for Q3.
Do NOT use this to switch the submitted signal; --submit-signal in task3_scores.py is fixed beforehand.

    python task3_evaluate.py --data-dir data --out-dir outputs
"""
import numpy as np
import pandas as pd

from cvb import cli, io, metrics


def main():
    ap = cli.base_parser(__doc__)
    ap.add_argument("--label-set", default="noisy", choices=["noisy", "synthetic"])
    args = ap.parse_args()
    data, _, paths = cli.setup(args)
    names = cli.class_names(args.data_dir)

    df = pd.read_csv(paths.reports / f"t3_{args.label_set}_signals.csv")
    assert (df["index"].to_numpy() == data.train_idx).all()
    y = df["given_label"].to_numpy()
    wrong = y != data.clean_labels_for_measurement("train")
    meta = io.load_json(paths.reports / f"t3_{args.label_set}_meta.json", {})
    k = int(wrong.sum())

    out = {}
    for s in ["nll_mean", "nll_final", "nll_classnorm", "disagree_frac"]:
        out[s] = {"auroc": metrics.auroc(df[s], wrong), "precision_at_k": metrics.precision_at_k(df[s], wrong, k)}
    print(f"true mislabeled k={k}   submitted signal: {meta.get('submit_signal')}")
    print(pd.DataFrame(out).T.round(4).to_string())

    sub = meta.get("submit_signal", "nll_mean")
    rows = []
    for c, n in enumerate(names):   # failure-mode view: how well does the submitted signal work per given-label class
        m = y == c
        if wrong[m].any() and (~wrong[m]).any():
            rows.append({"given_label": n, "noise_rate": wrong[m].mean(), "auroc": metrics.auroc(df[sub][m], wrong[m]),
                         "prec_at_k": metrics.precision_at_k(df[sub][m].to_numpy(), wrong[m])})
    pc = pd.DataFrame(rows)
    print("\nper given-label class (submitted signal):\n", pc.round(3).to_string(index=False))
    io.save_json(paths.reports / f"t3_{args.label_set}_eval.json", {"overall": out, "k": k})
    pc.to_csv(paths.reports / f"t3_{args.label_set}_eval_per_class.csv", index=False)


if __name__ == "__main__":
    main()
