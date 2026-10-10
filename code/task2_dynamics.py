"""Task 2 - training dynamics of the Task 1 noisy run (no retraining; reads the saved per-epoch logits).

Writes t2_dynamics.csv (exact 5 columns the grader wants), an extra table + plot, and t2_early_stop.csv.
The early-stopping epoch is chosen from NOISY validation accuracy only (trailing 3-epoch moving average,
ties -> earlier epoch). Clean labels are used only to measure the curves.

    python task2_dynamics.py --data-dir data --out-dir outputs
"""
import numpy as np
import pandas as pd

from cvb import cli, io, metrics
from cvb.engine import load_logits

COLS = ["epoch", "train_acc_correct", "train_acc_mislabeled", "val_acc_noisy", "val_acc_clean"]


def main():
    ap = cli.base_parser(__doc__)
    ap.add_argument("--ma-window", type=int, default=3)
    args = ap.parse_args()
    data, recipe, paths = cli.setup(args)
    run = paths.run("t1_noisy_ce")
    assert (run / "DONE").exists(), "run task1_baselines.py first"

    L_tr, L_va, L_te = (load_logits(run, n) for n in ("train", "val", "test"))
    rows = metrics.dynamics(L_tr, L_va, data.labels("noisy", "train"), data.clean_labels_for_measurement("train"),
                            data.labels("noisy", "val"), data.clean_labels_for_measurement("val"))
    df = pd.DataFrame(rows)
    df[COLS].to_csv(paths.preds / "t2_dynamics.csv", index=False)
    df.to_csv(paths.reports / "t2_dynamics_extended.csv", index=False)

    ep = metrics.pick_early_stop(df["val_acc_noisy"].to_numpy(), args.ma_window)
    io.write_predictions(paths.preds / "t2_early_stop.csv", L_te[ep - 1].argmax(1), data.n_test)

    # evidence for Q2 (descriptive; the early-stop choice above does NOT use these)
    m = df["train_acc_mislabeled"].to_numpy()
    run_min = np.minimum.accumulate(m)
    onset = {f"first_epoch_mislabeled_acc_rises_{int(100 * t)}pts_above_min":
             int(np.argmax(m - run_min >= t)) + 1 if (m - run_min >= t).any() else None for t in (0.05, 0.10, 0.20)}
    summary = {"early_stop_epoch": ep, "val_acc_noisy_at_early_stop": float(df.loc[ep - 1, "val_acc_noisy"]),
               "val_acc_noisy_final": float(df["val_acc_noisy"].iloc[-1]),
               "epoch_of_peak_val_acc_clean (measurement only)": int(df["val_acc_clean"].idxmax()) + 1,
               **onset}
    io.save_json(paths.reports / "t2_summary.json", summary)
    print(df[COLS].iloc[::max(1, len(df) // 12)].round(4).to_string(index=False))
    print(summary)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        for c, lab in [("train_acc_correct", "train acc, correctly-labeled"),
                       ("train_acc_mislabeled", "train acc, mislabeled (vs given label)"),
                       ("val_acc_noisy", "val acc (noisy labels)"), ("val_acc_clean", "val acc (clean labels)")]:
            ax.plot(df["epoch"], df[c], label=lab)
        ax.axvline(ep, ls="--", c="gray", label=f"early stop = epoch {ep}")
        ax.set_xlabel("epoch"); ax.set_ylabel("accuracy"); ax.legend(fontsize=7); fig.tight_layout()
        fig.savefig(paths.reports / "t2_dynamics.png", dpi=150)
    except Exception as e:  # plotting is optional
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
