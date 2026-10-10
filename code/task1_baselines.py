"""Task 1 - data & baselines.
  * noise rate + 10x10 confusion matrix (noisy vs clean)         [measurement: clean labels allowed]
  * clean-label reference model (CE)                              [clean labels allowed here]
  * noisy-label CE baseline (also saves per-epoch TRAIN logits for Task 2)
Predictions come from the FINAL epoch of each run.

    python task1_baselines.py --data-dir data --out-dir outputs
"""
import numpy as np
import pandas as pd

from cvb import cli, io, metrics
from cvb.engine import run_training, load_logits
from cvb.methods import make_method


def main():
    ap = cli.base_parser(__doc__)
    ap.add_argument("--only", choices=["both", "clean", "noisy", "stats"], default="both")
    args = ap.parse_args()
    data, recipe, paths = cli.setup(args)
    names = cli.class_names(args.data_dir)

    # ---- noise statistics (measurement) ----
    st = metrics.noise_stats(data.labels("noisy", "train"), data.clean_labels_for_measurement("train"))
    stv = metrics.noise_stats(data.labels("noisy", "val"), data.clean_labels_for_measurement("val"))
    pd.DataFrame(st["confusion"], index=[f"clean_{n}" for n in names],
                 columns=[f"noisy_{n}" for n in names]).to_csv(paths.reports / "t1_confusion_matrix.csv")
    pairs = metrics.top_noise_pairs(st["confusion"], names, k=10)
    io.save_json(paths.reports / "t1_noise_stats.json", {
        "train_noise_rate": st["noise_rate"], "val_noise_rate": stv["noise_rate"],
        "per_class_noise_rate": dict(zip(names, st["per_class_noise_rate"])), "top_pairs": pairs})
    print(f"noise rate  train={st['noise_rate']:.4f}  val={stv['noise_rate']:.4f}")
    for p in pairs[:6]:
        print(f"  {p['clean']:>10} -> {p['noisy']:<10} {p['count']:6d}  ({100 * p['share']:.1f}% of wrong labels)")
    if args.only == "stats":
        return

    log = print
    if args.only in ("both", "clean"):
        # clean-label REFERENCE: the one place clean labels may be used for training
        yc_tr, yc_va = data.clean_labels_for_measurement("train"), data.clean_labels_for_measurement("val")
        d = paths.run("t1_clean_ref")
        run_training(d, data, recipe, make_method("ce"), data.x_train, yc_tr,
                     eval_sets={"val": data.x_val, "test": data.x_test}, eval_labels={"val": yc_va}, log=log)
        io.write_predictions(paths.preds / "t1_clean_ref.csv", load_logits(d, "test")[-1].argmax(1), data.n_test)

    if args.only in ("both", "noisy"):
        d = paths.run("t1_noisy_ce")
        run_training(d, data, recipe, make_method("ce"), data.x_train, data.labels("noisy", "train"),
                     eval_sets={"train": data.x_train, "val": data.x_val, "test": data.x_test},
                     eval_labels={"val": data.labels("noisy", "val")}, log=log)
        io.write_predictions(paths.preds / "t1_noisy_ce.csv", load_logits(d, "test")[-1].argmax(1), data.n_test)
    print("Task 1 done")


if __name__ == "__main__":
    main()
