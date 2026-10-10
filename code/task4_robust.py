"""Task 4 - robust training. Pick 1-3 methods (budget), all under the exact Task 1 recipe.

  coteach  two nets, each trains on the small-loss samples chosen by the other (2 networks)
  filter   drop likely-mislabeled samples (Task 3 scores), retrain with CE
  relabel  like filter, but confidently-predicted dropped samples get the out-of-fold label instead
  gce / sce / ls   robust losses / label smoothing (single net)

The best method = highest NOISY validation accuracy at the final epoch; it is written to the summary and read by
task5_synthetic.py.

    python task4_robust.py --methods coteach filter gce --data-dir data --out-dir outputs
"""
import numpy as np

from cvb import cli, io
from cvb.runners import run_method, MethodOpts, ALL_METHODS


def main():
    ap = cli.base_parser(__doc__)
    ap.add_argument("--methods", nargs="+", required=True, choices=[m for m in ALL_METHODS if m != "ce"])
    ap.add_argument("--forget-rate", type=float, default=0.4)
    ap.add_argument("--drop-mode", default="gmm", choices=["gmm", "frac"])
    ap.add_argument("--drop-frac", type=float, default=0.35)
    ap.add_argument("--relabel-conf", type=float, default=0.8)
    ap.add_argument("--no-per-class", action="store_true")
    ap.add_argument("--force", action="store_true", help="allow more than 3 methods in total (breaks the rules)")
    args = ap.parse_args()
    data, recipe, paths = cli.setup(args)
    opts = MethodOpts(args.forget_rate, args.drop_mode, args.drop_frac, not args.no_per_class, args.relabel_conf)

    sp = paths.reports / "t4_summary.json"
    summ = io.load_json(sp, {"methods": {}})
    total = set(summ["methods"]) | set(args.methods)
    assert len(total) <= 3 or args.force, f"rules allow 1-3 methods; this would make {len(total)}: {sorted(total)}"

    for m in args.methods:
        res = run_method(m, "noisy", data, recipe, paths, "t4", opts)
        io.write_predictions(paths.preds / f"t4_{m}.csv", res["test_pred"], data.n_test)
        summ["methods"][m] = {"val_acc_noisy": res["val_acc"], **res["info"]}
        io.save_json(sp, summ)
        print(f"{m}: noisy val acc = {res['val_acc']:.4f}  {res['info']}")

    best = max(summ["methods"], key=lambda k: summ["methods"][k]["val_acc_noisy"])
    summ["best_method"] = best
    io.save_json(sp, summ)
    print(f"best so far (noisy val acc): {best}")


if __name__ == "__main__":
    main()
