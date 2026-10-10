"""Task 5 - same two experiments on SYNTHETIC (symmetric) noise: CE baseline and the best Task 4 method.
Selection/logging uses the synthetic validation labels only. Identical recipe and method settings as before.

If the best method is filter/relabel, first run:  python task3_scores.py --label-set synthetic ...

    python task5_synthetic.py --data-dir data --out-dir outputs          # best taken from t4_summary.json
"""
from cvb import cli, io
from cvb.runners import run_method, MethodOpts, ALL_METHODS


def main():
    ap = cli.base_parser(__doc__)
    ap.add_argument("--best", default=None, choices=ALL_METHODS, help="default: best_method in t4_summary.json")
    ap.add_argument("--forget-rate", type=float, default=0.4)
    ap.add_argument("--drop-mode", default="gmm", choices=["gmm", "frac"])
    ap.add_argument("--drop-frac", type=float, default=0.35)
    ap.add_argument("--relabel-conf", type=float, default=0.8)
    ap.add_argument("--no-per-class", action="store_true")
    args = ap.parse_args()
    data, recipe, paths = cli.setup(args)
    opts = MethodOpts(args.forget_rate, args.drop_mode, args.drop_frac, not args.no_per_class, args.relabel_conf)

    t4 = io.load_json(paths.reports / "t4_summary.json", {})
    best = args.best or t4.get("best_method")
    assert best, "run task4_robust.py first (or pass --best)"

    base = run_method("ce", "synthetic", data, recipe, paths, "t5", opts)
    io.write_predictions(paths.preds / "t5_baseline_synth.csv", base["test_pred"], data.n_test)
    res = run_method(best, "synthetic", data, recipe, paths, "t5", opts)
    io.write_predictions(paths.preds / "t5_best_synth.csv", res["test_pred"], data.n_test)

    from cvb.engine import load_logits
    real_base = float((load_logits(paths.run("t1_noisy_ce"), "val")[-1].argmax(1) == data.labels("noisy", "val")).mean())
    out = {"best_method": best,
           "synthetic": {"baseline_ce_val_acc_synth": base["val_acc"], "best_val_acc_synth": res["val_acc"],
                         "gain": res["val_acc"] - base["val_acc"]},
           "real": {"baseline_ce_val_acc_noisy": real_base,
                    "best_val_acc_noisy": t4.get("methods", {}).get(best, {}).get("val_acc_noisy")}}
    if out["real"]["best_val_acc_noisy"] is not None:
        out["real"]["gain"] = out["real"]["best_val_acc_noisy"] - real_base
    io.save_json(paths.reports / "t5_summary.json", out)
    print(out)


if __name__ == "__main__":
    main()
