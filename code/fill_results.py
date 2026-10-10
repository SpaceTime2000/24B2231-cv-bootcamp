"""Fill results.json from the saved run outputs, plus a report skeleton and an evidence digest for Q1-Q5.

    python fill_results.py --data-dir data --out-dir outputs

Writes  <out>/results.json         (refuses to overwrite an existing one unless --overwrite: you may have edited it)
        <out>/report.md            skeleton with Q1-Q5 (only if absent); you write the answers, <=100 words each
        <out>/reports/report_evidence.md   numbers to base the answers on
Accuracies are fractions in 0-1. NOTE: the template's *_noisy_labels__clean_ref_model field is the clean-label
reference model scored against the NOISY validation labels, so it is recomputed here from saved logits.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cvb import io
from cvb.config import Paths
from cvb.engine import load_logits

SIGNAL_TEXT = {"nll_mean": "mean over epochs of the held-out negative log-likelihood of the given label",
               "nll_final": "final-epoch held-out negative log-likelihood of the given label",
               "nll_classnorm": "class-normalized held-out negative log-likelihood of the given label",
               "disagree_frac": "fraction of epochs in which the held-out prediction disagrees with the given label"}


def val_acc(run, labels):
    return float((load_logits(run, "val")[-1].argmax(1) == labels).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--name", default="Tanish Kharbanda")
    ap.add_argument("--roll-no", default="24B2231")
    ap.add_argument("--gpu", default=None, help="GPU name; auto-detected if CUDA is available")
    ap.add_argument("--memorization-epoch", type=int, default=None,
                    help="your reading of the Task 2 curves; default = heuristic from t2_summary.json")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    paths = Paths(args.out_dir)
    todo = []

    # noisy validation labels (measurement-free: these are the labels model selection uses anyway)
    df = pd.read_csv(Path(args.data_dir) / "labels.csv").sort_values("index").set_index("index")
    noisy_val = df.loc[np.loadtxt(Path(args.data_dir) / "val_idx.txt", dtype=int), "noisy_label"].to_numpy()

    cfg = io.load_json(paths.run("t1_noisy_ce") / "config.json", {})
    if not cfg:
        raise SystemExit("run task1_baselines.py first")
    if cfg["optimizer"] == "sgd":
        opt = f"SGD, momentum {cfg['momentum']}{' (nesterov)' if cfg['nesterov'] else ''}, " \
              f"peak lr {cfg['lr']}, weight decay {cfg['weight_decay']}"
    else:
        opt = f"AdamW, peak lr {cfg['adamw_lr']}, weight decay {cfg['weight_decay']}"
    sched = (f"{cfg['warmup_epochs']:g}-epoch linear warmup, then cosine decay to 0 over {cfg['epochs']} epochs"
             if cfg["schedule"] == "cosine" else
             f"{cfg['warmup_epochs']:g}-epoch warmup, then step decay x0.1 at 50% and 75% of {cfg['epochs']} epochs")
    gpu = args.gpu
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except Exception:
        pass
    if not gpu:
        gpu = ""
        todo.append("training_setup.gpu_used (pass --gpu)")

    stats = io.load_json(paths.reports / "t1_noise_stats.json", {})
    t2 = io.load_json(paths.reports / "t2_summary.json", {})
    t3 = io.load_json(paths.reports / "t3_noisy_meta.json", {})
    t4 = io.load_json(paths.reports / "t4_summary.json", {})
    t5 = io.load_json(paths.reports / "t5_summary.json", {})

    mem = args.memorization_epoch or t2.get("first_epoch_mislabeled_acc_rises_10pts_above_min")
    if args.memorization_epoch is None and mem is not None:
        print(f"NOTE: memorization_start_epoch = {mem} is only a heuristic (first epoch the mislabeled-subset train "
              f"accuracy is 10 points above its running minimum). Check the plot and override with --memorization-epoch.")
    if mem is None:
        todo.append("task2.memorization_start_epoch (pass --memorization-epoch)")

    submitted_signal, sig_note = None, None
    sp, gp = paths.preds / "t3_noise_scores.csv", paths.reports / "t3_noisy_signals.csv"
    if sp.exists() and gp.exists():
        sub, sigs = pd.read_csv(sp).set_index("index")["score"], pd.read_csv(gp).set_index("index")
        for c in ["nll_mean", "nll_final", "nll_classnorm", "disagree_frac"]:
            if c in sigs and np.allclose(sub.reindex(sigs.index).to_numpy(), sigs[c].to_numpy(), atol=1e-5):
                submitted_signal = c
        sig_note = submitted_signal or "NONE of the saved signals matches the submitted file"
        print(f"submitted t3_noise_scores.csv == signal: {sig_note}")
        if submitted_signal and t3.get("submit_signal") != submitted_signal:
            print(f"NOTE: t3_noisy_meta.json says '{t3.get('submit_signal')}' but the file matches '{submitted_signal}'")
    if submitted_signal:
        t3["submit_signal"] = submitted_signal

    t4_methods = []
    for m, v in t4.get("methods", {}).items():
        f = f"t4_{m}.csv"
        if (paths.preds / f).exists():
            t4_methods.append({"name": m, "file": f, "val_acc_noisy_labels": v["val_acc_noisy"]})

    res = {
        "name": args.name, "roll_no": args.roll_no, "seed": cfg["seed"],
        "training_setup": {"optimizer": opt, "learning_rate_schedule": sched, "batch_size": cfg["batch_size"],
                           "epochs": cfg["epochs"],
                           "augmentation": f"random crop (zero-pad {cfg['crop_pad']}) + random horizontal flip; "
                                           f"per-channel mean/std normalization", "gpu_used": gpu},
        "task1": {"noise_rate": stats.get("train_noise_rate"),
                  "val_acc_noisy_labels__clean_ref_model": val_acc(paths.run("t1_clean_ref"), noisy_val)
                  if (paths.run("t1_clean_ref") / "DONE").exists() else None,
                  "val_acc_noisy_labels__noisy_ce_model": val_acc(paths.run("t1_noisy_ce"), noisy_val)},
        "task2": {"memorization_start_epoch": mem, "early_stop_epoch": t2.get("early_stop_epoch"),
                  "val_acc_noisy_labels_at_early_stop": t2.get("val_acc_noisy_at_early_stop")},
        "task3": {"auroc_self_check": (io.load_json(paths.reports / "t3_noisy_eval.json", {}).get("overall", {})
                                        .get(t3.get("submit_signal"), {}).get("auroc")),
                  "method": (f"{t3.get('folds')}-fold cross-validated out-of-fold scoring with ResNet-18 "
                             f"({t3.get('fold_epochs')} epochs/fold, noisy labels only); score = "
                             f"{SIGNAL_TEXT.get(t3.get('submit_signal'), t3.get('submit_signal'))}") if t3 else ""},
        "task4": {"methods": t4_methods, "best_method": t4.get("best_method", "")},
        "task5": {"val_acc_synthetic_labels__baseline": t5.get("synthetic", {}).get("baseline_ce_val_acc_synth"),
                  "val_acc_synthetic_labels__best_method": t5.get("synthetic", {}).get("best_val_acc_synth")},
    }
    if t5 and t5.get("best_method") != res["task4"]["best_method"]:
        todo.append("task5 used a different method than task4.best_method - check")

    def walk(d, pre=""):
        for k, v in d.items():
            if isinstance(v, dict):
                walk(v, pre + k + ".")
            elif v is None or v == "" or v == []:
                todo.append(pre + k)
    walk(res)

    rj = paths.root / "results.json"
    if rj.exists() and not args.overwrite:
        print(f"{rj} exists - not overwritten (use --overwrite). Freshly computed version printed below.")
        print(io.json.dumps(res, indent=2, default=float))
    else:
        io.save_json(rj, res)
        print(f"wrote {rj}")

    # ---- report skeleton + evidence digest -----------------------------------------------------------
    rp = paths.root / "report.md"
    if not rp.exists():
        rp.write_text("\n\n".join(f"Q{i}. TODO (<=100 words)" for i in range(1, 6)) + "\n")
    ev = ["# Evidence for report.md (not part of the submission)\n"]
    if stats:
        ev.append("## Q1 - noise structure\n" + "\n".join(
            f"- {p['clean']} -> {p['noisy']}: {p['count']} ({100 * p['share']:.1f}% of wrong labels)"
            for p in stats["top_pairs"][:8]) + f"\n- overall noise rate {stats['train_noise_rate']:.3f}\n")
    if t2:
        ev.append("## Q2 - memorization\n" + "\n".join(f"- {k}: {v}" for k, v in t2.items()) +
                  "\n- see reports/t2_dynamics.png and t2_dynamics_extended.csv\n")
    e3 = io.load_json(paths.reports / "t3_noisy_eval.json", {})
    if e3:
        ev.append("## Q3 - noise-score signals (AUROC / precision@k)\n" + "\n".join(
            f"- {s}: {v['auroc']:.3f} / {v['precision_at_k']:.3f}" for s, v in e3["overall"].items()) +
            "\n- per-class breakdown: reports/t3_noisy_eval_per_class.csv (look for the worst class = failure mode)\n")
    if t4:
        ev.append("## Q4 - Task 4 (noisy val acc)\n" + "\n".join(
            f"- {m}: {v['val_acc_noisy']:.4f}  {({k: x for k, x in v.items() if k != 'val_acc_noisy'})}"
            for m, v in t4["methods"].items()) + f"\n- baseline CE (Task 1): {res['task1']['val_acc_noisy_labels__noisy_ce_model']:.4f}\n")
    if t5:
        ev.append("## Q5 - real vs synthetic\n" + "\n".join(f"- {k}: {v}" for k, v in t5.items()) + "\n")
    (paths.reports / "report_evidence.md").write_text("\n".join(ev))
    print("evidence digest -> reports/report_evidence.md")
    if todo:
        print("STILL EMPTY in results.json:", ", ".join(todo))


if __name__ == "__main__":
    main()
