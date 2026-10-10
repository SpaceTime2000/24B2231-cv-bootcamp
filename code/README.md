# Inter-IIT CV Bootcamp - learning from noisy labels (CIFAR-10N)

Roll no: 24B2231. This folder (`code/`) holds all code; the repo root also has `predictions/`, `results.json`
and `report.md`.

## Setup
```
python -m venv .venv && .venv\Scripts\activate          # Windows
pip install -r requirements.txt                         # then install the CUDA build of torch from pytorch.org
python -c "import torch; print(torch.cuda.is_available())"   # must print True
```
Put the provided files in `data/` (`train_images.npy`, `test_images.npy`, `labels.csv`, `train_idx.txt`,
`val_idx.txt`, `classes.txt`) and keep the provided `model.py` in this folder (it is imported, never edited).

## Layout
```
model.py               provided ResNet-18 (untouched)
cvb/                   shared library (reused by every task)
  config.py            frozen Recipe (optimizer/schedule/aug) + output paths
  data.py              loading, GPU augmentation; clean labels only via clean_labels_for_measurement()
  losses.py            CE, label smoothing, GCE, SCE          methods.py   SingleNet, CoTeaching
  engine.py            resumable training loop, saves per-epoch logits for any eval set
  selection.py         label-free filter / relabel from Task 3 scores
  runners.py           run_method(): one call trains any Task 4/5 method on any label set
  metrics.py io.py cli.py
task1_baselines.py     noise stats + confusion matrix, clean reference, noisy CE
task2_dynamics.py      per-epoch curves, early-stop epoch + predictions (reads Task 1 logits, no retraining)
task3_scores.py        5-fold out-of-fold noise scores            task3_evaluate.py  AUROC / precision@k check
task4_robust.py        coteach / filter / relabel / gce / sce / ls
task5_synthetic.py     CE + best Task 4 method on synthetic noise
fill_results.py        fills results.json and an evidence digest from the saved runs
```

## Reproduce (run from this folder; every command resumes if interrupted)
```
python task1_baselines.py --data-dir data --out-dir outputs
python task2_dynamics.py  --data-dir data --out-dir outputs
python task3_scores.py    --data-dir data --out-dir outputs
python task3_evaluate.py  --data-dir data --out-dir outputs
python task4_robust.py    --data-dir data --out-dir outputs --methods coteach filter gce
python task5_synthetic.py --data-dir data --out-dir outputs
python fill_results.py    --data-dir data --out-dir outputs --gpu "<gpu name>" --memorization-epoch 40 --overwrite
```
Outputs: `outputs/predictions/*.csv` (the graded files, copied to `predictions/` in the submission),
`outputs/reports/*` (stats, curves, summaries), `outputs/runs/*` (logs, per-epoch logits, final weights).
`report.md` was written by hand from `outputs/reports/report_evidence.md`.

## What was run
- **Task 3:** 5-fold out-of-fold scoring (30 epochs per fold). The submitted score is `nll_mean`, the mean over
  epochs of the held-out negative log-likelihood of the given label (the script's default signal).
- **Task 4:** co-teaching (two networks, forget rate 0.4 taken from the ~40% noise stated in the problem),
  filtering (drops the high-score component of a per-class 2-component Gaussian mixture on the Task 3 scores,
  then retrains with CE) and GCE. Best on noisy validation accuracy: co-teaching.
- **Task 5:** CE and co-teaching retrained on `synthetic_label`, with the same recipe and settings.

## Fixed recipe (identical for every run)
SGD (momentum 0.9, nesterov), peak lr 0.1, wd 5e-4, batch 128, 2-epoch warmup + cosine, 60 epochs, random crop
(pad 4) + horizontal flip, mixed precision, normalization stats from the 45k train images only, seed 0.
Task 3 folds use 30 epochs (cap is 60). GPU runs are not bit-for-bit reproducible even with a fixed seed.

## Rule compliance
Clean labels: used for the Task 1 reference model and for measurements only (confusion matrix, Task 2 curves,
`task3_evaluate.py`). Tasks 2-5 train, filter and select with noisy (Task 5: synthetic) labels only. Task 1
predictions are from the final epoch; the early-stopping epoch is picked from noisy val accuracy only.