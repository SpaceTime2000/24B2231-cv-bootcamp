"""cvb: shared library for the CIFAR-10N noisy-label bootcamp.

Modules
  config     frozen training Recipe + output Paths
  data       dataset loading, GPU-side augmentation, label accessors (clean labels are measurement-only)
  losses     per-sample loss functions (CE, label smoothing, GCE, SCE)
  methods    training methods (single net with any loss, co-teaching with 2 nets)
  engine     resumable training loop; saves per-epoch logits for any eval set
  selection  label-free filtering / relabeling from Task 3 scores
  metrics    noise stats, confusion matrix, dynamics, AUROC, precision@k
  io         prediction / score / json writers with format checks
  cli        shared argparse options
"""
