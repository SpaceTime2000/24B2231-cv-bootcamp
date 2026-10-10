# Report - 24B2231

### Q1
No single pair dominates; the noise is diffuse. The largest are truck→automobile (6.9% of all wrong labels), automobile→truck (6.2%), cat→dog (5.1%), dog→cat (4.6%) and deer→horse (3.8%). These five cover 26.6% of the 18,109 mislabeled training images, and the top eight pairs only 33.5%. Errors are mostly between visually similar classes (vehicles, cats/dogs, deer/horse), with the rest spread thinly over other pairs.

### Q2
Memorization starts around epoch 40. Accuracy on the mislabeled training subset hovers near 13% through epoch ~30, stays at least 5 points above that baseline from epoch 40 on, and accelerates after epoch 46 (28% at epoch 46, 66% at epoch 60). Noisy validation accuracy meanwhile plateaus near 0.55 (peak 0.5568 at epoch 38) and falls to 0.4998 by epoch 60 as the network fits wrong labels.

### Q3
I submitted nll_mean (mean over epochs of the out-of-fold negative log-likelihood of the given label, the pre-set default): AUROC 0.942, precision@k 0.828. In my self-check disagree_frac scored slightly higher (0.953, 0.866). Failure mode: cat and dog. Per-class AUROC is lowest for cat (0.876) and dog (0.917) versus 0.98 for airplane and ship. They are about 48-49% mislabeled and mutually confusable, so wrong labels there likely look plausible and get lower loss.

### Q4
Co-teaching helped most: 0.586 vs 0.500 for plain CE on noisy validation labels. Each network trains on the small-loss samples its peer selects, so wrong labels likely enter the gradient less often. It beat GCE (0.567) and filtering (0.552; it dropped 17,933 samples, but the scores have precision@k of only 0.83, so some clean images were likely removed). Caveats: co-teaching predicts with the average of two networks, so part of the gain may be ensembling; all results are single-seed.

### Q5
No, the ranking did not change: co-teaching beat CE on both (real 0.586 vs 0.500; synthetic 0.531 vs 0.434), and the gain was slightly larger on synthetic noise (+9.8 vs +8.6 points, single seed). Only CE and co-teaching were run on synthetic labels, so I cannot compare GCE or filtering there. A plausible reason: small-loss selection assumes wrong labels are random, which symmetric noise satisfies; human noise is class-structured (cat↔dog, truck↔automobile), so some wrong labels look easy and slip through.
