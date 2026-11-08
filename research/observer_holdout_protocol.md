# Held-out observer evaluation

Specified 5 November 2026 before fitting this experiment. The existing OSIE observer arrays have 15 entries for every image. Source-order subject indices are treated as consistent across OSIE images, as in the data release; this assumption does not extend to EMOd's missing subject records.

Split observers into three fixed contiguous groups of five: source indices 0–4, 5–9, 10–14. For each fold, use only the other ten observers' events on the existing 420 training images. Select regularization from {0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2} using only those same ten observers on the 140 validation images. Fit geometry and full semantic linear models on human regions, plus geometry+confidence and full detected-class linear models on detector regions. Report proper categorical NLL, expected Brier and common-pixel IG on the 140 test images using the five completely held-out observers.

All models remain population predictors; no held-out observer supplies adaptation/support data. This tests transfer to unobserved source-order viewers within the same OSIE cohort, not personalization or a new recruited population. Average each observer's fixation distribution equally within an image for training, validation and evaluation. Keep an explicit background and the original duration >=100 ms filter and partition rules.

Each of 15 observers is tested exactly once; all 700 image identities keep their original split. Summarize each fold and an equal-fold mean. For the semantic/class contrast, bootstrap images after averaging paired differences across the three observer folds. Folds share training images and participants, so they are not independent replications. No observer-population confidence claim follows from an image-only bootstrap.

Do not reuse all-observer fitted coefficients for held-out-viewer evaluation. Write training/validation/test observer index lists and assert disjointness in every fold; retain fitted coefficients, regularization choices and per-image rows. This bounded experiment does not train a subject embedding or alter the prior neural comparison.
