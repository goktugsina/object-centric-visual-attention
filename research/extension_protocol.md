# Robustness and image-only extension

Specification recorded on 2 November 2026 after inspecting the original pilot. These analyses are exploratory extensions, not an independent confirmation or an externally preregistered study. All outcomes are reported, including unfavorable results.

## Research question

How much of the semantic benefit survives changes in image sampling, semantic-feature availability, and replacement of human object annotations by detections inferred from pixels?

The immediate deliverable is a reproducible comparison and error analysis, not a claim that a detector plus a conditional-choice model is a new architecture. The existing oracle-feature result is already consistent with Xu et al. (2014).

## Oracle robustness

Use five fixed image split seeds, 20261102 through 20261106, with the original 420/140/140 allocation. The splits overlap and are not independent datasets. Report each split and the descriptive range; do not pool their test images as independent observations or calculate a confidence interval across five seeds.

Select L2 separately for each model using only its validation images, from 0, 0.000001, 0.00001, 0.0001, 0.001 and 0.01. Compare geometry, all semantic attributes, removal of face, removal of text, removal of both face and text, and a within-image semantic-row permutation control. Permute only foreground object rows, retain all twelve attributes together, and leave geometry and background unchanged. A single such control per split is a diagnostic, not a permutation-test p-value.

Fit the original image-balanced, event-weighted target. Evaluate both that target and an equal-observer-within-image target, omitting only observer records with no eligible events. This reweighting does not hold observers out and does not establish generalization to new people. Use paired image bootstrap intervals conditional on each fitted model and the sampled cohort.

On the original split, additionally test semantic-vector omission at probabilities 0, 0.25, 0.5, 0.75 and 1. Replace selected foreground semantic vectors by zero at inference, with 20 deterministic repetitions. The fixed model is not refitted on test outcomes. This is a synthetic missing-information stress test, not a realistic estimate of detector errors.

## Image-only model

Use a frozen, explicitly versioned pretrained detector from an official model distribution. Profile one image before the complete run. The local installation has CPU-only PyTorch and approximately 3 GB of free disk space; use existing packages and a compact model rather than installing a CUDA environment.

Use Faster R-CNN MobileNetV3 Large 320 FPN, COCO_V1 weights, as a tractable initial detector. This supplies boxes, class labels and confidence from images. It is an established baseline, not a contemporary state-of-the-art claim. Use confidence threshold 0.5 and at most 50 detections, fixed before gaze evaluation. Rasterize boxes onto the original image; overlapping pixels go to the highest-confidence retained detection, ties to source order. Background covers every remaining pixel. Include images with zero detections.

Fit geometry-only and geometry-plus-detected-class conditional models using training fixations. Both models use the same predicted partition. At inference neither model accesses human object masks, human semantic attributes or test gaze. Class features are COCO class indicators; they are not estimates of OSIE's twelve semantic attributes.

For comparison across different partitions, expand each region probability into uniform mass over that region's pixels. Score the probability of the same eligible fixation pixel under every model. A raw categorical NLL across different region partitions is not comparable. Report information gain over a common uniform-pixel baseline and an image-independent center prior, with equal image weight. Human-mask models are annotation-conditioned references, not guaranteed performance upper bounds.

Stress-test predicted boxes by deterministic dilation/erosion and removal of detections. Hold the fitted parameters fixed and recompute partitions and features without using gaze. Report zero-detection scenes, foreground area, fixation coverage, and error-conditioned performance. Synthetic perturbations measure sensitivity to controlled changes, not natural detector accuracy.

## Publication threshold

Scope clarification added after the experiments: the paragraph below sets a threshold for strong performance/generalization claims, not a universal requirement for a small empirical preprint. The author's current version-1 goal is a bounded OSIE case study; see [publication_plan.md](publication_plan.md). This clarification changes the publication scope, not the original experimental specification or outcomes.

Before a strong publication claim: inspect overlays on sampled scenes; verify training-data overlap as far as available metadata permits; add a compatible image-only saliency baseline; evaluate across image splits and a second suitable dataset; and have the author independently review references, equations, code and results. Pretrained COCO weights do not by themselves prove that OSIE images were absent from pretraining.

Current policy references: [arXiv's generative AI and moderation policy](https://info.arxiv.org/help/moderation/index.html#policy-for-authors-use-of-generative-ai-language-tools) and [IEEE author guidance](https://open.ieee.org/author-guidelines-for-artificial-intelligence-ai-generated-text/). Significant assistance must be represented accurately. The author remains responsible for the submission; no submission or acceptance is implied by this repository.
