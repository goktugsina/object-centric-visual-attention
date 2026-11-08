# Frozen saliency comparison and external-data transfer

Specified 5 November 2026 before computing the new test outcomes. This continues `shared_plan_status.md` after the frozen ResNet/GAT experiment.

## OSIE comparator

Use the official [DeepGaze IIE](https://github.com/matthias-k/DeepGaze) implementation pinned at `c7db17e2d1d7ea6468ffdee2cfaddf141095dcff`, pretrained weights and supplied MIT1003 center-bias map. This is a static model; no previous human fixation is supplied. The [paper](https://arxiv.org/abs/2105.12441) describes SALICON pretraining and MIT1003 fine-tuning. These are external supervision, unlike our OSIE-only readouts. The model is a strong comparator, not a newly trained matched-data ablation or a claim to the latest state of the art. ImageNet/stylized-ImageNet and gaze-training image overlap is not exhaustively ruled out.

Resize the full OSIE photograph to 1024 x 768 (long edge 1024), without cropping. Supply RGB values in [0,255], following the upstream API. Resize the upstream log center-bias with nearest-neighbor interpolation, renormalize by logsumexp, and freeze it. Convert the output to a positive density, bilinearly resize it back to the original 800 x 600 support, and renormalize. Cache source hashes, package source hashes, weights, original/input sizes and predicted maps. No OSIE outcomes select these settings.

Evaluate exactly the existing 140 test images and duration >=100 ms fixation coordinates. Compare uniform pixels, the existing validation-selected center Gaussian (sigma .2), human-semantic linear reference, detected-class linear reference, and fixed detected full-self/GAT checkpoints (seed 11; not selected by test). Primary endpoint: equally image-weighted information gain over uniform pixels, and paired image-bootstrap differences from the center and detected linear references. The same observers recur across OSIE splits; intervals condition on them.

Secondary descriptive scores: NSS at eligible fixation events; a deterministic AUC-Judd implementation with thresholds at unique positive fixation-map values, no random jitter and explicit ties; CC and KL in bits against an empirical fixation density smoothed with Gaussian sigma=3% of the shorter image dimension, reflect boundary, and normalized to sum one. Report the bandwidth and metric conventions. These are common-map comparisons, not MIT benchmark submissions or metric-specific optimized saliency maps. Constant-map NSS/CC return zero by convention; constant-map AUC is .5. AUC uses unique fixation pixels, while likelihood/NSS retain repeated events.

## EMOd audit and transfer

Retrieve the images from the [official project page](https://www-users.cse.umn.edu/~qzhao/emotionalattention.html) and MAT annotations from the [author repository](https://github.com/Fanshaojing/emotionalattention), pinned at `d0d1758ac4eef9e7db6ceef78d1bfdf3dfa14d02`. Record hashes and inspect actual fields, dimensions, subject identities and coordinate/duration conventions before constructing the loader. Stop dependent transfer scoring if coordinates cannot be established. Keep original photographs and model weights outside Git; no redistribution license is inferred from public download access.

Before inspecting target scores, screen every EMOd image against all 700 OSIE images with file hashes, decoded RGB hashes, normalized 32 x 32 RGB exact matches and 64-bit perceptual hashes. Conservatively exclude any target with an OSIE perceptual-hash neighbor within four bits, as well as exact matches, from the primary transfer set. Report full-set sensitivity separately. This screening is not proof of absence of cropped or edited overlap. Do not assume identical image filenames mean identical images.

Transfer the original OSIE detected-class linear checkpoint and frozen ResNet full-self/GAT heads for seeds 11,22,33. Freeze detector, preprocessing, coefficients, feature normalization and all hyperparameters. Use the original OSIE center bandwidth .2 without fitting EMOd. No EMOd data train, calibrate or choose a model. Evaluate all eligible nonoverlapping images, with the same pixel metrics and image weighting. Do not map EMOd sentiment labels to OSIE's twelve semantics; the deployable arm only needs original photographs and gaze coordinates. The label-format audit is separate from predictive inputs.

DeepGaze IIE is first run on the 140 OSIE test scenes. A full EMOd DeepGaze run is a separate compute extension unless completed and explicitly reported. ResNet+U-Net and backbone fine-tuning are not implied by using a frozen external comparator. Independent observer holdout requires a separate retraining/evaluation protocol; descriptive observer groups do not fulfill it.

## Completion evidence

Keep per-image CSVs, paired contrasts, normalizations, preprocessing counts, excluded-image lists, exact model/source hashes, timings and tests. Record unsuccessful steps without inventing metrics. Report changes in the standalone continuation report before incorporating them into the paper.
