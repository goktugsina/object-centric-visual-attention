# Frozen visual features and object relations

Specified 3 November 2026 before fitting the new models. This continues the research branch in the shared conversation (private planning conversation omitted), rather than the unrelated admissions checklist. Existing results and the six-page version-1 paper remain historical artifacts.

## Question and scope

Does adding frozen visual information, then learned interaction between regions, improve object-event probabilities? This is a static free-viewing experiment, not a scanpath or neural-mechanism model. It cannot yet answer whether object models outperform strong pixel saliency networks.

Use the existing 700 OSIE images and 100 ms fixation filter. Keep the original 420/140/140 split (seed 20261102), not the conversation's illustrative 490/105/105 split. Preserve explicit background, smallest-mask human partition and confidence-resolved detector partition. The two arms are annotation-conditioned and image-only, respectively; human semantics are never deployable inputs.

## Visual representation

Freeze Torchvision ResNet50 IMAGENET1K_V2, eval mode, including batch-normalization statistics. Resize the entire photograph to 224 x 224, without center cropping, and apply ImageNet channel normalization. This deliberately differs from the weights' classification crop transform to retain edge objects. Area-downsample each disjoint region mask onto the 7 x 7 final feature grid; take its coverage-weighted mean of the 2048 channels. Area resampling preserves contributions from tiny regions. Background is pooled identically. These embeddings have large receptive fields and already contain scene context: the local head is independent only after feature extraction.

Cache image/annotation/detector/weight hashes, region identities, feature configuration and versions. No gaze labels enter feature extraction. Check cache provenance and region alignment when training. Normalize feature columns using training images only, each image contributing equal weight.

## Models and selection

For each partition, fit visual-only, visual+geometry and visual+geometry+semantics heads (12 human attributes or detector confidence and COCO classes). Geometry contains the existing six columns, including background. Full features also feed a one-head additive graph-attention layer on the complete within-image graph, including self edges and background. Compare learned edges to self-only edges and uniform all-node mixing in the same module. Self-only and uniform controls have the same nominal parameter count, but attention-score parameters are inactive; they are controls, not proof of equal effective capacity. A residual local encoder and shared prediction head are common to all variants. No edges cross images.

Use hidden width 64, AdamW learning rate 1e-4, weight decay 1e-3, batch size 16, maximum 50 epochs, early stopping patience 10, and independent initialization seeds 11, 22, 33. Select each checkpoint using validation image-average cross entropy only. Do not select a seed on test results. Report all seeds and their arithmetic mean; seed-averaged per-image contrasts are not an ensemble prediction. No hyperparameter tuning after test inspection in this run.

Primary scores: image-average fixation-event NLL and pixel information gain over uniform, using uniform density within each region. Also report expected one-hot Brier, equal-observer NLL, top-1/3 event coverage, MAE, Spearman (undefined constant ranks excluded and counted), and NDCG@3 with linear fixation-share relevance and average tie handling. Background participates in every region score. Compare to the existing linear model and validation-selected pixel center prior on identical test images. Bootstrap paired image differences, conditioning on the observed viewer cohort; these intervals do not include generalization to new observers or training runs. Report seed variability separately.

Observer agreement is mean Jensen-Shannon divergence between each observer's eligible fixation distribution and their equal-observer mean. Define low/medium/high groups from training-image tertiles; test observer data are used only to describe and score those groups, never as model inputs. This is descriptive observer-aware evaluation, not personalized prediction.

## Visual checks and stopping rule

Choose ten training images deterministically, before viewing prediction quality. Show retained fixation coordinates, source-mask boundaries, disjoint regions, per-observer fixation overlays and object target distributions. Keep photographs and derived overlays outside Git. Inspect all ten outputs.

Finish this bounded next stage with runnable extraction, training, inference, a Colab notebook, measured results and a plan-to-evidence checklist. Do not expand to backbone fine-tuning based on a disappointing test result. ResNet+U-Net, an external saliency comparator, EMOd format/overlap audit and cross-dataset evaluation remain separate later stages. Do not label these completed or infer publication novelty from this experiment.

Sources: [Torchvision ResNet50](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html), [Graph Attention Networks](https://arxiv.org/abs/1710.10903). Existing source papers and data provenance are in `literature_review.md` and `DATA_CARD.md`.
