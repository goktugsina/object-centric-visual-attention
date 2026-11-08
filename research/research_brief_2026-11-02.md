# Research status and recent developments

Checked 2 November 2026. This is a targeted primary-source update, not an exhaustive survey or a claim to identify the single newest paper. Publication issue dates, preprint dates and benchmark dates are different. Reported external performance has not been independently reproduced here.

## What problem are we studying?

A fixation is a period when the eyes remain directed at a location. Static saliency prediction estimates a probability map of where people look in an image. Scanpath prediction additionally models fixation order and often duration, previous fixations and task. Brain decoding predicts something from neural measurements such as fMRI. Our implemented work concerns static, overt gaze probabilities. It neither predicts fixation sequences nor analyzes fMRI, regardless of the workspace directory name.

The question is: how much does useful object information help attention prediction, and how does the result change when objects must be detected automatically instead of being supplied by human annotators?

## Recent work relevant to this question

| Work and source status | Main development | Implication for this project |
|---|---|---|
| [FastGaze](https://www.sciencedirect.com/science/article/pii/S0957417426017562), *Expert Systems with Applications*, volume 328, issue date 1 October 2026; [official code](https://github.com/JiaHongZ/FastGaze) | A compact task-conditioned scanpath model using an inhibition-of-return Transformer for free viewing and target-guided search. This is the most recent journal issue date among the directly relevant sources checked in this update, not proof it was the latest idea released. | Task guidance, efficient Transformers and inhibition of return are established research directions. We would need to implement sequential prediction before a direct comparison makes sense. |
| [DeepGaze3.5-VL](https://arxiv.org/abs/2607.02083), preprint submitted 2 July 2026 | Treats fixation coordinates as tokens in an autoregressive vision-language model, supporting task/viewer conditioning and durations. The authors report 2.18 bits IG on MIT1003 and a 46% improvement over DeepGaze III. | A substantially more expressive sequential model than ours. Its headline number cannot be compared with our OSIE score. We have not reproduced its results. |
| [ST-DiffEye](https://arxiv.org/abs/2606.15486), preprint submitted 13 June 2026; [author project page](https://st-diffeye.github.io/) labels it NeurIPS 2026 | Jointly models continuous gaze trajectories and discrete scanpaths using diffusion. It also proposes distribution-aware evaluation based on CRPS. | Modeling variability and using proper probabilistic scores are already active topics. The project-page venue label is distinguished from the arXiv version consulted. |
| [Why We Look Where We Look](https://arxiv.org/abs/2605.17823), preprint submitted 18 May 2026 | A foveated vision-language agent trained for scene comprehension develops several human-like fixation patterns. | A more substantive brain-inspired claim requires an explicit computational constraint and behavioral tests. Giving a model object features alone does not establish a neural mechanism. |
| [Modeling Saliency Dataset Bias / DeepGaze MSDB](https://openaccess.thecvf.com/content/ICCV2025/html/Kummerer_Modeling_Saliency_Dataset_Bias_ICCV_2025_paper.html), ICCV 2025 | Models interpretable dataset-specific spatial biases with fewer than 20 parameters; the authors report closing over 75% of their measured generalization gap through adaptation. | This is a more direct static-saliency comparator. Cross-dataset generalization and center bias must be evaluated explicitly. |
| [Explainable Saliency / XSal](https://openaccess.thecvf.com/content/CVPR2025/html/Chen_Explainable_Saliency_Articulating_Reasoning_with_Contextual_Prioritization_CVPR_2025_paper.html), CVPR 2025 | Uses vision-language reasoning and contextual prioritization to generate saliency maps with explanations. | Semantic explanations alone are not a novel contribution. Any claimed explanation should also be tested for predictive or intervention validity. |

The [official MIT/Tuebingen MIT300 leaderboard](https://saliency.tuebingen.ai/results.html) lists DeepGaze MSDB at 1.2463 IG. Its IG reference is an image-independent center-bias distribution. Our principal table uses a uniform-pixel reference on OSIE and a different image weighting/filtering protocol. Comparing 1.2463 directly with our 0.501 would be invalid. A fair comparison requires running the models on the same eligible fixations with identical splits and score definitions. Leaderboard rankings also vary by metric; this note does not assert a universal best model.

## What we actually implemented

The original pilot uses 700 OSIE scenes, 15 observers and 87,662 retained fixations of at least 100 ms. Each image belongs to exactly one of training, validation or test within a split (420/140/140 images). The same observers occur across splits. Human masks are converted into nonoverlapping object/background regions with an explicit overlap rule. A regularized conditional-choice model predicts a probability for each region from six geometric features and, optionally, twelve human semantic attributes.

The first extension evaluates six annotation-feature variants across five fixed image splits: geometry, full semantics, no face, no text, no face or text, and within-image shuffled semantics. It also evaluates equal observer weighting and inference-time semantic omission with 20 draws per level. The five splits overlap and therefore are not five independent replications.

The second extension adds an executable image-only system. A frozen Faster R-CNN MobileNetV3 detector produces boxes, confidence scores and COCO classes. A learned linear choice model assigns region probabilities. Pixels receive uniform density within their assigned region, including background. At inference this model reads the image and model weights, with no human object labels or gaze records. Detector provenance and image hashes are recorded. Three feature variants are evaluated across the same five splits. Box shrinking, expansion and deletion probe sensitivity to region assumptions.

Different region partitions cannot be compared using raw categorical NLL: changing the categories changes the event space. We therefore score human and detected models at the same fixation pixels. This is a comparability safeguard based on standard probabilistic evaluation, not a claim to have invented a new metric.

## Findings and their limits

| Finding | Measured result | Interpretation |
|---|---|---|
| Human semantics add useful signal | Gain over geometry: 0.101-0.149 bits per fixation across splits; equal-observer evaluation: 0.104-0.153 | Stable within this dataset and cohort; not proof about new observers or datasets. |
| Face and text explain only part of the gain | Removing both leaves 0.049-0.073 bits; shuffled semantics yields -0.007 to 0.002 | Correct semantic assignment matters. A single shuffle per split is a diagnostic, not a permutation-test p-value. |
| Detected class information adds a smaller gain | 0.011-0.036 bits over geometry and confidence; original-split gain 0.017, paired 95% interval [0.003, 0.033] | Classes help this detector-based representation. Not every split interval excludes zero. |
| The image-only baseline is weak against center bias | Original-split IG above uniform: 0.501 versus center prior 0.522; difference -0.021, interval [-0.137, 0.095] | No clear superiority to the center prior. The center difference is also stored for every split. |
| Human annotations give a much stronger reference | Human semantics: 1.515 bits; detected classes: 0.501; paired gap 1.013, interval [0.904, 1.122] | Joint differences in regions, geometry and semantic features contribute. The gap cannot all be attributed to detector mistakes. |
| Coverage is not sufficient for good density predictions | Shrinking box dimensions by 20% raises IG to 0.605 while reducing foreground fixation coverage; expansion lowers IG to 0.376 | Consistent with overly diffuse uniform-within-box predictions, but not an isolated causal test of that mechanism. |

All displayed absolute IG values in our result table use the same original 140-scene test set and uniform-pixel baseline. The intervals resample images and condition on the fitted models and observed cohort. Controlled omission is not a realistic natural-corruption benchmark. Box deletion uses one nested random draw, so its outcomes do not estimate variability over many deletion patterns. Pretraining overlap with OSIE source images is not fully audited.

## What is meaningful, and what remains unfinished?

Scope clarification: the target is now a small first research publication, as described in [the bounded publication plan](publication_plan.md). The larger experiments below are optional follow-ups for stronger claims, not a mandatory checklist before sharing a carefully reviewed OSIE case study.

The current contribution is a reproducible, controlled diagnosis of the gap between annotation-conditioned and image-only object attention. It includes negative results, semantic controls, probability normalization, common fixation support and a usable inference command. It is more substantial than a proposed architecture with unrun experiments. It does not establish a new state-of-the-art model, a new biological theory, or priority for the broad observation that semantics affect gaze.

The highest-value next study should separate object selection from within-object localization. Use a normalized spatial density inside each region instead of spreading its mass uniformly. Keep the detector and training split fixed, and compare uniform, Gaussian and learned spatial densities. Add matched region controls (human masks, bounding boxes derived from those masks, detector boxes) and matched semantic controls wherever available. This can distinguish losses from coarse boundaries, spatial density and missing semantic information more clearly than the present joint contrast.

Then evaluate a strong frozen static saliency model such as DeepGaze MSDB or DeepGaze IIE under exactly the same event filtering and scoring. Check its training datasets and image overlap before interpreting an external-data comparison. Replicate on a second eye-tracked free-viewing dataset with a locked protocol; do not silently pool it with a visual-search task. Hold out observers when making claims about new people. Repeat corruption seeds, report uncertainty and retain failed or null comparisons. A sequential model is a separate subsequent experiment, with DeepGaze3.5-VL, ST-DiffEye or FastGaze becoming relevant comparators.

The delivered artifact is a six-page IEEE-format manuscript with 20 references, source code, fitted parameters, per-image results and 18 passing contract tests. It is a candidate for a modest empirical preprint after author review and submission preparation. The larger comparisons above are needed to support broader claims, not automatically required for this bounded case study. Test passes verify code contracts, not novelty, external validity or venue acceptance.

Substantial assistance remains accurately recorded in the manuscript acknowledgment. [arXiv guidance](https://info.arxiv.org/help/moderation/index.html#policy-for-authors-use-of-generative-ai-language-tools) and [IEEE guidance](https://open.ieee.org/author-guidelines-for-artificial-intelligence-ai-generated-text/) require reporting relevant generative-tool use. Removing branding does not replace this disclosure or the author's responsibility to verify the work.
