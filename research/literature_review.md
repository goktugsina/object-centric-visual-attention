# Literature review and contribution audit

Search date: 2 November 2026. This is a targeted narrative review, not a systematic review or an exhaustive novelty search. Searches covered object-centric gaze, uncertainty, semantic attention, scanpath prediction, calibration, OSIE, COCO-Search18 and SALICON. Primary papers, author repositories, conference proceedings and dataset pages were preferred. Preprints are identified explicitly.

## What the roadmap needs to change

Combining objects, uncertainty and sequential attention is not by itself new. The defensible immediate product is a transparent replication-style pilot with calibrated object-event probabilities, followed by a carefully controlled image-only study. The current pilot tests a narrower estimand than saliency prediction: which annotated region contains a randomly selected eligible fixation in an unseen image.

| Prior work | Evidence relevant to this project | Consequence for the contribution |
|---|---|---|
| [Xu et al. 2014](https://www-users.cse.umn.edu/~qzhao/publications/pdf/saliency_jov14.pdf) | OSIE already combines pixel, object and semantic features. | A semantic-feature benefit is a replication-style finding, not a first discovery. |
| [Yang et al. 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Yang_Predicting_Goal-Directed_Human_Attention_Using_Inverse_Reinforcement_Learning_CVPR_2020_paper.html) | Goal-conditioned inverse reinforcement learning with a simulated fovea. | Foveation and task guidance require direct baselines, not novelty by naming. |
| [Chen et al. 2021](https://www.nature.com/articles/s41598-021-87715-9) | COCO-Search18 records target-present and target-absent visual search. | Keep search separate from OSIE free viewing; do not assume same-image multi-target counterfactual annotations exist. |
| [Kummerer et al. 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9055565/) | DeepGaze III predicts conditional fixation distributions from image and history. | Sequential probabilities and variability already have strong precedents. |
| [Fang et al. 2024](https://arxiv.org/abs/2407.13335) | OAT represents visual-search scanpaths as sequences of object fixations. | Object-level scanpath modeling cannot be claimed as the new contribution. Source consulted is the arXiv preprint. |
| [Yang et al. 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Yang_Unifying_Top-down_and_Bottom-up_Scanpath_Prediction_Using_Transformers_CVPR_2024_paper.html) | HAT combines task-conditioned transformers and foveated memory for search and free viewing. | A ResNet plus U-Net is a debugging baseline, not a sufficient research comparator. |
| [Chen et al. 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Chen_Beyond_Average_Individualized_Visual_Scanpath_Prediction_CVPR_2024_paper.html) | Observer embeddings and personalized feature integration. | Personalization needs held-out observers and support/query separation. |
| [Mengers et al. 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC11812614/) | Object segmentation uncertainty actively guides scanpaths in dynamic scenes. | Distinguish segmentation uncertainty from predictive calibration of object selection. |
| [Xue et al. 2025](https://arxiv.org/abs/2504.05499) | Few-shot adaptation to unseen viewers. | A same-observer image split cannot support claims about new people. |
| [Quan et al. 2026](https://arxiv.org/abs/2604.20361) | Language-referred object search with a vision-language model; arXiv record lists ICMR 2026. | Task-conditioned semantic fusion is an active, crowded direction. |
| [Zhao et al. 2026](https://arxiv.org/abs/2606.15486) | Preprint proposing joint trajectory/scanpath diffusion and distribution-aware scoring. | Distributional evaluation is not an unoccupied research niche. |
| [Agrawal et al. 2026](https://arxiv.org/abs/2607.02083) | Preprint modeling scanpaths as autoregressive tokens with flexible conditioning. | Update comparisons before submission; do not call 2022 models the current frontier. |

## Dataset decisions

OSIE is the pilot because human masks and semantic annotations enable a clean conditional analysis without detector training. Its 700 scenes and 15 viewers are small for broad generalization. The source paper reports 5,551 annotated objects; the script checks the actual files. The `gazed` attribute means an object looked at by a person or animal depicted in the scene, not the observers' measured fixation outcome (Xu et al., Table 1). It is not a direct encoding of the response.

COCO-Search18 should be the next goal-directed experiment. The [official project page](https://sites.google.com/view/cocosearch/) provides train/validation annotations and withholds benchmark test fixations. Its opening release text is older than its resource list, which also links target-absent data. Record the exact download/version rather than relying on the introductory text alone. All COCO-derived sources must be deduplicated by original COCO image ID across pretraining and evaluation.

[SALICON](https://complexity.cecs.ucf.edu/salicon/) uses mouse-contingent annotations; it is useful for spatial pretraining but is not interchangeable with eye-tracked scanpath timing. Neither OSIE nor COCO-Search18, as used here, supplies fMRI measurements. The workspace name does not establish an fMRI reconstruction experiment.

## Probability evaluation

[Gneiting and Raftery 2007](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf) motivates proper scores. [Guo et al. 2017](https://proceedings.mlr.press/v70/guo17a.html) motivates validation-only temperature scaling. [Gal and Ghahramani 2016](https://proceedings.mlr.press/v48/gal16.html) and [Lakshminarayanan et al. 2017](https://proceedings.neurips.cc/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html) describe neural uncertainty methods, but the implemented bootstrap of a convex linear choice model is neither of those methods. Entropy, observer disagreement and epistemic uncertainty are related but not interchangeable quantities.

## Candidate research contribution after the pilot

Status update, 2 November 2026: the annotation-robustness and image-only baseline experiments have now been implemented and run. Their results are in [extension_results.md](extension_results.md). The [research briefing](research_brief_2026-11-02.md) updates the comparison with DeepGaze MSDB, XSal, foveated vision-language modeling and FastGaze (October 2026 journal issue). These additions sharpen the outstanding comparison and generalization requirements; they do not establish publication readiness or novelty priority. The paragraph below remains a proposal for a stronger follow-up, not a description of completed experiments.

Test whether an image-only object distribution remains calibrated across object-detector errors, task changes, and observer variation. Pair each neural model with an equally expressive version without object semantics, use common visual features, and account for object area and image center. Test a learned recency term against zero and fixed inhibition, allowing return fixations. A positive result would need held-out image and observer evidence, neural baselines and sensitivity to boundary assignment. No priority or publication guarantee is made.
