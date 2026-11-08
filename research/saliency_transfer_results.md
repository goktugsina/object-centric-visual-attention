# Saliency comparison, external transfer and excluded observers

Completed 5 November 2026. All values below are generated from completed result JSON.

### 5.7 Visual features and relation mixing

Thirty neural fits completed on the original split. Across three seeds, full local and GAT heads achieve 1.383 and 1.380 bits of IG with annotated regions and 0.367 and 0.328 bits with detected regions. GAT minus local improvement is -0.003 bits with paired image interval [-0.011, 0.005] for annotated regions and -0.039 [-0.062, -0.020] for detections. Uniform mixing performs similarly to GAT. Every neural variant underperforms the corresponding prior full-feature linear reference. These results do not support adding this mixing layer under the tested frozen features and training settings.

### 5.8 Strong static comparator and domain transfer

Table III reports the common-pixel comparison. DeepGaze IIE improves over the detected linear model by 2.194 bits, with paired interval [2.083, 2.305]. Its NSS is 3.771, AUC-Judd 0.932, CC 0.809 and KL 0.697 bits. This establishes a substantial gap to a pretrained spatial predictor under the documented preprocessing, without isolating architecture from training-data differences.

Table III  Static prediction on the same 140 OSIE test scenes

| Model | IG bits | NSS |
|---|---|---|
| Center prior | 0.522 | 0.842 |
| Detected linear | 0.501 | 0.884 |
| Detected local seed 11 | 0.380 | 0.749 |
| Detected GAT seed 11 | 0.318 | 0.733 |
| Human semantic linear | 1.515 | 2.302 |
| DeepGaze IIE | 2.695 | 3.771 |

On the 698 public EMOd scenes, 102,760 eligible events remain. Center and OSIE-trained detected linear IG are 0.590 and 0.206 bits. Their paired linear-minus-center difference is -0.384 [-0.421, -0.346]. The fixed linear model therefore underperforms the center prior on this target subset. Mean seed-specific IG for local and GAT heads is 0.088 and 0.064 bits. These are averages of scores, not prediction ensembles. There are 243 scenes without retained detections (34.8 percent); their region models return uniform density. All scenes remain in evaluation. This is zero-shot transfer to the available subset, not evaluation of all 1019 EMOd images.

### 5.9 Prediction for excluded observers

Full-minus-geometry log-score improvement for held-out observers is 0.149 bits with image interval [0.121, 0.177] for annotated regions, and 0.017 [0.002, 0.033] for detected classes beyond geometry and confidence. Full-model IG is 1.525 and 0.506 bits, respectively. The semantic association persists under this cohort holdout. The experiment does not learn individual preferences or establish adaptation to a new person.

## Secondary metrics

| Dataset | Model | IG bits | NSS | AUC Judd | CC | KL bits |
|---|---|---:|---:|---:|---:|---:|
| OSIE test 140 | center | 0.522 | 0.842 | 0.742 | 0.253 | 2.381 |
| OSIE test 140 | deepgaze_iie | 2.695 | 3.771 | 0.932 | 0.809 | 0.697 |
| OSIE test 140 | detected_linear | 0.501 | 0.884 | 0.683 | 0.245 | 2.433 |
| OSIE test 140 | full_gat_11 | 0.318 | 0.733 | 0.679 | 0.200 | 2.635 |
| OSIE test 140 | full_self_11 | 0.380 | 0.749 | 0.679 | 0.204 | 2.569 |
| OSIE test 140 | human_semantic_linear | 1.515 | 2.302 | 0.819 | 0.522 | 1.812 |
| OSIE test 140 | uniform | -0.000 | 0.000 | 0.500 | 0.000 | 2.879 |
| EMOd public 698 | center | 0.590 | 0.958 | 0.755 | 0.396 | 1.542 |
| EMOd public 698 | full_gat_11 | 0.074 | 0.411 | 0.583 | 0.136 | 2.071 |
| EMOd public 698 | full_gat_22 | 0.044 | 0.408 | 0.582 | 0.135 | 2.102 |
| EMOd public 698 | full_gat_33 | 0.073 | 0.420 | 0.584 | 0.140 | 2.072 |
| EMOd public 698 | full_self_11 | 0.104 | 0.418 | 0.582 | 0.140 | 2.038 |
| EMOd public 698 | full_self_22 | 0.067 | 0.408 | 0.581 | 0.135 | 2.078 |
| EMOd public 698 | full_self_33 | 0.094 | 0.417 | 0.582 | 0.139 | 2.048 |
| EMOd public 698 | linear | 0.206 | 0.493 | 0.588 | 0.169 | 1.928 |
| EMOd public 698 | uniform | 0.000 | 0.000 | 0.500 | 0.000 | 2.109 |

## Interpretation and limits

The completed comparisons preserve the distinction between privileged human annotations and image-only prediction. DeepGaze IIE demonstrates that the current region-uniform models leave considerable spatial information unused. It has a different pretraining regime, so its advantage is not a controlled estimate of one architectural component. Frozen coarse visual features and graph mixing did not improve the linear references. The finding is specific to these representations, settings and finite seeds.

The external test expands the evidence to the 698 available EMOd images, while excluding the unavailable IAPS subset. Missing observer records prevent assuming stable cross-image identities in EMOd. OSIE observer holdout supports transfer within the observed cohort but does not provide population-level intervals. The original test split was already examined before these exploratory follow-ups. Splits overlap, bootstrap intervals condition on fitted models and observed viewers, multiple contrasts are descriptive, and pretrained-data overlap remains incompletely audited. No claim of state-of-the-art performance, personalization or a validated neural mechanism follows.

Protocols: [saliency and transfer](saliency_transfer_protocol.md), [observer holdout](observer_holdout_protocol.md). Per-image scores, selected regularization and provenance are in `results/saliency_comparison/`, `results/emod_transfer/`, `results/observer_holdout/` and `results/emod_audit/`. Reproduction commands are in the README. No model was tuned on EMOd outcomes. No hosted or Colab execution is claimed.
