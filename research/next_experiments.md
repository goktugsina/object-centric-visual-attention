# Follow-up experiments with decision gates

Update, 5 November 2026: DeepGaze IIE common-pixel evaluation and three-fold OSIE observer exclusion are complete; EMOd coordinate/schema/overlap auditing is complete for 698 public photographs, with zero-shot transfer completed without target fitting. The current evidence is tracked in [shared-plan status](shared_plan_status.md). These completed components supersede their earlier future-work descriptions below. Learned within-region density, a matched pixel decoder, personalization and sequential prediction remain separate experiments.

Update, 3 November 2026: the user resumed the object-centric branch in the shared plan. Frozen ResNet50 features, local/uniform/GAT heads, three-seed training, ten scene audits and observer-disagreement strata have now been run. See [shared-plan status](shared_plan_status.md) and [measured neural results](relational_results.md). GAT did not improve over the local head; the neural variants underperformed the previous linear references. The larger sequential roadmap below remains a separate future direction.

The immediate goal is a small first publication. Follow [publication_plan.md](publication_plan.md) for the finite version-1 finishing steps. The stages and sixteen-week schedule below describe an optional larger project; they are not prerequisites for submitting the current narrow case study as a preprint.

Status after the 2 November extension: five split sensitivities, equal-observer evaluation, face/text ablations, a semantic permutation control, synthetic semantic omission, a frozen detector, an image-only predictor, shared-pixel scoring and fixed box perturbations are complete. Their results are in `extension_results.md`. The stages below remain a longer-term roadmap; completed components should not be described as future-only work.

## Stage 1 Verify the reference study

Reproduce the CPU pilot, review region assignments on sampled scenes, repeat with several independent image splits, and report observer-weighted as well as fixation-weighted outcomes. Compare smallest-mask, nearest-centroid and boundary-tolerance assignments; report background frequency and masked-out object counts. Use a two-way image/observer bootstrap for inference beyond the sampled observer cohort. Do not treat the current image bootstrap as covering observer-population uncertainty.

## Stage 2 Replace human annotations

Freeze a documented detector/segmenter and visual encoder. Compare geometry, visual features, predicted semantics and human semantics under the same training split. Human semantics are an oracle-feature reference, not a deployable input. Inspect unmatched objects, nested objects and empty detections. Assign every fixation to exactly one candidate or to background. Report mask recall, segmentation quality, object-event log score and calibration together. If gains disappear with predicted masks, the current obstacle is perception reliability, not a reason to inflate the oracle result.

## Stage 3 Predict sequences and test mechanisms

Use a shared pixel encoder and object tokens in a history-conditioned decoder. Estimate p(object at t | image, task, previous fixations) followed by a within-region fixation density and a termination probability. Evaluate teacher-forced conditional log likelihood separately from free-running scanpaths. Compare history-free, history-conditioned, fixed-inhibition, learned-recency and foveated variants with matched encoders and parameter budgets. Return fixations remain possible; hard exclusion of every visited object is not a universal model of human gaze.

## Stage 4 Evaluate search and uncertainty

Use COCO-Search18 target-present and target-absent train/validation splits separately. Audit exact COCO image overlap with SALICON and detector pretraining. Select parameters without hidden test gaze; submit only a frozen final model to the official benchmark when ready. Use HAT and an appropriate current scanpath model as comparators. Evaluate free-viewing adaptation on OSIE as a different task, not as direct evidence of identical search behavior. Counterfactual target changes are model interventions unless matched human trials are actually available.

Compare a single calibrated network, MC dropout and independently trained deep ensembles. Fit all calibration parameters on validation data. Measure NLL and Brier scores over a fixed region partition, conditional log likelihood over pixels, top-k event coverage and reliability. Add scanpath similarity metrics as secondary behavior summaries, with duration metrics only when timing is modeled. Stratify by object recall and observer disagreement using boundaries selected without test-label tuning.

## Resources and stopping rules

The pilot runs on CPU. Neural training needs a separately measured compute budget; no cloud resources are provisioned by this repository. Profile a small training batch before selecting GPU time. Stop architectural expansion if the new component fails to improve the primary proper score across independent splits, or if an equally expressive control explains the gain. A negative result is publishable evidence only when the comparison itself is sound; no venue acceptance is promised.

## Suggested sixteen-week schedule

| Weeks | Deliverable | Exit criterion |
|---|---|---|
| 1-2 | Reproduction and extended split/mask audit | Stable implementation and documented sensitivity |
| 3-4 | Detector and feature cache | No split overlap; measured mask coverage |
| 5-6 | Matched static image-only baselines | Frozen validation-selected comparisons |
| 7-9 | History and foveation ablations | Conditional and free-running evaluations separated |
| 10-11 | Calibration and neural ensembles | Proper scores and reliability agree or discrepancies explained |
| 12-14 | Search and observer transfer | Task, image and observer leakage checks complete |
| 15-16 | Revised manuscript and independent reproduction | Every claim linked to an actual result |
