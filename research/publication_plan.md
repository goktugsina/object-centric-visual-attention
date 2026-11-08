# A bounded first research publication

Updated 7 November 2026. The author selected a preprint-first route. The target is a short empirical case study, not a new state-of-the-art architecture. The current review and outstanding author decisions are recorded in [publication review](publication_review.md).

## Version 1 scope

Keep the seven-page paper, **Robustness of Object Attention Prediction With Annotated and Detected Objects**, and position it explicitly as an exploratory OSIE and public-EMOd study. The existing title describes the tested representations; no brain-activity decoding is claimed.

Its central finding is that semantic features improve region-choice probabilities within both tested representations, but the implemented detector-based model does not clearly outperform a center prior. Human annotations provide a much stronger reference on common fixation pixels. This diagnoses limitations of this particular pipeline; it does not isolate detector error as the cause of the gap.

The completed continuation adds 30 frozen-feature neural fits, a stronger DeepGaze IIE comparator, zero-shot evaluation on 698 public EMOd images and three OSIE observer-exclusion folds. Graph mixing does not improve the linear references; the transferred detected linear model underperforms the center prior on EMOd. Human semantic attributes and detector categories are different feature sets, not interchangeable semantic measurements.

The general observation that semantics influence gaze is established prior work. Neither the model family nor the probability metric is new. The contribution is the controlled empirical analysis, including its negative results. Whether it is of sufficient archival interest remains a judgment for readers and, for arXiv, moderators.

## Finishing steps

1. **Keep claims bounded.** The paper already states the negative center comparison, mismatched semantic features, overlapping splits, sampled-cohort intervals and one-draw deletion analysis. Avoid generalizing these results to all detectors, new observers, natural corruption or neural mechanisms.
2. **Complete author review.** Read the paper alongside the scripts and source papers. Be able to explain the 420/140/140 split, region probabilities, uniform-pixel reference and why the representation gap is not a causal estimate of detector quality. Faculty feedback would be useful; it is a recommendation, not a requirement to add a co-author.
3. **Finalize submission details.** Sole author Göktuğ Sina Bekçioğulları and current Marmara University Computer Engineering student affiliation are recorded, together with ORCID 0009-0001-2026-1019. Account/endorsement status, distribution license and reader access to code remain submission decisions. Repository visibility must be checked before promising public code access. A local metadata draft and technical PDF checks are in `output/submission/`; no preprint has been submitted.

The second-dataset test, frozen neural controls, strong saliency comparator and within-cohort observer holdout are complete. Learned spatial density, matched-budget encoders, sequential models, additional deletion draws and support/query personalization remain extensions. Expanded competitive, causal or population-level claims require corresponding new evidence.

## Publication routes

**First choice: arXiv preprint.** `cs.CV` is a candidate category for this image-based study. arXiv moderates submissions but does not peer-review them. It expects substantive research and scholarly interest; it may decline work lacking these, including course projects without substantive research. A small case study can be proposed on its merits, but acceptance is not guaranteed. [Moderation policy](https://info.arxiv.org/help/moderation/index.html)

First submissions and new categories require endorsement. An eligible researcher familiar with the subject, such as a faculty member, is a practical person to approach. Do not assume an institutional email alone provides automatic endorsement. [Endorsement guidance](https://info.arxiv.org/help/endorsement.html)

The manuscript was authored in Word, so an exported PDF is an applicable submission format. There is no need to convert it to LaTeX solely for arXiv. Local file checks do not guarantee successful platform processing. [PDF submission guidance](https://info.arxiv.org/help/submit_pdf.html)

**Alternative: Zenodo report or preprint.** Zenodo can archive the manuscript and assigns a DOI when a record is published. This makes the work citable, not peer-reviewed. Label it accurately as a report or preprint rather than an accepted journal article. [Quick start](https://help.zenodo.org/docs/get-started/quickstart/), [content policy](https://support.zenodo.org/help/en-gb/2-content/141-what-content-is-not-suitable-for-zenodo)

A student research venue or suitable workshop is another possible later route for external review. A specific venue needs a current call that welcomes this type of small empirical study. IEEE formatting itself is not IEEE publication.

## Draft and stopping point

`output/submission/metadata.json` contains the current manuscript's title, abstract, confirmed author name and student affiliation, supplied ORCID, suggested comments, candidate category and exact PDF checksum. The author reports no arXiv account and has explicitly left the license undecided. No journal acceptance or publication DOI is asserted. See the [publication route assessment](publication_routes.md) for the preprint recommendation and a possible later journal.

The stopping point is a reviewed, internally consistent case study with reproducible evidence and a clear limitation on its conclusions. Retain the accurate assistance acknowledgment. The author's understanding and verification matter more than the size of the model.
