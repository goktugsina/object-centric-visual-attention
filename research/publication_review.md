# Preprint readiness review

Review date: 7 November 2026. Scope: the current seven-page paper, **Robustness of Object Attention Prediction With Annotated and Detected Objects**. The sole author is **Göktuğ Sina Bekçioğulları**, a fourth-year Computer Engineering undergraduate at Marmara University, with supplied ORCID **0009-0001-2026-1019**. The author reports no university support. Current student affiliation does not assert funding, supervision or institutional endorsement. The author selected a preprint-first route; the license remains undecided.

## Decision

The completed experiments support a bounded empirical preprint about object-based prediction of recorded gaze, representation assumptions and transfer limitations. They do not support brain decoding, a new neural mechanism, personalization or state-of-the-art performance. No additional architecture is required for these narrow claims. This review is a local consistency audit, not independent peer review or a prediction of venue acceptance.

The current title remains accurate. The previously suggested broader title is an alternative, not an adopted manuscript change.

## Claims and evidence

| Claim | Saved evidence | Interpretation required in the paper |
|---|---|---|
| OSIE semantic annotations improve on geometry | `results/primary/`, `results/robustness/`; gain 0.101-0.149 bits over five overlapping splits | Association under privileged annotations; splits are not independent replications |
| Image-only linear predictor has limited spatial performance | `results/saliency_comparison/`; IG 0.501 bits versus center 0.522 | No claim of superiority to the center prior |
| Additional graph mixing does not improve the tested frozen-feature model | `results/relational/`; 30 fits and saved replay verification | Specific to these features, training settings and seeds |
| A stronger pretrained spatial predictor leaves a substantial gap | DeepGaze IIE IG 2.695 bits on the same 140 OSIE scenes | Different pretraining budgets; not an isolated architectural effect |
| OSIE-trained object models transfer poorly to the available EMOd subset | `results/emod_transfer/`; linear IG 0.206 versus center 0.590 on 698 images | Excludes 321 unavailable IAPS images; includes empty detections |
| Semantic benefit survives exclusion of OSIE observer groups | `results/observer_holdout/`; annotated gain 0.149 bits | Cohort-conditional holdout, not personalized predictions or population-level uncertainty |

These boundaries are stated in the current abstract, methods, discussion and conclusion. The main text was reviewed against the saved reports. No experimental outcome was changed in this review.

## Fresh checks

Run `python scripts/verify_publication.py`. The deterministic output is [publication_verification.json](../output/submission/publication_verification.json).

- 159 checks passed: saved CSV means against summary JSON, unique model/image rows, shared image coverage, seed-score means, observer fold pairing, manuscript title and abstract, citation order, PDF checksum and persistent author metadata.
- Recomputed aggregates from 980 OSIE saliency rows, 6,282 EMOd rows and 1,680 observer-holdout rows. These use the existing recorded predictions. They do not rerun the detector or model training.
- Observer gains are computed within each image/fold, then averaged over folds and images. The audit does not recompute confidence intervals or treat folds as independent samples.
- All 24 references occur in first-citation order. This is a structural check, not a fresh verification of every bibliographic fact.
- The submission metadata build inspected the existing PDF: seven pages, eight embedded fonts, extractable body text, no encryption and no embedded JavaScript. The abstract contains 170 whitespace-separated words.
- After the confirmed author spelling and student affiliation were added, Word exported a fresh PDF and Poppler rendered all seven pages. All pages were visually inspected, including Turkish glyphs in the author block. No clipping or missing glyphs were observed. The packaged renderer was attempted and failed because LibreOffice is absent; Word supplied the export. This is not a full accessibility audit; equations remain raster illustrations.
- The ORCID checksum is valid. The public profile could not be read through the web tool; profile ownership and affiliation were not independently verified.
- The earlier 31-test pass and model replay checks remain recorded in [validation](validation.md). This review does not claim a new scientific-suite run or remote CI success.

## Reference spot checks

Primary-source checks on 7 November confirmed the titles/authors and stated scope of [ScanVLA](https://arxiv.org/abs/2604.20361), [ST-DiffEye](https://arxiv.org/abs/2606.15486), and [DeepGaze IIE](https://arxiv.org/abs/2105.12441). The first concerns language-referred search, the second joint trajectory/scanpath generation, and the third static saliency. These do not change the current study's static free-viewing scope.

The direct arXiv fetch for DeepGaze3.5-VL failed. Its [author repository](https://github.com/Susmit-A/DeepGaze3.5-VL) links that identifier and describes autoregressive scanpaths; it labels the project ECCV 2026. The manuscript cites the arXiv version. A proceedings record was not independently checked, so no final publication citation was invented. This is a targeted refresh, not a new exhaustive literature review.

## Corrections made

The publication plan described a six-page OSIE-only draft and listed the second dataset and extra neural comparisons as future work. It now describes the completed seven-page OSIE/EMOd study. The metadata generator now reads explicit author settings from `paper/submission_details.json`, preventing regeneration from erasing the supplied ORCID. A hard-coded claim that the repository is currently private was replaced with an instruction to verify visibility before promising public access.

## Remaining author decisions

1. Read and approve the current paper. Sole-author spelling and student affiliation are now recorded; journal-specific affiliation requirements should be checked if a later journal is selected.
2. Register an arXiv account, then check the category endorsement shown for that account. The author reports that they do not yet have an account. ORCID alone does not establish endorsement.
3. Choose a paper distribution license after considering any intended later venue. The author explicitly chose to leave this undecided for now.
4. Decide whether and how readers can access the code. A GitHub push does not by itself establish public visibility or a reuse license.
5. Review the platform-processed preview and complete the final submission. No submission, endorsement request, email or public release was performed here.

The local handoff is [preprint_handoff.md](../output/submission/preprint_handoff.md).
