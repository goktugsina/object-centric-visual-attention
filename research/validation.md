# Validation record

## Author correction and preprint checks, 7 November 2026

The sole author confirmed the name Göktuğ Sina Bekçioğulları and current Computer Engineering student affiliation at Marmara University. These appear in the manuscript, regenerated DOCX/PDF and submission metadata; no university support is claimed. The PDF remains seven pages with eight embedded fonts. Word exported the document after the packaged renderer failed because LibreOffice was unavailable; Poppler rendered all pages and all seven were visually inspected, including the Turkish author-name glyphs. The manuscript body and experiment results were not changed. The publication audit passed all 159 checks after regeneration. The bundled document environment lacks matplotlib, so the author block was updated in both the manuscript template and resolved Markdown without regenerating unchanged equations. No new scientific test run, remote CI success or submission is claimed.

## Saliency, transfer and observer continuation, 5 November 2026

All 31 scientific tests pass. Five new common-pixel tests cover normalization, repeated events, tie-aware positive-threshold AUC-Judd, equal-observer scoring and invalid inputs. An asymmetric non-square EMOd fixture checks the source axis reversal, duration filtering, invalid coordinates and unavailable images. All Python sources and notebook code cells parse.

Frozen DeepGaze IIE was evaluated on all 140 fixed OSIE test images. The two existing linear pixel references replay with maximum errors below 1e-15 bits. Three deterministic test-image prediction panels were visually inspected. DeepGaze IG is 2.695259 bits versus 0.501077 for the detected linear predictor; the paired improvement is 2.194183 bits, interval [2.083075, 2.304917]. This is a comparison of delivered predictors with different pretraining, not equal-data architectural evidence.

The EMOd audit contains 698 public images, 11,148 subject-image records and 102,760 eligible events after duration/coordinate filtering. Its reversed axis names were checked against author binary fixation maps on all 698 images. Exact and perceptual-hash screening against all 700 OSIE images found no candidate at the pre-specified threshold. This is not an exhaustive cropped-image or external-pretraining audit. The unavailable 321 IAPS images are excluded.

Zero-shot transfer completed 6,282 image-model rows. Center IG is 0.589749 bits; detected linear IG is 0.206201, with paired difference -0.383548 [-0.421259, -0.346279]. Mean seed-specific local and GAT IG are 0.088224 and 0.063511 bits. There are 243 images without retained detections; all remain in the analysis. No target outcomes fit or select models. An interrupted extraction was resumed from 515 cached predictions, followed by complete scoring.

The three observer folds exclude five of fifteen OSIE observers from both training and validation. Twelve selected linear fits produce 1,680 per-image rows, and every observer is tested once. Gains over geometry are 0.149129 [0.121072, 0.177229] bits for annotations and 0.017409 [0.001864, 0.033320] for classes beyond detector geometry/confidence. Image intervals condition on this cohort and do not support a population or personalization claim.

`results/continuation_verification.json` records 1,412 source/reference/cache hash checks, all observer exclusions, and replay of seven model probabilities/scores on three deterministic EMOd images from cached visual features. Maximum IG replay difference is 4.46e-7 bits (CPU thread arithmetic); pixel-map mass error is 2.22e-16. This replay does not re-extract every detector/encoder output.

The current seven-page DOCX/PDF incorporates all these results, with three tables, one figure, four numbered equations and 24 references. The prior six-page artifacts are archived locally. The packaged DOCX renderer was attempted and diagnosed: bundled runtime metadata reports no LibreOffice, and rendering fails because `soffice.exe` is absent. Microsoft Word exported the actual DOCX, bundled Poppler generated all seven page PNGs, and every page was visually inspected. No clipping, missing glyphs or table overflow was observed. The retained IEEE template checksum and its styles, numbering, font table and theme were preserved. Eight PDF fonts are embedded, body text is extractable, and no encryption or embedded JavaScript was found. Local submission metadata has the current PDF hash; no external submission or remote CI execution occurred.

The refreshed Colab source archive includes the saliency requirements and baseline coefficients but no original photographs, fixation sequences, pretrained weights, neural checkpoints or credentials. Notebook cells were syntax-checked; Colab execution remains unverified. Historical sections below describe earlier milestones and do not override this record.

Original study completed 2 November 2026. Neural continuation added 3 November 2026; see the dated section below.

## Scientific checks

Eighteen unit tests pass. They check disjoint object/background assignment, stable overlap ties, nearest-centroid assignment, one-based coordinate conversion with out-of-bounds rejection, deterministic image-disjoint splitting, the expected one-hot Brier score, the analytic training gradient against finite differences, temperature scaling invariants and a degenerate bootstrap interval. Extension tests cover observer reweighting, semantics-only perturbations without input mutation, detector-box overlap and empty detections, probability-map normalization, partition-invariant uniform pixel scoring, and agreement between packed and reference training objectives and gradients with unequal image weights.

Three full real-data analyses completed:

| Analysis | Semantic log-score gain over geometry | Paired image-bootstrap 95% interval |
|---|---|---|
| Primary smallest-mask partition | 0.148802 bits/fixation | 0.120644 to 0.176767 |
| First recorded fixation removed | 0.144501 bits/fixation | 0.115407 to 0.174256 |
| Nearest-centroid partition | 0.124997 bits/fixation | 0.100033 to 0.149525 |

The primary audit reports 700 images, 5,551 source objects, 5,549 nonempty object regions, 10,500 observer-image records and 87,662 retained fixation events. All analyses use the same image identities in each split. Sensitivity outcomes have their own event counts and model fits.

Both annotation files match the documented upstream SHA-256 checksums. The manuscript's numeric fields are generated from result JSON. All optimization calls terminated successfully. The extension uses a frozen pretrained neural object detector and a learned linear region-choice model; no neural attention network was trained. All gaze outcomes are real OSIE records. Synthetic omission and box interventions modify model inputs, not human observations.

## Extension execution

Six annotation-feature variants and three detected-region variants were evaluated across five image splits. Each split has 420 training, 140 validation and 140 test images, with validation-only regularization selection. All 700 photographs were verified against pinned Git blob hashes; SHA-256 hashes, detector weight provenance and software versions are recorded. The image-only arm retains the same 87,662 events, including seven scenes with no detections. Human and detected partitions are scored using the same fixation pixels. The standalone image inference command ran successfully and its pixel probabilities summed to one within floating-point tolerance.

Semantic gains across splits are 0.101 to 0.149 bits per fixation; detector class gains over geometry and confidence are 0.011 to 0.036 bits. Original-split image-only IG is 0.501 bits above uniform, compared with 0.522 for the center prior and 1.515 for the privileged human-annotation model. Details and uncertainty intervals are in `extension_results.md` and the committed result files.

## Document checks

The packaged `render_docx.py` was attempted and diagnosed: this Windows runtime has no bundled LibreOffice and `soffice.exe` was absent from PATH. Microsoft Word exported the actual DOCX to PDF. Bundled Poppler rasterized that PDF, and all six final IEEE-format pages were visually inspected. The original three-page IEEE template was also rendered and inspected. Title styling, equation sizes and number alignment, table widths, figure captions, citation order and reference-heading placement were checked before final inspection. No clipping, overlap or missing mathematical glyphs was observed in the delivered render. Poppler reported optional Unicode-map path warnings on this Windows installation; the PNG output completed and the manuscript glyphs were inspected directly.

The final artifacts are a six-page US Letter PDF and its editable DOCX source, using the retained IEEE conference template. The body has two columns, Roman-numbered major headings, lettered subsections, numbered equations, 20 references in first-citation order and no page numbers or conference-specific copyright notice. The PDF contains selectable body text. Figures and equations are raster illustrations embedded at deliberate sizes; vector chart PDFs and formula source strings are included separately. The Markdown manuscript and generation scripts are included for later editing. Template provenance, checksum and allowed adaptations are recorded in `research/ieee_format.md`.

## Scope of verification

The tests verify implementation contracts, not the truth of the proposed brain-inspired architecture. Image-bootstrap intervals condition on the sampled viewer cohort. The local scientific protocol was written before model fitting, while the nearest-centroid check was added after observing mask-overlap counts and before inspecting its outcomes. Neither is an externally preregistered study. The extension protocol was written after the original pilot but before extended outcomes. Five overlapping image splits are sensitivity checks, not independent replications. Equal-observer evaluation is not evaluation on new people. A second dataset, held-out observers, a stronger saliency comparator, repeated box-deletion draws and an audit of pretrained-detector data overlap remain outstanding. The image-only baseline does not consistently beat the center prior.

Remote GitHub Actions status should be read from the repository itself; local results do not imply that remote CI has run.

## Neural continuation, 3 November 2026

Twenty-five local scientific tests pass, including seven new neural/visual contracts: permutation equivariance, padding isolation, no cross-image mixing, single-background behavior and finite gradients, equal-image loss, training-only image-balanced normalization, observer disagreement and tied ranks, independent local logits, and survival of a one-pixel region under feature pooling. The original eighteen tests remain unchanged. Neural tests skip explicitly without Torch/Torchvision; a separate CPU neural job was added to GitHub Actions, but remote execution was not performed in this task.

Frozen ResNet50 features were extracted for all 700 photographs in 162 seconds on local CPU. The five pre-specified heads were fitted on both partitions with three seeds (30 fits) in 178 seconds. Every model used the original split and validation-selected epochs. Results and complete training histories are in `results/relational/`. The negative GAT and neural-versus-linear findings are reported in `relational_results.md`; the version-1 paper is unchanged.

Ten training scenes were selected with seed 20261103 before viewing outputs: 1109, 1135, 1213, 1276, 1282, 1475, 1607, 1616, 1666 and 1681. Their full-image fixation overlays, source mask contours, disjoint region views, two individual-observer overlays and target distributions were inspected in the contact sheet. All ten independently recomputed region counts exactly match the established loader. No coordinate/mask alignment failure was observed. This is a sample inspection, not a claim of manually validating every annotation. Photographs and overlays remain outside Git.

The paired-contrast figure was visually checked for readable labels, intervals and the zero reference. The image-only GAT checkpoint ran on `1001.jpg`, producing four regions and a finite pixel map summing to exactly 1.0; its input/partition/probability preview was inspected. This image is an OSIE example, not independent transfer evidence. All notebook code cells were parsed, but the notebook has not been run on a Colab GPU.

`scripts/verify_relational.py` successfully replayed all 30 saved checkpoints and all 4,200 per-image predictions: maximum absolute difference in the recorded NLL/IG scores was 0.0. Training-only normalization and all recorded source/checkpoint hashes also matched. Its machine-readable completion record is `results/relational/verification.json`.

The Colab source ZIP was built and inspected: all 55 expected source/reference files are readable, all code cells parse, and the archive contains no raw data, trained weights or credential directories. This checks packaging and syntax, not actual remote runtime execution.
