# Object-Centric Visual Attention Prediction

An independent research project by **Göktuğ Sina Bekçioğulları**, Computer Engineering undergraduate at Marmara University. This project predicts recorded human gaze from image content; it does not decode brain activity.

**Python · PyTorch · NumPy · SciPy · computer vision · eye tracking**

## Start here

- [Research paper (PDF)](output/pdf/object_attention_paper.pdf) — complete manuscript, not yet published or peer-reviewed.
- [Technical walkthrough](research/technical_walkthrough.md) — architecture, equations, experimental reasoning and results.
- [Source code](src/attention) · [Experiment scripts](scripts) · [Tests](tests) · [Result tables](results)
- [Data provenance and limitations](DATA_CARD.md)

## What I built

- A reproducible pipeline for **700 OSIE scenes** and zero-shot evaluation on **698 public EMOd images**.
- Object-region probability models, frozen Faster R-CNN detections, ResNet50 visual features and local/uniform/graph-attention controls.
- **30 neural fits**, a frozen DeepGaze IIE comparison, observer holdout and sensitivity analyses.
- **31 scientific tests** plus a separate publication-consistency audit.

## Main findings

Semantic annotations improve region prediction beyond geometry. The tested graph-attention models do not improve the linear references. On OSIE, DeepGaze IIE achieves **2.695 bits** of information gain versus **0.501** for the detected linear model. On EMOd, that fixed linear model achieves **0.206 bits**, below the center prior (**0.590**). These are comparisons of the documented pipelines, not matched-pretraining architectural claims.

## Reuse and data

This public copy includes code, aggregate results and the manuscript. Raw photographs, individual fixation sequences, model weights, caches and credentials are excluded. Download instructions retrieve data from the original sources under their own terms. No general reuse license has been selected for the original project materials; public visibility alone is not an open-source license. The retained IEEE template is attributed separately in [format provenance](research/ieee_format.md).

The sections below contain the full experiment and reproduction documentation.

## Research details

A reproducible study of object-attention probabilities with human annotations and image-only detections. It tests semantic feature contributions, repeated image splits, observer reweighting, missing semantic information and changes to predicted object regions.

**Measured result:** On 140 held-out OSIE scenes, semantic features improve image-averaged log score by **0.149 bits per fixation** over geometry (paired image-bootstrap 95% interval **0.121 to 0.177**). The improvement remains positive after excluding the first fixation (**0.145 bits**) and using nearest-centroid overlap assignment (**0.125 bits**). Temperature scaling and a ten-member bootstrap ensemble show no clear additional log-score benefit.

**New image-only result:** A frozen Faster R-CNN MobileNetV3 detector plus a learned region-choice model runs without human annotations at inference. On the original split it achieves **0.501 bits** of pixel information gain over uniform pixels, compared with **0.522 bits** for a center prior and **1.515 bits** for the human-annotation semantic reference. Detected classes add **0.011–0.036 bits** over geometry and confidence across five overlapping splits. The image-only model does not consistently outperform the center prior.

This is an exploratory OSIE and public EMOd study with executable baselines. The annotation-conditioned semantic gain remains **0.101–0.149 bits** across the five splits. Original and extended analyses share data; repeated splits are not independent replications. Author review and external submission remain outstanding. Sequential prediction and personalization remain future directions; the study does not claim state-of-the-art performance. See the [bounded publication plan](research/publication_plan.md).

## Paper and research

- [Complete technical walkthrough with equations, experimental logic and results](research/technical_walkthrough.md)

**5 November continuation:** frozen DeepGaze IIE reaches **2.695 bits** of IG on the same 140 OSIE test images, versus **0.501** for the detected linear model. Zero-shot transfer on **698 public EMOd images** gives that linear model **0.206 bits**, below the fixed center prior (**0.590**). Three observer-exclusion folds preserve the annotated semantic gain (**0.149 bits**). All **31 tests** pass. These findings and their limits are included in the revised **seven-page** paper and [continuation report](research/saliency_transfer_results.md).

**3 November continuation of the shared research plan:** frozen ResNet50 region features, self-only/uniform/GAT heads, three initialization seeds, ten scene audits and a Colab workflow are now implemented and evaluated. GAT does not improve the annotated-region model (paired gain -0.003 bits; 95% interval -0.011 to 0.005) and underperforms the self-only detector-region model (-0.039 bits; -0.062 to -0.020). The new neural heads also underperform the previous full-feature linear references. These exploratory follow-up results are incorporated into the revised paper alongside the 5 November saliency, transfer and observer experiments. See the [measured report](research/relational_results.md) and [shared-plan status, including unfinished stages](research/shared_plan_status.md).

- [Paper PDF](output/pdf/object_attention_paper.pdf)
- [Editable Word paper](output/docx/object_attention_paper.docx)
- [Manuscript source](paper/manuscript.md)
- [Literature review and novelty audit](research/literature_review.md)
- [Recent research, our findings and publication gaps (2 November 2026)](research/research_brief_2026-11-02.md)
- [A bounded first-publication plan](research/publication_plan.md)
- [Preprint readiness review and author decisions](research/publication_review.md)
- [Preprint submission handoff](output/submission/preprint_handoff.md)
- [Publication routes and journal-fit assessment](research/publication_routes.md)
- [Analysis protocol](research/protocol.md)
- [Extension protocol](research/extension_protocol.md)
- [Extended results and limitations](research/extension_results.md)
- [Follow-up experiments](research/next_experiments.md)
- [Data card](DATA_CARD.md)

The revised manuscript follows the IEEE conference US Letter template, with two columns, four numbered equations, 24 references in citation order and measured results from both model families. See [template provenance and formatting](research/ieee_format.md). Recent 2026 preprints are identified as preprints. Synthetic omission/deletion interventions are explicitly distinguished from natural detector errors.

## Reproduce the experiments

Python 3.12 or newer is required. The recorded experiment ran on Python 3.14.3 with the versions pinned below. Create a virtual environment using your usual platform activation command, then run from the repository root:

```bash
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python scripts/download_osie.py
python -m unittest discover -s tests -v
python scripts/run_pilot.py
python scripts/run_pilot.py --drop-first
python scripts/run_pilot.py --overlap nearest
python scripts/run_robustness.py
python scripts/plot_results.py
```

The downloader retrieves about 46 MB of public annotation files from the pinned official repository and verifies their SHA-256 checksums. It downloads no photographs. Each original pilot analysis took roughly 2.5-3.3 minutes on the execution machine; the expanded robustness run took about 13 minutes. Timings depend on hardware and are recorded in result metadata. No GPU, account token or paid service is needed to reproduce the analysis.

Training, validation and test use 420, 140 and 140 disjoint image identities, with all observer records for an image kept together. Test observers also occur in training. The primary run retains 87,662 fixations of at least 100 ms, including 17,516 in the test set. All probabilities include an explicit background category. Read the protocol before comparing scores with another implementation.

## Outputs and provenance

The original `primary`, `drop_first` and `primary_nearest` directories under `results/` contain:

- `audit.json`: counts, preprocessing decisions and raw-source hashes.
- `splits.json`: exact train/validation/test image names.
- `summary.json`: proper scores, validation choices, paired bootstrap intervals, package versions and elapsed time.
- `coefficients.json`: fitted conditional-choice and bootstrap parameters.
- `per_image_metrics.csv`: image-level scores for every evaluated method.
- `uncertainty.json`: empirical entropy, descriptive observer disagreement and bootstrap model disagreement.

Models optimize image-average cross entropy, not pooled-event cross entropy. Multiclass Brier is the expected squared error against a one-hot event, not squared distance to empirical fixation fractions. Top-k measures fixation-event coverage; background participates in ranking. The bootstrap intervals resample images and do not represent uncertainty about a new observer population.

## Reproduce the image-only extension

Install a compatible official PyTorch/Torchvision build; the local CPU versions are recorded in `requirements-vision.txt` and the detector metadata. No GPU or paid service is required. The following downloads about 49 MB of photographs from the pinned upstream tree and about 74 MB of official detector weights. Both remain outside version control.

```bash
python -m pip install -r requirements-vision.txt
python scripts/download_osie_images.py
python scripts/extract_detections.py
python scripts/run_image_only.py
python scripts/score_oracle_pixels.py
python scripts/build_extension.py
python scripts/build_manuscript.py
```

`results/robustness/summary.json` records all five splits, feature ablations, validation selections, fitted coefficients and omission tests. `results/image_only/summary.json` records detector provenance, pixel scores and region perturbations; its companion files include image checksums, a runnable attention-model checkpoint, per-image scores, region coverage and the paired representation contrast. Different region partitions are compared on the same fixation pixels, not by incomparable categorical NLL values. Pretraining overlap with OSIE photographs is not exhaustively ruled out.

Run the trained model on a photograph:

```bash
python scripts/predict_attention.py path/to/image.jpg
```

It writes a normalized probability array, region probabilities and a preview under `output/predictions/`, which is ignored by Git. This inference command loads no human object annotations or gaze data. The model assigns uniform density within each predicted region; the resulting maps can be coarse.

## Reproduce the ResNet50 and relation experiment

Install the core and vision requirements and the local package as above. Reuse the downloaded OSIE photographs and frozen detector cache. Then run:

```bash
python scripts/audit_visual_scenes.py
python scripts/extract_visual_features.py
python scripts/run_relational.py
python scripts/report_relational.py
python scripts/verify_relational.py
python scripts/predict_relational.py data/raw/stimuli/1001.jpg
```

The feature extractor downloads the official ResNet50 ImageNet V2 weights (about 98 MB), freezes the encoder and caches features for both partitions. Training uses five models per partition and three seeds; checkpoints are selected using validation NLL. Results include all seeds, proper scores, object ranks, observer disagreement groups, paired intervals and hashes. The scripts preserve the original 420/140/140 split. `run_relational.py` refuses an existing run directory; pass a new `--output results/another_run` to repeat it. Learned heads are stored locally under `output/relational_checkpoints/<run-directory-name>/`. Default image inference uses detected GAT seed 11, fixed in advance, and is a runnable example rather than a recommendation of the best-performing model.

For Colab, open [the notebook](notebooks/object_attention_relations.ipynb). `python scripts/prepare_colab_bundle.py` produces `output/colab/object_attention_source.zip` for its upload cell, including the two previous baseline tables needed for comparison. Raw photographs and weights are downloaded from their documented sources. The notebook's code cells were syntax-checked locally; the completed experiment ran on local CPU, not Colab GPU. The [protocol](research/relational_protocol.md) distinguishes this static region experiment from strong pixel saliency or cross-dataset evaluation.

## Build the documents

Generate the current continuation fields before rebuilding the paper. After the original and relational workflows above:

```bash
python -m pip install --no-deps -r requirements-saliency.txt
python scripts/download_emod.py
python scripts/audit_emod.py
python scripts/extract_deepgaze.py
python scripts/compare_saliency.py
python scripts/run_emod_transfer.py
python scripts/run_observer_holdout.py
python scripts/verify_continuation.py
python scripts/report_continuation.py
python scripts/build_manuscript.py
```

DeepGaze downloads roughly 0.9 GB of official weights and uses its static IIE model without OSIE fitting. The import adapter avoids initializing the unrelated MSDB model and its CLIP dependency. EMOd downloads roughly 100 MB of public images and annotations; the unavailable 321 IAPS images are excluded. Raw data, weights and prediction caches remain outside Git. The source ZIP contains baseline coefficients but no neural checkpoints: run relational training before transfer on a fresh machine. These experiments ran on local CPU; Colab execution has not been verified. The earlier six-page manuscript is archived locally under `output/archive/2026-11-02/`.

The editable document builder needs `python-docx`, `lxml` and `Pillow`. Equation and figure generation use Matplotlib. After generating the manuscript fields above, run:

```bash
python scripts/build_documents.py
```

On Windows with Microsoft Word installed, export the PDF:

```powershell
.\scripts\export_word_pdf.ps1
```

`build_documents.py` invokes the IEEE builder. The retained original template is verified by checksum; `prepare_ieee_template.py` normalizes its Strict OOXML syntax into a temporary working copy. The release documents were exported with Microsoft Word and rasterized with Poppler for page-by-page inspection. See [validation record](research/validation.md). Manuscript numbers are generated from result JSON by the manuscript builders.

## Attribution and status

Local submission metadata is in `output/submission/metadata.json`, with technical PDF checks in `output/submission/pdf_checks.json`. To refresh them after a manuscript/PDF revision, install `pypdf` in the document environment and run `python scripts/prepare_submission.py`. This prepares a draft; it does not upload files or choose a publication license.

Project owner: **Goktug Sina Bekciogullari** (`goktugsina`). Code, analysis and drafting were assisted by OpenAI Codex. Author review is required before scientific submission. No institutional affiliation is asserted.

OSIE source: Xu et al. (2014), *Predicting Human Gaze Beyond Pixels*, [doi:10.1167/14.1.28](https://doi.org/10.1167/14.1.28). Raw data and original photographs are not committed. The upstream repository declares the MIT license; original image rights remain with their owners. No redistribution license is granted for this private project's new manuscript or code at this stage.

The GitHub Actions configuration runs the scientific contract tests on Python 3.12. Local test results are recorded separately; a remote CI pass should only be claimed after its workflow completes.
