# Object-Centric Visual Attention Prediction

Research code and results by **Göktuğ Sina Bekçioğulları**. Predicts human gaze from images using object-region models, frozen Faster R-CNN detections, ResNet50 features and graph attention. This is eye-tracking research, not brain-activity decoding.

## Results

Experiments use 700 OSIE images (420 train / 140 validation / 140 test) and zero-shot transfer to 698 public EMOd images.

| Model | OSIE test IG | EMOd IG |
|---|---:|---:|
| Center prior | 0.522 | 0.590 |
| Detected-object linear model | 0.501 | 0.206 |
| Human-annotation semantic model | 1.515 | — |
| DeepGaze IIE | 2.695 | — |

Information gain is measured in bits per fixation over uniform pixels, averaged equally over images. Human annotations are privileged inputs; DeepGaze uses different pretraining. The tested graph-attention heads do not improve the linear references. Results support a comparison of these pipelines, not state-of-the-art or causal claims.

- [`src/attention/`](src/attention): models, data processing and metrics.
- [`scripts/`](scripts): downloads, training, evaluation and plots.
- [`results/`](results): per-image CSVs, summary JSON, coefficients, split manifests, provenance and figures.
- [`tests/`](tests): 31 scientific contract tests.
- [`notebooks/`](notebooks): optional Colab workflow.

## Run

Use Python 3.12+ in a virtual environment:

```bash
pip install -r requirements.txt
pip install --no-deps -e .
python -m unittest discover -s tests -v
python scripts/verify_results.py
```

The base test run skips neural tests unless the vision requirements are installed. `verify_results.py` checks the committed aggregate scores without downloading data.

```bash
# Original region-choice and robustness experiments
python scripts/download_osie.py
python scripts/run_pilot.py
python scripts/run_pilot.py --drop-first
python scripts/run_pilot.py --overlap nearest
python scripts/run_robustness.py

# Image-only and relational models
pip install -r requirements-vision.txt
python scripts/download_osie_images.py
python scripts/extract_detections.py
python scripts/run_image_only.py
python scripts/score_oracle_pixels.py
python scripts/extract_visual_features.py
python scripts/run_relational.py --output results/relational_rerun

# Figures from committed results
python scripts/plot_results.py
python scripts/plot_extension.py
python scripts/plot_relational.py
```

Neural training refuses to overwrite an existing run; use a new output directory. Checkpoints and feature caches are generated locally. For saliency and transfer, use `extract_deepgaze.py`, `compare_saliency.py`, `download_emod.py`, `audit_emod.py`, `run_emod_transfer.py` and `run_observer_holdout.py`; inspect `--help` where available. Transfer needs trained checkpoints and caches. Historical source hashes describe the original experiments; checkpoint replay requires the matching source/data artifacts. Numerical aggregate checks are available independently through `verify_results.py`.

## Data and limitations

Data comes from [OSIE](https://github.com/NUS-VIP/predicting-human-gaze-beyond-pixels) and [EMOd](https://github.com/Fanshaojing/emotionalattention). Downloaders pin sources and verify hashes. Fixations require at least 100 ms and valid coordinates. EMOd's 321 unavailable IAPS images are excluded. Original images, individual fixation sequences, model weights and caches are not redistributed; upstream terms apply.

Image-bootstrap intervals condition on the sampled observers; overlapping splits are not independent replications. Pretraining overlap is not exhaustively excluded. The study is not yet peer-reviewed. Code and analysis were assisted by OpenAI Codex. No general reuse license has been selected.
