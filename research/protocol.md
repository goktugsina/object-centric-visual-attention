# OSIE pilot analysis protocol

Protocol written before fitting the pilot models on 2 November 2026. This is a local analysis specification, not an externally preregistered study.

The unit is a scene and the outcome is the object region containing a randomly chosen eligible fixation. Background is a real category. The pilot uses human masks and semantic annotations; it is not an image-only model or a state-of-the-art comparison.

1. Pin the two official OSIE MATLAB files to commit `4a467890fcb40a2fec04cdded552760942bac7a4` and check SHA-256. Match records by image name.
2. Transpose HDF5 masks to height by width. Resolve overlap by smallest original mask area, breaking ties by original object index. Remove empty regions; retain background if nonempty. This differs from the original paper's tolerance-based matching.
3. Keep finite fixations with duration at least 100 ms and coordinates within the one-based image domain. Round halves upward and subtract one. Do not clip out-of-image gaze. Primary analysis retains the first recorded fixation; a specified sensitivity analysis removes it before other filtering.
4. Sort filenames, permute with NumPy default_rng seed 20261102, and use 60/20/20 percent train/validation/test images. Every observer for an image stays in the same split. No observer-generalization claim.
5. Compare uniform regions, uniform pixels integrated over regions, a Gaussian center prior integrated over regions, geometry-only conditional softmax, and geometry plus 12 semantic attributes. Geometry is log area fraction, centered x and y, squared x and y, and a background indicator.
6. Fit softmax with equal image weight and L2 coefficient selected from 0.0001, 0.001, 0.01 using validation NLL. Center bandwidth is selected from 0.10, 0.15, 0.20, 0.30, 0.40 in normalized coordinates using validation NLL. Do not standardize features.
7. Use ten training-image bootstrap fits of the semantic model for a bootstrap ensemble. Reuse the selected L2 coefficient. Average probabilities. Fit scalar temperature on validation data within [0.25, 4]. This is not a deep ensemble or an exact Bayesian posterior.
8. Primary endpoint: semantic minus geometry improvement in image-average log score, bits/fixation. Report paired 5000-resample image bootstrap confidence intervals, held-out NLL, multiclass Brier score and top-1/top-3 event coverage. Background participates in ranking. ECE is a secondary 10-bin top-label diagnostic, not a proper score.
9. Report empirical outcome entropy separately from ensemble disagreement. Neither alone identifies intrinsic observer variability. Do not call a high-entropy prediction calibrated merely because it is diverse.
10. Report negative calibration results and all prespecified model comparisons. No GPU model, foveation, scanpath-generation or fMRI claim will be made from this pilot.

## Exploratory addition after the data audit

The raw-data audit found 24,775 eligible fixation events within more than one source mask, before inspecting held-out model scores. A second complete analysis uses nearest original-mask centroid among masks containing the pixel, with pixel-distance ties broken by source order. No tolerance expansion is added. This sensitivity check is exploratory and its event space differs from the primary partition; compare the semantic-versus-geometry contrast within each analysis rather than comparing absolute NLL across partitions.
