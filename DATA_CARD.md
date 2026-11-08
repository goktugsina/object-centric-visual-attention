# OSIE and public EMOd data card

## EMOd continuation, 5 November 2026

The [official EMOd project](https://www-users.cse.umn.edu/~qzhao/emotionalattention.html) provides 698 author-collected images in its public archive. The other 321 IAPS images are unavailable here and excluded. Annotations and binary fixation maps come from the [author repository](https://github.com/Fanshaojing/emotionalattention), pinned to `d0d1758ac4eef9e7db6ceef78d1bfdf3dfa14d02`. Downloads are verified by SHA-256; hashes and schemas are in `results/emod_audit/summary.json` and the local download manifest.

The source fields are reversed relative to their names: `fix_y` is horizontal and `fix_x` vertical. The chosen conversion was checked against official binary maps for all 698 photographs. Duration >=100 ms and finite in-image coordinates are required. Subject arrays sometimes contain fewer than sixteen records; stable cross-image observer identity is not assumed for EMOd. The loaded object annotations differ from OSIE's twelve semantic attributes and are not used as automatic-model inputs.

Before transfer scoring, all public EMOd images were screened against all 700 OSIE photographs using exact file/RGB/thumbnail hashes and 64-bit perceptual hash distance <=4. No candidates were found. Cropped or edited overlap and external pretrained-model data overlap are not exhaustively excluded. Public access does not grant a new redistribution license: original images, annotations, fixation sequences and model weights remain outside version control. OSIE-trained model parameters are frozen; target gaze does not select or calibrate them.

## OSIE source and processing

Source: Juan Xu, Ming Jiang, Shuo Wang, Mohan Kankanhalli and Qi Zhao, *Predicting Human Gaze Beyond Pixels*, Journal of Vision 14(1):28 (2014), https://doi.org/10.1167/14.1.28.

Official repository: https://github.com/NUS-VIP/predicting-human-gaze-beyond-pixels

Pinned revision: `4a467890fcb40a2fec04cdded552760942bac7a4`.

| File | SHA-256 |
|---|---|
| attrs.mat | 51e9747fc6364102080cd5bac8921a279fc3f9e2a022119ae691c4bee4e8b74e |
| fixations.mat | 2aab9133a407e4eff3f57ab5799872b162dbc77ca462c0391762eb5e67deadb9 |

The source repository declares the MIT license. Original photographs can carry separate rights. This repository distributes neither the photos nor the raw annotations. The download script retrieves public annotation files directly from their original source. Derived aggregate metrics and image split names are committed; individual gaze sequences and learned per-observer profiles are not.

OSIE contains static free-viewing eye tracking and manual object/semantic annotations. The original pilot uses masks, attributes and fixation coordinates/durations. The image-only extension additionally downloads 700 source photographs, verifies their pinned Git blob hashes, and records SHA-256 checksums in `results/image_only/image_manifest.json`. It is not fMRI data, a clinical sample, a language-conditioned search dataset or a record of people's thoughts.

All records associated with an image remain in one split. The same observer cohort contributes to all splits. The study therefore measures new-image performance for that cohort, not generalization to unseen observers. Human object annotations are privileged inputs in the reference arm. The image-only model uses predicted boxes, classes and confidence, and accesses neither human object annotations nor gaze at inference. Background includes every remaining pixel, not necessarily a psychologically uniform category.

Coordinate conversion, filtering and overlap resolution are specified in `research/protocol.md` and `research/extension_protocol.md`. The MATLAB mask array is stored with reversed dimensions. Original categorical scores cannot be compared directly with pixel metrics. The extension scores a shared pixel event space, but comparisons with external publications still require matching fixation filtering, splits and spatial-density assumptions. Detector pretraining uses COCO; possible source-image overlap with OSIE is not exhaustively audited.
