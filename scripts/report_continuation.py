"""Resolve the saliency, transfer and observer report from completed experiments."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    read = lambda folder: json.loads((ROOT/'results'/folder/'summary.json').read_text())
    sal, emod, obs, audit = [read(p) for p in ('saliency_comparison', 'emod_transfer', 'observer_holdout', 'emod_audit')]
    assert all(x['status'] == 'complete' for x in (sal, emod, obs, audit))
    transfer = emod['evaluation']['primary_nonoverlap']
    f = lambda x: f'{x:.3f}'
    ci = lambda x: '[' + ', '.join(f(v) for v in x) + ']'
    dg = sal['models']['deepgaze_iie']
    delta = sal['contrasts']['deepgaze_iie_over_detected_linear']
    center_delta = transfer['contrasts']['linear_over_center']
    ag, de = obs['aggregate']['annotated'], obs['aggregate']['detected']
    relational = read('relational')['arms']
    ra, rd = relational['annotated'], relational['detected']
    gain_a = ra['contrasts']['full_gat_over_full_self']
    gain_d = rd['contrasts']['full_gat_over_full_self']
    methods = """### 4.6 Frozen visual and relational controls

A frozen ImageNet ResNet50 [24] encodes each full photograph resized to 224 by 224 pixels. Area-weighted pooling of its 7 by 7 feature map produces 2048 features for each region. Training-only, image-balanced normalization precedes a 64-unit head. We compare visual features, visual plus geometry, and full features with local, uniform-context or learned additive graph-attention mixing [23]. Both partitions use seeds 11, 22 and 33, fixed AdamW settings, and validation-selected stopping within 50 epochs. The local and uniform controls share nominal parameters, but their attention-score parameters are inactive. ResNet receptive fields already include context; this experiment tests additional mixing, not the existence of relational human attention.

### 4.7 Static saliency and external transfer

DeepGaze IIE [21] supplies a strong frozen static image-only comparator on the same 140 OSIE test scenes. The pinned official implementation uses its released ensemble weights and MIT1003 center prior without OSIE fitting. RGB input is resized to 1024 by 768; output density is bilinearly resized to 800 by 600 and renormalized. Its SALICON and MIT1003 saliency training and ImageNet encoders differ from our training budget. Training-data overlap has not been exhaustively excluded. Thus the comparison measures delivered predictors, not equal-data architectural effects. IG remains primary. Secondary NSS, AUC-Judd without jitter, CC and target-to-prediction KL use identical fixation pixels. CC and KL use Gaussian target smoothing with sigma 0.03 times the shorter image side. The local/GAT saliency comparison fixes seed 11 in advance.

EMOd [22] provides an external free-viewing test. Its public image archive contains 698 author-collected photographs; the other 321 IAPS photographs require separate access and are not evaluated. Source fixation fields have reversed axis names: fix_y is horizontal and fix_x vertical. The chosen conversion matches the supplied binary fixation maps on all 698 images. We retain the same 100 ms duration filter. Exact file, decoded-RGB and resized-image hashes and a 64-bit perceptual-hash distance threshold of four find no OSIE matches; edited or cropped overlap is not exhaustively ruled out. The OSIE detector, visual encoder, linear coefficients, six neural checkpoints and center bandwidth remain frozen. No EMOd gaze selects parameters. EMOd object labels are not mapped to OSIE's twelve semantic attributes. DeepGaze is evaluated on OSIE only.

### 4.8 Held out observers

Three disjoint groups of five of OSIE's fifteen observer indices serve in turn as test observers. Each model fits on the other ten observers in the original 420 training images; regularization uses only those ten in the 140 validation images. Evaluation uses the five excluded observers in the 140 test images. Within-image targets weight eligible observers equally. Geometry and full linear models are reselected separately for annotated and detected regions using the six-value L2 grid. We average the three fold contrasts per test image before image bootstrapping. This tests prediction for unseen members of the observed cohort, not personalization or uncertainty over a new observer population."""
    table = ['| Model | IG bits | NSS |', '|---|---|---|']
    for label, key in [('Center prior', 'center'), ('Detected linear', 'detected_linear'),
                       ('Detected local seed 11', 'full_self_11'), ('Detected GAT seed 11', 'full_gat_11'),
                       ('Human semantic linear', 'human_semantic_linear'), ('DeepGaze IIE', 'deepgaze_iie')]:
        r = sal['models'][key]; table.append(f"| {label} | {f(r['ig_uniform_bits'])} | {f(r['nss'])} |")
    results = f"""### 5.7 Visual features and relation mixing

Thirty neural fits completed on the original split. Across three seeds, full local and GAT heads achieve {f(ra['models']['full_self']['mean_test']['ig_uniform_bits'])} and {f(ra['models']['full_gat']['mean_test']['ig_uniform_bits'])} bits of IG with annotated regions and {f(rd['models']['full_self']['mean_test']['ig_uniform_bits'])} and {f(rd['models']['full_gat']['mean_test']['ig_uniform_bits'])} bits with detected regions. GAT minus local improvement is {f(gain_a['improvement_bits'])} bits with paired image interval {ci(gain_a['image_bootstrap_ci95'])} for annotated regions and {f(gain_d['improvement_bits'])} {ci(gain_d['image_bootstrap_ci95'])} for detections. Uniform mixing performs similarly to GAT. Every neural variant underperforms the corresponding prior full-feature linear reference. These results do not support adding this mixing layer under the tested frozen features and training settings.

### 5.8 Strong static comparator and domain transfer

Table III reports the common-pixel comparison. DeepGaze IIE improves over the detected linear model by {f(delta['improvement_bits'])} bits, with paired interval {ci(delta['ci95'])}. Its NSS is {f(dg['nss'])}, AUC-Judd {f(dg['auc_judd'])}, CC {f(dg['cc'])} and KL {f(dg['kl_bits'])} bits. This establishes a substantial gap to a pretrained spatial predictor under the documented preprocessing, without isolating architecture from training-data differences.

Table III  Static prediction on the same 140 OSIE test scenes

{chr(10).join(table)}

On the {audit['primary_image_count']} public EMOd scenes, {emod['data_audit']['retained_fixations']:,} eligible events remain. Center and OSIE-trained detected linear IG are {f(transfer['models']['center']['ig_uniform_bits'])} and {f(transfer['models']['linear']['ig_uniform_bits'])} bits. Their paired linear-minus-center difference is {f(center_delta['improvement_bits'])} {ci(center_delta['ci95'])}. The fixed linear model therefore underperforms the center prior on this target subset. Mean seed-specific IG for local and GAT heads is {f(transfer['models']['full_self_seed_mean']['ig_uniform_bits'])} and {f(transfer['models']['full_gat_seed_mean']['ig_uniform_bits'])} bits. These are averages of scores, not prediction ensembles. There are {emod['zero_detection_images']} scenes without retained detections ({100*emod['zero_detection_images']/audit['primary_image_count']:.1f} percent); their region models return uniform density. All scenes remain in evaluation. This is zero-shot transfer to the available subset, not evaluation of all 1019 EMOd images.

### 5.9 Prediction for excluded observers

Full-minus-geometry log-score improvement for held-out observers is {f(ag['full_over_geometry_bits'])} bits with image interval {ci(ag['image_ci95'])} for annotated regions, and {f(de['full_over_geometry_bits'])} {ci(de['image_ci95'])} for detected classes beyond geometry and confidence. Full-model IG is {f(ag['full_ig_bits'])} and {f(de['full_ig_bits'])} bits, respectively. The semantic association persists under this cohort holdout. The experiment does not learn individual preferences or establish adaptation to a new person."""
    limitations = """The completed comparisons preserve the distinction between privileged human annotations and image-only prediction. DeepGaze IIE demonstrates that the current region-uniform models leave considerable spatial information unused. It has a different pretraining regime, so its advantage is not a controlled estimate of one architectural component. Frozen coarse visual features and graph mixing did not improve the linear references. The finding is specific to these representations, settings and finite seeds.

The external test expands the evidence to the 698 available EMOd images, while excluding the unavailable IAPS subset. Missing observer records prevent assuming stable cross-image identities in EMOd. OSIE observer holdout supports transfer within the observed cohort but does not provide population-level intervals. The original test split was already examined before these exploratory follow-ups. Splits overlap, bootstrap intervals condition on fitted models and observed viewers, multiple contrasts are descriptive, and pretrained-data overlap remains incompletely audited. No claim of state-of-the-art performance, personalization or a validated neural mechanism follows."""
    abstract = ("Object-based gaze models often rely on human-defined regions and semantic labels that are absent at image-only inference. We compare annotated and detected representations using 700 OSIE scenes with an image-disjoint 420/140/140 split. Semantic annotations improve log score beyond geometry by 0.101 to 0.149 bits per fixation across five overlapping split sensitivities. On the original test set, human semantic and detected linear models achieve 1.515 and 0.501 bits of information gain over uniform pixels, versus 0.522 for a center prior. Frozen ResNet50 features with local or graph-attention heads do not improve these linear references. "
        f"A frozen DeepGaze IIE comparator achieves {f(dg['ig_uniform_bits'])} bits under documented common-pixel evaluation. Zero-shot evaluation on 698 public EMOd images gives the OSIE-trained detected linear model {f(transfer['models']['linear']['ig_uniform_bits'])} bits versus {f(transfer['models']['center']['ig_uniform_bits'])} for the center prior. Excluding five OSIE observers from training and validation preserves a {f(ag['full_over_geometry_bits'])}-bit semantic benefit for annotated regions. "
        "The evidence supports a reproducible analysis of representation limits and transfer, rather than a new attention mechanism. Pretraining differences, partial EMOd access and cohort-conditioned uncertainty limit broader conclusions.")
    fields = dict(continuation_methods=methods, continuation_results=results, extension_limitations=limitations,
                  extension_abstract=abstract, test_count='Thirty-one')
    (ROOT/'paper/continuation_fields.json').write_text(json.dumps(fields, indent=2)+'\n', encoding='utf-8')
    report = ['# Saliency comparison, external transfer and excluded observers', '', 'Completed 5 November 2026. All values below are generated from completed result JSON.', '', results,
              '', '## Secondary metrics', '', '| Dataset | Model | IG bits | NSS | AUC Judd | CC | KL bits |', '|---|---|---:|---:|---:|---:|---:|']
    for dataset, group in [('OSIE test 140', sal['models']), ('EMOd public 698', transfer['models'])]:
        for model, r in group.items():
            if 'nss' not in r: continue
            report.append('| ' + ' | '.join([dataset, model]+[f(r[k]) for k in ('ig_uniform_bits','nss','auc_judd','cc','kl_bits')]) + ' |')
    report += ['', '## Interpretation and limits', '', limitations, '',
               'Protocols: [saliency and transfer](saliency_transfer_protocol.md), [observer holdout](observer_holdout_protocol.md). Per-image scores, selected regularization and provenance are in `results/saliency_comparison/`, `results/emod_transfer/`, `results/observer_holdout/` and `results/emod_audit/`. Reproduction commands are in the README. No model was tuned on EMOd outcomes. No hosted or Colab execution is claimed.']
    (ROOT/'research/saliency_transfer_results.md').write_text('\n'.join(report)+'\n', encoding='utf-8')
    print('Resolved continuation report and manuscript fields')


if __name__ == '__main__': main()
