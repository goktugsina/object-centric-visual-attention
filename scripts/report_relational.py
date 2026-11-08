"""Build a standalone measured-results report; do not rewrite the version-1 paper."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=ROOT/'results/relational')
    parser.add_argument('--output', type=Path, default=ROOT/'research/relational_results.md')
    args = parser.parse_args()
    s = json.loads((args.results/'summary.json').read_text())
    if s['status'] != 'complete':
        raise ValueError('Experiment is incomplete')
    result_location = args.results.resolve().relative_to(ROOT).as_posix()
    checkpoint_location = f'output/relational_checkpoints/{args.results.name}'
    lines = ['# Frozen visual features and object relations: measured results', '',
             'Protocol dated 3 November 2026. See [protocol](relational_protocol.md) and [plan status](shared_plan_status.md).', '',
             f'The full experiment completed in {s["elapsed_seconds"]/60:.1f} minutes on {s["configuration"]["device"]}. '
             'Both region partitions use the original 420/140/140 image split. Each model was trained with initialization seeds '
             + ', '.join(map(str, s['configuration']['seeds'])) + '. Checkpoints were selected on validation NLL; test results did not select epochs or seeds.', '',
             f'The common validation-selected pixel center prior scores **{s["center_ig_bits"]:.4f} bits** above uniform pixels '
             f'(bandwidth {s["center_bandwidth"]}). Higher information gain (IG) and lower NLL/Brier are better.', '',
             f'![Paired contrasts](../{result_location}/figures/contrasts.png)', '']
    for arm, a in s['arms'].items():
        lines += [f'## {arm.capitalize()} regions', '',
                  '| Model | NLL bits | IG bits | Brier | NDCG@3 | NLL seed SD |',
                  '|---|---:|---:|---:|---:|---:|']
        for name, model in a['models'].items():
            m = model['mean_test']; sd = model['test_nll_seed_std']
            lines.append(f'| {name} | {m["nll_bits"]:.4f} | {m["ig_uniform_bits"]:.4f} | {m["brier"]:.4f} | {m["ndcg3"]:.4f} | {sd:.4f} |' if sd is not None else
                         f'| {name} | {m["nll_bits"]:.4f} | {m["ig_uniform_bits"]:.4f} | {m["brier"]:.4f} | {m["ndcg3"]:.4f} | n/a |')
        lines += ['', 'Paired improvements, averaged over initialization seeds before resampling test images:', '',
                  '| Contrast | Improvement bits | Image-bootstrap 95% interval | Per-seed improvements |',
                  '|---|---:|---|---|']
        for name, c in a['contrasts'].items():
            lo, hi = c['image_bootstrap_ci95']
            seeds = ', '.join(f'{v:.4f}' for v in c['improvement_by_seed'])
            lines.append(f'| {name} | {c["improvement_bits"]:.4f} | [{lo:.4f}, {hi:.4f}] | {seeds} |')
        lines += ['', f'Previous full-feature linear reference NLL: {a["linear_reference_nll_bits"]:.4f} bits. '
                  'All reference comparisons align exactly the same 140 test image identities.', '',
                  '| Model | Gain over linear reference | 95% interval | Gain over center | 95% interval |',
                  '|---|---:|---|---:|---|']
        for name, m in a['models'].items():
            l, c = m['over_linear_reference'], m['over_center']
            lines.append(f'| {name} | {l["improvement_bits"]:.4f} | [{l["image_bootstrap_ci95"][0]:.4f}, {l["image_bootstrap_ci95"][1]:.4f}] | '
                         f'{c["improvement_bits"]:.4f} | [{c["image_bootstrap_ci95"][0]:.4f}, {c["image_bootstrap_ci95"][1]:.4f}] |')
        lines += ['', 'Observer disagreement groups use training-image Jensen–Shannon tertiles: '
                  + ', '.join(f'{v:.4f}' for v in a['disagreement_training_tertiles']) + ' bits.', '',
                  '| Group | Test images | Full self NLL | Full GAT NLL |', '|---|---:|---:|---:|']
        for group, row in a['models']['full_gat']['observer_groups'].items():
            other = a['models']['full_self']['observer_groups'][group]
            lines.append(f'| {group} | {row["images"]} | {other["nll_bits"]:.4f} | {row["nll_bits"]:.4f} |')
        contrast = a['contrasts']['full_gat_over_full_self']
        lo, hi = contrast['image_bootstrap_ci95']
        conclusion = ('The paired image interval is positive for GAT over the self-only control.' if lo > 0 else
                      'The paired image interval favors the self-only control.' if hi < 0 else
                      'The paired image interval crosses zero; this run does not establish a GAT gain over the self-only control.')
        lines += ['', conclusion, '']
    lines += ['## Interpretation and limits', '',
              'These are exploratory follow-up results on an already examined OSIE split, not a new independent replication. '
              'The intervals condition on the sampled observer cohort and fitted seeds; they omit full training uncertainty. '
              'Multiple descriptive contrasts are reported without a multiplicity correction. Seed SD is across training runs, not a confidence interval.', '',
              'The annotated arm uses human masks and semantic attributes; the detected arm uses photographs and frozen detections. '
              'ResNet receptive fields already contain scene context. Adding GAT measures an extra mixing layer, not whether humans use relations. '
              'The self-only and uniform controls have equal nominal parameter counts, but their attention-score parameters are inactive. '
              'No claim of equal effective capacity or causal interpretation follows. '
              'In the detected arm, full features add confidence and class indicators jointly; the contrast does not isolate class information alone.', '',
              'NDCG uses linear fixation-share relevance and averages tied predicted ranks. Background participates in ranking. '
              'Undefined Spearman correlations (constant targets or predictions, including one-region scenes) are excluded and counted in each run. '
              'Observer groups use test gaze only for descriptive evaluation; the model is not personalized. '
              'Pretraining overlap is not exhaustively excluded. Tiny objects share a coarse 7 x 7 feature grid.', '',
              'ResNet+U-Net, a stronger externally pretrained saliency comparator, EMOd compatibility/overlap checking, '
              'cross-dataset evaluation, held-out-observer prediction and backbone fine-tuning have not been completed by this stage. '
              'These results do not establish superiority to conventional pixel saliency models. '
              'The existing six-page paper has not been revised to incorporate this follow-up.', '',
              '## Reproduction and artifacts', '',
              f'- `{result_location}/summary.json`: all seeds, validation histories, selected epochs, contrasts and source hashes.',
              f'- `{result_location}/per_image_metrics.csv`: per-image proper scores, ranks and observer groups.',
              f'- `{result_location}/visual_provenance.json`: encoder, detector, image and annotation provenance.',
              f'- `{checkpoint_location}/`: trained heads and training-only normalization (local, ignored by Git).',
              '- `notebooks/object_attention_relations.ipynb`: Colab-compatible reproduction workflow.',
              '- `output/visual_audit/`: ten local scene audits; manifest in `results/visual_audit/manifest.json`.', '']
    args.output.write_text('\n'.join(lines), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    contrasts = [('visual_geometry_over_visual', 'Add geometry'),
                 ('full_self_over_visual_geometry', 'Add attributes / confidence+class'),
                 ('full_gat_over_full_self', 'GAT vs self-only'),
                 ('full_gat_over_full_uniform', 'GAT vs uniform mixing')]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained', sharey=True)
    for ax, (arm, a) in zip(axes, s['arms'].items()):
        for i, (key, label) in enumerate(contrasts):
            c = a['contrasts'][key]; mean = c['improvement_bits']; lo, hi = c['image_bootstrap_ci95']
            ax.errorbar(mean, i, xerr=[[mean-lo], [hi-mean]], fmt='o', color='#28648b', capsize=4)
        ax.axvline(0, linestyle='--', linewidth=1, color='#666666')
        ax.set_yticks(range(len(contrasts)), [label for _, label in contrasts])
        ax.set_title(arm.capitalize()+' regions')
        ax.set_xlabel('Improvement (bits / fixation; positive is better)')
        ax.grid(axis='x', alpha=.2)
    axes[0].invert_yaxis()
    fig.suptitle('Frozen ResNet50: paired region-choice contrasts\nSeed-average differences; image-bootstrap 95% intervals')
    folder = args.results/'figures'; folder.mkdir(exist_ok=True)
    fig.savefig(folder/'contrasts.png', dpi=160)
    fig.savefig(folder/'contrasts.pdf')
    plt.close(fig)
    print(args.output)


if __name__ == '__main__':
    main()
