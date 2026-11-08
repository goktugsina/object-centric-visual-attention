"""Score frozen external and object models on exactly the same OSIE pixels."""
import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy.special import softmax
import torch

from attention.data import load_osie
from attention.visual import human_regions, cache_entry, sha256
from attention.image_only import region_features, probability_map, fixation_records, center_log_ratio
from attention.frozen_heads import load_heads, head_probabilities
from attention.saliency import fixation_target, pixel_scores
from attention.metrics import bootstrap_mean_ci

ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1); started = time.monotonic()
    out = ROOT/'results/saliency_comparison'; out.mkdir(parents=True, exist_ok=True)
    provenance = json.loads((out/'deepgaze_provenance.json').read_text())
    if provenance['status'] != 'complete':
        raise ValueError('Complete all 140 DeepGaze predictions first')
    names = json.loads((ROOT/'results/primary/splits.json').read_text())['test']
    fixations = fixation_records(ROOT/'data/raw/fixations.mat')
    scenes, _ = load_osie(ROOT/'data/raw'); scenes = {s['name']: s for s in scenes}
    linear = json.loads((ROOT/'results/image_only/model.json').read_text())
    human_coefficients = json.loads((ROOT/'results/primary/coefficients.json').read_text())['semantics']
    heads, head_hashes = load_heads(ROOT, seeds=(11,))
    rows = []; predictions = {}
    for i, (name, _, human_labels, human_active) in enumerate(human_regions(ROOT/'data/raw/attrs.mat', set(names)), 1):
        with np.load(ROOT/'data/processed/deepgaze_osie'/(Path(name).stem+'.npz'), allow_pickle=False) as entry:
            deepgaze = entry['density'].copy()
            if str(entry['image_sha256']) != sha256(ROOT/'data/raw/stimuli'/name):
                raise ValueError('DeepGaze source image changed')
        deep_path = ROOT/'data/processed/deepgaze_osie'/(Path(name).stem+'.npz')
        if sha256(deep_path) != provenance['predictions_sha256'][name]:
            raise ValueError('DeepGaze prediction changed')
        detection = json.loads((ROOT/'data/processed/detections'/(Path(name).stem+'.json')).read_text())
        labels, active, area, auxiliary = region_features(detection, len(linear['detector']['categories']))
        visual = cache_entry(ROOT/'data/processed/visual_resnet50/detected'/(Path(name).stem+'.npz'), name, active)
        ps = dict(detected_linear=softmax(auxiliary@np.asarray(linear['coefficients'])),
                  **head_probabilities(visual, auxiliary, heads))
        scene = scenes[name]; assert np.array_equal(scene['active_ids'], human_active)
        human_mass = softmax(scene['features']@np.asarray(human_coefficients))
        target = fixation_target(fixations[name], 800, 600)
        maps = dict(uniform=np.ones((600, 800))/(800*600),
                    center=np.exp2(center_log_ratio(800, 600, .2))/(800*600),
                    human_semantic_linear=probability_map(human_labels, human_active, human_mass),
                    deepgaze_iie=deepgaze,
                    **{model: probability_map(labels, active, mass) for model, mass in ps.items()})
        for model, density in maps.items():
            rows.append(dict(image=name, model=model, fixations=len(target['x']), **pixel_scores(density, target)))
        if name in names[:3]:
            predictions[name] = {k: v.astype(np.float32) for k, v in maps.items() if k in ('center', 'deepgaze_iie', 'detected_linear', 'human_semantic_linear')}
        if i % 25 == 0: print(f'Common-pixel scoring {i}/140', flush=True)
    summary = dict(status='complete', images=len(names), fixations=sum(len(x) for name in names for x, y in fixations[name]),
                   metrics=dict(primary='image-average IG over uniform, bits', auc='Judd positive-pixel thresholds, no jitter; tie-aware',
                                cc_kl_target_sigma_fraction_of_short_side=.03, kl_units='bits'),
                   models={}, contrasts={}, checkpoint_sha256=head_hashes,
                   source_sha256={p:sha256(ROOT/p) for p in ['scripts/compare_saliency.py','src/attention/saliency.py','src/attention/frozen_heads.py']},
                   references_sha256={p:sha256(ROOT/p) for p in ['results/primary/splits.json','results/primary/coefficients.json','results/image_only/model.json']},
                   deepgaze_provenance_sha256=sha256(out/'deepgaze_provenance.json'))
    records = {}
    for model in sorted({r['model'] for r in rows}):
        rs = sorted([r for r in rows if r['model'] == model], key=lambda r: r['image']); records[model] = rs
        summary['models'][model] = {k:float(np.mean([r[k] for r in rs])) for k in
                                   ('ig_uniform_bits','pixel_nll_bits','equal_observer_ig_bits','nss','auc_judd','cc','kl_bits')}
        summary['models'][model]['ig_ci95'] = bootstrap_mean_ci([r['ig_uniform_bits'] for r in rs])
    for model in records:
        for reference in ('center', 'detected_linear'):
            if model == reference: continue
            a, b = records[model], records[reference]
            assert [r['image'] for r in a] == [r['image'] for r in b]
            delta = np.array([x['ig_uniform_bits']-y['ig_uniform_bits'] for x,y in zip(a,b)])
            summary['contrasts'][model+'_over_'+reference] = dict(improvement_bits=float(delta.mean()), ci95=bootstrap_mean_ci(delta))
    old = json.loads((ROOT/'results/image_only/oracle_pixel_reference.json').read_text())
    reference = {r['image']:r['ig_uniform_bits'] for r in old['models']['semantics']['per_image']}
    error = max(abs(r['ig_uniform_bits']-reference[r['image']]) for r in records['human_semantic_linear'])
    if error > 1e-10: raise ValueError('Human reference scoring changed')
    with (ROOT/'results/image_only/per_image_metrics.csv').open(newline='') as f:
        original = {r['image']:float(r['ig_uniform_bits']) for r in csv.DictReader(f) if r['seed']=='20261102' and r['model']=='detected_class'}
    detector_error = max(abs(r['ig_uniform_bits']-original[r['image']]) for r in records['detected_linear'])
    if detector_error > 1e-10: raise ValueError('Detector reference scoring changed')
    summary['reference_replay_max_error'] = dict(human=error, detected=detector_error)
    summary['elapsed_seconds'] = time.monotonic()-started
    with (out/'per_image_metrics.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    # Deterministic test examples; photographs stay outside version control.
    from PIL import Image
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory = ROOT/'output/predictions/saliency_comparison'; directory.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 4, figsize=(14, 8), layout='constrained')
    for row, name in enumerate(names[:3]):
        with Image.open(ROOT/'data/raw/stimuli'/name) as photograph: axes[row,0].imshow(photograph)
        axes[row,0].set_title(name)
        for column, model in enumerate(('detected_linear','deepgaze_iie','human_semantic_linear'), 1):
            axes[row,column].imshow(predictions[name][model], cmap='magma')
            axes[row,column].set_title(model.replace('_',' '))
        for ax in axes[row]: ax.axis('off')
    fig.savefig(directory/'examples.png', dpi=140); plt.close(fig)
    print(json.dumps(summary['models'], indent=2), flush=True)


if __name__ == '__main__': main()
