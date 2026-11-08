"""Replay learned checkpoints and verify published rows without retraining."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from attention.data import load_osie
from attention.image_only import fixation_records, make_scene, pixel_information_gain
from attention.relational import RegionNetwork, fit_normalizer, predict_network
from attention.visual import cache_entry, sha256
from run_relational import feature_arrays

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=ROOT/'results/relational')
    args = parser.parse_args()
    torch.set_num_threads(1)
    summary = json.loads((args.results/'summary.json').read_text())
    if summary['status'] != 'complete':
        raise ValueError('Incomplete experiment')
    for name, digest in summary['source_sha256'].items():
        if sha256(ROOT/name) != digest:
            raise ValueError(f'Source has changed since this experiment: {name}')
    splits = json.loads((args.results/'splits.json').read_text())
    with (args.results/'per_image_metrics.csv').open(newline='') as f:
        table = list(csv.DictReader(f))
    keys = [(r['arm'], r['model'], int(r['seed']), r['image']) for r in table]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate per-image rows')
    rows = dict(zip(keys, table))
    annotated, _ = load_osie(ROOT/'data/raw')
    fixation = fixation_records(ROOT/'data/raw/fixations.mat')
    folder = ROOT/'data/processed/detections'
    config = json.loads((folder/'config.json').read_text())
    detected = [make_scene(json.loads(p.read_text()), fixation[p.stem+'.jpg'], len(config['categories']))
                for p in sorted(folder.glob('*.json')) if p.name != 'config.json']
    checked = 0; max_error = 0.; checkpoints = 0
    for arm, scenes in [('annotated', annotated), ('detected', detected)]:
        byname = {s['name']: s for s in scenes}
        train, test = [[byname[n] for n in splits[k]] for k in ('train', 'test')]
        cache = {s['name']: cache_entry(ROOT/'data/processed/visual_resnet50'/arm/(Path(s['name']).stem+'.npz'),
                                        s['name'], s['active_ids']) for s in train+test}
        for variant, model in summary['arms'][arm]['models'].items():
            for run in model['runs']:
                path = ROOT/run['checkpoint']
                if sha256(path) != run['checkpoint_sha256']:
                    raise ValueError(f'Checkpoint changed: {path}')
                checkpoint = torch.load(path, map_location='cpu', weights_only=True)
                tx = feature_arrays(train, cache, checkpoint['feature_kind'])
                mean, std = fit_normalizer(tx)
                np.testing.assert_array_equal(mean, checkpoint['mean'].numpy())
                np.testing.assert_array_equal(std, checkpoint['std'].numpy())
                network = RegionNetwork(checkpoint['input_dim'], checkpoint['hidden'], checkpoint['relation'])
                network.load_state_dict(checkpoint['state_dict'])
                ex = feature_arrays(test, cache, checkpoint['feature_kind'])
                predictions = predict_network(network, ex, test, mean, std)
                values = []
                for s, p in zip(test, predictions):
                    key = (arm, variant, run['initialization_seed'], s['name'])
                    row = rows[key]; q = s['counts']/s['counts'].sum()
                    nll = float(-q @ np.log2(p)); ig = pixel_information_gain(q, p, s['area'])
                    error = max(abs(nll-float(row['nll_bits'])), abs(ig-float(row['ig_uniform_bits'])))
                    if error > 1e-6:
                        raise ValueError(f'Checkpoint replay differs: {key}, error={error}')
                    max_error = max(error, max_error); checked += 1; values.append(nll)
                np.testing.assert_allclose(np.mean(values), run['test']['nll_bits'], atol=1e-6, rtol=0)
                checkpoints += 1
    if checked != len(rows):
        raise ValueError('Unexpected rows outside the experiment design')
    verification = dict(status='passed', checkpoints_replayed=checkpoints, image_predictions_checked=checked,
                        max_score_absolute_error=max_error, training_only_normalizers_verified=True,
                        source_hashes_verified=True, checkpoint_hashes_verified=True,
                        summary_sha256=sha256(args.results/'summary.json'),
                        table_sha256=sha256(args.results/'per_image_metrics.csv'))
    (args.results/'verification.json').write_text(json.dumps(verification, indent=2)+'\n')
    print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    main()
