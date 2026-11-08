"""Audit frozen provenance, observer exclusion and sampled transfer replay."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from scipy.special import softmax
from attention.visual import sha256
from attention.frozen_heads import load_heads, head_probabilities
from attention.image_only import region_features, probability_map
from attention.emod import load_emod
from attention.saliency import fixation_target, pixel_scores

ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1)
    read = lambda p: json.loads((ROOT/p).read_text())
    sal = read('results/saliency_comparison/summary.json')
    emod = read('results/emod_transfer/summary.json')
    obs = read('results/observer_holdout/summary.json')
    checked = 0
    for manifest in [sal['source_sha256'], sal['references_sha256'], emod['provenance']['source_sha256'], obs['source_sha256']]:
        for path, digest in manifest.items():
            assert sha256(ROOT/path) == digest, path
            checked += 1
    for name, hashes in emod['predictions_sha256'].items():
        for key, suffix in [('probabilities_sha256', '.npz'), ('detection_sha256', '.json')]:
            assert sha256(ROOT/'data/processed/emod_transfer'/(Path(name).stem+suffix)) == hashes[key]
            checked += 1
    test_observers = []
    for fold in obs['folds']:
        train, val, test = [set(fold[k]) for k in ('train_observers', 'validation_observers', 'test_observers')]
        assert train == val and not train & test and train | test == set(range(15))
        test_observers.extend(test)
    assert sorted(test_observers) == list(range(15))
    split = read('results/primary/splits.json')
    assert sha256(ROOT/'results/primary/splits.json') == obs['split_sha256']
    with (ROOT/'results/observer_holdout/per_image_metrics.csv').open(newline='') as stream:
        observer_rows = list(csv.DictReader(stream))
    assert len(observer_rows) == 3*2*2*140
    assert {r['image'] for r in observer_rows} == set(split['test'])
    with (ROOT/'results/emod_transfer/per_image_metrics.csv').open(newline='') as stream:
        rows = {(r['image'], r['model']):r for r in csv.DictReader(stream)}
    heads, hashes = load_heads(ROOT)
    assert hashes == emod['provenance']['checkpoints']
    linear = read('results/image_only/model.json')
    data, _ = load_emod(ROOT/'data/raw/emod')
    names = sorted(data); sampled = [names[0], names[len(names)//2], names[-1]]
    error = 0.; normalization_error = 0.
    for name in sampled:
        cache = ROOT/'data/processed/emod_transfer'/Path(name).stem
        detection = read(str(cache.with_suffix('.json').relative_to(ROOT)))
        labels, active, area, auxiliary = region_features(detection, len(linear['detector']['categories']))
        with np.load(cache.with_suffix('.npz'), allow_pickle=False) as entry:
            assert str(entry['image_sha256']) == sha256(ROOT/'data/raw/emod/stimuli'/name)
            ps = dict(linear=softmax(auxiliary@np.asarray(linear['coefficients'])),
                      **head_probabilities(entry['visual'], auxiliary, heads))
            for key, mass in ps.items(): np.testing.assert_allclose(mass, entry[key], atol=1e-7, rtol=1e-6)
        record = data[name]; target = fixation_target(record['subjects'], record['width'], record['height'])
        for key, mass in ps.items():
            density = probability_map(labels, active, mass)
            normalization_error = max(normalization_error, abs(density.sum()-1))
            actual = pixel_scores(density, target)
            error = max(error, abs(actual['ig_uniform_bits']-float(rows[name, key]['ig_uniform_bits'])))
    assert error < 1e-6 and normalization_error < 1e-6
    assert len(rows) == 698*9
    report = dict(status='complete', hash_checks=checked, held_out_observers=15,
                  observer_rows=len(observer_rows), emod_rows=len(rows), sampled_images=sampled,
                  sampled_replay_max_ig_error=error, probability_normalization_max_error=normalization_error,
                  osie_reference_replay=sal['reference_replay_max_error'],
                  scope='All transfer cache hashes and observer partitions; 3 transfer images replayed from cached visual features, not full detector extraction.')
    (ROOT/'results/continuation_verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
