"""Validation-selected neural heads, frozen features, paired image evaluation."""
import argparse
import csv
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from attention.data import load_osie
from attention.image_only import (fixation_records, make_scene, pixel_information_gain,
                                  center_information_gain)
from attention.metrics import scores, bootstrap_mean_ci
from attention.robustness import observer_target
from attention.visual import cache_entry, detection_manifest, sha256
from attention.relational import (RegionNetwork, image_cross_entropy, fit_normalizer,
                                  batch_tensors, predict_network, observer_disagreement, rank_scores)

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {'visual': ('visual', 'self'), 'visual_geometry': ('geometry', 'self'),
            'full_self': ('full', 'self'), 'full_uniform': ('full', 'uniform'),
            'full_gat': ('full', 'learned')}


def feature_arrays(scenes, cache, kind):
    output = []
    for s in scenes:
        v = cache[s['name']]
        if kind != 'visual':
            v = np.concatenate([v, s['features'][:, :6] if kind == 'geometry' else s['features']], axis=1)
        output.append(v.astype(np.float32))
    return output


def fit_one(train, val, tx, vx, relation, seed, args):
    torch.manual_seed(seed)
    if args.device.startswith('cuda'):
        torch.cuda.manual_seed_all(seed)
    mean, std = fit_normalizer(tx)
    model = RegionNetwork(len(mean), args.hidden, relation).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = np.random.default_rng(seed)
    best, best_epoch, state, history = np.inf, 0, None, []
    for epoch in range(1, args.epochs+1):
        model.train()
        order = rng.permutation(len(train))
        total = 0.
        for start in range(0, len(order), args.batch_size):
            ids = order[start:start+args.batch_size]
            x, q, valid = batch_tensors(tx, train, ids, mean, std, args.device)
            optimizer.zero_grad(set_to_none=True)
            loss = image_cross_entropy(model(x, valid), q, valid)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training objective')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            total += float(loss.detach())*len(ids)
        ps = predict_network(model, vx, val, mean, std, args.device)
        nll = float(np.mean([-s['counts']/s['counts'].sum() @ np.log2(p) for s, p in zip(val, ps)]))
        history.append(dict(epoch=epoch, train_nll_bits=total/len(train)/np.log(2), validation_nll_bits=nll))
        if nll < best:
            best, best_epoch = nll, epoch
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch-best_epoch >= args.patience:
            break
    model.load_state_dict(state)
    metadata = dict(initialization_seed=seed, selected_epoch=best_epoch, epochs_run=len(history),
                    validation_nll_bits=best, parameters=sum(p.numel() for p in model.parameters()),
                    inactive_attention_parameters=(2*args.hidden if relation != 'learned' else 0), history=history)
    return model, mean, std, metadata


def evaluate(scenes, ps, center, cutoffs):
    rows = []
    for s, p in zip(scenes, ps):
        q = s['counts']/s['counts'].sum()
        disagreement = observer_disagreement(s['subject_counts'])
        rows.append(dict(image=s['name'], regions=len(q), fixations=int(s['counts'].sum()),
                         ig_uniform_bits=pixel_information_gain(q, p, s['area']),
                         center_ig_bits=center[s['name']],
                         equal_observer_nll_bits=float(-observer_target(s['subject_counts']) @ np.log2(p)),
                         observer_js_bits=disagreement,
                         disagreement_group=('low', 'medium', 'high')[int(np.searchsorted(cutoffs, disagreement, side='right'))],
                         **scores(q, p), **rank_scores(q, p)))
    return rows


def summarize(rows):
    keys = ['nll_bits', 'ig_uniform_bits', 'brier', 'top1', 'top3', 'equal_observer_nll_bits', 'ndcg3', 'mae']
    summary = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    rho = [r['spearman'] for r in rows if r['spearman'] is not None]
    summary.update(spearman=float(np.mean(rho)) if rho else None, spearman_defined_images=len(rho))
    return summary


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-3)
    parser.add_argument('--seeds', type=int, nargs='+', default=[11, 22, 33])
    parser.add_argument('--output', type=Path, default=ROOT/'results/relational')
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1 or not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error('Positive epochs/patience and distinct seeds required')
    out = args.output.resolve()
    if (out/'summary.json').exists():
        raise ValueError('Output already contains a run; choose a new --output directory')
    out.mkdir(parents=True, exist_ok=True)
    checkpoints = ROOT/'output/relational_checkpoints'/out.name
    checkpoints.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    start = time.monotonic()
    cache_dir = ROOT/'data/processed/visual_resnet50'
    config = json.loads((cache_dir/'config.json').read_text())
    index = json.loads((cache_dir/'index.json').read_text())
    if config['attrs_sha256'] != sha256(ROOT/'data/raw/attrs.mat'):
        raise ValueError('Annotation hash mismatch')
    if config['detections_sha256'] != detection_manifest(ROOT/'data/processed/detections'):
        raise ValueError('Detection cache hash mismatch')
    for name, digest in config['images_sha256'].items():
        if sha256(ROOT/'data/raw/stimuli'/name) != digest:
            raise ValueError(f'Image hash mismatch: {name}')
    split = json.loads((ROOT/'results/primary/splits.json').read_text())
    assert [len(split[k]) for k in ('train', 'validation', 'test')] == [420, 140, 140]
    assert len(set(sum(split.values(), []))) == 700
    write_json(out/'splits.json', split)
    write_json(out/'visual_provenance.json', config)
    fixation = fixation_records(ROOT/'data/raw/fixations.mat')
    center_grid = {sigma: {n: center_information_gain(fixation[n], 800, 600, sigma)
                          for n in fixation} for sigma in (.1, .15, .2, .3, .4)}
    sigma = max(center_grid, key=lambda v: np.mean([center_grid[v][n] for n in split['validation']]))
    center = center_grid[sigma]
    annotated, audit = load_osie(ROOT/'data/raw')
    detection_dir = ROOT/'data/processed/detections'
    detector = json.loads((detection_dir/'config.json').read_text())
    detected = [make_scene(json.loads(p.read_text()), fixation[p.stem+'.jpg'], len(detector['categories']))
                for p in sorted(detection_dir.glob('*.json')) if p.name != 'config.json']
    source_files = ['src/attention/relational.py', 'src/attention/visual.py', 'src/attention/data.py',
                    'src/attention/image_only.py', 'scripts/run_relational.py',
                    'scripts/extract_visual_features.py']
    summary = dict(status='in_progress', protocol='static region choice; image-disjoint 420/140/140 split; validation-only model selection',
                   configuration={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                   environment=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__),
                   source_sha256={p: sha256(ROOT/p) for p in source_files},
                   fixation_sha256=sha256(ROOT/'data/raw/fixations.mat'),
                   feature_index_sha256=sha256(cache_dir/'index.json'),
                   center_bandwidth=sigma, center_ig_bits=float(np.mean([center[n] for n in split['test']])),
                   arms={}, audit=audit)
    allrows = []
    for arm, scenes in [('annotated', annotated), ('detected', detected)]:
        byname = {s['name']: s for s in scenes}
        assert set(byname) == set(sum(split.values(), []))
        cache = {}
        for s in scenes:
            path = cache_dir/arm/(Path(s['name']).stem+'.npz')
            if sha256(path) != index[s['name']][arm]:
                raise ValueError(f'Feature checksum mismatch: {path}')
            cache[s['name']] = cache_entry(path, s['name'], s['active_ids'])
        train, val, test = [[byname[n] for n in split[k]] for k in ('train', 'validation', 'test')]
        cutoffs = np.quantile([observer_disagreement(s['subject_counts']) for s in train], [1/3, 2/3])
        result = dict(disagreement_training_tertiles=cutoffs.tolist(), models={}, contrasts={})
        summary['arms'][arm] = result
        predictions = {}
        for variant, (kind, relation) in VARIANTS.items():
            tx, vx, ex = [feature_arrays(subset, cache, kind) for subset in (train, val, test)]
            runs = []
            predictions[variant] = []
            for seed in args.seeds:
                model, mean, std, meta = fit_one(train, val, tx, vx, relation, seed, args)
                checkpoint = checkpoints/f'{arm}_{variant}_{seed}.pt'
                payload = dict(state_dict={k: v.cpu() for k, v in model.state_dict().items()},
                               mean=torch.tensor(mean), std=torch.tensor(std), input_dim=len(mean),
                               hidden=args.hidden, relation=relation, feature_kind=kind, arm=arm,
                               seed=seed, visual_config=config, detector=detector,
                               split_sha256=sha256(out/'splits.json'), selected_epoch=meta['selected_epoch'])
                torch.save(payload, checkpoint)
                meta.update(checkpoint=str(checkpoint.relative_to(ROOT)).replace('\\', '/'),
                            checkpoint_sha256=sha256(checkpoint))
                ps = predict_network(model, ex, test, mean, std, args.device)
                rows = evaluate(test, ps, center, cutoffs)
                predictions[variant].append(rows)
                meta['test'] = summarize(rows)
                runs.append(meta)
                allrows.extend(dict(arm=arm, model=variant, seed=seed, **r) for r in rows)
                print(f'{arm} {variant} seed={seed}: selected epoch {meta["selected_epoch"]}; validation NLL={meta["validation_nll_bits"]:.4f}', flush=True)
                write_json(out/'progress.json', dict(arm=arm, model=variant, seed=seed, elapsed_seconds=time.monotonic()-start))
            result['models'][variant] = dict(runs=runs,
                mean_test={k: float(np.mean([r['test'][k] for r in runs])) for k in
                           ('nll_bits', 'ig_uniform_bits', 'brier', 'top1', 'top3', 'equal_observer_nll_bits', 'ndcg3', 'mae')},
                test_nll_seed_std=float(np.std([r['test']['nll_bits'] for r in runs], ddof=1)) if len(runs)>1 else None)
        for better, reference in [('visual_geometry', 'visual'), ('full_self', 'visual_geometry'),
                                  ('full_gat', 'full_self'), ('full_gat', 'full_uniform')]:
            deltas = [[b['nll_bits']-a['nll_bits'] for a, b in zip(run_a, run_b)]
                      for run_a, run_b in zip(predictions[better], predictions[reference])]
            delta = np.mean(deltas, axis=0)
            result['contrasts'][f'{better}_over_{reference}'] = dict(
                improvement_bits=float(delta.mean()), image_bootstrap_ci95=bootstrap_mean_ci(delta),
                improvement_by_seed=np.mean(deltas, axis=1).tolist())
        for variant, records in predictions.items():
            delta = np.mean([[r['ig_uniform_bits']-r['center_ig_bits'] for r in rows] for rows in records], axis=0)
            result['models'][variant]['over_center'] = dict(improvement_bits=float(delta.mean()), image_bootstrap_ci95=bootstrap_mean_ci(delta))
            result['models'][variant]['observer_groups'] = {
                group: dict(images=sum(r['disagreement_group'] == group for r in records[0]),
                            nll_bits=float(np.mean([r['nll_bits'] for rows in records for r in rows if r['disagreement_group'] == group])))
                for group in ('low', 'medium', 'high') if any(r['disagreement_group'] == group for r in records[0])}
        # Align the previous full-feature linear reference by exact image identity.
        old_path = ROOT/('results/primary/per_image_metrics.csv' if arm == 'annotated' else 'results/image_only/per_image_metrics.csv')
        with old_path.open(newline='') as f:
            old = {r['image']: r for r in csv.DictReader(f) if
                   (r['model'] == 'semantics' if arm == 'annotated' else
                    r['model'] == 'detected_class' and r['seed'] == '20261102')}
        if set(old) != set(split['test']):
            raise ValueError('Previous linear baseline split mismatch')
        old_nll = np.array([float(old[s['name']]['nll_bits']) if arm == 'annotated' else
                           -float(old[s['name']]['ig_uniform_bits'])-
                           float(np.sum((s['counts']/s['counts'].sum())*np.log2(s['area']))) for s in test])
        result['linear_reference_nll_bits'] = float(old_nll.mean())
        for variant in predictions:
            new_nll = np.mean([[r['nll_bits'] for r in rows] for rows in predictions[variant]], axis=0)
            delta = old_nll-new_nll
            result['models'][variant]['over_linear_reference'] = dict(improvement_bits=float(delta.mean()), image_bootstrap_ci95=bootstrap_mean_ci(delta))
        write_json(out/'summary.json', summary)
    with (out/'per_image_metrics.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(allrows[0])); writer.writeheader(); writer.writerows(allrows)
    summary.update(status='complete', elapsed_seconds=time.monotonic()-start)
    write_json(out/'summary.json', summary)
    print(f'Completed {len(allrows)} per-image model/seed records in {summary["elapsed_seconds"]:.1f}s', flush=True)


if __name__ == '__main__':
    main()
