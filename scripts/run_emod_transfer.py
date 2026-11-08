"""Zero-shot transfer of locked OSIE models to the public EMOd subset."""
import csv
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.special import softmax
import torch
from torchvision.models.detection import (fasterrcnn_mobilenet_v3_large_320_fpn,
                                         FasterRCNN_MobileNet_V3_Large_320_FPN_Weights)

from attention.emod import load_emod
from attention.image_only import region_features, probability_map, center_log_ratio
from attention.visual import load_backbone, image_tensor, pool_regions, sha256
from attention.frozen_heads import load_heads, head_probabilities
from attention.saliency import fixation_target, pixel_scores
from attention.metrics import bootstrap_mean_ci

ROOT = Path(__file__).resolve().parents[1]


def summarize_rows(rows, primary_names):
    output = {}
    for subset, selected in [('primary_nonoverlap', set(primary_names)), ('all_available', {r['image'] for r in rows})]:
        output[subset] = {'models': {}, 'contrasts': {}, 'images': len(selected)}
        records = {}
        for model in sorted({r['model'] for r in rows}):
            rs = sorted([r for r in rows if r['model'] == model and r['image'] in selected], key=lambda r: r['image'])
            records[model] = rs
            metrics = ['ig_uniform_bits', 'pixel_nll_bits', 'equal_observer_ig_bits', 'nss', 'auc_judd', 'cc', 'kl_bits']
            output[subset]['models'][model] = {key: float(np.mean([r[key] for r in rs if r[key] is not None])) for key in metrics}
            output[subset]['models'][model]['ig_ci95'] = bootstrap_mean_ci([r['ig_uniform_bits'] for r in rs])
        for model in records:
            if model == 'center': continue
            for reference in ('center', 'linear'):
                if model == reference: continue
                a, b = records[model], records[reference]
                assert [r['image'] for r in a] == [r['image'] for r in b]
                delta = np.array([x['ig_uniform_bits']-y['ig_uniform_bits'] for x, y in zip(a, b)])
                output[subset]['contrasts'][model+'_over_'+reference] = dict(improvement_bits=float(delta.mean()), ci95=bootstrap_mean_ci(delta))
        # Average seed-specific scores before bootstrapping images (not an ensemble).
        for variant in ('full_self', 'full_gat'):
            means = np.mean([[r['ig_uniform_bits'] for r in records[f'{variant}_{seed}']] for seed in (11, 22, 33)], axis=0)
            reference = np.array([r['ig_uniform_bits'] for r in records['linear']])
            output[subset]['models'][variant+'_seed_mean'] = dict(ig_uniform_bits=float(means.mean()),
                gain_over_linear_bits=float((means-reference).mean()), paired_ci95=bootstrap_mean_ci(means-reference))
    return output


def main():
    torch.set_num_threads(4)
    started = time.monotonic(); out = ROOT/'results/emod_transfer'; out.mkdir(parents=True, exist_ok=True)
    cache = ROOT/'data/processed/emod_transfer'; cache.mkdir(parents=True, exist_ok=True)
    audit_path = ROOT/'results/emod_audit/summary.json'
    audit = json.loads(audit_path.read_text()); data, data_audit = load_emod(ROOT/'data/raw/emod')
    trained = json.loads((ROOT/'results/image_only/model.json').read_text()); config = trained['detector']
    coefficients = np.asarray(trained['coefficients'])
    torch.hub.set_dir(str(ROOT/'data/raw/model_cache'))
    weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1
    detector = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights, progress=False).eval().requires_grad_(False)
    weight_path = ROOT/'data/raw/model_cache/checkpoints'/weights.url.rsplit('/', 1)[1]
    if sha256(weight_path) != config['weights_sha256']:
        raise ValueError('Detector weights differ from the OSIE model')
    encoder, visual_config = load_backbone(ROOT)
    heads, head_hashes = load_heads(ROOT)
    for _, metadata in heads.values():
        for key, value in visual_config.items():
            if metadata['visual_config'][key] != value:
                raise ValueError('Visual representation differs from training')
    cache_config = dict(detector=config, visual=visual_config, checkpoints=head_hashes,
                        audit_sha256=sha256(audit_path), linear_sha256=sha256(ROOT/'results/image_only/model.json'),
                        torch=str(torch.__version__), source_sha256={p:sha256(ROOT/p) for p in
                            ('src/attention/frozen_heads.py','src/attention/emod.py','src/attention/saliency.py','scripts/run_emod_transfer.py')})
    config_path = cache/'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != cache_config:
        raise ValueError('Transfer cache configuration mismatch')
    config_path.write_text(json.dumps(cache_config, indent=2)+'\n')
    rows = []; zero_detections = 0; predictions = {}
    for i, (name, record) in enumerate(sorted(data.items()), 1):
        image_path = ROOT/'data/raw/emod/stimuli'/name
        digest = sha256(image_path)
        if audit['images'][name]['file_sha256'] != digest:
            raise ValueError('Target image changed after overlap audit')
        target_path = cache/(Path(name).stem+'.npz'); det_path = cache/(Path(name).stem+'.json')
        if not target_path.exists() or not det_path.exists():
            with Image.open(image_path) as source: image = source.convert('RGB')
            with torch.inference_mode():
                d = detector([weights.transforms()(image)])[0]
                keep = torch.nonzero(d['scores'] >= config['threshold']).flatten()[:config['max_detections']]
                detection = dict(image=name, width=image.width, height=image.height, image_sha256=digest,
                                 boxes=d['boxes'][keep].tolist(), scores=d['scores'][keep].tolist(), classes=d['labels'][keep].tolist())
                labels, active, area, auxiliary = region_features(detection, len(config['categories']))
                fmap = encoder(image_tensor(image)[None])[0]
                visual = pool_regions(fmap, labels, active).numpy()
            ps = dict(linear=softmax(auxiliary@coefficients), **head_probabilities(visual, auxiliary, heads))
            np.savez_compressed(target_path, image_sha256=digest, active_ids=active, visual=visual, **ps)
            det_path.write_text(json.dumps(detection)+'\n')
        detection = json.loads(det_path.read_text())
        labels, active, area, auxiliary = region_features(detection, len(config['categories']))
        with np.load(target_path, allow_pickle=False) as entry:
            if str(entry['image_sha256']) != digest or not np.array_equal(entry['active_ids'], active):
                raise ValueError('Transfer prediction cache mismatch')
            ps = {key: entry[key].copy() for key in ['linear', *heads]}
        zero_detections += int(len(active) == 1 and active[0] == 0)
        target = fixation_target(record['subjects'], record['width'], record['height'])
        for model in ['uniform', 'center', *ps]:
            if model == 'uniform': density = np.ones(labels.shape)/labels.size
            elif model == 'center': density = np.exp2(center_log_ratio(record['width'], record['height'], .2))/labels.size
            else: density = probability_map(labels, active, ps[model])
            rows.append(dict(image=name, model=model, fixations=len(target['x']), regions=len(active),
                             primary_nonoverlap=name in audit['primary_images'], **pixel_scores(density, target)))
        predictions[name] = dict(probabilities_sha256=sha256(target_path), detection_sha256=sha256(det_path))
        if i == 1 or i % 50 == 0:
            print(f'EMOd {i}/{len(data)}; elapsed {time.monotonic()-started:.1f}s', flush=True)
    with (out/'per_image_metrics.csv').open('w', newline='', encoding='utf-8') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = dict(status='complete', data_audit=data_audit, provenance=cache_config, predictions_sha256=predictions,
                   zero_detection_images=zero_detections, elapsed_seconds=time.monotonic()-started,
                   evaluation=summarize_rows(rows, audit['primary_images']),
                   limits=['698 public images; IAPS 321 unavailable', 'No target training or calibration',
                           'No pHash candidate <=4; cropped/edited overlap not exhaustively excluded',
                           'No stable cross-image observer identities asserted for incomplete subject arrays'])
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(f'Completed EMOd transfer: {len(rows)} rows', flush=True)


if __name__ == '__main__': main()
