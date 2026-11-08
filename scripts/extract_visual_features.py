"""Cache frozen ResNet50 features for human and detected disjoint regions."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image

from attention.visual import (human_regions, load_backbone, image_tensor, pool_regions,
                              sha256, detection_manifest)
from attention.image_only import region_features

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    start = time.monotonic()
    folder = ROOT/'data/processed/visual_resnet50'
    for arm in ('annotated', 'detected'):
        (folder/arm).mkdir(parents=True, exist_ok=True)
    encoder, config = load_backbone(ROOT, args.device)
    detections = ROOT/'data/processed/detections'
    categories = json.loads((detections/'config.json').read_text())['categories']
    config.update(torch=torch.__version__, torchvision=torchvision.__version__,
                  device=args.device, attrs_sha256=sha256(ROOT/'data/raw/attrs.mat'),
                  detections_sha256=detection_manifest(detections),
                  images_sha256={p.name: sha256(p) for p in sorted((ROOT/'data/raw/stimuli').glob('*.jpg'))})
    config_path = folder/'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Visual cache configuration changed; use a fresh cache directory')
    config_path.write_text(json.dumps(config, indent=2)+'\n')
    index = {}
    with torch.inference_mode():
        for i, (name, _, labels, active) in enumerate(human_regions(ROOT/'data/raw/attrs.mat'), 1):
            destinations = {arm: folder/arm/(Path(name).stem+'.npz') for arm in ('annotated', 'detected')}
            if not all(p.exists() for p in destinations.values()):
                with Image.open(ROOT/'data/raw/stimuli'/name) as image:
                    fmap = encoder(image_tensor(image)[None].to(args.device))[0]
                detection = json.loads((detections/(Path(name).stem+'.json')).read_text())
                detlabels, detactive, _, _ = region_features(detection, len(categories))
                for arm, partition, ids in [('annotated', labels, active), ('detected', detlabels, detactive)]:
                    visual = pool_regions(fmap, partition, ids).cpu().numpy()
                    np.savez_compressed(destinations[arm], image=name, active_ids=ids, visual=visual)
            index[name] = {arm: sha256(p) for arm, p in destinations.items()}
            if i == 1 or i % 50 == 0:
                print(f'{i}/700 images; elapsed {time.monotonic()-start:.1f}s', flush=True)
    if len(index) != 700:
        raise ValueError(f'Expected 700 images, found {len(index)}')
    (folder/'index.json').write_text(json.dumps(index, indent=2)+'\n')
    (folder/'completed.json').write_text(json.dumps(dict(images=700, elapsed_seconds=time.monotonic()-start), indent=2)+'\n')
    print('Both visual feature caches complete', flush=True)


if __name__ == '__main__':
    main()
