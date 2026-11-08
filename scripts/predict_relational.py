"""Image-only neural inference: no human masks, semantic labels or gaze loaded."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.models.detection import (fasterrcnn_mobilenet_v3_large_320_fpn,
                                         FasterRCNN_MobileNet_V3_Large_320_FPN_Weights)

from attention.image_only import region_features, probability_map
from attention.relational import RegionNetwork
from attention.visual import load_backbone, image_tensor, pool_regions, sha256

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=Path)
    parser.add_argument('--checkpoint', type=Path,
                        default=ROOT/'output/relational_checkpoints/relational/detected_full_gat_11.pt')
    parser.add_argument('--output', type=Path, default=ROOT/'output/predictions/relational')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if checkpoint['arm'] != 'detected':
        raise ValueError('Only detected-region checkpoints support image-only inference')
    config = checkpoint['detector']
    encoder, backbone_config = load_backbone(ROOT, args.device)
    for key, value in backbone_config.items():
        if checkpoint['visual_config'][key] != value:
            raise ValueError(f'Backbone configuration mismatch: {key}')
    weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1
    detector = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights, progress=False).to(args.device).eval()
    weight_file = ROOT/'data/raw/model_cache/checkpoints'/weights.url.rsplit('/', 1)[1]
    if sha256(weight_file) != config['weights_sha256']:
        raise ValueError('Detector weight hash mismatch')
    with Image.open(args.image) as source:
        image = source.convert('RGB')
    model = RegionNetwork(checkpoint['input_dim'], checkpoint['hidden'], checkpoint['relation']).to(args.device)
    model.load_state_dict(checkpoint['state_dict']); model.eval()
    with torch.inference_mode():
        result = detector([weights.transforms()(image).to(args.device)])[0]
        keep = torch.nonzero(result['scores'] >= config['threshold']).flatten()[:config['max_detections']]
        detection = dict(image=args.image.name, width=image.width, height=image.height,
                         boxes=result['boxes'][keep].cpu().tolist(), scores=result['scores'][keep].cpu().tolist(),
                         classes=result['labels'][keep].cpu().tolist())
        labels, active, area, auxiliary = region_features(detection, len(config['categories']))
        fmap = encoder(image_tensor(image)[None].to(args.device))[0]
        visual = pool_regions(fmap, labels, active).cpu().numpy()
        kind = checkpoint['feature_kind']
        x = visual if kind == 'visual' else np.concatenate([visual, auxiliary[:, :6] if kind == 'geometry' else auxiliary], axis=1)
        x = torch.as_tensor(x.astype(np.float32))
        x = ((x-checkpoint['mean'])/checkpoint['std'])[None].to(args.device)
        valid = torch.ones(x.shape[:2], dtype=torch.bool, device=args.device)
        p = model(x, valid).softmax(-1)[0].cpu().numpy().astype(float)
    p = np.maximum(p, 1e-12); p /= p.sum()
    density = probability_map(labels, active, p)
    if not np.isclose(density.sum(), 1.) or not np.isfinite(density).all():
        raise ValueError('Invalid probability map')
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output/(args.image.stem+'_probability.npy'), density)
    regions = [dict(source_detection=int(i)-1,
                    category='background' if i == 0 else config['categories'][detection['classes'][i-1]],
                    probability=float(mass), area_fraction=float(fraction)) for i, mass, fraction in zip(active, p, area)]
    summary = dict(image=str(args.image), image_sha256=sha256(args.image),
                   checkpoint=str(args.checkpoint), checkpoint_sha256=sha256(args.checkpoint),
                   seed=checkpoint['seed'], probability_sum=float(density.sum()), regions=regions,
                   note='Uniform density within detector regions; static attention, not a scanpath.')
    (args.output/(args.image.stem+'_attention.json')).write_text(json.dumps(summary, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout='constrained')
    axes[0].imshow(image); axes[0].set_title('Input photograph')
    axes[1].imshow(image)
    axes[1].imshow(np.ma.masked_where(labels == 0, labels), cmap='turbo', alpha=.45)
    axes[1].set_title('Detected disjoint regions')
    shown = axes[2].imshow(density, cmap='magma'); axes[2].set_title('Neural probability per pixel')
    for ax in axes:
        ax.axis('off')
    fig.colorbar(shown, ax=axes[2], fraction=.04)
    fig.savefig(args.output/(args.image.stem+'_attention.png'), dpi=140); plt.close(fig)
    print(json.dumps(dict(output=str(args.output), probability_sum=float(density.sum()), regions=len(active))))


if __name__ == '__main__':
    main()
