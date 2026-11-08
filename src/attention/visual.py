"""Frozen full-image ResNet features; extraction never reads fixation records."""
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms.functional import to_tensor, normalize

from attention.data import partition_masks
from attention.image_only import region_features


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def human_regions(path, names=None):
    with h5py.File(path, 'r') as data:
        def chars(ds):
            return ''.join(chr(int(c)) for c in ds[:].ravel())
        for ref in data['attrs'][:].ravel():
            record = data[ref]
            name = chars(record['img'])
            if names is not None and name not in names:
                continue
            masks = np.array([data[r]['map'][:].T.astype(bool)
                              for r in record['objs'][:].ravel()])
            labels = partition_masks(masks)
            yield name, masks, labels, np.unique(labels)


def load_backbone(root, device='cpu'):
    torch.hub.set_dir(str(Path(root)/'data/raw/model_cache'))
    weights = ResNet50_Weights.IMAGENET1K_V2
    network = resnet50(weights=weights, progress=False)
    encoder = torch.nn.Sequential(*list(network.children())[:-2]).to(device).eval()
    encoder.requires_grad_(False)
    checkpoint = Path(root)/'data/raw/model_cache/checkpoints'/weights.url.rsplit('/', 1)[1]
    return encoder, dict(model='resnet50', weights='IMAGENET1K_V2',
                         weights_url=weights.url, weights_sha256=sha256(checkpoint),
                         resize=[224, 224], crop=False, feature_grid=[7, 7],
                         pooling='area-resampled disjoint mask weighted mean',
                         normalization_mean=[.485, .456, .406],
                         normalization_std=[.229, .224, .225])


def image_tensor(image):
    image = image.convert('RGB').resize((224, 224), Image.Resampling.BILINEAR)
    return normalize(to_tensor(image), [.485, .456, .406], [.229, .224, .225])


def pool_regions(feature_map, labels, active):
    """Area pooling preserves sub-grid objects and returns rows in active order."""
    if feature_map.ndim != 3:
        raise ValueError('Expected channel x height x width feature map')
    masks = torch.as_tensor(np.array([labels == i for i in active], dtype=np.float32),
                            device=feature_map.device)
    coverage = F.interpolate(masks[:, None], size=feature_map.shape[-2:], mode='area')[:, 0]
    mass = coverage.sum((1, 2))
    if torch.any(mass <= 0):
        raise ValueError('Empty active region')
    return torch.einsum('nhw,chw->nc', coverage, feature_map)/mass[:, None]


def detection_manifest(folder):
    return {p.name: sha256(p) for p in sorted(Path(folder).glob('*.json'))}


def cache_entry(path, name, active):
    with np.load(path, allow_pickle=False) as entry:
        if str(entry['image']) != name or not np.array_equal(entry['active_ids'], active):
            raise ValueError(f'Feature/region identity mismatch: {name}')
        visual = entry['visual'].copy()
    if visual.shape != (len(active), 2048) or not np.isfinite(visual).all():
        raise ValueError(f'Invalid visual features: {name}')
    return visual
