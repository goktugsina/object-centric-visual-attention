"""Reuse the recorded neural heads without fitting or target-data calibration."""
from pathlib import Path
import numpy as np
import torch
from attention.relational import RegionNetwork
from attention.visual import sha256


def load_heads(root, seeds=(11, 22, 33)):
    heads = {}; hashes = {}
    for variant in ('full_self', 'full_gat'):
        for seed in seeds:
            name = f'{variant}_{seed}'
            path = Path(root)/f'output/relational_checkpoints/relational/detected_{name}.pt'
            trained = torch.load(path, map_location='cpu', weights_only=True)
            if trained['arm'] != 'detected' or trained['feature_kind'] != 'full':
                raise ValueError('Unexpected transfer checkpoint')
            model = RegionNetwork(trained['input_dim'], trained['hidden'], trained['relation'])
            model.load_state_dict(trained['state_dict']); model.eval().requires_grad_(False)
            heads[name] = (model, trained); hashes[name] = sha256(path)
    return heads, hashes


@torch.inference_mode()
def head_probabilities(visual, auxiliary, heads):
    x = torch.tensor(np.concatenate([visual, auxiliary], axis=1).astype(np.float32))
    result = {}
    for name, (model, trained) in heads.items():
        normalized = (x-trained['mean'])/trained['std']
        valid = torch.ones((1, len(x)), dtype=torch.bool)
        p = model(normalized[None], valid).softmax(-1)[0].numpy().astype(float)
        p = np.maximum(p, 1e-12); result[name] = p/p.sum()
    return result
