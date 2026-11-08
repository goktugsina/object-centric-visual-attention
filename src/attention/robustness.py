"""Controlled feature interventions and explicit observer reweighting."""
import numpy as np


def observer_target(counts):
    counts = np.asarray(counts, float)
    valid = counts.sum(axis=1) > 0
    if not valid.any():
        raise ValueError('No observer has eligible events')
    return (counts[valid] / counts[valid].sum(axis=1, keepdims=True)).mean(axis=0)


def project(scenes, columns):
    return [dict(s, features=s['features'][:, columns].copy()) for s in scenes]


def permute_semantics(scenes, seed):
    rng = np.random.default_rng(seed)
    output = []
    for s in sorted(scenes, key=lambda x: x['name']):
        x = s['features'].copy()
        rows = np.flatnonzero(s['active_ids'] != 0)
        x[rows, 6:] = x[rng.permutation(rows), 6:]
        output.append(dict(s, features=x))
    return output


def omit_semantics(scenes, probability, seed):
    if not 0 <= probability <= 1:
        raise ValueError('Omission probability must be in [0,1]')
    rng = np.random.default_rng(seed)
    output = []
    for s in scenes:
        x = s['features'].copy()
        selected = (rng.random(len(x)) < probability) & (s['active_ids'] != 0)
        x[selected, 6:] = 0
        output.append(dict(s, features=x))
    return output
