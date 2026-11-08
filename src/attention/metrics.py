"""Proper scores for a random fixation event, averaged equally over images."""
import numpy as np


def entropy(p):
    p = np.asarray(p, float)
    return float(-np.sum(p[p > 0] * np.log2(p[p > 0])))


def scores(q, p):
    q, p = np.asarray(q, float), np.asarray(p, float)
    if q.shape != p.shape or np.any(q < 0) or np.any(p <= 0):
        raise ValueError("Probabilities must share support and predictions be positive")
    if not np.isclose(q.sum(), 1) or not np.isclose(p.sum(), 1):
        raise ValueError("Probabilities must sum to one")
    ranking = np.argsort(-p, kind="stable")
    return dict(nll_bits=float(-q @ np.log2(p)),
                brier=float(1 + p @ p - 2 * q @ p),
                top1=float(q[ranking[0]]), top3=float(q[ranking[:3]].sum()),
                confidence=float(p[ranking[0]]),
                predictive_entropy=entropy(p), empirical_entropy=entropy(q))


def bootstrap_mean_ci(values, seed=20261102, samples=5000):
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(samples, len(values)))].mean(axis=1)
    return [float(x) for x in np.quantile(means, [.025, .975])]


def calibration(rows, bins=10):
    """Image-balanced top-label ECE using observed top-label fixation fractions."""
    conf = np.array([r["confidence"] for r in rows])
    actual = np.array([r["top1"] for r in rows])
    assignments = np.minimum((conf * bins).astype(int), bins - 1)
    table, ece = [], 0.
    for i in range(bins):
        use = assignments == i
        if use.any():
            c, a = float(conf[use].mean()), float(actual[use].mean())
            ece += use.mean() * abs(c - a)
            table.append(dict(bin=i, images=int(use.sum()), confidence=c, observed=a))
    return dict(ece=float(ece), bins=table)
