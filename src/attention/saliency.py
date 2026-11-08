"""Common-pixel saliency scoring with explicit bandwidth and tie conventions."""
import numpy as np
from scipy.ndimage import gaussian_filter


def normalize_density(density):
    p = np.asarray(density, dtype=float)
    if p.ndim != 2 or not np.isfinite(p).all() or np.any(p < 0) or p.sum() <= 0:
        raise ValueError('Expected a finite nonnegative two-dimensional density')
    p = np.maximum(p, 1e-300)
    return p/p.sum()


def fixation_target(subjects, width, height, sigma_fraction=.03):
    usable = [(np.asarray(x, int), np.asarray(y, int)) for x, y in subjects if len(x)]
    if not usable:
        raise ValueError('No eligible fixation events')
    x = np.concatenate([s[0] for s in usable]); y = np.concatenate([s[1] for s in usable])
    if np.any((x < 0) | (x >= width) | (y < 0) | (y >= height)):
        raise ValueError('Fixation outside pixel support')
    histogram = np.bincount(y*width+x, minlength=width*height).reshape(height, width).astype(float)
    empirical = gaussian_filter(histogram, sigma=sigma_fraction*min(height, width), mode='reflect')
    return dict(x=x, y=y, binary=histogram > 0, density=normalize_density(empirical),
                subjects=usable, sigma_pixels=sigma_fraction*min(height, width))


def auc_judd(density, binary):
    """Positive-pixel thresholds, deterministic ties, endpoints; no jitter."""
    p = np.asarray(density); mask = np.asarray(binary, bool)
    if p.shape != mask.shape:
        raise ValueError('AUC support mismatch')
    positives = p[mask]; negatives = p[~mask]
    if not len(positives) or not len(negatives):
        return None
    thresholds = np.unique(positives)[::-1]
    positive_sorted = np.sort(positives); negative_sorted = np.sort(negatives)
    tp = (len(positives)-np.searchsorted(positive_sorted, thresholds, side='left'))/len(positives)
    fp = (len(negatives)-np.searchsorted(negative_sorted, thresholds, side='left'))/len(negatives)
    return float(np.trapezoid(np.r_[0., tp, 1.], np.r_[0., fp, 1.]))


def pixel_scores(density, target):
    p = normalize_density(density)
    q = target['density']
    if p.shape != q.shape:
        raise ValueError('Target/prediction dimensions differ')
    x, y = target['x'], target['y']
    ig_map = np.log2(p*p.size)
    std = float(p.std())
    nss = float(((p[y, x]-p.mean())/std).mean()) if std > 1e-15 else 0.
    centered, qcentered = p-p.mean(), q-q.mean()
    denominator = np.sqrt(np.sum(centered**2)*np.sum(qcentered**2))
    cc = float(np.sum(centered*qcentered)/denominator) if denominator > 1e-20 else 0.
    return dict(ig_uniform_bits=float(ig_map[y, x].mean()),
                pixel_nll_bits=float(-np.log2(p[y, x]).mean()),
                equal_observer_ig_bits=float(np.mean([ig_map[sy, sx].mean() for sx, sy in target['subjects']])),
                nss=nss, auc_judd=auc_judd(p, target['binary']), cc=cc,
                kl_bits=float(np.sum(q*(np.log2(q)-np.log2(p)))))
