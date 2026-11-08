"""OSIE loading, MATLAB coordinates, disjoint regions, and image-only splits."""
from pathlib import Path
import hashlib
import json
import h5py
import numpy as np
from scipy.io import loadmat


def partition_masks(masks, policy='smallest'):
    """Background=0; overlaps go to smallest original mask, then source index.

    Produces disjoint regions so both fixation labels and integrated priors have
    exactly the same support. Empty regions are removed downstream.
    """
    masks = np.asarray(masks, dtype=bool)
    if masks.ndim != 3 or len(masks) == 0:
        raise ValueError("Expected nonempty [objects, height, width] masks")
    if policy not in ('smallest', 'nearest'):
        raise ValueError('Unknown overlap policy')
    areas = masks.sum(axis=(1, 2))
    labels = np.zeros(masks.shape[1:], dtype=np.int32)
    if policy == 'nearest':
        yy, xx = np.mgrid[:masks.shape[1], :masks.shape[2]]
        best = np.full(labels.shape, np.inf)
        for i, mask in enumerate(masks):
            if not mask.any():
                continue
            distance = (xx-xx[mask].mean())**2 + (yy-yy[mask].mean())**2
            use = mask & (distance < best)
            labels[use], best[use] = i+1, distance[use]
        return labels
    order = sorted(range(len(masks)), key=lambda i: (-int(areas[i]), -i))
    for i in order:
        labels[masks[i]] = i + 1
    return labels


def fixation_pixels(x, y, width, height):
    """Reject nonfinite/out-of-range continuous coordinates; round half up.

    OSIE/MATLAB coordinates are one based. We do not clip off-image samples
    onto the border. Valid samples are converted to zero-based pixel indices.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    valid = np.isfinite(x) & np.isfinite(y) & (x >= 1) & (x <= width) & (y >= 1) & (y <= height)
    return (np.floor(x[valid] + .5).astype(int) - 1,
            np.floor(y[valid] + .5).astype(int) - 1, valid)


def split_images(names, seed=20261102):
    names = np.array(sorted(names))
    ids = np.random.default_rng(seed).permutation(len(names))
    a, b = int(.6 * len(names)), int(.8 * len(names))
    return {k: sorted(names[v].tolist()) for k, v in
            zip(("train", "validation", "test"), (ids[:a], ids[a:b], ids[b:]))}


def load_osie(raw_dir, drop_first=False, overlap_policy='smallest'):
    raw_dir = Path(raw_dir)
    fdata = loadmat(raw_dir / "fixations.mat", simplify_cells=True)["fixations"]
    fixations = {str(item["img"]): item["subjects"] for item in fdata}
    scenes = []
    audit = dict(images=0, annotated_objects=0, active_objects=0, subject_records=0,
                 raw_fixations=0, dropped_first=0, rejected_duration=0,
                 rejected_coordinates=0, retained_fixations=0, background_fixations=0,
                 overlap_fixations=0, overlap_pixels=0, total_pixels=0)
    with h5py.File(raw_dir / "attrs.mat", "r") as data:
        def chars(ds):
            return "".join(chr(int(c)) for c in ds[:].ravel())
        attributes = [chars(data[r]) for r in data["attrNames"][:].ravel()]
        for ref in data["attrs"][:].ravel():
            record = data[ref]
            name = chars(record["img"])
            objects = [data[r] for r in record["objs"][:].ravel()]
            # MATLAB HDF5 reverses dimension order: stored [width,height].
            masks = np.array([obj["map"][:].T.astype(bool) for obj in objects])
            semantic = np.array([obj["features"][:].ravel() for obj in objects])
            labels = partition_masks(masks, overlap_policy)
            overlap = masks.sum(axis=0) > 1
            height, width = labels.shape
            full_area = np.bincount(labels.ravel(), minlength=len(objects)+1)
            active = np.flatnonzero(full_area > 0)
            yy, xx = np.mgrid[:height, :width]
            nx, ny = (xx + .5) / width - .5, (yy + .5) / height - .5
            area = full_area[active] / labels.size
            cx = np.bincount(labels.ravel(), weights=nx.ravel(), minlength=len(full_area))[active] / full_area[active]
            cy = np.bincount(labels.ravel(), weights=ny.ravel(), minlength=len(full_area))[active] / full_area[active]
            features = np.column_stack([np.log(area), cx, cy, cx**2, cy**2, active == 0,
                                        np.vstack([np.zeros(12), semantic])[active]])
            priors = {}
            for sigma in (.10, .15, .20, .30, .40):
                weight = np.exp(-(nx**2 + ny**2) / (2*sigma*sigma))
                mass = np.bincount(labels.ravel(), weights=weight.ravel(), minlength=len(full_area))[active]
                priors[str(sigma)] = mass / mass.sum()
            per_subject = []
            subjects = fixations.pop(name)
            for subject in subjects:
                x, y, duration = [np.atleast_1d(subject[k]) for k in ("fix_x", "fix_y", "fix_duration")]
                if not len(x) == len(y) == len(duration):
                    raise ValueError(f"Unequal fixation lengths in {name}")
                audit["raw_fixations"] += len(x)
                audit["subject_records"] += 1
                if drop_first and len(x):
                    x, y, duration = x[1:], y[1:], duration[1:]
                    audit["dropped_first"] += 1
                keep_duration = np.isfinite(duration) & (duration >= 100)
                audit["rejected_duration"] += int((~keep_duration).sum())
                px, py, valid = fixation_pixels(x[keep_duration], y[keep_duration], width, height)
                audit["rejected_coordinates"] += int((~valid).sum())
                chosen = labels[py, px]
                per_subject.append(np.bincount(chosen, minlength=len(full_area))[active])
                audit["retained_fixations"] += len(chosen)
                audit["background_fixations"] += int((chosen == 0).sum())
                audit["overlap_fixations"] += int(overlap[py, px].sum())
            counts = np.sum(per_subject, axis=0)
            if counts.sum() == 0:
                raise ValueError(f"No eligible fixations: {name}")
            scenes.append(dict(name=name, features=features, counts=counts,
                               subject_counts=np.array(per_subject), active_ids=active,
                               area=area, center_priors=priors))
            audit["images"] += 1
            audit["annotated_objects"] += len(objects)
            audit["active_objects"] += int((active != 0).sum())
            audit["total_pixels"] += labels.size
            audit["overlap_pixels"] += int(overlap.sum())
        if fixations:
            raise ValueError("Unmatched fixation records")
    audit["semantic_attributes"] = attributes
    audit['overlap_policy'] = overlap_policy
    audit["source_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(raw_dir.glob("*.mat"))}
    return sorted(scenes, key=lambda s: s["name"]), audit
