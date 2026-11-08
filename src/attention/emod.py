"""EMOd loader: published fix_x is row, fix_y is column (binary-map audited)."""
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image
from scipy.io import loadmat

from attention.data import fixation_pixels


def load_emod(root):
    root = Path(root)
    records = loadmat(root/'fixations1019_renamed.mat', simplify_cells=True)['fixations1019']
    audit = Counter(); output = {}; missing = []
    for record in records:
        name = str(record['img']); path = root/'stimuli'/name
        if not path.exists():
            missing.append(name); continue
        with Image.open(path) as image:
            width, height = image.size
        subjects = []
        for subject in record['subjects']:
            # Source names are misleading: fix_y is horizontal, fix_x vertical.
            x, y, duration = [np.atleast_1d(subject[k]) for k in ('fix_y', 'fix_x', 'fix_duration')]
            if not len(x) == len(y) == len(duration):
                raise ValueError(f'Unequal fixation lengths: {name}')
            audit['raw_fixations'] += len(x); audit['subject_records'] += 1
            keep = np.isfinite(duration) & (duration >= 100)
            audit['rejected_duration'] += int((~keep).sum())
            px, py, valid = fixation_pixels(x[keep], y[keep], width, height)
            audit['rejected_coordinates'] += int((~valid).sum())
            audit['retained_fixations'] += len(px)
            subjects.append((px, py))
        if not sum(len(x) for x, y in subjects):
            raise ValueError(f'No eligible events: {name}')
        output[name] = dict(subjects=subjects, width=width, height=height)
    return output, dict(audit, available_images=len(output), missing_image_count=len(missing),
                        missing_images=missing, annotation_images=len(records),
                        coordinates='horizontal=fix_y, vertical=fix_x; one-based round-half-up; reject outside',
                        subject_identity='source subject-array positions; identity across missing records not assumed')
