"""Validate coordinates against author binary maps and screen source overlap."""
import io
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile
from collections import Counter

import numpy as np
from PIL import Image
from scipy.fft import dctn
from scipy.io import loadmat
from attention.emod import load_emod

ROOT = Path(__file__).resolve().parents[1]


def signatures(path):
    with Image.open(path) as source:
        image = source.convert('RGB'); rgb = np.asarray(image)
        thumb = image.resize((32, 32), Image.Resampling.LANCZOS)
        gray = np.asarray(thumb.convert('L'), float)
    coefficients = dctn(gray, norm='ortho')[:8, :8].ravel()
    bits = coefficients > np.median(coefficients[1:])
    phash = int.from_bytes(np.packbits(bits).tobytes(), 'big')
    return dict(file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                rgb_sha256=hashlib.sha256(str(rgb.shape).encode()+rgb.tobytes()).hexdigest(),
                thumbnail_sha256=hashlib.sha256(np.asarray(thumb).tobytes()).hexdigest(), phash=phash)


def main():
    root = ROOT/'data/raw/emod'; out = ROOT/'results/emod_audit'; out.mkdir(parents=True, exist_ok=True)
    data, audit = load_emod(root)
    records = loadmat(root/'fixations1019_renamed.mat', simplify_cells=True)['fixations1019']
    coordinate_rows = []; total_hits = Counter(); total_valid = Counter()
    with ZipFile(root/'FixationMap.zip') as archive:
        for record in records:
            name = record['img']
            if name not in data: continue
            binary = np.asarray(Image.open(io.BytesIO(archive.read('FixationMap/Binary_map/'+name)))) > 128
            rows = np.concatenate([np.atleast_1d(s['fix_x']) for s in record['subjects']])
            cols = np.concatenate([np.atleast_1d(s['fix_y']) for s in record['subjects']])
            score = {}
            for orientation, x, y in [('source_names', rows, cols), ('swapped', cols, rows)]:
                for rule in ('round1', 'floor0', 'round0', 'ceil1'):
                    def convert(v):
                        return (np.floor(v+.5)-1 if rule == 'round1' else np.floor(v) if rule == 'floor0'
                                else np.floor(v+.5) if rule == 'round0' else np.ceil(v)-1).astype(int)
                    px, py = convert(x), convert(y)
                    valid = np.isfinite(x) & np.isfinite(y) & (px >= 0) & (px < binary.shape[1]) & (py >= 0) & (py < binary.shape[0])
                    key = orientation+'_'+rule
                    hits = int(binary[py[valid], px[valid]].sum())
                    total_hits[key] += hits; total_valid[key] += int(valid.sum())
                    score[key] = hits/max(1, int(valid.sum()))
            coordinate_rows.append(dict(image=name, **score))
    coord = {k: dict(binary_map_hit_fraction=total_hits[k]/total_valid[k], valid_events=total_valid[k]) for k in total_hits}
    if coord['swapped_round1']['binary_map_hit_fraction'] < .98:
        raise ValueError('Coordinate mapping does not match official fixation maps')
    source = {p.name: signatures(p) for p in sorted((ROOT/'data/raw/stimuli').glob('*.jpg'))}
    targets = {}; excluded = []
    for name in sorted(data):
        signature = signatures(root/'stimuli'/name)
        near = []
        for source_name, other in source.items():
            distance = (signature['phash'] ^ other['phash']).bit_count()
            exact = [key for key in ('file_sha256', 'rgb_sha256', 'thumbnail_sha256') if signature[key] == other[key]]
            if exact or distance <= 4:
                near.append(dict(osie_image=source_name, phash_distance=distance, exact_match_types=exact))
        signature['phash'] = f'{signature["phash"]:016x}'
        targets[name] = dict(**signature, osie_candidates=near)
        if near: excluded.append(name)
    labels = loadmat(root/'AllLabelData1019.mat', simplify_cells=True)['AllLabelData']
    object_count = 0; fields = Counter()
    for record in labels:
        objects = record['annotation'].get('object', [])
        if isinstance(objects, dict): objects = [objects]
        object_count += len(objects)
        for obj in objects: fields.update(obj.keys())
    result = dict(status='complete', data=audit, coordinate_validation=coord,
                  coordinate_validation_images=len(coordinate_rows),
                  annotation_schema=dict(images=len(labels), objects=object_count, object_fields=dict(fields),
                                         semantics='LabelMe polygons, category names and sentiment; not OSIE 12 attributes'),
                  overlap_rule='Exclude exact/file/RGB/32x32 matches or 64-bit pHash distance <=4 against all 700 OSIE images',
                  excluded_images=excluded, primary_images=sorted(set(data)-set(excluded)),
                  primary_image_count=len(data)-len(excluded), images=targets,
                  sources=json.loads((root/'download_manifest.json').read_text()))
    (out/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    (out/'coordinate_checks.json').write_text(json.dumps(coordinate_rows, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('images', 'primary_images', 'sources', 'data')}, indent=2))


if __name__ == '__main__':
    main()
