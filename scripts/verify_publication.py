"""Audit saved aggregate scores and manuscript metadata without retraining models.

Run from any directory with Python's standard library. This verifies arithmetic
from existing per-image CSVs, not the upstream data or independent reproduction.
"""
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]


def main():
    checks = []
    inputs = {}

    def read(relative):
        data = (ROOT / relative).read_bytes()
        inputs[relative] = hashlib.sha256(data).hexdigest()
        return data.decode('utf-8-sig')

    def require(condition, description):
        if not condition:
            raise ValueError(description)
        checks.append(description)

    def close(actual, expected, description):
        require(math.isfinite(actual) and abs(actual - expected) < 1e-10, description)

    facts = {}
    metrics = ('ig_uniform_bits', 'pixel_nll_bits', 'equal_observer_ig_bits',
               'nss', 'auc_judd', 'cc', 'kl_bits')
    for folder, expected_images, expected_models in (
        ('saliency_comparison', 140, 7), ('emod_transfer', 698, 9)
    ):
        summary = json.loads(read(f'results/{folder}/summary.json'))
        require(summary['status'] == 'complete', f'{folder}: completed result')
        rows = list(csv.DictReader(read(f'results/{folder}/per_image_metrics.csv').splitlines()))
        groups = defaultdict(list)
        for row in rows:
            groups[row['model']].append(row)
        require(len(groups) == expected_models, f'{folder}: model count')
        reference_images = {row['image'] for row in rows}
        require(len(reference_images) == expected_images, f'{folder}: image count')
        recorded = summary['models'] if folder == 'saliency_comparison' else summary['evaluation']['primary_nonoverlap']['models']
        for model, group in groups.items():
            require(len(group) == expected_images and {r['image'] for r in group} == reference_images,
                    f'{folder}/{model}: exactly one row per shared image')
            for metric in metrics:
                actual = mean(float(row[metric]) for row in group)
                close(actual, recorded[model][metric], f'{folder}/{model}/{metric}: CSV mean equals summary')
            facts[f'{folder}/{model}/ig_uniform_bits'] = recorded[model]['ig_uniform_bits']
        if folder == 'emod_transfer':
            require(all(r['primary_nonoverlap'] == 'True' for r in rows), 'EMOd: all reported images in primary nonoverlap subset')
            for family in ('full_self', 'full_gat'):
                actual = mean(recorded[f'{family}_{seed}']['ig_uniform_bits'] for seed in (11, 22, 33))
                close(actual, recorded[f'{family}_seed_mean']['ig_uniform_bits'], f'EMOd/{family}: mean of seed scores')

    observer = json.loads(read('results/observer_holdout/summary.json'))
    rows = list(csv.DictReader(read('results/observer_holdout/per_image_metrics.csv').splitlines()))
    require(len(rows) == 1680, 'Observer holdout: 1680 rows')
    indexed = {(r['arm'], r['fold'], r['image'], r['model']): r for r in rows}
    require(len(indexed) == len(rows), 'Observer holdout: unique fold/image/model rows')
    images = {r['image'] for r in rows}
    require(len(images) == 140, 'Observer holdout: 140 shared images')
    for arm in ('annotated', 'detected'):
        gains, full_ig = [], []
        for image in sorted(images):
            fold_gains, fold_ig = [], []
            for fold in ('0', '1', '2'):
                geometry = indexed[arm, fold, image, 'geometry']
                full = indexed[arm, fold, image, 'full']
                fold_gains.append(float(geometry['nll_bits']) - float(full['nll_bits']))
                fold_ig.append(float(full['ig_uniform_bits']))
            gains.append(mean(fold_gains))
            full_ig.append(mean(fold_ig))
        close(mean(gains), observer['aggregate'][arm]['full_over_geometry_bits'], f'Observers/{arm}: paired fold-then-image mean gain')
        close(mean(full_ig), observer['aggregate'][arm]['full_ig_bits'], f'Observers/{arm}: full-model IG')
        facts[f'observer_holdout/{arm}/gain_bits'] = mean(gains)

    manuscript = read('paper/manuscript.md')
    metadata = json.loads(read('output/submission/metadata.json'))
    details = json.loads(read('paper/submission_details.json'))
    require(metadata['publication_route'] == details['publication_route'], 'Metadata: selected publication route preserved')
    require(metadata['authors'] == [{'name': details['author_name'], 'affiliation': details['affiliation'], 'orcid': details['orcid']}], 'Metadata: author details preserved')
    require(metadata['license'] == details['license'], 'Metadata: license choice preserved without an invented default')
    body, references = manuscript.split('## References', 1)
    require('{{' not in manuscript, 'Manuscript: no unresolved template fields')
    require(manuscript.splitlines()[0].removeprefix('# ').strip() == metadata['title'], 'Metadata: title matches manuscript')
    abstract = manuscript.split('### Abstract', 1)[1].split('Keywords:', 1)[0].strip()
    require(abstract == metadata['abstract'], 'Metadata: abstract matches manuscript')
    citation_ids = list(dict.fromkeys(int(x) for x in re.findall(r'\[(\d+)\]', body)))
    bibliography_ids = [int(x) for x in re.findall(r'^\[(\d+)\]', references, re.M)]
    require(citation_ids == bibliography_ids == list(range(1, 25)), 'References: all 24 entries cited in first-use order')
    pdf_path = metadata['pdf']
    pdf_hash = hashlib.sha256((ROOT / pdf_path).read_bytes()).hexdigest()
    inputs[pdf_path] = pdf_hash
    require(pdf_hash == metadata['pdf_sha256'], 'PDF: checksum matches submission metadata')
    require(metadata['status'] == 'local_draft_not_submitted', 'Submission: draft status remains explicit')
    for key in ('saliency_comparison/deepgaze_iie/ig_uniform_bits',
                'saliency_comparison/detected_linear/ig_uniform_bits',
                'emod_transfer/center/ig_uniform_bits', 'emod_transfer/linear/ig_uniform_bits',
                'observer_holdout/annotated/gain_bits', 'observer_holdout/detected/gain_bits'):
        require(f'{facts[key]:.3f}' in body, f'Manuscript contains rounded headline value: {key}')

    result = {'status': 'complete', 'checks_passed': len(checks), 'checks': checks,
              'headline_values': facts, 'input_sha256': inputs,
              'scope': 'Saved CSV aggregate arithmetic, observer pairing, citation numbering and manuscript/PDF metadata consistency. Does not rerun models, bootstrap intervals, PDF rendering, full bibliography verification or remote CI; does not establish submission approval.'}
    out = ROOT / 'output/submission/publication_verification.json'
    out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'checks_passed': len(checks), 'report': str(out)}))


if __name__ == '__main__':
    main()
