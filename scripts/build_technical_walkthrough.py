"""Build the explanatory report's numeric tables from saved experimental records."""
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |',
                      '| '+' | '.join(['---']*len(headers))+' |'] +
                     ['| '+' | '.join(map(str, row))+' |' for row in rows])


def main():
    read = lambda name: json.loads((ROOT/'results'/name/'summary.json').read_text(encoding='utf-8'))
    primary, neural, saliency, transfer, observer = [read(p) for p in
        ('primary', 'relational', 'saliency_comparison', 'emod_transfer', 'observer_holdout')]
    for result in (neural, saliency, transfer, observer):
        assert result['status'] == 'complete'
    rows = []
    for name, label in [('uniform_regions','Uniform regions'), ('area','Area prior'),
                         ('center','Integrated center prior'), ('geometry','Geometry'),
                         ('semantics','Semantics')]:
        r = primary['models'][name]
        rows.append([label, f"{r['nll_bits']:.4f}", f"{r['brier']:.4f}",
                     f"{100*r['top1']:.2f}%", f"{100*r['top3']:.2f}%"])
    fields = {'PRIMARY_TABLE':table(['Model','Region NLL bits','Brier','Top-1 coverage','Top-3 coverage'], rows)}
    rows = []; epochs = []
    for arm, r in neural['arms'].items():
        for variant, model in r['models'].items():
            s = model['mean_test']
            rows.append([arm, variant, f"{s['nll_bits']:.4f}", f"{s['ig_uniform_bits']:.4f}",
                         f"{s['ndcg3']:.4f}", f"{model['test_nll_seed_std']:.4f}"])
            epochs.append([arm, variant, *[run['selected_epoch'] for run in model['runs']]])
    fields['NEURAL_TABLE'] = table(['Partition','Model','Mean region NLL','Mean pixel IG','NDCG@3','NLL seed SD'], rows)
    fields['EPOCH_TABLE'] = table(['Partition','Model','Seed 11','Seed 22','Seed 33'], epochs)
    rows = []
    for key, label in [('uniform','Uniform'), ('center','Center'), ('detected_linear','Detected linear'),
                        ('full_self_11','Detected local, seed 11'), ('full_gat_11','Detected GAT, seed 11'),
                        ('human_semantic_linear','Human semantic linear'), ('deepgaze_iie','DeepGaze IIE')]:
        r = saliency['models'][key]
        rows.append([label]+[f'{r[k]:.4f}' for k in ('ig_uniform_bits','nss','auc_judd','cc','kl_bits')])
    fields['SALIENCY_TABLE'] = table(['Predictor','IG bits','NSS','AUC-Judd','CC','KL bits'], rows)
    r = transfer['evaluation']['primary_nonoverlap']['models']
    rows = []
    for key, label in [('uniform','Uniform'), ('center','Center'), ('linear','Detected linear'),
                        ('full_self_seed_mean','Local, mean of three seed scores'),
                        ('full_gat_seed_mean','GAT, mean of three seed scores')]:
        rows.append([label, f"{r[key]['ig_uniform_bits']:.6f}"])
    fields['TRANSFER_TABLE'] = table(['Predictor on 698 EMOd images','IG bits'], rows)
    rows = []
    for arm, r in observer['aggregate'].items():
        rows.append([arm, f"{r['full_ig_bits']:.6f}", f"{r['full_over_geometry_bits']:.6f}",
                     '['+', '.join(f'{v:.6f}' for v in r['image_ci95'])+']'])
    fields['OBSERVER_TABLE'] = table(['Partition','Full-model IG','Full-minus-reference gain bits','Paired image interval'], rows)
    coefficients = json.loads((ROOT/'results/primary/coefficients.json').read_text())['semantics']
    attributes = json.loads((ROOT/'results/primary/audit.json').read_text())['semantic_attributes']
    labels = ['Log area','Centered x','Centered y','Centroid x squared','Centroid y squared','Background']+attributes
    fields['COEFFICIENT_TABLE'] = table(['Feature','Coefficient','exp(coefficient)'],
        [[label, f'{value:.6f}', f'{math.exp(value):.4f}'] for label,value in zip(labels, coefficients)])
    report = (ROOT/'research/technical_walkthrough.template.md').read_text(encoding='utf-8')
    headings = re.findall(r'^## (.+)$', report, re.M)
    fields['CONTENTS'] = 'Contents\n\n' + '\n'.join(
        f"- [{heading}](#{re.sub(r'[^a-z0-9 -]', '', heading.lower()).replace(' ', '-')})"
        for heading in headings)
    for key, value in fields.items(): report = report.replace('{{'+key+'}}', value)
    assert not re.search(r'\{\{.*?\}\}', report), 'Unresolved report field'
    destination = ROOT/'research/technical_walkthrough.md'
    destination.write_text(report, encoding='utf-8', newline='\n')
    links = re.findall(r'\[[^\]]*\]\(([^)]+)\)', report)
    for link in links:
        if not link.startswith(('https://','http://','#')):
            assert (destination.parent/link.split('#')[0]).exists(), f'Missing local reference: {link}'
    print(f'Built {destination}; {len(report.split()):,} whitespace-separated words; {len(links)} links and anchors checked')


if __name__ == '__main__': main()
