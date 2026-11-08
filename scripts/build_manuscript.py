"""Resolve manuscript numbers directly from committed experiment outputs."""
import json
import re
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'Times New Roman', 'mathtext.fontset': 'stix'})

ROOT=Path(__file__).resolve().parents[1]
read=lambda name:json.loads((ROOT/'results'/name/'summary.json').read_text())
d=read('primary'); drop=read('drop_first'); near=read('primary_nearest')
a=json.loads((ROOT/'results/primary/audit.json').read_text())
m=d['models']; pairs=d['paired']
g=pairs['semantics_over_geometry']; tg=pairs['semantics_calibrated_over_semantics']; eg=pairs['ensemble_over_semantics']
fmt=lambda v:f'{v:.3f}'
ci=lambda v:f'[{v[0]:.3f}, {v[1]:.3f}]'
ci6=lambda v:f'[{v[0]:.6f}, {v[1]:.6f}]'
values={
 'retained':f"{a['retained_fixations']:,}", 'raw_fixations':f"{a['raw_fixations']:,}",
 'active_objects':f"{a['active_objects']:,}", 'duration_rejected':f"{a['rejected_duration']:,}",
 'background_pct':f"{100*a['background_fixations']/a['retained_fixations']:.2f}",
 'overlap_pct':f"{100*a['overlap_fixations']/a['retained_fixations']:.2f}",
 'geometry_nll':fmt(m['geometry']['nll_bits']), 'semantics_nll':fmt(m['semantics']['nll_bits']),
 'gain':fmt(g['improvement_bits']), 'gain_ci':ci(g['ci95']),
 'geometry_top1':f"{100*m['geometry']['top1']:.2f}", 'semantics_top1':f"{100*m['semantics']['top1']:.2f}",
 'top1_gain':f"{100*(m['semantics']['top1']-m['geometry']['top1']):.2f}",
 'geometry_brier':fmt(m['geometry']['brier']), 'semantics_brier':fmt(m['semantics']['brier']),
 'train_fixations':f"{d['split_fixations']['train']:,}", 'val_fixations':f"{d['split_fixations']['validation']:,}",
 'test_fixations':f"{d['split_fixations']['test']:,}", 'l2':str(d['selection']['semantics']['l2']),
 'bandwidth':str(d['selection']['center']['bandwidth']),
 'temperature':f"{d['selection']['semantics_temperature']:.4f}",
 'temperature_gain':f"{tg['improvement_bits']:.6f}", 'temperature_ci':ci6(tg['ci95']),
 'ensemble_gain':f"{eg['improvement_bits']:.6f}", 'ensemble_ci':ci6(eg['ci95']),
 'disagreement':f"{d['mean_ensemble_disagreement_bits']:.5f}",
 'nearest_gain':fmt(near['paired']['semantics_over_geometry']['improvement_bits']), 'test_count':'Eighteen',
}
labels={'uniform_regions':'Uniform regions','area':'Area prior','center':'Center prior','geometry':'Geometry',
 'semantics':'Semantics','semantics_calibrated':'Semantics plus temperature','ensemble':'Bootstrap ensemble',
 'ensemble_calibrated':'Ensemble plus temperature'}
lines=['| Model | NLL bits | Brier | Top 1 percent | ECE |','|---|---|---|---|---|']
for k,label in labels.items():
    v=m[k];lines.append(f"| {label} | {v['nll_bits']:.4f} | {v['brier']:.4f} | {100*v['top1']:.2f} | {v['calibration']['ece']:.4f} |")
values['results_table']='\n'.join(lines)
ng=near['paired']['semantics_over_geometry']; dg=drop['paired']['semantics_over_geometry']
values['sensitivity_text']=(f"Removing the first recorded fixation retains {sum(drop['split_fixations'].values()):,} events and yields a semantic log-score improvement of {fmt(dg['improvement_bits'])} bits, with a paired 95 percent interval of {ci(dg['ci95'])}. "
 f"The exploratory nearest-centroid partition yields an improvement of {fmt(ng['improvement_bits'])} bits, with an interval of {ci(ng['ci95'])}. "
 "Both contrasts favor semantic features. The magnitude depends on how nested and overlapping regions are defined, so the primary estimate should not be interpreted independently of its assignment policy.")
template=(ROOT/'paper/manuscript.template.md').read_text(encoding='utf-8')
values.update(json.loads((ROOT/'paper/extension_fields.json').read_text(encoding='utf-8')))
values.update(json.loads((ROOT/'paper/continuation_fields.json').read_text(encoding='utf-8')))
for k,v in values.items():template=template.replace('{{'+k+'}}',v)
if re.search(r'\{\{.*?\}\}',template):raise ValueError('Unresolved manuscript field')
# Number the IEEE bibliography by first citation rather than by source ID.
body = template.split('## References')[0]
refs = json.loads((ROOT/'paper/references_ieee.json').read_text(encoding='utf-8'))
mapping = {}
def citation(match):
    ids = []
    for group in match.group(1).split(','):
        limits = [int(x) for x in group.split('-')]
        ids.extend(range(limits[0], limits[-1] + 1))
    for old in ids:
        if old not in mapping:
            mapping[old] = len(mapping) + 1
    return ', '.join(f'[{mapping[old]}]' for old in ids)
body = re.sub(r'\[(\d+(?:[-,]\d+)*)\]', citation, body)
assert len(mapping) == len(refs)
body = re.sub(r'^## A reproducible.*\n', '', body, flags=re.M)
body = re.sub(r'^\d+ November 2026.*\n', '', body, flags=re.M)
body = body.replace('Table 1', 'Table I').replace('Table 2', 'Table II')
figure_ids = re.findall(r'!\[Figure (\d+)', body)
figure_order = {old: str(i+1) for i,old in enumerate(figure_ids)}
body = re.sub(r'Figure (\d+)', lambda m: 'Fig. ' + figure_order[m.group(1)], body)
body = body.replace('Semantics plus temperature', 'Semantics + T')
body = body.replace('Ensemble plus temperature', 'Ensemble + T')
body = body.replace('| NLL bits | Brier | Top 1 percent | ECE |', '| NLL | Brier | Top 1 (%) | ECE |')
body = body.replace('Bootstrap ensemble |', 'Ensemble |')
body = body.replace('Every measure first averages', 'T denotes temperature scaling. Every measure first averages')
body += '## References\n\n' + '\n\n'.join(f'[{new}] {refs[str(old)]}' for old,new in mapping.items()) + '\n'
(ROOT/'paper/manuscript.md').write_text(body,encoding='utf-8')
equations={
 'softmax':r'$p_{ij}=\frac{\exp(\theta^{\mathsf{T}}x_{ij})}{\sum_{k\in\mathcal{R}_i}\exp(\theta^{\mathsf{T}}x_{ik})}$',
 'nll':r'$\mathrm{NLL}= -\frac{1}{N}\sum_{i=1}^{N}\sum_{j\in\mathcal{R}_i}q_{ij}\log_2 p_{ij}$',
 'brier':r'$\mathrm{BS}_i=1+\sum_j p_{ij}^{2}-2\sum_j q_{ij}p_{ij}$',
 'ig':r'$\mathrm{IG}_i=\sum_j q_{ij}\log_2\!\left(\frac{p_{ij}}{a_{ij}}\right)$',
}
for name,eq in equations.items():
    fig=plt.figure(figsize=(6,.65));fig.text(.5,.5,eq,ha='center',va='center',fontsize=15)
    fig.savefig(ROOT/f'paper/figures/eq_{name}.png',dpi=240,bbox_inches='tight',pad_inches=.08);plt.close(fig)
print('Resolved manuscript and equations')
