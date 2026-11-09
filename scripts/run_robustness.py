"""Run split, semantic-feature and observer-weighting robustness experiments."""
import csv
import json
import platform
import time
from pathlib import Path
import numpy as np
import scipy
from attention.data import load_osie, split_images
from attention.model import fit, predict
from attention.metrics import scores, bootstrap_mean_ci
from attention.robustness import observer_target, project, permute_semantics, omit_semantics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/robustness'
GRID = (0., 1e-6, 1e-5, 1e-4, 1e-3, 1e-2)
SEEDS = list(range(20261102, 20261107))


def rows(scenes, ps, equal_observers=False):
    return [dict(image=s['name'], **scores(observer_target(s['subject_counts']) if equal_observers
            else s['counts']/s['counts'].sum(), p)) for s, p in zip(scenes, ps)]


def main():
    started = time.monotonic(); OUT.mkdir(parents=True, exist_ok=True)
    scenes, audit = load_osie(ROOT/'data/raw')
    print('Loaded 700 scenes', flush=True)
    (OUT/'audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    attributes = audit['semantic_attributes']
    face, text = 6+attributes.index('face'), 6+attributes.index('text')
    variants = {'geometry': list(range(6)), 'semantics': list(range(18)),
                'without_face': [i for i in range(18) if i != face],
                'without_text': [i for i in range(18) if i != text],
                'without_face_text': [i for i in range(18) if i not in (face,text)],
                'permuted_semantics': list(range(18))}
    summary = {'status':'in_progress', 'seeds':SEEDS, 'l2_grid':GRID, 'runs':[],
               'environment': {'python':platform.python_version(), 'numpy':np.__version__, 'scipy':scipy.__version__},
               'interpretation':'Exploratory; overlapping splits are not independent replications.'}
    csvrows = []
    for seed in SEEDS:
        split = split_images([s['name'] for s in scenes], seed)
        run = {'seed':seed, 'splits':split, 'models':{}, 'contrasts':{}}
        metrics = {}
        for name, columns in variants.items():
            base = permute_semantics(scenes, seed+100) if name=='permuted_semantics' else scenes
            byname = {s['name']:s for s in project(base, columns)}
            train,val,test = [[byname[n] for n in split[k]] for k in ('train','validation','test')]
            fits = {l2:fit(train,len(columns),l2) for l2 in GRID}
            selection = {l2:float(np.mean([r['nll_bits'] for r in rows(val,predict(val,theta))])) for l2,theta in fits.items()}
            chosen = min(selection,key=selection.get); theta = fits[chosen]; ps = predict(test,theta)
            model = {'columns':columns,'l2':chosen,'validation_nll':selection,'coefficients':theta.tolist(),'scores':{}}
            for target,equal in [('event',False),('observer',True)]:
                records = rows(test,ps,equal); metrics[name,target] = records
                model['scores'][target] = {k:float(np.mean([r[k] for r in records])) for k in ('nll_bits','brier','top1')}
                csvrows.extend(dict(seed=seed,model=name,target=target,**r) for r in records)
            run['models'][name] = model
            print(f'{seed} {name}: L2={chosen:g}, NLL={model["scores"]["event"]["nll_bits"]:.6f}',flush=True)
            if seed==SEEDS[0] and name=='semantics':
                original_test, original_theta = test,theta
        for target in ('event','observer'):
            for name in variants:
                if name=='geometry':continue
                delta=np.array([a['nll_bits']-b['nll_bits'] for a,b in zip(metrics['geometry',target],metrics[name,target])])
                run['contrasts'][f'{name}_over_geometry_{target}']={'improvement_bits':float(delta.mean()),'ci95':bootstrap_mean_ci(delta)}
        summary['runs'].append(run)
        (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    omission=[]
    for probability in (0.,.25,.5,.75,1.):
        draws=[]
        for repeat in range(20):
            altered=omit_semantics(original_test,probability,20261200+repeat)
            draws.append([r['nll_bits'] for r in rows(altered,predict(altered,original_theta))])
        per_image=np.mean(draws,axis=0)
        omission.append({'omission_probability':probability,'repetitions':20,'nll_bits':float(per_image.mean()),'nll_ci95':bootstrap_mean_ci(per_image)})
    summary.update(status='complete',semantic_omission=omission,elapsed_seconds=time.monotonic()-started)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (OUT/'per_image_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(csvrows[0]));writer.writeheader();writer.writerows(csvrows)
    print('Completed robustness study',flush=True)


if __name__=='__main__':main()
